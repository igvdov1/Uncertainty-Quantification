"""
A4 — разметка label_faithful / label_factual (dump_schema_v1.md).

Two-track design (см. A4_dataset.md):
  - short-form (NQ/TriviaQA/PopQA/WebQuestions): factuality — сверка с
    gold-answer (EM/F1, стандартная QA-нормализация); faithfulness — NLI
    ответа против retrieved-пассажей.
  - long-form (FRANQ, 76 вопросов): переиспользуем их собственную
    разметку целиком (экспертная + GPT-4o-фактчекинг) — ничего не
    считаем заново, только парсим их формат. Ответ — их оригинальный
    текст, teacher-forced через нашу модель (generation.teacher_force).

V1: faithfulness через общедоступную NLI-модель (не AlignScore — тот
отдельный репозиторий, RoBERTa, дообученная специально под faithfulness;
интеграция — следующий шаг, см. cluster_runbook.md). Factuality для
short-form — EM/F1 против gold, не LLM-judge — тоже следующий шаг,
откалиброванный на 76 вопросах FRANQ (см. A4_dataset.md).
"""
from __future__ import annotations

import re
import string
from dataclasses import dataclass

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_WHITESPACE = re.compile(r"\s+")
_AUTO_LABEL_RE = re.compile(r'\("([^"]+)",\s*"(True|False)"\)')


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = _ARTICLES.sub(" ", text)
    text = _WHITESPACE.sub(" ", text).strip()
    return text


def exact_match(prediction: str, gold_answers: list[str]) -> bool:
    pred_norm = normalize_answer(prediction)
    return any(pred_norm == normalize_answer(g) for g in gold_answers)


def f1_score(prediction: str, gold: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = set(pred_tokens) & set(gold_tokens)
    n_common = sum(min(pred_tokens.count(t), gold_tokens.count(t)) for t in common)
    if n_common == 0:
        return 0.0
    precision = n_common / len(pred_tokens)
    recall = n_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def factuality_shortform(prediction: str, gold_answers: list[str], f1_threshold: float = 0.5) -> int:
    """1 = факт верен. Основной критерий — gold-ответ СОДЕРЖИТСЯ в
    сгенерированном тексте (Asai et al. 2024, тот же метод, что описан в
    ragu README про --acc: 'whether the gold answer is contained in the
    generated answer'). Не SQuAD-style F1/точное совпадение всей строки —
    наша модель отвечает полным предложением ("The color ... is white."),
    а не extractive-спаном, и token-level F1 против короткого gold почти
    всегда топит precision даже при полностью верном ответе.

    Найдено на реальном дампе (dump_pilot.jsonl): со старой F1-логикой
    ВСЕ 524 short-form записи получили label_factual=0 без исключений,
    включая явно правильные ответы (напр. ответ содержит "white",
    gold=["white"], но F1 между всем предложением и одним словом < 0.5).
    Это же объясняло аномальный AUROC на A5 — сигнал мерил не
    factual/non-factual, а long-form/short-form (полный конфаунд)."""
    if exact_match(prediction, gold_answers):
        return 1
    pred_norm = normalize_answer(prediction)
    for g in gold_answers:
        if normalize_answer(g) in pred_norm:
            return 1
    best_f1 = max((f1_score(prediction, g) for g in gold_answers), default=0.0)
    return int(best_f1 >= f1_threshold)


@dataclass
class NLIFaithfulnessScorer:
    """Обёртка над общедоступной NLI-моделью. max_entailment по всем
    пассажам — то же, что MaxNLI в FRANQ (§2.2 статьи), только без
    интеграции их конкретного чекпоинта."""
    pipeline: object  # transformers.pipeline("text-classification", ...)
    entail_label: str = "ENTAILMENT"

    def score(self, answer: str, passages: list[str]) -> float:
        """max по пассажам P(entailment), а не "top-1 метка была entailment".
        Пайплайн по умолчанию отдаёт только argmax-метку — для многотемного
        long-form ответа против одного узкого пассажа top-1 почти всегда
        'neutral', даже когда реальная entailment-вероятность заметна
        (например 0.04, а не 0). Со старым кодом (проверка top-1) это давало
        alignscore=0.0 систематически на long-form записях — найдено на
        реальном дампе (dump_smoke20.jsonl), не на синтетике."""
        if not passages:
            return float("nan")
        best = 0.0
        for p in passages:
            results = self.pipeline(f"{p}", text_pair=answer, truncation=True, top_k=None)
            for r in results:
                if r["label"].upper().startswith(self.entail_label[:3]):
                    best = max(best, r["score"])
        return best

    def label(self, answer: str, passages: list[str], threshold: float = 0.5) -> int:
        s = self.score(answer, passages)
        return int(s >= threshold) if s == s else 0  # NaN-safe


def load_nli_scorer(device: str = "cpu") -> NLIFaithfulnessScorer:
    from transformers import pipeline
    pipe = pipeline(
        "text-classification",
        model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
        device=device,
    )
    return NLIFaithfulnessScorer(pipeline=pipe, entail_label="ENTAILMENT")


# --- long-form: парсинг готовой разметки FRANQ ------------------------

def parse_franq_auto_label(auto_label: str) -> tuple[int | None, int | None]:
    """auto_labels[i] у FRANQ — '(\"faithful\"|\"unfaithful-contra\"|
    \"unfaithful-neutral\", \"True\"|\"False\")', иногда с рассуждением
    перед скобками (берём последнее совпадение). Возвращает
    (label_faithful, label_factual), None если не разобралось."""
    matches = _AUTO_LABEL_RE.findall(auto_label)
    if not matches:
        return None, None
    faith_str, fact_str = matches[-1]
    label_faithful = 1 if faith_str == "faithful" else 0
    label_factual = 1 if fact_str == "True" else 0
    return label_faithful, label_factual


def franq_longform_claims(ex: dict) -> list[dict]:
    """ex — запись из claim_level/dataset/*.json (одна запись = один
    вопрос). Возвращает claims в формате dump_schema_v1 (без token_span —
    он у FRANQ per-claim не выровнен на наши токены; наш teacher_force
    токенизирует их текст заново, спаны клеймов на нём не совпадают
    один-в-один с их span-разметкой — известное ограничение V1, чинится
    выравниванием текста клейма на новые токены при необходимости)."""
    claims = []
    for i, text in enumerate(ex.get("decoded_claims", ex.get("claims", []))):
        lf, lfact = parse_franq_auto_label(ex["auto_labels"][i])
        if lf is None:
            continue
        claims.append({"cid": f"{ex.get('_qid', '?')}_c{i}", "text": text,
                        "label_faithful": lf, "label_factual": lfact})
    return claims


def franq_longform_answer_labels(claims: list[dict]) -> tuple[int, int]:
    """Агрегация уровня ответа из клеймов: faithful/factual, если ВСЕ
    клеймы такие (консервативно — один плохой клейм портит весь ответ,
    та же логика, что подразумевает бинарный label_faithful/label_factual
    на уровне записи в dump_schema_v1)."""
    if not claims:
        return 1, 1
    label_faithful = int(all(c["label_faithful"] == 1 for c in claims))
    label_factual = int(all(c["label_factual"] == 1 for c in claims))
    return label_faithful, label_factual
