"""
GPU-инференс готовых моделей для карточек банка B (researcher_folder/B2_cards.md),
которым не нужны ни генерации LLM, ни обучение. Дорогая часть отдельным
проходом; результат — sidecar со скалярами на qid, скринер подмешивает его
через run_screener --sidecar и считает как обычные сигналы.

Карточки:
  selfcheckgpt-consistency  — SelfCheckGPT-NLI (Manakul et al. 2023), как
      SelfCheckNLI в их пакете: модель potsawee/deberta-v3-large-mnli
      (2 класса: entailment/contradiction), для каждого предложения жадного
      ответа P(contradiction | предложение, сэмпл), среднее по сэмплам, затем
      по предложениям. Выше = больше неопределённость.
  non-contradiction-probability — NCP из UQLM (Chen & Mueller 2024), по их
      описанию: microsoft/deberta-large-mnli, P(contradiction) в обе стороны
      между жадным ответом и каждым сэмплом, усредняется; NCP = среднее по
      сэмплам (1 - P(contradiction)). Выше = увереннее. Плюс вариант из
      карточки: жадный rag-ответ против closed-book сэмплов (спорит ли
      параметрическое знание с контекстным).
  lettucedetect-lightweight — LettuceDetect (arXiv 2502.17125), токен-
      классификатор KRLabsOrg/lettucedect-base-modernbert-en-v1 (обучен на
      RAGTruth): вероятность «галлюцинация» для каждого токена rag-ответа
      при данных пассажах. Скоры: max и mean по токенам, доля токенов с
      pred=1. Выше = больше неопределённость.

Запуск:
    # локально, CPU: дамп (ГБ) -> лёгкий вход
    python -m dump_assembly.gpu_cards extract --dump dump_pilot_v4.jsonl --out gpu_cards_input.jsonl
    # GPU (pip install lettucedetect sentencepiece protobuf)
    python -m dump_assembly.gpu_cards run --input gpu_cards_input.jsonl --out gpu_cards.jsonl
    # скринер
    python -m screener.run_screener --input dump_pilot_v4.jsonl \
        --sidecar semantic_clusters.jsonl --sidecar gpu_cards.jsonl --faithful-exclude-refusals

Выход, одна строка на запись: {"qid", "models": {...}, "signals": {имя: float}}.
Resume по qid: перезапуск досчитывает недостающие записи.

Оговорка по long-form: rag.answer у long-form — оригинальный текст FRANQ
(teacher-forced), а сэмплы — генерации нашей модели, обрезанные на 50 токенах.
Сравнение ответ-vs-сэмплы там описывает разные тексты; интерпретировать
отдельно (скринер и так печатает long/short раздельно).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Callable

import numpy as np

MODES = ("rag", "closed_book")
SELFCHECK_MODEL = "potsawee/deberta-v3-large-mnli"
NCP_MODEL = "microsoft/deberta-large-mnli"
LETTUCE_MODEL = "KRLabsOrg/lettucedect-base-modernbert-en-v1"

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def iter_jsonl(path: Path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def split_sentences(text: str) -> list[str]:
    """SelfCheckGPT режет ответ на предложения (у них spacy); здесь — регулярка
    по концу предложения и переводам строк (long-form ответы — markdown-списки)."""
    sents = [s.strip() for s in _SENT_SPLIT.split(text)]
    return [s for s in sents if len(s) > 1] or ([text.strip()] if text.strip() else [])


# ---- формулы карточек (без моделей) ---------------------------------------
# p_contra: список (premise, hypothesis) -> P(contradiction) для каждой пары

def selfcheck_nli(answer: str, samples: list[str], p_contra: Callable[[list[tuple[str, str]]], list[float]]) -> float:
    sents = split_sentences(answer)
    samples = [s for s in samples if s.strip()]
    if not sents or not samples:
        return float("nan")
    pairs = [(sent, smp) for sent in sents for smp in samples]
    probs = np.array(p_contra(pairs)).reshape(len(sents), len(samples))
    return float(probs.mean(axis=1).mean())


def non_contradiction(answer: str, samples: list[str], p_contra: Callable[[list[tuple[str, str]]], list[float]]) -> float:
    samples = [s for s in samples if s.strip()]
    if not answer.strip() or not samples:
        return float("nan")
    pairs = [(answer, s) for s in samples] + [(s, answer) for s in samples]
    p = np.array(p_contra(pairs))
    both = (p[:len(samples)] + p[len(samples):]) / 2
    return float(np.mean(1.0 - both))


def lettuce_scores(token_probs: list[dict]) -> dict[str, float]:
    if not token_probs:
        return {"lettuce_max": float("nan"), "lettuce_mean": float("nan"), "lettuce_frac": float("nan")}
    probs = np.array([t["prob"] for t in token_probs])
    preds = np.array([t["pred"] for t in token_probs])
    return {"lettuce_max": float(probs.max()), "lettuce_mean": float(probs.mean()),
            "lettuce_frac": float((preds == 1).mean())}


def card_signals(rec: dict, sc_contra, ncp_contra, lettuce_predict) -> dict[str, float]:
    out: dict[str, float] = {}
    for mode, suf in (("rag", "rag"), ("closed_book", "cb")):
        ans, smp = rec[mode]["answer"], rec[mode]["samples"]
        out[f"selfcheck_nli_{suf}"] = selfcheck_nli(ans, smp, sc_contra)
        out[f"ncp_{suf}"] = non_contradiction(ans, smp, ncp_contra)
    out["ncp_cross"] = non_contradiction(rec["rag"]["answer"], rec["closed_book"]["samples"], ncp_contra)
    if rec["passages"]:
        out.update(lettuce_scores(lettuce_predict(rec["passages"], rec["question"], rec["rag"]["answer"])))
    else:
        out.update(lettuce_scores([]))
    return out


# ---- модели --------------------------------------------------------------

class ContradictionNLI:
    """P(contradiction) из NLI-классификатора; индекс класса — по id2label."""

    # модели с безымянными классами (label_0/label_1): индекс contradiction.
    # potsawee/deberta-v3-large-mnli — как в SelfCheckNLI (probs[:, 1]; 0 = entailment)
    KNOWN_CONTRA_IDX = {"potsawee/deberta-v3-large-mnli": 1}

    def __init__(self, model_name: str, batch_size: int, device: str):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()
        if device == "cuda":
            self.model.half()
        labels = {i: l.lower() for i, l in self.model.config.id2label.items()}
        contra = [i for i, l in labels.items() if l.startswith("contra")]
        if not contra and model_name in self.KNOWN_CONTRA_IDX:
            contra = [self.KNOWN_CONTRA_IDX[model_name]]
        if len(contra) != 1:
            raise ValueError(f"{model_name}: не нашёл класс contradiction в {labels}")
        self.contra_idx = contra[0]
        self.batch_size = batch_size

    def __call__(self, pairs: list[tuple[str, str]]) -> list[float]:
        out: list[float] = []
        for start in range(0, len(pairs), self.batch_size):
            chunk = pairs[start:start + self.batch_size]
            enc = self.tokenizer([a for a, _ in chunk], [b for _, b in chunk], return_tensors="pt",
                                 padding=True, truncation=True, max_length=512).to(self.device)
            with self.torch.no_grad():
                probs = self.model(**enc).logits.float().softmax(dim=-1)[:, self.contra_idx]
            out.extend(probs.tolist())
        return out


class Lettuce:
    def __init__(self, model_name: str):
        from lettucedetect.models.inference import HallucinationDetector
        self.detector = HallucinationDetector(method="transformer", model_path=model_name)

    def __call__(self, passages: list[str], question: str, answer: str) -> list[dict]:
        return self.detector.predict(context=passages, question=question, answer=answer, output_format="tokens")


# ---- команды -------------------------------------------------------------

def cmd_extract(args) -> None:
    n = 0
    with open(args.out, "w") as fout:
        for r in iter_jsonl(args.dump):
            fout.write(json.dumps({
                "qid": r["qid"], "source": r.get("source"), "question": r["question"],
                "passages": [p["text"] for p in r.get("passages", [])],
                **{m: {"answer": r[m]["answer"], "samples": r[m].get("samples", [])} for m in MODES},
            }) + "\n")
            n += 1
    print(f"Готово: {n} записей -> {args.out}")


def cmd_run(args) -> None:
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    done = {j["qid"] for j in iter_jsonl(args.out)} if args.out.exists() else set()
    todo = [r for r in iter_jsonl(args.input) if r["qid"] not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(done)} уже посчитано, к обработке {len(todo)} (device={device})")
    if not todo:
        return
    sc = ContradictionNLI(args.selfcheck_model, args.batch_size, device)
    ncp = ContradictionNLI(args.ncp_model, args.batch_size, device)
    lettuce = Lettuce(args.lettuce_model)
    models = {"selfcheck": args.selfcheck_model, "ncp": args.ncp_model, "lettuce": args.lettuce_model}
    with open(args.out, "a") as fout:
        for i, r in enumerate(todo):
            fout.write(json.dumps({"qid": r["qid"], "models": models,
                                   "signals": card_signals(r, sc, ncp, lettuce)}) + "\n")
            if (i + 1) % 25 == 0:
                fout.flush()
                print(f"  {i + 1}/{len(todo)}")
    print(f"Готово -> {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("extract")
    p.add_argument("--dump", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.set_defaults(fn=cmd_extract)

    p = sub.add_parser("run")
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--selfcheck-model", default=SELFCHECK_MODEL)
    p.add_argument("--ncp-model", default=NCP_MODEL)
    p.add_argument("--lettuce-model", default=LETTUCE_MODEL)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--limit", type=int, default=None, help="смоук: первые N записей")
    p.set_defaults(fn=cmd_run)

    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
