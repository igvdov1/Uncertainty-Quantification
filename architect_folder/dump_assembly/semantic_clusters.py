"""
Семантические кластеры сэмплов для semantic entropy — дорогая (NLI) часть,
отдельным проходом поверх собранного дампа. Сама энтропия считается дёшево
в скринере (screener/registry.py) из cluster_ids + sample_logprobs.

Метод — Kuhn, Gal, Farquhar, "Semantic Uncertainty", ICLR 2023; версия
Farquhar et al., "Detecting hallucinations in large language models using
semantic entropy", Nature 2024, референсный код
github.com/jlko/semantic_uncertainty. Повторяем их дефолты:
  - NLI: microsoft/deberta-v2-xlarge-mnli (их EntailmentDeberta);
  - к ответу приписывается вопрос (их --condition_on_question);
  - эквивалентность не строгая: ни в одну сторону нет contradiction и
    не обе стороны neutral (strict_entailment=False); --strict — только
    entailment в обе стороны;
  - жадная кластеризация как в их get_semantic_ids.
Отступление: побайтно одинаковые сэмплы считаются эквивалентными без
вызова NLI (на short-form их много) — на результат практически не влияет,
экономит большую часть вызовов.

Запуск:
    # локально, CPU: из дампа (ГБ) -> лёгкий файл только с вопросами и сэмплами
    python -m dump_assembly.semantic_clusters extract \
        --dump dump_pilot_v3.jsonl --out se_input.jsonl

    # на GPU
    python -m dump_assembly.semantic_clusters cluster \
        --input se_input.jsonl --out semantic_clusters.jsonl

    # скринер подхватывает sidecar
    python -m screener.run_screener --input dump_pilot_v4.jsonl --sidecar semantic_clusters.jsonl

Выход: одна строка на запись
    {"qid", "nli_model", "strict", "condition_on_question",
     "rag": {"cluster_ids": [...]}, "closed_book": {"cluster_ids": [...]}}
cluster_ids выровнены с {mode}.samples дампа.

Оговорка по long-form: сэмплы в дампе обрезаны MAX_NEW_TOKENS=50, а метод
создан для коротких ответов — на long-form кластеры описывают начало
ответа, не весь ответ. Считаем, но интерпретируем отдельно.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

MODES = ("rag", "closed_book")
DEFAULT_NLI = "microsoft/deberta-v2-xlarge-mnli"
CONTRADICTION, NEUTRAL, ENTAILMENT = 0, 1, 2


def iter_jsonl(path: Path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# ---- алгоритм (без модели) ---------------------------------------------

def equivalent_from_labels(l12: int, l21: int, strict: bool) -> bool:
    """Как в jlko/semantic_uncertainty (get_semantic_ids.are_equivalent)."""
    if strict:
        return l12 == ENTAILMENT and l21 == ENTAILMENT
    return CONTRADICTION not in (l12, l21) and (l12, l21) != (NEUTRAL, NEUTRAL)


def get_semantic_ids(strings: list[str], equivalent: Callable[[int, int], bool]) -> list[int]:
    """Жадная кластеризация (их get_semantic_ids): i-й ещё не размеченный
    сэмпл открывает новый кластер и забирает всех последующих, эквивалентных
    ему. equivalent(i, j) — по индексам."""
    ids = [-1] * len(strings)
    next_id = 0
    for i in range(len(strings)):
        if ids[i] != -1:
            continue
        ids[i] = next_id
        for j in range(i + 1, len(strings)):
            if ids[j] == -1 and equivalent(i, j):
                ids[j] = next_id
        next_id += 1
    return ids


def needed_pairs(texts: list[str]) -> list[tuple[str, str]]:
    """Все упорядоченные пары различных текстов — с запасом покрывает любые
    сравнения, которые сделает get_semantic_ids. Считать их одним батчем
    быстрее, чем гонять NLI последовательно внутри жадного цикла."""
    uniq = list(dict.fromkeys(texts))
    return [(a, b) for a in uniq for b in uniq if a != b]


def cluster_samples(question: str, samples: list[str], nli_predict: Callable[[list[tuple[str, str]]], list[int]],
                    strict: bool = False, condition_on_question: bool = True) -> list[int]:
    """nli_predict: список (premise, hypothesis) -> метки CONTRADICTION/NEUTRAL/ENTAILMENT."""
    if not samples:
        return []
    texts = [f"{question} {s}" if condition_on_question else s for s in samples]
    pairs = needed_pairs(texts)
    labels = dict(zip(pairs, nli_predict(pairs))) if pairs else {}

    def equivalent(i: int, j: int) -> bool:
        a, b = texts[i], texts[j]
        if a == b:
            return True
        return equivalent_from_labels(labels[(a, b)], labels[(b, a)], strict)

    return get_semantic_ids(texts, equivalent)


# ---- NLI-модель ----------------------------------------------------------

class DebertaNLI:
    def __init__(self, model_name: str, batch_size: int, device: str | None = None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device).eval()
        if self.device == "cuda":
            self.model.half()
        self.batch_size = batch_size
        # индекс выхода модели -> наша метка; у MNLI-чекпоинтов порядок разный
        id2label = {i: l.lower() for i, l in self.model.config.id2label.items()}
        name_to_ours = {"contradiction": CONTRADICTION, "neutral": NEUTRAL, "entailment": ENTAILMENT}
        self.out_to_label = {i: name_to_ours[l] for i, l in id2label.items()}

    def __call__(self, pairs: list[tuple[str, str]]) -> list[int]:
        out: list[int] = []
        for start in range(0, len(pairs), self.batch_size):
            chunk = pairs[start:start + self.batch_size]
            enc = self.tokenizer([p for p, _ in chunk], [h for _, h in chunk], return_tensors="pt",
                                 padding=True, truncation=True, max_length=512).to(self.device)
            with self.torch.no_grad():
                pred = self.model(**enc).logits.argmax(dim=-1).tolist()
            out.extend(self.out_to_label[i] for i in pred)
        return out


# ---- команды ------------------------------------------------------------

def cmd_extract(args) -> None:
    n = 0
    with open(args.out, "w") as fout:
        for r in iter_jsonl(args.dump):
            fout.write(json.dumps({
                "qid": r["qid"], "source": r.get("source"), "question": r["question"],
                **{m: {"samples": r[m].get("samples", [])} for m in MODES},
            }) + "\n")
            n += 1
    print(f"Готово: {n} записей -> {args.out}")


def cmd_cluster(args) -> None:
    done = {j["qid"] for j in iter_jsonl(args.out)} if args.out.exists() else set()
    todo = [r for r in iter_jsonl(args.input) if r["qid"] not in done]
    print(f"{len(done)} уже посчитано, к обработке {len(todo)}")
    if not todo:
        return
    nli = DebertaNLI(args.nli_model, args.batch_size)
    condition = not args.no_condition_on_question
    with open(args.out, "a") as fout:
        for i, r in enumerate(todo):
            row = {"qid": r["qid"], "nli_model": args.nli_model, "strict": args.strict,
                   "condition_on_question": condition}
            for m in MODES:
                row[m] = {"cluster_ids": cluster_samples(r["question"], r[m]["samples"], nli,
                                                         strict=args.strict, condition_on_question=condition)}
            fout.write(json.dumps(row) + "\n")
            if (i + 1) % 50 == 0:
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

    p = sub.add_parser("cluster")
    p.add_argument("--input", required=True, type=Path, help="se_input.jsonl из extract (или сам дамп)")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--nli-model", default=DEFAULT_NLI)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--strict", action="store_true", help="эквивалентность = entailment в обе стороны")
    p.add_argument("--no-condition-on-question", action="store_true")
    p.set_defaults(fn=cmd_cluster)

    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
