"""
A4 — загрузка и нормализация вопросов пилота, до ретривала и генерации.

Источники (решение — A4_dataset.md):
  - short-form: NQ-open, TriviaQA, PopQA, WebQuestions — те же четыре
    бенчмарка, на которых сам FRANQ тестирует short-form трек.
  - long-form: все 76 вопросов FRANQ (claim_level) — с уже готовыми
    пассажами (передаются напрямую, без ретривала: RAGTruth-источники,
    не Wikipedia).

Не требует GPU и не требует тяжёлого Wikipedia-корпуса — тестируется
локально. Ретривал для short-form (Contriever-MSMARCO + Wikipedia-2018)
и генерация — отдельные модули, им нужен кластер.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Question:
    qid: str
    question: str
    source: str                      # "nq" | "triviaqa" | "popqa" | "webq" | "franq_longform"
    gold_answers: list[str] = field(default_factory=list)   # short-form: список допустимых ответов
    passages: list[dict] | None = None                       # long-form: готовые пассажи (без ретривала)
    franq_raw: dict | None = None                             # long-form: оригинальная запись FRANQ целиком
                                                                # (answer, claims, auto_labels) — для teacher_force
                                                                # и переиспользования разметки, см. labeling.py
    split: str = "dev"
    meta: dict = field(default_factory=dict)                  # PopQA: prop, s_pop; все: row (индекс в сплите HF)


SHORTFORM_SOURCES = ("nq", "triviaqa", "popqa", "webq")


def load_nq(n: int | None = None, split: str = "validation") -> list[Question]:
    from datasets import load_dataset
    spec = f"{split}[:{n}]" if n else split
    ds = load_dataset("google-research-datasets/nq_open", split=spec)
    return [
        Question(qid=f"nq_{i}", question=ex["question"], source="nq", gold_answers=list(ex["answer"]),
                 meta={"row": i})
        for i, ex in enumerate(ds)
    ]


def load_triviaqa(n: int | None = None, split: str = "validation") -> list[Question]:
    from datasets import load_dataset
    spec = f"{split}[:{n}]" if n else split
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split=spec)
    out = []
    for i, ex in enumerate(ds):
        answers = [ex["answer"]["value"]] + list(ex["answer"].get("aliases", []))
        out.append(Question(qid=f"triviaqa_{i}", question=ex["question"], source="triviaqa",
                             gold_answers=sorted(set(answers)), meta={"row": i}))
    return out


def load_popqa(n: int | None = None, split: str = "test") -> list[Question]:
    from datasets import load_dataset
    spec = f"{split}[:{n}]" if n else split
    ds = load_dataset("akariasai/PopQA", split=spec)
    out = []
    for i, ex in enumerate(ds):
        answers = json.loads(ex["possible_answers"])
        out.append(Question(qid=f"popqa_{i}", question=ex["question"], source="popqa", gold_answers=answers,
                            meta={"prop": ex["prop"], "s_pop": int(ex["s_pop"]), "row": i}))
    return out


def load_webq(n: int | None = None, split: str = "test") -> list[Question]:
    from datasets import load_dataset
    spec = f"{split}[:{n}]" if n else split
    ds = load_dataset("Stanford/web_questions", split=spec)
    return [
        Question(qid=f"webq_{i}", question=ex["question"], source="webq", gold_answers=list(ex["answers"]),
                 meta={"row": i})
        for i, ex in enumerate(ds)
    ]


SHORTFORM_LOADERS = {"nq": load_nq, "triviaqa": load_triviaqa, "popqa": load_popqa, "webq": load_webq}


_PASSAGE_RE = re.compile(r"passage (\d+):", re.IGNORECASE)


def parse_franq_retrieval(retrieval_text: str) -> list[dict]:
    """'passage 1:...\\n\\npassage 2:...' -> [{"rank":1,"text":...}, ...]"""
    matches = list(_PASSAGE_RE.finditer(retrieval_text))
    passages = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(retrieval_text)
        text = retrieval_text[start:end].strip()
        passages.append({"rank": int(m.group(1)), "text": text, "doc_id": f"franq_p{m.group(1)}",
                          "is_golden": None, "score": None})  # не из ретривера — score недоступен
    return passages


def load_franq_longform(repo_claim_level_dataset_dir: Path, model_file: str = "Falcon3-3B-Base.json") -> list[Question]:
    with open(Path(repo_claim_level_dataset_dir) / model_file) as f:
        data = json.load(f)
    out = []
    for key in sorted((k for k in data.keys() if k.isdigit()), key=int):
        ex = data[key]
        passages = parse_franq_retrieval(ex["retrieval"])
        out.append(Question(
            qid=f"franq_lf_{key}",
            question=ex["question"].strip(),
            source="franq_longform",
            passages=passages,
            franq_raw=ex,
            split="dev",
        ))
    return out


_POOL_SIZE_PER_SOURCE = 1000  # фиксированный размер пула на источник — НЕ зависит от
                               # dev_size/test_size. Нужно, чтобы перемешивание было одним и тем
                               # же при разных вызовах, и меньшая выборка (смоук) всегда была
                               # префиксом большей (реальный прогон пилота) — иначе смоук-тест
                               # тянет qid, для которых ретривал ещё не посчитан (найдено на
                               # практике: --dev-size 96 против уже готового ретривала под 300/300).


def build_pilot_question_set(
    franq_dataset_dir: Path,
    dev_size: int = 300,
    test_size: int = 300,
    seed: int = 0,
) -> list[Question]:
    """Состав пилота — A4_dataset.md: все 76 long-form FRANQ в dev целиком,
    остаток dev+test поровну между 4 short-form источниками.

    Вложенность подмножеств: при одном seed выборка для dev_size=96 —
    префикс выборки для dev_size=300 (см. _POOL_SIZE_PER_SOURCE)."""
    longform = load_franq_longform(franq_dataset_dir)
    assert len(longform) == 76, f"ожидались все 76 вопросов FRANQ, получено {len(longform)}"

    remaining_dev = dev_size - len(longform)
    if remaining_dev < 0:
        raise ValueError(f"dev_size={dev_size} меньше, чем 76 long-form вопросов")

    per_source_dev = remaining_dev // len(SHORTFORM_SOURCES)
    per_source_test = test_size // len(SHORTFORM_SOURCES)
    if per_source_dev + per_source_test > _POOL_SIZE_PER_SOURCE:
        raise ValueError(
            f"на источник нужно {per_source_dev + per_source_test} вопросов, "
            f"больше фиксированного пула {_POOL_SIZE_PER_SOURCE} — увеличьте _POOL_SIZE_PER_SOURCE"
        )

    shortform_dev, shortform_test = [], []
    for source in SHORTFORM_SOURCES:
        rng = np.random.default_rng(seed)  # свой генератор на источник, не завязан на порядок цикла
        pool = SHORTFORM_LOADERS[source](n=_POOL_SIZE_PER_SOURCE)
        idx = rng.permutation(len(pool))
        pool = [pool[i] for i in idx]
        dev_part, test_part = pool[:per_source_dev], pool[per_source_dev:per_source_dev + per_source_test]
        for q in dev_part:
            q.split = "dev"
        for q in test_part:
            q.split = "test"
        shortform_dev += dev_part
        shortform_test += test_part

    return longform + shortform_dev + shortform_test


# ---- расширенный набор short-form (C9 §0, §11, §16) -------------------------
# Пилотный build_pilot_question_set оставлен как есть — он воспроизводит v1–v5.
# Новый отбор: пул — ВЕСЬ сплит (а не первые 1000 строк), дубли вопросов
# выкинуты, PopQA стратифицирован по 16 отношениям и квантилям популярности
# s_pop, без long-form. Результат пишется в questions.jsonl — единственный
# источник вопросов и gold для всех следующих шагов (C9 §12: никто больше не
# пересэмплирует вопросы заново).

N_POP_BINS = 5


def _norm_question(text: str) -> str:
    return " ".join(text.lower().split())


def dedupe_questions(pool: list[Question]) -> list[Question]:
    """Убирает повторы одного и того же вопроса (сравнение без регистра и
    лишних пробелов) и вопросы без gold; оставляет первое вхождение."""
    seen, out = set(), []
    for q in pool:
        key = _norm_question(q.question)
        if key in seen or not q.gold_answers:
            continue
        seen.add(key)
        out.append(q)
    return out


def allocate_evenly(capacity: dict, n: int) -> dict:
    """Делит n поровну между группами с ограничением ёмкости: группе, где
    вопросов меньше доли, отдаётся всё, остаток — поровну остальным."""
    alloc = {g: 0 for g in capacity}
    left, open_groups = n, sorted(g for g in capacity if capacity[g] > 0)
    while left > 0 and open_groups:
        share = max(left // len(open_groups), 1)
        for g in list(open_groups):
            take = min(share, capacity[g] - alloc[g], left)
            alloc[g] += take
            left -= take
            if alloc[g] >= capacity[g]:
                open_groups.remove(g)
            if left == 0:
                break
    return alloc


def stratified_popqa(pool: list[Question], n: int, rng: np.random.Generator,
                     n_bins: int = N_POP_BINS) -> list[Question]:
    """Поровну по отношениям (prop), внутри отношения — поровну по квантилям
    s_pop (квантили считаются внутри отношения)."""
    by_prop: dict[str, list[Question]] = {}
    for q in pool:
        by_prop.setdefault(q.meta["prop"], []).append(q)
    per_prop = allocate_evenly({p: len(v) for p, v in by_prop.items()}, n)
    chosen = []
    for prop, qs in sorted(by_prop.items()):
        qs = sorted(qs, key=lambda q: (q.meta["s_pop"], q.qid))
        bins = [list(b) for b in np.array_split(np.arange(len(qs)), min(n_bins, len(qs)))]
        per_bin = allocate_evenly({i: len(b) for i, b in enumerate(bins)}, per_prop[prop])
        for i, b in enumerate(bins):
            for j in rng.permutation(len(b))[: per_bin[i]]:
                q = qs[b[j]]
                q.meta["pop_bin"] = i
                chosen.append(q)
    return chosen


def build_question_set(per_source: int = 2000, dev_frac: float = 0.4, seed: int = 0,
                       sources: tuple = SHORTFORM_SOURCES, loaders: dict | None = None) -> list[Question]:
    """per_source вопросов из каждого источника (или весь сплит, если он
    меньше), dev_frac из них — dev, остальное — test. Порядок результата
    перемешан с seed: --limit на последующих шагах даёт смешанную подвыборку
    всех источников и сплитов (для смоука)."""
    loaders = loaders or SHORTFORM_LOADERS
    out = []
    for source in sources:
        rng = np.random.default_rng([seed, sorted(SHORTFORM_SOURCES).index(source)])
        pool = dedupe_questions(loaders[source](n=None))
        if source == "popqa":
            chosen = stratified_popqa(pool, min(per_source, len(pool)), rng)
        else:
            chosen = [pool[i] for i in rng.permutation(len(pool))[:per_source]]
        chosen = [chosen[i] for i in rng.permutation(len(chosen))]
        n_dev = int(round(dev_frac * len(chosen)))
        for i, q in enumerate(chosen):
            q.split = "dev" if i < n_dev else "test"
        out += chosen
    rng = np.random.default_rng(seed)
    return [out[i] for i in rng.permutation(len(out))]


def question_to_json(q: Question) -> dict:
    return {"qid": q.qid, "question": q.question, "source": q.source, "split": q.split,
            "gold_answers": q.gold_answers, "meta": q.meta}


def write_question_file(questions: list[Question], path: Path) -> None:
    with open(path, "w") as f:
        for q in questions:
            f.write(json.dumps(question_to_json(q), ensure_ascii=False) + "\n")


def read_question_file(path: Path, limit: int | None = None) -> list[Question]:
    out = []
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out.append(Question(qid=r["qid"], question=r["question"], source=r["source"],
                                    gold_answers=r["gold_answers"], split=r["split"], meta=r.get("meta", {})))
    return out[:limit] if limit else out


def main() -> None:
    """python -m dump_assembly.questions --per-source 2000 --dev-frac 0.4 --out questions.jsonl"""
    import argparse
    from collections import Counter

    parser = argparse.ArgumentParser()
    parser.add_argument("--per-source", type=int, default=2000)
    parser.add_argument("--dev-frac", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    qs = build_question_set(args.per_source, args.dev_frac, args.seed)
    write_question_file(qs, args.out)
    print(f"Готово: {len(qs)} вопросов -> {args.out}")
    print("  по источникам и сплитам:", dict(sorted(Counter((q.source, q.split) for q in qs).items())))
    props = Counter(q.meta["prop"] for q in qs if q.source == "popqa")
    print(f"  PopQA: {len(props)} отношений, min {min(props.values(), default=0)}, max {max(props.values(), default=0)}")


if __name__ == "__main__":
    main()
