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


SHORTFORM_SOURCES = ("nq", "triviaqa", "popqa", "webq")


def load_nq(n: int | None = None, split: str = "validation") -> list[Question]:
    from datasets import load_dataset
    spec = f"{split}[:{n}]" if n else split
    ds = load_dataset("google-research-datasets/nq_open", split=spec)
    return [
        Question(qid=f"nq_{i}", question=ex["question"], source="nq", gold_answers=list(ex["answer"]))
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
                             gold_answers=sorted(set(answers))))
    return out


def load_popqa(n: int | None = None, split: str = "test") -> list[Question]:
    from datasets import load_dataset
    spec = f"{split}[:{n}]" if n else split
    ds = load_dataset("akariasai/PopQA", split=spec)
    out = []
    for i, ex in enumerate(ds):
        answers = json.loads(ex["possible_answers"])
        out.append(Question(qid=f"popqa_{i}", question=ex["question"], source="popqa", gold_answers=answers))
    return out


def load_webq(n: int | None = None, split: str = "test") -> list[Question]:
    from datasets import load_dataset
    spec = f"{split}[:{n}]" if n else split
    ds = load_dataset("Stanford/web_questions", split=spec)
    return [
        Question(qid=f"webq_{i}", question=ex["question"], source="webq", gold_answers=list(ex["answers"]))
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
