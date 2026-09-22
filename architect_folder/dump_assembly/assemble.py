"""
A4 — сборка одной записи дампа (dump_schema_v1.md) из question + результатов
retrieval/generation/labeling. Не содержит тяжёлой логики сама по себе —
только склейка того, что уже посчитали generation.py/retrieval.py/labeling.py
в нужный JSON-объект.

CLI: run_assembly.py (в этой же папке).
"""
from __future__ import annotations

from .questions import Question


def assemble_record(
    q: Question,
    rag_fields: dict,
    cb_fields: dict,
    samples_rag: dict,
    samples_cb: dict,
    paraphrases: list[dict],
    retriever_alt: dict,
    passages_top20_scores: list[float],
    tokenizer_name: str,
    label_faithful: int,
    label_factual: int,
    claims: list[dict],
    signals: dict,
) -> dict:
    rag = {**rag_fields, **samples_rag, "tokenizer_name": tokenizer_name}
    closed_book = {**cb_fields, **samples_cb, "tokenizer_name": tokenizer_name}
    # closed-book по схеме не несёт prompt_spans/attention (не RAG-специфичные поля)
    closed_book.pop("prompt_spans", None)
    closed_book.pop("prompt_raw", None)
    closed_book.pop("attention_by_group", None)
    closed_book.pop("attention_by_passage", None)
    closed_book.pop("attention_by_head", None)

    record = {
        "qid": q.qid,
        "question": q.question,
        "split": q.split,
        "source": q.source,  # доп. поле, не было в исходной схеме брифа — дёшево добавить сейчас

        "passages": q.passages or [],
        "passages_top20_scores": passages_top20_scores,
        "passage_embeddings": rag_fields.pop("passage_embeddings", []),

        "rag": rag,
        "closed_book": closed_book,

        "signals": signals,

        "perturbations": {
            "query_paraphrases": paraphrases,
            "retriever_alt": retriever_alt,
        },

        "claims": claims,

        "label_faithful": label_faithful,
        "label_factual": label_factual,
    }
    return record
