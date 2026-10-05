"""rebuild_v2 без модели: подготовка perturbations и Question из записи v4."""
from __future__ import annotations

import pytest

from dump_assembly import rebuild_v2 as rb


def test_strip_paraphrase_answers_keeps_retrieval():
    pert = {"query_paraphrases": [{"text": "p", "topk_doc_ids": ["1"], "topk_scores": [0.5],
                                   "cb_answer": "x", "cb_samples": ["x"], "rag_answer": "y", "rag_samples": []}],
            "retriever_alt": {"bm25": {"topk_doc_ids": ["2"]}}}
    out = rb.strip_paraphrase_answers(pert)
    assert out["query_paraphrases"][0] == {"text": "p", "topk_doc_ids": ["1"], "topk_scores": [0.5]}
    assert out["retriever_alt"] == pert["retriever_alt"]
    assert pert["query_paraphrases"][0]["cb_answer"] == "x"          # исходник не мутирован


def test_question_from_record_requires_gold():
    r = {"qid": "nq_1", "question": "q?", "source": "nq", "split": "test", "passages": [{"text": "t"}]}
    q = rb.question_from_record(r, {"nq_1": ["a"]})
    assert q.gold_answers == ["a"] and q.split == "test" and q.passages == [{"text": "t"}]
    with pytest.raises(KeyError):
        rb.question_from_record(r, {})
