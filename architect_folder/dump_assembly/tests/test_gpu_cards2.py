"""gpu_cards2 без моделей: наборы ответов и энтропии на мок-NLI."""
from __future__ import annotations

import numpy as np
import pytest

from dump_assembly import gpu_cards2 as g2
from dump_assembly.semantic_clusters import CONTRADICTION as C, ENTAILMENT as E


def city_nli(pairs):
    return [E if ("Paris" in a) == ("Paris" in b) else C for a, b in pairs]


def _rec(source="nq"):
    return {"qid": "q", "source": source, "question": "Capital?", "passages": ["Paris is the capital."],
            "rag_answer": "Paris", "rag_samples": ["Paris", "Paris"],
            "cb_answer": "Paris", "cb_samples": ["Lyon", "Paris"],
            "paraphrases": [{"cb_answer": "Lyon", "cb_samples": ["Lyon"], "rag_answer": "Paris", "rag_samples": ["Paris"]},
                            {"cb_answer": "Paris", "cb_samples": [], "rag_answer": None, "rag_samples": []}]}


def test_answer_sets_and_longform_has_no_rag_sets():
    s = g2.answer_sets(_rec())
    assert s["mem_greedy"] == ["Paris", "Lyon", "Paris"]
    assert len(s["mem_all"]) == 3 + 2 + 1 and s["sre_greedy"] == ["Paris", "Paris"]
    assert "sre_greedy" not in g2.answer_sets(_rec("franq_longform"))


def test_paraphrase_signals_entropies():
    sig = g2.paraphrase_signals(_rec(), city_nli)
    p = np.array([2, 1]) / 3
    assert sig["mem_entropy_greedy"] == pytest.approx(-(p * np.log(p)).sum())
    assert sig["sre_entropy_greedy"] == pytest.approx(0.0)
    lf = g2.paraphrase_signals(_rec("franq_longform"), city_nli)
    assert np.isnan(lf["sre_entropy_all"]) and not np.isnan(lf["mem_entropy_all"])


def test_cluster_entropy_edge():
    assert np.isnan(g2.cluster_entropy([0]))
    assert g2.cluster_entropy([0, 0, 0]) == 0.0
