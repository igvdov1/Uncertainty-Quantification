"""semantic_clusters без модели: NLI — мок по таблице, плюс энтропии реестра
против значений, посчитанных вручную по формулам jlko/semantic_uncertainty."""
from __future__ import annotations

import math

import numpy as np
import pytest

from dump_assembly import semantic_clusters as sc
from dump_assembly.semantic_clusters import CONTRADICTION as C, ENTAILMENT as E, NEUTRAL as N
from screener import registry


def test_equivalence_rules_match_reference():
    assert sc.equivalent_from_labels(E, E, strict=False)
    assert sc.equivalent_from_labels(E, N, strict=False)          # не строго: одна сторона neutral — ок
    assert not sc.equivalent_from_labels(N, N, strict=False)
    assert not sc.equivalent_from_labels(E, C, strict=False)
    assert not sc.equivalent_from_labels(E, N, strict=True)
    assert sc.equivalent_from_labels(E, E, strict=True)


def test_greedy_ids_non_transitive_case():
    # 0~1, 1~2, но 0 !~ 2: жадный алгоритм кладёт 2 в новый кластер (сравнивает с 0, не с 1)
    eq = {(0, 1), (1, 2)}
    ids = sc.get_semantic_ids(["a", "b", "c"], lambda i, j: (i, j) in eq)
    assert ids == [0, 0, 1]


class TableNLI:
    """Мок: 'Paris' ~ 'the capital is Paris', 'Lyon' противоречит обоим."""
    def __init__(self):
        self.seen = []

    def __call__(self, pairs):
        self.seen.extend(pairs)
        out = []
        for p, h in pairs:
            city_p, city_h = ("Paris" in p), ("Paris" in h)
            out.append(E if city_p == city_h else C)
        return out


def test_cluster_samples_uses_question_and_dedups():
    nli = TableNLI()
    samples = ["Paris.", "Paris.", "The capital is Paris.", "Lyon."]
    ids = sc.cluster_samples("Capital of France?", samples, nli)
    assert ids == [0, 0, 0, 1]
    assert all(p.startswith("Capital of France? ") for p, _ in nli.seen)
    assert len(nli.seen) == 4      # кластер «Paris»: 2 других уникальных текста x 2 стороны; «Lyon» — один в кластере
    assert sc.cluster_samples("q", [], nli) == []


def _rec(ids, lps):
    return {"qid": "x", "rag": {"samples": ["s"] * len(ids), "sample_logprobs": lps},
            "derived": {"semantic_clusters": {"rag": {"cluster_ids": ids}}}}


def test_semantic_entropy_matches_hand_computation():
    lps = [[-0.1, -0.3], [-0.2], [-2.0, -1.0], [-0.5]]
    ids = [0, 0, 1, 0]
    r = _rec(ids, lps)
    loglik = np.array([np.mean(l) for l in lps])
    w = np.exp(loglik) / np.exp(loglik).sum()
    p0, p1 = w[[0, 1, 3]].sum(), w[2]
    expected = -(p0 * math.log(p0) + p1 * math.log(p1))
    assert registry.semantic_entropy(r, "rag") == pytest.approx(expected)
    assert registry.semantic_entropy_discrete(r, "rag") == pytest.approx(-(0.75 * math.log(0.75) + 0.25 * math.log(0.25)))


def test_semantic_entropy_edge_cases():
    assert registry.semantic_entropy(_rec([0, 0, 0], [[-1.0], [-2.0], [-0.5]]), "rag") == pytest.approx(0.0)
    # пустой сэмпл выпадает из правдоподобия, остаётся один -> 0
    assert registry.semantic_entropy(_rec([0, 1], [[-1.0], []]), "rag") == 0.0
    with pytest.raises(IndexError):
        registry.semantic_entropy(_rec([0, 1], [[-1.0]]), "rag")
    with pytest.raises(KeyError):
        registry.semantic_entropy({"qid": "x", "rag": {"samples": [], "sample_logprobs": []}}, "rag")


def test_cluster_samples_matches_reference_greedy_ids():
    """Батч-версия даёт те же id, что эталонный get_semantic_ids с попарной эквивалентностью."""
    import random
    rng = random.Random(0)
    for _ in range(50):
        samples = [rng.choice(["a", "b", "c", "d", "ab", "ba"]) for _ in range(rng.randint(1, 12))]

        def nli(pairs):
            # «эквивалентны», если делят хотя бы одну букву (нетранзитивно — проверяет жадность)
            return [E if set(p.split()[-1]) & set(h.split()[-1]) else C for p, h in pairs]

        got = sc.cluster_samples("q", samples, nli)
        texts = [f"q {s}" for s in samples]
        ref = sc.get_semantic_ids(texts, lambda i, j: texts[i] == texts[j] or
                                  bool(set(samples[i]) & set(samples[j])))
        assert got == ref
