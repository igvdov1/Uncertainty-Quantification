"""local_cards на синтетических векторах: Махаланобис, порог SEP, split-логика."""
from __future__ import annotations

import numpy as np
import pytest

from dump_assembly import local_cards as lc


def test_best_split_threshold_separates_two_modes():
    v = np.array([0.1, 0.2, 0.15, 2.0, 2.1, 1.9])
    t = lc.best_split_threshold(v)
    assert 0.2 < t < 1.9


def test_mahalanobis_larger_for_outliers():
    rng = np.random.default_rng(0)
    dev = rng.normal(size=(200, 30))
    md = lc.Mahalanobis(n_components=10).fit(dev)
    near, far = md.score(rng.normal(size=(20, 30))), md.score(rng.normal(size=(20, 30)) + 4)
    assert far.mean() > near.mean() * 1.5


def _feat(qid, split, source, rng, shift, se):
    f = {"qid": qid, "split": split, "source": source}
    for suf in ("rag", "cb"):
        v = rng.normal(size=16) + shift
        f[f"mean_{suf}"], f[f"slt_{suf}"], f[f"se_{suf}"] = v, v, se
    return f


def test_compute_fits_on_dev_and_scores_only_short_test():
    rng = np.random.default_rng(1)
    feats = []
    for i in range(60):
        hi = i % 2
        feats.append(_feat(f"d{i}", "dev", "nq", rng, 3 * hi, 2.0 if hi else 0.1))
    for i in range(20):
        hi = i % 2
        feats.append(_feat(f"t{i}", "test", "nq", rng, 3 * hi, 2.0 if hi else 0.1))
    feats.append(_feat("lf", "dev", lc.LONGFORM, rng, 0, 1.0))
    rows, rep = lc.compute(feats)
    assert {r["qid"] for r in rows} == {f"t{i}" for i in range(20)}
    assert rep["n_dev"] == 60 and rep["n_test"] == 20
    assert rep["sep_rag_spearman_with_se_test"] > 0.7
    s = {r["qid"]: r["signals"] for r in rows}
    assert s["t1"]["sep_rag"] > s["t0"]["sep_rag"]
    assert s["t0"]["md_diff"] == pytest.approx(s["t0"]["md_rag"] - s["t0"]["md_cb"])


def test_cp_internal_report_runs():
    rng = np.random.default_rng(3)
    feats = []
    for i in range(160):
        lab = i % 2
        split = "dev" if i < 80 else "test"
        v = rng.normal(size=16) + 2 * (1 - lab)
        feats.append({"qid": f"q{i}", "split": split, "source": "nq", "label_factual": lab,
                      "len_norm_rag": float(rng.normal() + (1 - lab)), "mean_rag": v})
    rep = lc.cp_internal_report(feats, alphas=(0.3,))
    assert rep["n"] == {"A": 40, "B": 40, "test": 80}
    assert set(rep) >= {"hidden (Махаланобис)", "logprob (NLL)"}
