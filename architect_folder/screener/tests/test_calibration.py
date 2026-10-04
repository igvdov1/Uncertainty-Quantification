"""Этап калибровки на смоделированных данных: гарантия LTT, ECE, Mondrian, traq."""
from __future__ import annotations

import numpy as np

from screener import calibration as cal


def _data(n, rng):
    risk = rng.normal(size=n)
    err = (rng.random(n) < 1 / (1 + np.exp(-2 * risk))).astype(int)   # риск честно предсказывает ошибку
    return risk, err


def test_ece_zero_for_perfect_and_positive_for_overconfident():
    rng = np.random.default_rng(0)
    p = rng.random(20000)
    err = (rng.random(20000) < p).astype(int)
    assert cal.ece(p, err) < 0.02
    assert cal.ece(np.full(20000, 0.05), err) > 0.3


def test_ltt_controls_selective_error_on_fresh_data():
    rng = np.random.default_rng(1)
    viol = []
    for _ in range(30):
        rd, ed = _data(224, rng)
        rt, et = _data(2000, rng)
        tau = cal.ltt_threshold(rd, ed, alpha=0.2)
        s = cal.selective(rt, et, tau)
        if s["coverage"] > 0:
            viol.append(s["sel_error"] > 0.2)
    assert np.mean(viol) <= 0.15                      # δ = 0.1 с запасом на шум симуляции


def test_ltt_abstains_when_alpha_unreachable():
    risk, err = np.zeros(100), np.ones(100, dtype=int)
    assert cal.ltt_threshold(risk, err, alpha=0.1) == -np.inf


def test_calibrate_signal_runs_with_groups():
    rng = np.random.default_rng(2)
    rd, ed = _data(224, rng)
    rt, et = _data(300, rng)
    gd, gt = np.array(["a", "b"] * 112), np.array(["a", "b"] * 150)
    golden = rng.random(300) < 0.6
    out = cal.calibrate_signal(rd, ed, rt, et, golden, gd, gt)
    assert 0.6 < out["auroc_raw"] < 0.95 and out["ece_platt"] < 0.15
    c = out["conformal"][0.2]
    assert 0 < c["coverage"] <= 1 and 0 <= c["coverage_mondrian"] <= 1


def test_traq_retrieval_coverage():
    rng = np.random.default_rng(3)

    def make(n):
        scores, golden = [], []
        for _ in range(n):
            s = sorted(rng.normal(size=5), reverse=True)
            g = [False] * 5
            if rng.random() < 0.8:
                g[rng.integers(0, 5)] = True
            scores.append(s)
            golden.append(g)
        return scores, golden

    sd, gd = make(400)
    st, gt = make(2000)
    out = cal.traq_retrieval(sd, gd, st, gt, rng.integers(0, 2, 2000), alpha=0.3)
    assert abs(out["coverage_test"] - 0.7) < 0.05
    assert cal.traq_retrieval(sd, gd, st, gt, rng.integers(0, 2, 2000), alpha=0.1)["tau"] == float("-inf")
