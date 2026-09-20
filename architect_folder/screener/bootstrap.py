"""
Парный bootstrap (бриф, раздел 7, "закладывается в A3, а не потом") —
сравнение двух методов на одних и тех же qid, а не по пересекающимся
доверительным интервалам.
"""
from __future__ import annotations

from typing import Callable

import numpy as np


def paired_bootstrap(
    score_a: np.ndarray,
    score_b: np.ndarray,
    label: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 1000,
    seed: int = 0,
) -> dict:
    """Ресэмплирует qid с возвращением n_boot раз, каждый раз пересчитывая
    metric_fn(score, label) для обоих методов на одной и той же выборке
    qid, и смотрит на распределение разности metric_a - metric_b.

    Возвращает: point-estimate разности (на полной выборке), среднюю
    разность по bootstrap, 95%-й доверительный интервал и долю
    ресэмплов, где метод A лучше метода B (аналог one-sided p-value).
    """
    score_a = np.asarray(score_a, dtype=float)
    score_b = np.asarray(score_b, dtype=float)
    label = np.asarray(label)
    n = len(label)
    if not (len(score_a) == len(score_b) == n):
        raise ValueError("score_a, score_b и label должны быть одной длины (одни qid)")

    rng = np.random.default_rng(seed)
    point_diff = metric_fn(score_a, label) - metric_fn(score_b, label)

    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        m_a = metric_fn(score_a[idx], label[idx])
        m_b = metric_fn(score_b[idx], label[idx])
        diffs[b] = m_a - m_b

    diffs = diffs[~np.isnan(diffs)]
    if len(diffs) == 0:
        return {
            "point_diff": point_diff,
            "mean_diff": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "p_a_better": float("nan"),
            "n_boot_valid": 0,
        }

    return {
        "point_diff": float(point_diff),
        "mean_diff": float(diffs.mean()),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
        "p_a_better": float((diffs > 0).mean()),
        "n_boot_valid": int(len(diffs)),
    }
