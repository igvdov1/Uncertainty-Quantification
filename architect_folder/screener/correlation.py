"""
Корреляционная матрица сигналов (бриф, раздел 6) — обязательный артефакт,
основной инструмент финального синтеза: интересны сильные И
некоррелированные сигналы, а не просто сильные.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def spearman_matrix(signal_values: dict[str, np.ndarray]) -> tuple[list[str], np.ndarray]:
    """signal_values: {имя_сигнала: массив значений по вопросам, той же
    длины и порядка qid для всех сигналов}. NaN-записи (сигнал не
    посчитался для части вопросов) исключаются попарно.

    Возвращает (имена, матрица) — матрица[i][j] = Spearman rho(signal_i, signal_j).
    """
    names = list(signal_values.keys())
    n = len(names)
    matrix = np.full((n, n), np.nan)

    for i in range(n):
        xi = np.asarray(signal_values[names[i]], dtype=float)
        for j in range(i, n):
            xj = np.asarray(signal_values[names[j]], dtype=float)
            mask = ~(np.isnan(xi) | np.isnan(xj))
            if mask.sum() < 2 or np.std(xi[mask]) == 0 or np.std(xj[mask]) == 0:
                rho = float("nan")
            else:
                rho, _ = spearmanr(xi[mask], xj[mask])
            matrix[i, j] = rho
            matrix[j, i] = rho

    return names, matrix
