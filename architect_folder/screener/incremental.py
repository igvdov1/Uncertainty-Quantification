"""
Добавочная ценность сигнала поверх бесплатных бейзлайнов (решение 2026-10-05).

Одиночный AUROC не показывает, полезен ли сигнал в комбинации: слабый, но
ортогональный NLL сигнал (ncp_cross, QPP, retriever-disagreement) может
добавить больше, чем сильный, но дублирующий. Здесь — логистическая регрессия
«база» против «база + сигнал», AUROC на out-of-fold предсказаниях
(StratifiedKFold 5 x 5 повторов, предсказания усредняются по повторам), Δ
с парным bootstrap по записям.

Базы: len_norm_rag (NLL RAG-ответа) и len_norm_rag + mean_sample_len_rag
(NLL + длина). Признаки — ранги (риск-ориентация уже сделана в
run_screener.to_risk_scores): у NLL/расстояний тяжёлые хвосты, ранговое
преобразование без меток утечки не даёт.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from . import bootstrap, metrics

BASES = {"NLL": ["len_norm_rag"], "NLL+длина": ["len_norm_rag", "mean_sample_len_rag"]}
N_SPLITS, N_REPEATS = 5, 5


def _ranks(x: np.ndarray) -> np.ndarray:
    return (rankdata(x) - 0.5) / len(x)


def oof_risk(features: np.ndarray, labels: np.ndarray, seed: int = 0) -> np.ndarray:
    """Out-of-fold риск (P(ошибка)), усреднённый по повторам CV. labels: 1 = хорошо."""
    y = 1 - labels  # положительный класс — ошибка
    x = np.column_stack([_ranks(f) for f in features.T])
    out = np.zeros(len(y))
    for rep in range(N_REPEATS):
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed + rep)
        for tr, te in skf.split(x, y):
            clf = LogisticRegression(max_iter=1000).fit(x[tr], y[tr])
            out[te] += clf.predict_proba(x[te])[:, 1]
    return out / N_REPEATS


_BASE_CACHE: dict[tuple, np.ndarray] = {}


def _base_oof(risk: dict[str, np.ndarray], base: list[str], m: np.ndarray, lab: np.ndarray, seed: int) -> np.ndarray:
    """OOF базы одинаков для всех сигналов с той же маской — считаем один раз."""
    key = (tuple(base), m.tobytes(), lab.tobytes(), seed, *(risk[c][m].tobytes() for c in base))
    if key not in _BASE_CACHE:
        _BASE_CACHE[key] = oof_risk(np.column_stack([risk[c][m] for c in base]), lab, seed)
    return _BASE_CACHE[key]


def incremental_value(risk: dict[str, np.ndarray], labels: np.ndarray, signal: str,
                      n_boot: int = 1000, seed: int = 0) -> dict[str, dict] | None:
    cols = {c for base in BASES.values() for c in base} | {signal}
    if any(c not in risk for c in cols):
        return None
    m = labels >= 0
    for c in cols:
        m &= ~np.isnan(risk[c])
    if m.sum() < 50 or len(set(labels[m])) < 2:
        return None
    lab = labels[m]
    out = {}
    for name, base in BASES.items():
        if signal in base:
            continue
        r_base = _base_oof(risk, base, m, lab, seed)
        r_full = oof_risk(np.column_stack([risk[c][m] for c in base + [signal]]), lab, seed)
        bs = bootstrap.paired_bootstrap(r_full, r_base, lab, metrics.auroc, n_boot=n_boot, seed=seed)
        out[name] = {"base": metrics.auroc(r_base, lab), "with": metrics.auroc(r_full, lab),
                     "delta": bs["point_diff"], "ci": (bs["ci_low"], bs["ci_high"]),
                     "p_better": bs["p_a_better"], "n": int(m.sum())}
    return out
