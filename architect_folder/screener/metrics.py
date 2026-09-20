"""
Метрики дискриминации и калибровки (бриф, раздел 6).

Соглашение по направлению: везде на вход подаётся risk-score — чем выше,
тем более вероятна ошибка (unfaithful / non-factual, label == 0).
Сигналы с полярностью "confidence" (см. registry.signal_polarity)
инвертируются перед вызовом этих функций — это делает
run_screener.compute_metrics_table, сами функции ничего не знают про
полярность.

label: 1 = хорошо (faithful/factual), 0 = плохо. Это то же соглашение,
что и в схеме дампа (label_faithful, label_factual).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import kendalltau
from sklearn.metrics import roc_auc_score


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    """np.trapz исчез в numpy>=2.0 (переименован в trapezoid, но не везде
    есть) — считаем вручную, чтобы не зависеть от версии numpy на кластере."""
    y, x = np.asarray(y, dtype=float), np.asarray(x, dtype=float)
    return float(np.sum((y[1:] + y[:-1]) / 2.0 * np.diff(x)))


def auroc(risk_score: np.ndarray, label: np.ndarray) -> float:
    """AUROC для задачи 'risk_score отличает label==0 от label==1'.
    Положительный класс — ошибка (label==0), как в Table 2 FRANQ."""
    y = 1 - np.asarray(label)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, risk_score))


def risk_coverage_curve(risk_score: np.ndarray, label: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Сортирует по возрастанию риска (сначала оставляем самые уверенные
    ответы) и возвращает (coverage, risk) — риск среди принятых на каждом
    уровне покрытия. label==0 считается ошибкой."""
    order = np.argsort(risk_score)
    errors = (1 - np.asarray(label))[order]
    n = len(errors)
    cum_errors = np.cumsum(errors)
    coverage = np.arange(1, n + 1) / n
    risk = cum_errors / np.arange(1, n + 1)
    return coverage, risk


def aurc(risk_score: np.ndarray, label: np.ndarray) -> float:
    """Area Under Risk-Coverage curve — чем меньше, тем лучше."""
    coverage, risk = risk_coverage_curve(risk_score, label)
    return float(_trapz(risk, coverage))


def prr(risk_score: np.ndarray, label: np.ndarray) -> float:
    """Prediction Rejection Ratio: (AURC_random - AURC_model) / (AURC_random - AURC_oracle).
    AURC_random — площадь при случайном ранжировании (общий risk на всех
    уровнях покрытия), AURC_oracle — при идеальном ранжировании (все
    ошибки отброшены первыми)."""
    y_err = 1 - np.asarray(label)
    n = len(y_err)
    base_risk = y_err.mean()
    aurc_random = base_risk  # random ranking: risk постоянен на всех покрытиях
    aurc_model = aurc(risk_score, label)

    n_err = int(y_err.sum())
    # оракул: сначала все правильные (risk=0), ошибки откладываются в самый конец
    oracle_order = np.concatenate([np.zeros(n - n_err), np.ones(n_err)])
    cum_errors = np.cumsum(oracle_order)
    coverage = np.arange(1, n + 1) / n
    risk_oracle_curve = cum_errors / np.arange(1, n + 1)
    aurc_oracle = float(_trapz(risk_oracle_curve, coverage))

    denom = aurc_random - aurc_oracle
    if denom <= 0:
        return float("nan")
    return float((aurc_random - aurc_model) / denom)


def coverage_at_risk(risk_score: np.ndarray, label: np.ndarray, max_risk: float = 0.05) -> float:
    """Максимальное покрытие, при котором риск среди принятых <= max_risk."""
    coverage, risk = risk_coverage_curve(risk_score, label)
    ok = coverage[risk <= max_risk]
    return float(ok.max()) if len(ok) else 0.0


def brier_score(prob: np.ndarray, label: np.ndarray) -> float:
    """label==1 - хороший ответ; prob должен быть P(label==1)."""
    prob = np.asarray(prob, dtype=float)
    y = np.asarray(label, dtype=float)
    return float(np.mean((prob - y) ** 2))


def ece(prob: np.ndarray, label: np.ndarray, n_bins: int = 10) -> float:
    prob = np.asarray(prob, dtype=float)
    y = np.asarray(label, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(prob)
    err = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (prob >= lo) & (prob < hi) if hi < 1.0 else (prob >= lo) & (prob <= hi)
        if not mask.any():
            continue
        acc = y[mask].mean()
        conf = prob[mask].mean()
        err += (mask.sum() / total) * abs(acc - conf)
    return float(err)


def per_question_kendall_tau(scores: list[float], labels: list[int]) -> float:
    """Внутри-инстансное ранжирование (клеймы одного ответа): корреляция
    между risk-score и (1 - label) по клеймам одного вопроса. Возвращает
    NaN, если меньше двух клеймов или нет вариации меток."""
    if len(scores) < 2 or len(set(labels)) < 2:
        return float("nan")
    errors = [1 - l for l in labels]
    tau, _ = kendalltau(scores, errors)
    return float(tau)
