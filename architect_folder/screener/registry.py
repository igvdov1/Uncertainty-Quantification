"""
Реестр дешёвых бейзлайнов (бриф, раздел 5) — чистые функции над записью
дампа. Ничего не считает по-настоящему дорого: semantic_entropy берёт готовые
NLI-кластеры сэмплов из sidecar (dump_assembly/semantic_clusters.py,
run_screener --sidecar) — без sidecar эти сигналы NaN. lexical_entropy —
прежний лексический прокси (точное совпадение строк), оставлен для сравнения.

Все сигналы, зависящие от режима (rag/closed_book), кладутся в выходной
словарь с суффиксом _rag / _cb. Для каждой пары с обоими режимами
дополнительно считается uplift_<name> = value_rag - value_cb (раздел 5
брифа, "разность rag / closed_book").

Полярность (нужна метрикам для правильного знака AUROC):
  "uncertainty" — выше значение = больше неопределённости / хуже ответ
  "confidence"  — выше значение = увереннее / вероятнее правильный ответ
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.special import logsumexp

from . import schema
from .schema import Record

MODE_SUFFIX = {"rag": "rag", "closed_book": "cb"}


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max()
    exp = np.exp(logits)
    return exp / exp.sum()


# ---- сигналы на token_logprobs ---------------------------------------

def mean_nll(record: Record, mode: str) -> float:
    lp = np.asarray(schema.token_logprobs(record, mode), dtype=float)
    return float(-lp.mean())


def max_nll(record: Record, mode: str) -> float:
    lp = np.asarray(schema.token_logprobs(record, mode), dtype=float)
    return float(-lp.min())


def len_norm_nll(record: Record, mode: str) -> float:
    """Length-normalized NLL — основной ноль брифа (раздел 5). Формула
    совпадает с mean_nll; оставлена отдельной registry-записью, потому
    что так она названа в брифе, и синтетика/скринер должны уметь
    выдать это имя как самостоятельную строку таблицы."""
    lp = np.asarray(schema.token_logprobs(record, mode), dtype=float)
    return float(-lp.sum() / len(lp))


def predictive_entropy(record: Record, mode: str) -> float:
    ent = schema.token_entropy(record, mode)
    if ent is not None:
        return float(np.mean(ent))
    topk = schema.branch(record, mode).get("token_topk")
    if topk is None:
        raise KeyError(
            f"{record.get('qid', '?')}/{mode}: нет ни token_entropy, ни token_topk"
        )
    entropies = []
    for logits in topk:
        probs = _softmax(np.asarray(logits, dtype=float))
        entropies.append(-float(np.sum(probs * np.log(probs + 1e-12))))
    return float(np.mean(entropies))


def lexical_entropy(record: Record, mode: str) -> float:
    """Энтропия распределения сэмплов по классам точного совпадения
    нормализованной строки. Раньше называлась semantic_entropy (прокси
    до NLI-версии) — оставлена как дешёвый ориентир для сравнения."""
    smp = schema.samples(record, mode)
    if len(smp) < 2:
        return 0.0
    normalized = [s.strip().lower() for s in smp]
    counts = Counter(normalized)
    n = len(normalized)
    probs = np.array([c / n for c in counts.values()])
    return float(-np.sum(probs * np.log(probs + 1e-12)))


def semantic_entropy_discrete(record: Record, mode: str) -> float:
    """Энтропия частот семантических кластеров — cluster_assignment_entropy
    у Farquhar et al. 2024 (кластеры: dump_assembly/semantic_clusters.py)."""
    ids = schema.semantic_cluster_ids(record, mode)
    if len(ids) < 2:
        return 0.0
    counts = np.array(list(Counter(ids).values()), dtype=float)
    probs = counts / counts.sum()
    return float(-np.sum(probs * np.log(probs)))


def semantic_entropy(record: Record, mode: str) -> float:
    """Semantic entropy (Kuhn et al. 2023; Farquhar et al. 2024) — как их
    основная метрика в jlko/semantic_uncertainty: лог-правдоподобие сэмпла =
    средний логпроб токена, вероятность кластера = logsumexp по его сэмплам,
    нормированная на все сэмплы (agg='sum_normalized'), энтропия —
    predictive_entropy_rao: -sum p*log p. Пустые сэмплы (EOS сразу) без
    логпробов в правдоподобии не участвуют."""
    ids = schema.semantic_cluster_ids(record, mode)
    lps = schema.sample_logprobs(record, mode)
    if len(ids) != len(lps):
        raise IndexError(f"{record.get('qid', '?')}/{mode}: cluster_ids и sample_logprobs разной длины")
    pairs = [(c, float(np.mean(lp))) for c, lp in zip(ids, lps) if len(lp) > 0]
    if len(pairs) < 2:
        return 0.0
    cids = np.array([c for c, _ in pairs])
    loglik = np.array([l for _, l in pairs])
    log_total = logsumexp(loglik)
    log_p = np.array([logsumexp(loglik[cids == c]) - log_total for c in np.unique(cids)])
    return float(-np.sum(np.exp(log_p) * log_p))


# ---- длина ответа ----------------------------------------------------
# Не UQ-метод, а конфаунд, который нужно видеть в таблице: на пилоте v3
# средняя длина сэмплов одна даёт factual AUROC 0.769 на short-form (модель
# отвечает коротко, когда знает, и многословно, когда нет), а лексическая
# энтропия коррелирует с ней по Спирмену 0.89. Карточку, которая не
# обгоняет длину, «живой» считать нельзя.

def answer_len(record: Record, mode: str) -> float:
    return float(len(schema.token_logprobs(record, mode)))


def mean_sample_len(record: Record, mode: str) -> float:
    lps = schema.sample_logprobs(record, mode)
    if not lps:
        raise KeyError(f"{record.get('qid', '?')}/{mode}: нет сэмплов")
    return float(np.mean([len(lp) for lp in lps]))


# ---- сигналы на passages (только общие, без деления rag/cb) ----------

def top1_cos(record: Record) -> float:
    p = schema.passages(record)
    return float(p[0]["score"])


def margin_12(record: Record) -> float:
    p = schema.passages(record)
    if len(p) < 2:
        return 0.0
    return float(p[0]["score"] - p[1]["score"])


def topk_spread(record: Record) -> float:
    scores = schema.passages_top20_scores(record) or [p["score"] for p in schema.passages(record)]
    return float(np.std(scores))


# ---- сигналы на signals-блоке -----------------------------------------

def p_true(record: Record, mode: str) -> float:
    s = schema.signals(record)
    return float(s[f"p_true_{MODE_SUFFIX[mode]}"])


def verbalized_conf(record: Record, mode: str) -> float:
    s = schema.signals(record)
    key = f"verbalized_conf_{MODE_SUFFIX[mode]}"
    if key not in s and mode == "rag":
        key = "verbalized_conf_rag"
    return float(s[key])


def alignscore(record: Record) -> float:
    return float(schema.signals(record)["alignscore"])


@dataclass
class SignalSpec:
    name: str
    fn: Callable
    per_mode: bool          # True: считается отдельно для rag и closed_book
    polarity: str            # "uncertainty" | "confidence"


PER_MODE_SPECS: list[SignalSpec] = [
    SignalSpec("mean_nll", mean_nll, True, "uncertainty"),
    SignalSpec("max_nll", max_nll, True, "uncertainty"),
    SignalSpec("len_norm", len_norm_nll, True, "uncertainty"),
    SignalSpec("predictive_entropy", predictive_entropy, True, "uncertainty"),
    SignalSpec("lexical_entropy", lexical_entropy, True, "uncertainty"),
    SignalSpec("semantic_entropy", semantic_entropy, True, "uncertainty"),
    SignalSpec("semantic_entropy_discrete", semantic_entropy_discrete, True, "uncertainty"),
    SignalSpec("answer_len", answer_len, True, "uncertainty"),
    SignalSpec("mean_sample_len", mean_sample_len, True, "uncertainty"),
    SignalSpec("p_true", p_true, True, "confidence"),
    SignalSpec("verbalized_conf", verbalized_conf, True, "confidence"),
]

SHARED_SPECS: list[SignalSpec] = [
    SignalSpec("top1_cos", top1_cos, False, "confidence"),
    SignalSpec("margin_12", margin_12, False, "confidence"),
    SignalSpec("topk_spread", topk_spread, False, "uncertainty"),
    SignalSpec("alignscore", alignscore, False, "confidence"),
]


def compute_all_signals(record: Record) -> dict[str, float]:
    """Считает весь реестр для одной записи: per-mode сигналы (с суффиксами
    _rag/_cb), общие retrieval-сигналы и uplift_* для каждой per-mode пары."""
    out: dict[str, float] = {}

    # TypeError ловит в т.ч. score=None у пассажей без ретривера (long-form
    # FRANQ, пассажи распарсены из текста, не из скорингового ретривера) —
    # найдено прогоном настоящего собранного дампа через скринер, не на синтетике.
    _SKIP = (KeyError, IndexError, ZeroDivisionError, TypeError)

    for spec in PER_MODE_SPECS:
        for mode, suffix in MODE_SUFFIX.items():
            try:
                out[f"{spec.name}_{suffix}"] = spec.fn(record, mode)
            except _SKIP:
                out[f"{spec.name}_{suffix}"] = float("nan")
        rag_key, cb_key = f"{spec.name}_rag", f"{spec.name}_cb"
        if not (np.isnan(out[rag_key]) or np.isnan(out[cb_key])):
            out[f"uplift_{spec.name}"] = out[rag_key] - out[cb_key]
        else:
            out[f"uplift_{spec.name}"] = float("nan")

    for spec in SHARED_SPECS:
        try:
            out[spec.name] = spec.fn(record)
        except _SKIP:
            out[spec.name] = float("nan")

    return out


def signal_polarity() -> dict[str, str]:
    """Статическая карта имя_сигнала -> полярность, для всех имён, которые
    может произвести compute_all_signals. uplift_* наследует полярность
    базового сигнала (упрощение, см. докстринг модуля)."""
    polarity: dict[str, str] = {}
    for spec in PER_MODE_SPECS:
        for suffix in MODE_SUFFIX.values():
            polarity[f"{spec.name}_{suffix}"] = spec.polarity
        polarity[f"uplift_{spec.name}"] = spec.polarity
    for spec in SHARED_SPECS:
        polarity[spec.name] = spec.polarity
    return polarity
