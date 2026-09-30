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


# ---- бесплатные карточки банка B (researcher_folder/B2_cards.md) -----------
# Всё — прямо по полям дампа, ноль генераций и моделей.

def nll_ratio(record: Record) -> float:
    """uplift-contrast-baseline, вариант «отношение»: len_norm_rag / len_norm_cb
    (разность — это uplift_len_norm, считается в compute_all_signals)."""
    cb = len_norm_nll(record, "closed_book")
    if cb == 0:
        raise ZeroDivisionError
    return len_norm_nll(record, "rag") / cb


def _context_share(record: Record) -> np.ndarray:
    """attention-context-ratio: доля внимания текущей позиции на контекст по
    токенам ответа. attention_by_group = [instruction, context, question,
    generated] (generation._group_for_position), последний слой, среднее по
    головам."""
    g = np.asarray(record["rag"]["attention_by_group"], dtype=float)
    if g.ndim != 2 or g.shape[1] != 4 or len(g) == 0:
        raise KeyError("attention_by_group")
    return g[:, 1] / np.clip(g.sum(axis=1), 1e-12, None)


def attn_ctx_mean(record: Record) -> float:
    return float(_context_share(record).mean())


def attn_ctx_min(record: Record) -> float:
    """Самая «оторванная от контекста» позиция ответа — галлюцинация локальна."""
    return float(_context_share(record).min())


def attn_ctx_vs_param(record: Record) -> float:
    """Контекст против параметрической части промпта (инструкция + вопрос),
    без учёта уже сгенерированного: среднее log(ctx / (instr + question))."""
    g = np.asarray(record["rag"]["attention_by_group"], dtype=float)
    return float(np.mean(np.log((g[:, 1] + 1e-6) / (g[:, 0] + g[:, 2] + 1e-6))))


def _dense_scores(record: Record) -> np.ndarray:
    s = np.asarray(schema.passages_top20_scores(record), dtype=float)
    # long-form FRANQ: пассажи не из ретривера, скоры — нули-заглушки
    if len(s) < 2 or not np.any(s):
        raise KeyError("нет скоров ретривера")
    return np.sort(s)[::-1]


def qpp_nqc(record: Record) -> float:
    """score-distribution-qpp: NQC (Shtok et al.) = std(top-k) / |mean(top-k)|
    — без скора всего корпуса нормируем на среднее топа. В дампе топ-5, не 20."""
    s = _dense_scores(record)
    return float(s.std() / abs(s.mean()))


def qpp_top1_tail(record: Record) -> float:
    """score-distribution-qpp: отношение первого скора к среднему хвоста."""
    s = _dense_scores(record)
    return float(s[0] / s[1:].mean())


def _ids(ids) -> list[str]:
    return [str(i) for i in ids]


def _jaccard(a: list[str], b: list[str]) -> float:
    a, b = set(a), set(b)
    if not a or not b:
        raise KeyError("пустой топ")
    return len(a & b) / len(a | b)


def retriever_agree_jaccard(record: Record) -> float:
    """retriever-disagreement: пересечение топа dense и BM25 (выше = согласнее)."""
    dense = _ids(p["doc_id"] for p in schema.passages(record))
    bm25 = _ids(record["perturbations"]["retriever_alt"]["bm25"]["topk_doc_ids"])
    return _jaccard(dense, bm25)


def retriever_agree_top1(record: Record) -> float:
    """retriever-disagreement: топ-1 каждого ретривера есть в топе другого (среднее из двух)."""
    dense = _ids(p["doc_id"] for p in schema.passages(record))
    bm25 = _ids(record["perturbations"]["retriever_alt"]["bm25"]["topk_doc_ids"])
    if not dense or not bm25:
        raise KeyError("пустой топ")
    return float(((dense[0] in bm25) + (bm25[0] in dense)) / 2)


def _paraphrase_jaccards(record: Record) -> list[float]:
    orig = _ids(p["doc_id"] for p in schema.passages(record))
    out = []
    for para in record.get("perturbations", {}).get("query_paraphrases", []):
        ids = _ids(para.get("topk_doc_ids", []))
        if ids and orig:
            out.append(_jaccard(orig, ids))
    if not out:
        raise KeyError("нет ретривала по перефразам")
    return out


def para_rank_stability_mean(record: Record) -> float:
    """paraphrase-rank-stability: среднее пересечение топа оригинала и перефраза."""
    return float(np.mean(_paraphrase_jaccards(record)))


def para_rank_stability_min(record: Record) -> float:
    return float(np.min(_paraphrase_jaccards(record)))


# ---- карточки банка B из sidecar (dump_assembly/gpu_cards.py) --------------

def _derived(name: str) -> Callable:
    def fn(record: Record) -> float:
        v = schema.derived_signal(record, name)
        if v is None:
            raise KeyError(name)
        return float(v)
    fn.__name__ = name
    return fn


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

FREE_CARD_SPECS: list[SignalSpec] = [
    # uplift-contrast-baseline (разность — uplift_len_norm)
    SignalSpec("nll_ratio", nll_ratio, False, "uncertainty"),
    # attention-context-ratio
    SignalSpec("attn_ctx_mean", attn_ctx_mean, False, "confidence"),
    SignalSpec("attn_ctx_min", attn_ctx_min, False, "confidence"),
    SignalSpec("attn_ctx_vs_param", attn_ctx_vs_param, False, "confidence"),
    # score-distribution-qpp
    SignalSpec("qpp_nqc", qpp_nqc, False, "confidence"),
    SignalSpec("qpp_top1_tail", qpp_top1_tail, False, "confidence"),
    # retriever-disagreement
    SignalSpec("retriever_agree_jaccard", retriever_agree_jaccard, False, "confidence"),
    SignalSpec("retriever_agree_top1", retriever_agree_top1, False, "confidence"),
    # paraphrase-rank-stability
    SignalSpec("para_rank_stability_mean", para_rank_stability_mean, False, "confidence"),
    SignalSpec("para_rank_stability_min", para_rank_stability_min, False, "confidence"),
]

# card id (B2_cards.md) -> сигналы; без sidecar — NaN
CARD_SPECS: list[SignalSpec] = FREE_CARD_SPECS + [
    # selfcheckgpt-consistency
    SignalSpec("selfcheck_nli_rag", _derived("selfcheck_nli_rag"), False, "uncertainty"),
    SignalSpec("selfcheck_nli_cb", _derived("selfcheck_nli_cb"), False, "uncertainty"),
    # non-contradiction-probability
    SignalSpec("ncp_rag", _derived("ncp_rag"), False, "confidence"),
    SignalSpec("ncp_cb", _derived("ncp_cb"), False, "confidence"),
    SignalSpec("ncp_cross", _derived("ncp_cross"), False, "confidence"),
    # density-based-dev-embeddings (dump_assembly/local_cards.py, только test)
    SignalSpec("md_rag", _derived("md_rag"), False, "uncertainty"),
    SignalSpec("md_cb", _derived("md_cb"), False, "uncertainty"),
    SignalSpec("md_diff", _derived("md_diff"), False, "uncertainty"),
    # semantic-entropy-probes (dump_assembly/local_cards.py, только test)
    SignalSpec("sep_rag", _derived("sep_rag"), False, "uncertainty"),
    SignalSpec("sep_cb", _derived("sep_cb"), False, "uncertainty"),
    # lettucedetect-lightweight
    SignalSpec("lettuce_max", _derived("lettuce_max"), False, "uncertainty"),
    SignalSpec("lettuce_mean", _derived("lettuce_mean"), False, "uncertainty"),
    SignalSpec("lettuce_frac", _derived("lettuce_frac"), False, "uncertainty"),
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

    for spec in SHARED_SPECS + CARD_SPECS:
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
    for spec in SHARED_SPECS + CARD_SPECS:
        polarity[spec.name] = spec.polarity
    return polarity
