"""
Вердикты по карточкам банка B (бриф, раздел 10; researcher_folder/B_handoff_to_C.md §1).

Оценка — только short-form (524 записи; long-form — 76 FRANQ с другой
природой ответа, в смеси AUROC награждает за отличие long от short).
Основной target — factual; faithful вторичный (метки LLM-судьи, отказы
исключены, см. runner_folder/C1_faithful_label_leakage.md §9).

Линейка — простой NLL RAG-ответа (len_norm_rag) для ВСЕХ сигналов, включая
closed-book: метки стоят на RAG-ответ, и вопрос скрининга — даёт ли сигнал
лучший предсказатель его ошибки, чем бесплатный NLL этого же ответа
(решение 2026-10-04; раньше cb-сигналы сравнивались с len_norm_cb — слишком
мягко). Рядом — длина RAG-сэмплов (mean_sample_len_rag): на пилоте она одна
даёт factual 0.759, сигнал, не обгоняющий длину, помечается.

Правило (фиксировано до просмотра результатов карточек):
  жива    — Δ AUROC к NLL >= +0.02 и P(лучше NLL) >= 0.9 по парному bootstrap
  в пул   — не жива, Δ > -0.03 (примерно вровень) и |Spearman с NLL| < 0.5
  убита   — иначе
Вердикт карточки — по лучшему из её сигналов (сигналов у карточки
несколько -> лёгкое множественное сравнение, держать в уме).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from . import bootstrap, incremental, metrics

ALIVE_DELTA, ALIVE_P, POOL_DELTA, POOL_CORR = 0.02, 0.9, -0.03, 0.5
# добавочная ценность — тысячи обучений логрегрессии на ~50 сигналов; тесты на
# синтетике её отключают (саму её проверяет test_incremental_value_*)
WITH_INCREMENTAL = True

# card id -> [(сигнал, ветка сигнала)]. Ветка — справочно: линейка у всех rag-NLL.
CARDS: dict[str, list[tuple[str, str]]] = {
    "uplift-contrast-baseline": [("uplift_len_norm", "rag"), ("nll_ratio", "rag")],
    "attention-context-ratio": [("attn_ctx_mean", "rag"), ("attn_ctx_min", "rag"), ("attn_ctx_vs_param", "rag")],
    "score-distribution-qpp": [("qpp_nqc", "rag"), ("qpp_top1_tail", "rag"), ("top1_cos", "rag"),
                               ("margin_12", "rag")],
    "retriever-disagreement": [("retriever_agree_jaccard", "rag"), ("retriever_agree_top1", "rag")],
    "paraphrase-rank-stability": [("para_rank_stability_mean", "rag"), ("para_rank_stability_min", "rag")],
    "density-based-dev-embeddings": [("md_rag", "rag"), ("md_cb", "cb"), ("md_diff", "rag")],
    "two-sample-verbal-consistency": [("two_sample_verbal", "rag"), ("two_sample_agree", "rag")],
    "rag-query-variant-qpp": [("rqv_best_nqc", "rag"), ("rqv_margin", "rag")],
    "memory-strength-paraphrase": [("mem_entropy_greedy", "cb"), ("mem_entropy_all", "cb")],
    "semantic-reformulation-entropy": [("sre_entropy_greedy", "rag"), ("sre_entropy_all", "rag")],
    "context-sufficiency-self-eval": [("ctx_sufficiency", "rag")],
    "semantic-entropy-probes": [("sep_rag", "rag"), ("sep_cb", "cb")],
    "ccp-faithful-substitution": [("ccp_rag", "rag"), ("ccp_mean_rag", "rag"), ("ccp_cb", "cb"),
                                  ("ccp_mean_cb", "cb"), ("ccp_diff", "rag")],
    "semantic-entropy (бейзлайн брифа)": [("semantic_entropy_rag", "rag"), ("semantic_entropy_cb", "cb")],
    "selfcheckgpt-consistency": [("selfcheck_nli_rag", "rag"), ("selfcheck_nli_cb", "cb")],
    "non-contradiction-probability": [("ncp_rag", "rag"), ("ncp_cb", "cb"), ("ncp_cross", "rag")],
    "lettucedetect-lightweight": [("lettuce_max", "rag"), ("lettuce_mean", "rag"), ("lettuce_frac", "rag")],
}


# Собственные критерии убийства карточек из B2 (корреляционные):
# card -> [(сигнал карточки, другой сигнал, порог |ρ|, что значит превышение)]
KILL_CHECKS: dict[str, list[tuple[str, str, float, str]]] = {
    "paraphrase-rank-stability": [("para_rank_stability_mean", "qpp_nqc", 0.8, "дублирует форму скоров")],
    "retriever-disagreement": [("retriever_agree_jaccard", "qpp_nqc", 0.7, "дублирует форму скоров")],
    "density-based-dev-embeddings": [("md_rag", "answer_len_rag", 0.6, "мерит длину"),
                                     ("md_rag", "mean_sample_len_rag", 0.6, "мерит длину")],
    "ccp-faithful-substitution": [("ccp_rag", "len_norm_rag", 0.85, "дублирует NLL")],
    "semantic-reformulation-entropy": [("sre_entropy_all", "para_rank_stability_mean", 0.7,
                                        "объясняется нестабильностью ретривера")],
    "semantic-entropy-probes": [("sep_rag", "semantic_entropy_rag", -0.7, "проба не держит SE (ρ ниже порога)"),
                                ("sep_cb", "semantic_entropy_cb", -0.7, "проба не держит SE (ρ ниже порога)")],
}


def kill_checks(risk: dict[str, np.ndarray], form_mask: np.ndarray) -> dict[str, list[dict]]:
    """Порог > 0: убивает |ρ| выше порога; порог < 0: убивает ρ ниже |порога|."""
    out: dict[str, list[dict]] = {}
    for card, checks in KILL_CHECKS.items():
        rows = []
        for mine, other, thr, meaning in checks:
            if mine not in risk or other not in risk:
                continue
            rho = _rho(risk[mine][form_mask], risk[other][form_mask])
            killed = (abs(rho) > thr) if thr > 0 else (rho < -thr)
            rows.append({"signal": mine, "other": other, "rho": rho, "thr": thr, "killed": bool(killed),
                         "meaning": meaning})
        out[card] = rows
    return out


def _rho(a: np.ndarray, b: np.ndarray) -> float:
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < 3 or np.std(a[m]) == 0 or np.std(b[m]) == 0:
        return float("nan")
    return float(spearmanr(a[m], b[m]).correlation)


def verdict(delta: float, p_better: float, rho_nll: float) -> str:
    if delta >= ALIVE_DELTA and p_better >= ALIVE_P:
        return "жива"
    if delta > POOL_DELTA and abs(rho_nll) < POOL_CORR:
        return "в пул"
    return "убита"


def signal_row(risk: dict[str, np.ndarray], labels: np.ndarray, name: str, branch: str,
               n_boot: int, seed: int) -> dict | None:
    s = risk.get(name)
    if s is None:
        return None
    nll, length = risk["len_norm_rag"], risk["mean_sample_len_rag"]
    m = ~(np.isnan(s) | np.isnan(nll)) & (labels >= 0)
    if m.sum() < 10 or len(set(labels[m])) < 2:
        return None
    bs = bootstrap.paired_bootstrap(s[m], nll[m], labels[m], metrics.auroc, n_boot=n_boot, seed=seed)
    a, a_nll, a_len = (metrics.auroc(x[m], labels[m]) for x in (s, nll, length))
    rho_nll, rho_len = _rho(s[m], nll[m]), _rho(s[m], length[m])
    return {"signal": name, "n": int(m.sum()), "auroc": a, "nll": a_nll, "delta": a - a_nll,
            "ci": (bs["ci_low"], bs["ci_high"]), "p_better": bs["p_a_better"], "len": a_len,
            "rho_nll": rho_nll, "rho_len": rho_len, "verdict": verdict(a - a_nll, bs["p_a_better"], rho_nll),
            "incremental": incremental.incremental_value(risk, labels, name, n_boot=n_boot, seed=seed)
            if WITH_INCREMENTAL else None}


def card_report(risk: dict[str, np.ndarray], labels_by_target: dict[str, np.ndarray], form_mask: np.ndarray,
                n_boot: int = 1000, seed: int = 0) -> dict:
    out: dict = {}
    for card, sigs in CARDS.items():
        out[card] = {}
        for target, labels in labels_by_target.items():
            sub = {k: v[form_mask] for k, v in risk.items()}
            rows = [r for name, br in sigs if (r := signal_row(sub, labels[form_mask], name, br, n_boot, seed))]
            out[card][target] = rows
    out["_kill_checks"] = kill_checks(risk, form_mask)
    return out


def print_card_report(report: dict) -> None:
    print("\n=== Вердикты по карточкам (short-form; линейка — len_norm_rag для всех) ===")
    checks = report.get("_kill_checks", {})
    for card, by_target in report.items():
        if card.startswith("_"):
            continue
        print(f"\n## {card}")
        for target in ("factual", "faithful"):
            rows = by_target.get(target, [])
            if not rows:
                print(f"  [{target}] нет данных (нет sidecar?)")
                continue
            best = max(rows, key=lambda r: r["delta"])
            tag = "основной" if target == "factual" else "вторичный"
            print(f"  [{target}, {tag}] вердикт: {best['verdict'].upper()} (по {best['signal']})")
            for r in rows:
                flag = "  <= длины" if r["auroc"] <= r["len"] else ""
                print(f"    {r['signal']:22s} auroc={r['auroc']:.3f}  NLL={r['nll']:.3f}  "
                      f"Δ={r['delta']:+.3f} [{r['ci'][0]:+.3f},{r['ci'][1]:+.3f}] P>NLL={r['p_better']:.2f}  "
                      f"длина={r['len']:.3f}  ρ(NLL)={r['rho_nll']:+.2f} ρ(длина)={r['rho_len']:+.2f}  "
                      f"n={r['n']}  -> {r['verdict']}{flag}")
                for base, inc in (r.get("incremental") or {}).items():
                    print(f"      + к {base:9s}: {inc['base']:.3f} -> {inc['with']:.3f}  Δ={inc['delta']:+.3f} "
                          f"[{inc['ci'][0]:+.3f},{inc['ci'][1]:+.3f}] P={inc['p_better']:.2f}  n={inc['n']}")
        for c in checks.get(card, []):
            cond = f"|ρ|>{c['thr']}" if c["thr"] > 0 else f"ρ<{-c['thr']}"
            status = "СРАБОТАЛ" if c["killed"] else "не сработал"
            print(f"  критерий B2: ρ({c['signal']}, {c['other']})={c['rho']:+.2f}, убивает при {cond} "
                  f"({c['meaning']}) -> {status}")
