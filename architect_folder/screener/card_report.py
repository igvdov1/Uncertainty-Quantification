"""
Вердикты по карточкам банка B (бриф, раздел 10; researcher_folder/B_handoff_to_C.md §1).

Оценка — только short-form (524 записи; long-form — 76 FRANQ с другой
природой ответа, в смеси AUROC награждает за отличие long от short).
Основной target — factual; faithful вторичный (метки LLM-судьи, отказы
исключены, см. runner_folder/C1_faithful_label_leakage.md §9).

Линейка — простой NLL своей ветки (len_norm_rag / len_norm_cb). Рядом —
длина сэмплов своей ветки (mean_sample_len_*): на пилоте она одна даёт
factual 0.759, сигнал, не обгоняющий длину, помечается.

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

from . import bootstrap, metrics

ALIVE_DELTA, ALIVE_P, POOL_DELTA, POOL_CORR = 0.02, 0.9, -0.03, 0.5

# card id -> [(сигнал, ветка линейки)]
CARDS: dict[str, list[tuple[str, str]]] = {
    "semantic-entropy (бейзлайн брифа)": [("semantic_entropy_rag", "rag"), ("semantic_entropy_cb", "cb")],
    "selfcheckgpt-consistency": [("selfcheck_nli_rag", "rag"), ("selfcheck_nli_cb", "cb")],
    "non-contradiction-probability": [("ncp_rag", "rag"), ("ncp_cb", "cb"), ("ncp_cross", "rag")],
    "lettucedetect-lightweight": [("lettuce_max", "rag"), ("lettuce_mean", "rag"), ("lettuce_frac", "rag")],
}


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
    nll, length = risk[f"len_norm_{branch}"], risk[f"mean_sample_len_{branch}"]
    m = ~(np.isnan(s) | np.isnan(nll)) & (labels >= 0)
    if m.sum() < 10 or len(set(labels[m])) < 2:
        return None
    bs = bootstrap.paired_bootstrap(s[m], nll[m], labels[m], metrics.auroc, n_boot=n_boot, seed=seed)
    a, a_nll, a_len = (metrics.auroc(x[m], labels[m]) for x in (s, nll, length))
    rho_nll, rho_len = _rho(s[m], nll[m]), _rho(s[m], length[m])
    return {"signal": name, "n": int(m.sum()), "auroc": a, "nll": a_nll, "delta": a - a_nll,
            "ci": (bs["ci_low"], bs["ci_high"]), "p_better": bs["p_a_better"], "len": a_len,
            "rho_nll": rho_nll, "rho_len": rho_len, "verdict": verdict(a - a_nll, bs["p_a_better"], rho_nll)}


def card_report(risk: dict[str, np.ndarray], labels_by_target: dict[str, np.ndarray], form_mask: np.ndarray,
                n_boot: int = 1000, seed: int = 0) -> dict:
    out: dict = {}
    for card, sigs in CARDS.items():
        out[card] = {}
        for target, labels in labels_by_target.items():
            sub = {k: v[form_mask] for k, v in risk.items()}
            rows = [r for name, br in sigs if (r := signal_row(sub, labels[form_mask], name, br, n_boot, seed))]
            out[card][target] = rows
    return out


def print_card_report(report: dict) -> None:
    print("\n=== Вердикты по карточкам (short-form; линейка — len_norm своей ветки) ===")
    for card, by_target in report.items():
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
