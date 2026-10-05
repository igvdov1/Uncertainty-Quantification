"""
task-dependent-reliance-modifier (Task Matters, arXiv 2506.06485): зависит ли
связь сигналов «контекст vs память» с правильностью от типа вопроса.
Операционализация (решение 2026-10-06): категория = источник; в Task Matters
категории тоже берутся из датасетов, а не размечаются классификатором.
PopQA (длинный хвост, параметрической памяти мало) против NQ/TriviaQA/WebQ.

Критерий убийства карточки: категоризация не меняет связь uplift-контраста с
правильностью между категориями — разница AUROC (PopQA − остальные) с 95% CI
по bootstrap накрывает 0.
"""
from __future__ import annotations

import numpy as np

from . import metrics

SIGNALS = ["uplift_len_norm", "nll_ratio", "len_norm_rag", "len_norm_cb", "ctx_sufficiency", "ncp_cross",
           "mem_entropy_all"]
TAIL = "popqa"


def _auc(s, lab):
    m = ~np.isnan(s) & (lab >= 0)
    return metrics.auroc(s[m], lab[m]) if m.sum() > 10 and len(set(lab[m])) == 2 else float("nan")


def task_dependent_report(risk: dict[str, np.ndarray], labels: np.ndarray, sources: np.ndarray,
                          n_boot: int = 1000, seed: int = 0) -> dict:
    short = sources != "franq_longform"
    rng = np.random.default_rng(seed)
    out = {"per_source": {}, "popqa_vs_rest": {}}
    for name in SIGNALS:
        if name not in risk:
            continue
        s = risk[name]
        out["per_source"][name] = {src: _auc(s[sources == src], labels[sources == src])
                                   for src in sorted(set(sources[short]))}
        tail, rest = short & (sources == TAIL), short & (sources != TAIL)
        d0 = _auc(s[tail], labels[tail]) - _auc(s[rest], labels[rest])
        ti, ri = np.where(tail)[0], np.where(rest)[0]
        boots = []
        for _ in range(n_boot):
            bt, br = rng.choice(ti, len(ti)), rng.choice(ri, len(ri))
            boots.append(_auc(s[bt], labels[bt]) - _auc(s[br], labels[br]))
        boots = np.array([b for b in boots if np.isfinite(b)])
        out["popqa_vs_rest"][name] = (d0, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
    out["factual_rate"] = {src: float((labels[sources == src] == 1).mean()) for src in sorted(set(sources[short]))}
    return out


def print_task_dependent(target: str, rep: dict) -> None:
    print(f"\n=== task-dependent (target={target}): AUROC по источникам и разница PopQA − остальные ===")
    print("  доля верных: " + ", ".join(f"{k} {v:.2f}" for k, v in rep["factual_rate"].items()))
    for name, per in rep["per_source"].items():
        d, lo, hi = rep["popqa_vs_rest"][name]
        flag = "  <- различается" if lo > 0 or hi < 0 else ""
        print(f"  {name:18s} " + "  ".join(f"{k}={v:.3f}" for k, v in per.items())
              + f"  | PopQA−ост.={d:+.3f} [{lo:+.3f},{hi:+.3f}]{flag}")
