"""
Линейка и базы сравнения — в одном месте (C9 §15). Раньше len_norm_rag был
зашит отдельно в card_report и synthesis.

RULER — линейка вердиктов карточек (Δ AUROC и парный bootstrap против неё).
INCREMENTAL_BASES — базы добавочной ценности (логрегрессия «база» против
«база + сигнал»):
  NLL, NLL+длина — как раньше;
  NLL+max+длина — усиленная база (C9 §3): на v5 без отказов max_nll_rag
      сильнее len_norm_rag, и сравнение только с len_norm завышало ATS/INTRYGUE;
  NLL+отказ, NLL+отказ+длина — только если в данных есть отказы (C9 §2):
      флаг «ответ — отказ» сам даёт заметный AUROC, и сигнал, узнающий отказ,
      иначе получает прирост бесплатно.
"""
from __future__ import annotations

import numpy as np

RULER = "len_norm_rag"
LENGTH = "mean_sample_len_rag"
MAX_NLL = "max_nll_rag"
REFUSAL = "refusal_flag"

INCREMENTAL_BASES = {
    "NLL": [RULER],
    "NLL+длина": [RULER, LENGTH],
    "NLL+max+длина": [RULER, MAX_NLL, LENGTH],
    "NLL+отказ": [RULER, REFUSAL],
    "NLL+отказ+длина": [RULER, REFUSAL, LENGTH],
}
# база, к которой считается «вердикт к усиленной базе»
STRONG_BASE = "NLL+max+длина"


def active_bases(risk: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, list[str]]:
    """Базы с отказом — только когда флаг отказа в выборке не константа."""
    out = {}
    for name, cols in INCREMENTAL_BASES.items():
        if any(c not in risk for c in cols):
            continue
        if REFUSAL in cols:
            r = risk[REFUSAL][mask]
            r = r[~np.isnan(r)]
            if len(r) == 0 or r.min() == r.max():
                continue
        out[name] = cols
    return out
