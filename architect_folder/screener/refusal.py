"""
Детектор отказов вида «в пассажах нет информации» по тексту ответа.

Зачем: LLM-судья (dump_assembly/relabel_faithfulness.py, промпт v2 — метки
dump_pilot_v4) ставил верным отказам unfaithful-neutral (89 из 111), и
faithful-метка на short-form частично мерила «это отказ» (отказы длинные и
с высоким NLL -> простой NLL давал faithful AUROC 0.82). Промпт v3 это
чинит вердиктом abstain, но третий прогон судьи решили не делать — вместо
этого отказы исключаются из оценки faithful (run_screener
--faithful-exclude-refusals). См. runner_folder/C1_faithful_label_leakage.md §8-9.

Грубая регулярка, не классификатор: цель — прозрачное и воспроизводимое
исключение, а не точная разметка.
"""
from __future__ import annotations

import re

REFUSAL_RE = re.compile(
    r"not (explicitly )?(mention|state|provide|contain|specif)"
    r"|no (direct |specific )?(information|mention)"
    r"|not aware|couldn't find|could not find|unable to|don't have|do not have"
    r"|cannot (determine|find)|isn't (mentioned|provided)|does not (say|include)",
    re.IGNORECASE,
)


def is_refusal(answer: str) -> bool:
    return bool(REFUSAL_RE.search(answer))
