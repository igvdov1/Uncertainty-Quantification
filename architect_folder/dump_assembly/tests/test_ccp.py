"""CCP без моделей: формула и сборка NLI-пар на моках."""
from __future__ import annotations

import numpy as np
import pytest

from dump_assembly import ccp
from dump_assembly.ccp import CONTRA, ENTAIL, NEUTRAL


def test_ccp_word_formula():
    # жадный 0.5; альтернативы: синоним 0.2 (entail), другой факт 0.2 (contra), мусор 0.1 (neutral)
    assert ccp.ccp_word(0.5, [0.2, 0.2, 0.1], [ENTAIL, CONTRA, NEUTRAL]) == pytest.approx(0.7 / 0.9)
    assert ccp.ccp_word(0.5, [0.3], [NEUTRAL]) == 1.0                 # нейтральные не считаются
    assert ccp.ccp_word(0.9, [], []) == 1.0


def test_ccp_scores():
    s = ccp.ccp_scores([0.5, 0.8])
    assert s["ccp"] == pytest.approx(1 - 0.4) and s["ccp_mean"] == pytest.approx(1 - 0.65)
    assert np.isnan(ccp.ccp_scores([])["ccp"])


def test_answer_ccp_builds_pairs_and_filters_low_prob():
    vocab = {1: "July", 2: " 24", 3: "June", 4: " 25", 5: "zzz"}
    decode = lambda ids: "".join(vocab[i] for i in ids)
    seen = []

    def nli(pairs):
        seen.extend(pairs)
        # «June»/«25» противоречат, остальное — следует
        return [CONTRA if ("June" in a or "25" in a) else ENTAIL for a, _ in pairs]

    steps = [([1, 3, 5], [0.6, 0.3, 1e-4]), ([2, 4], [0.9, 0.1])]
    words = ccp.answer_ccp("When?", [1, 2], steps, [0.6, 0.9], decode, nli)
    assert words == pytest.approx([0.6 / 0.9, 0.9 / 1.0])
    assert len(seen) == 2                                               # zzz (P<1e-3) отброшен
    assert seen[0] == ("When? June", "When? July")
    assert seen[1] == ("When? July 25", "When? July 24")
