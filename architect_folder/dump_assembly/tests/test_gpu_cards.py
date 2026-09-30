"""gpu_cards без моделей: NLI и LettuceDetect — моки."""
from __future__ import annotations

import json

import numpy as np
import pytest

from dump_assembly import gpu_cards as gc
from screener import registry, run_screener


def contra_if_lyon(pairs):
    """Противоречие, если ровно одна сторона про Lyon."""
    return [0.9 if ("Lyon" in a) != ("Lyon" in b) else 0.1 for a, b in pairs]


def test_split_sentences():
    assert gc.split_sentences("Paris. It is big!\n- Lyon too") == ["Paris.", "It is big!", "- Lyon too"]
    assert gc.split_sentences("Real Madrid") == ["Real Madrid"]
    assert gc.split_sentences("   ") == []


def test_selfcheck_nli_mean_over_sentences_and_samples():
    s = gc.selfcheck_nli("Paris is the capital. It is in France.", ["Paris.", "Lyon.", "Paris."], contra_if_lyon)
    assert s == pytest.approx((0.1 + 0.9 + 0.1) / 3)
    assert np.isnan(gc.selfcheck_nli("Paris.", ["", " "], contra_if_lyon))


def test_ncp_symmetric_and_bounded():
    v = gc.non_contradiction("Paris.", ["Paris.", "Lyon."], contra_if_lyon)
    assert v == pytest.approx(((1 - 0.1) + (1 - 0.9)) / 2)
    seen = []
    gc.non_contradiction("A", ["B"], lambda p: seen.extend(p) or [0.5] * len(p))
    assert seen == [("A", "B"), ("B", "A")]


def test_lettuce_scores():
    toks = [{"prob": 0.1, "pred": 0}, {"prob": 0.8, "pred": 1}, {"prob": 0.3, "pred": 0}, {"prob": 0.6, "pred": 1}]
    s = gc.lettuce_scores(toks)
    assert s == {"lettuce_max": 0.8, "lettuce_mean": pytest.approx(0.45), "lettuce_frac": 0.5}
    assert np.isnan(gc.lettuce_scores([])["lettuce_max"])


def test_card_signals_and_screener_sidecar(tmp_path):
    rec = {"qid": "q1", "question": "Capital?", "passages": ["Paris is the capital of France."],
           "rag": {"answer": "Paris.", "samples": ["Paris.", "Paris."]},
           "closed_book": {"answer": "Lyon.", "samples": ["Lyon.", "Paris."]}}
    sig = gc.card_signals(rec, contra_if_lyon, contra_if_lyon, lambda p, q, a: [{"prob": 0.2, "pred": 0}])
    assert set(sig) == {"selfcheck_nli_rag", "selfcheck_nli_cb", "ncp_rag", "ncp_cb", "ncp_cross",
                        "lettuce_max", "lettuce_mean", "lettuce_frac"}
    assert sig["ncp_cross"] == pytest.approx(((1 - 0.9) + (1 - 0.1)) / 2)

    side = tmp_path / "gpu.jsonl"
    side.write_text(json.dumps({"qid": "q1", "models": {}, "signals": sig}) + "\n")
    by_qid = run_screener.load_sidecars([side])
    r = next(run_screener.attach_sidecars(iter([{"qid": "q1"}]), by_qid))
    assert registry.CARD_SPECS[0].fn(r) == sig["selfcheck_nli_rag"]
    assert registry.signal_polarity()["ncp_rag"] == "confidence"
