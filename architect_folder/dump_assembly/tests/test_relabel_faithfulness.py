"""
relabel_faithfulness без GPU: судья — мок, дамп — пара рукописных записей
в формате dump_schema_v1 (только поля, которые скрипт читает).
"""
from __future__ import annotations

import json

import pytest

from dump_assembly import labeling, relabel_faithfulness as rf


def _short(qid, answer, alignscore, old_label):
    return {"qid": qid, "source": "nq", "question": f"q {qid}",
            "passages": [{"text": "Paris is the capital of France."}, {"text": "Berlin is in Germany."}],
            "rag": {"answer": answer}, "signals": {"alignscore": alignscore},
            "claims": [{"cid": f"{qid}_c0", "text": answer, "label_faithful": old_label, "label_factual": 1}],
            "label_faithful": old_label, "label_factual": 1}


def _long(qid, claim_labels):
    return {"qid": qid, "source": "franq_longform", "question": f"q {qid}",
            "passages": [{"text": "some passage"}], "rag": {"answer": "long answer"}, "signals": {},
            "claims": [{"cid": f"{qid}_c{i}", "text": f"claim {i}", "label_faithful": l, "label_factual": 1}
                       for i, l in enumerate(claim_labels)],
            "label_faithful": int(all(claim_labels)), "label_factual": 1}


@pytest.fixture
def records():
    return [
        _short("nq_1", "Paris is the capital of France.", 0.9, 1),
        _short("nq_2", "The capital is Lyon.", 0.8, 1),          # NLI сказал faithful, судья — contra
        _long("franq_lf_0", [1] * 9 + [0]),                       # 0.9 faithful: AND -> 0, мягко -> 1
        _long("franq_lf_1", [1, 0, 0, 1]),                        # 0.5 -> 0 при любой агрегации
    ]


class MockJudge:
    """Возвращает ответ в формате судьи; вердикт по ключевому слову в утверждении."""
    def __init__(self):
        self.calls = 0

    def __call__(self, batch_messages):
        self.calls += 1
        outs = []
        for m in batch_messages:
            stmt = m[1]["content"].split("Statement to check:\n", 1)[1].split("\n\nClassify", 1)[0]
            if "Lyon" in stmt:
                outs.append("Passages say Paris, not Lyon; this is a faithful-sounding but wrong claim.\n"
                            "VERDICT: unfaithful-contra")
            elif stmt == "claim 0":
                outs.append("Not supported.\nVERDICT: unfaithful-neutral")
            else:
                outs.append("Supported by passage 1.\nVERDICT: faithful")
        return outs


def test_parse_verdict_takes_last_and_handles_markdown():
    raw = "It is not unfaithful-contra, it is supported.\n**VERDICT:** faithful"
    assert rf.parse_verdict(raw) == "faithful"
    assert rf.parse_verdict("VERDICT: faithful\nWait, no.\nVERDICT: unfaithful-neutral") == "unfaithful-neutral"
    assert rf.parse_verdict("I think it's fine.") is None
    assert rf.verdict_to_label("unfaithful-contra") == 0 and rf.verdict_to_label(None) is None


def test_extract_items_kinds(records):
    short_items = rf.extract_items(records[0])
    assert [it["kind"] for it in short_items] == ["answer"]
    assert short_items[0]["item_id"] == "nq_1_answer"
    long_items = rf.extract_items(records[2])
    assert len(long_items) == 10 and all(it["kind"] == "claim" for it in long_items)
    assert long_items[-1]["ref_label_faithful"] == 0


def test_prompt_contains_passages_and_statement(records):
    msgs = rf.build_messages(rf.extract_items(records[1])[0])
    assert "[1] Paris is the capital of France." in msgs[1]["content"]
    assert "The capital is Lyon." in msgs[1]["content"]


def _judge_all(records, tmp_path, batch_size=3):
    items = [it for r in records for it in rf.extract_items(r)]
    out = tmp_path / "labels.jsonl"
    rf.run_judge(items, MockJudge(), out, batch_size=batch_size)
    return items, out


def test_apply_replaces_shortform_with_judge_and_softens_longform(records, tmp_path):
    _, out = _judge_all(records, tmp_path)
    judged = rf.load_judge_labels(out)
    results = [rf.apply_to_record(r, judged, labeling.LONGFORM_FAITHFUL_MIN_FRAC) for r in records]
    assert results[0] == (1, 1)
    assert results[1] == (1, 0)                       # утечка NLI исправлена судьёй
    assert records[1]["claims"][0]["label_faithful"] == 0
    assert results[2] == (0, 1)                       # мягкая агрегация, 9/10 >= 0.8
    assert results[3] == (0, 0)
    # long-form метка — из разметки FRANQ, не из судьи
    assert records[2]["claims"][0]["label_faithful"] == 1


def test_apply_fails_loudly_without_verdict(records):
    with pytest.raises(KeyError):
        rf.apply_to_record(records[0], {}, 0.8)


def test_calibration_report(records, tmp_path):
    items, out = _judge_all(records, tmp_path)
    rep = rf.calibration_report(items, rf.load_judge_labels(out))
    # мок-судья говорит unfaithful только про "claim 0" (у обоих long-form ref=1)
    assert rep["n_claims"] == 14
    assert rep["confusion_ref_x_judge"] == {"ref=1,judge=1": 9, "ref=1,judge=0": 2,
                                             "ref=0,judge=1": 3, "ref=0,judge=0": 0}
    assert rep["accuracy"] == pytest.approx(9 / 14)
    assert rep["judge_verdicts_on_ref_unfaithful"] == {"faithful": 3}
    # record-level: lf_0 ref 9/10 -> 1, judge 9/10 -> 1; lf_1 ref 2/4 -> 0, judge 3/4 -> 0
    assert rep["n_records"] == 2 and rep["record_accuracy"] == 1.0


def test_judge_resume_skips_done_items(records, tmp_path):
    items = [it for r in records for it in rf.extract_items(r)]
    inp, out = tmp_path / "in.jsonl", tmp_path / "labels.jsonl"
    inp.write_text("".join(json.dumps(it) + "\n" for it in items))
    rf.run_judge(items[:5], MockJudge(), out, batch_size=2)       # "оборванный" прогон

    judge = MockJudge()
    args = type("A", (), dict(input=inp, out=out, only_kind=None, limit=None, backend="hf",
                              model="x", dtype="bfloat16", max_new_tokens=8, batch_size=4))()
    import dump_assembly.relabel_faithfulness as mod
    orig = mod.HFJudge
    mod.HFJudge = lambda *a, **k: judge
    try:
        rf.cmd_judge(args)
    finally:
        mod.HFJudge = orig
    assert judge.calls == 3                                        # 11 оставшихся / 4
    assert set(rf.load_judge_labels(out)) == {it["item_id"] for it in items}


def test_cohen_kappa_bounds():
    assert rf.cohen_kappa([(1, 1), (0, 0)] * 5) == pytest.approx(1.0)
    assert rf.cohen_kappa([(1, 0), (0, 1)] * 5) == pytest.approx(-1.0)


def test_longform_aggregation_strict_mode_reproduces_old_and():
    claims = [{"label_faithful": l, "label_factual": 1} for l in [1] * 9 + [0]]
    assert labeling.franq_longform_answer_labels(claims, faithful_min_frac=1.0) == (0, 1)
    assert labeling.franq_longform_answer_labels(claims) == (1, 1)


def test_extract_substitutes_full_franq_claims(records):
    full = {"franq_lf_0": [f"Full sentence {i}." for i in range(10)]}
    items = rf.extract_items(records[2], full)
    assert [it["text"] for it in items][:2] == ["Full sentence 0.", "Full sentence 1."]
    assert rf.extract_items(records[3], full)[0]["text"] == "claim 0"      # нет в словаре -> как в дампе


def test_resume_reasks_items_judged_on_old_prompt(records, tmp_path):
    items = [it for r in records for it in rf.extract_items(r)]
    inp, out = tmp_path / "in.jsonl", tmp_path / "labels.jsonl"
    rf.run_judge(items, MockJudge(), out, batch_size=4)             # все размечены на старом тексте
    changed = [dict(it, text="Paris.") if it["item_id"] == "nq_2_answer" else it for it in items]
    inp.write_text("".join(json.dumps(it) + "\n" for it in changed))

    judge = MockJudge()
    args = type("A", (), dict(input=inp, out=out, only_kind=None, limit=None, backend="hf",
                              model="x", dtype="bfloat16", max_new_tokens=8, batch_size=4))()
    import dump_assembly.relabel_faithfulness as mod
    orig = mod.HFJudge
    mod.HFJudge = lambda *a, **k: judge
    try:
        rf.cmd_judge(args)
    finally:
        mod.HFJudge = orig
    assert judge.calls == 1                                          # перепрошен только изменённый
    fresh = rf.load_judge_labels(out, changed)
    assert fresh["nq_2_answer"]["label_faithful"] == 1               # новый вердикт, не старый contra
    assert rf.load_judge_labels(out)["nq_2_answer"]["label_faithful"] == 1


def test_franq_longform_claims_uses_full_text():
    ex = {"_qid": "franq_lf_7", "claims": ["Magnesium reacts with halogens."],
          "decoded_claims": [" reacts with halogens"], "auto_labels": ['("faithful", "True")']}
    c = labeling.franq_longform_claims(ex)[0]
    assert c["text"] == "Magnesium reacts with halogens." and c["text_decoded"] == " reacts with halogens"
