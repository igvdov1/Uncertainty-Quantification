"""
Гейт A3 (бриф, раздел 7): "Скринер считает весь реестр и матрицу, не видя
настоящего дампа." Всё здесь работает только на synthetic.py — ни одного
обращения к реальным данным.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from screener import bootstrap, correlation, metrics, registry, run_screener, schema, synthetic

N = 150
SEED = 42


@pytest.fixture(scope="module")
def records():
    return synthetic.generate_synthetic_dataset(n=N, seed=SEED)


def test_synthetic_records_pass_schema_validation(records):
    for r in records:
        schema.validate_record(r)  # не должно кидать


def test_schema_validation_catches_missing_field(records):
    broken = dict(records[0])
    del broken["closed_book"]
    with pytest.raises(KeyError):
        schema.validate_record(broken)


def test_registry_computes_full_signal_set(records):
    all_names = set(registry.signal_polarity().keys())
    for r in records[:20]:
        values = registry.compute_all_signals(r)
        assert set(values.keys()) == all_names
        # verbalized_conf_cb и его uplift — единственные ожидаемо-NaN поля
        # в синтетике (в схеме нет verbalized_conf_cb, см. registry.py)
        allowed_nan = {"verbalized_conf_cb", "uplift_verbalized_conf"}
        bad_nan = {k for k, v in values.items() if math.isnan(v) and k not in allowed_nan}
        assert not bad_nan, f"неожиданный NaN в сигналах: {bad_nan}"


def test_metrics_table_populated_for_both_targets(records):
    result = run_screener.run(records, n_boot=200, seed=0)
    table = result["metrics_table"]
    assert set(table.keys()) == {"faithful", "factual"}
    all_names = set(registry.signal_polarity().keys()) - {"verbalized_conf_cb", "uplift_verbalized_conf"}
    for target in ("faithful", "factual"):
        assert set(table[target].keys()) == all_names
        for name, row in table[target].items():
            assert 0.0 <= row["auroc"] <= 1.0, f"{target}/{name}: auroc вне [0,1]"
            assert row["n"] == N


def test_correlation_matrix_shape_and_diagonal(records):
    table, _, _, _, _, _ = run_screener.build_signal_table_and_claims(records)
    risk = run_screener.to_risk_scores(table, registry.signal_polarity())
    names, matrix = correlation.spearman_matrix(risk)
    n = len(names)
    assert matrix.shape == (n, n)
    assert np.allclose(matrix, matrix.T, equal_nan=True)
    diag = np.diag(matrix)
    finite_diag = diag[~np.isnan(diag)]
    assert np.allclose(finite_diag, 1.0)


def test_paired_bootstrap_matches_point_estimate(records):
    table, _, labels, _, _, _ = run_screener.build_signal_table_and_claims(records)
    risk = run_screener.to_risk_scores(table, registry.signal_polarity())
    target = "factual"
    a, b = risk["mean_nll_rag"], risk["p_true_rag"]
    result = bootstrap.paired_bootstrap(a, b, labels[target], metrics.auroc, n_boot=300, seed=1)
    expected_point = metrics.auroc(a, labels[target]) - metrics.auroc(b, labels[target])
    assert result["point_diff"] == pytest.approx(expected_point, abs=1e-9)
    assert result["ci_low"] <= result["ci_high"]
    assert 0.0 <= result["p_a_better"] <= 1.0
    assert result["n_boot_valid"] > 0


def test_per_question_kendall_tau_is_finite(records):
    _, _, _, claim_data, _, _ = run_screener.build_signal_table_and_claims(records)
    tau = run_screener.claim_level_kendall(claim_data, target="factual")
    assert not math.isnan(tau)
    assert -1.0 <= tau <= 1.0


def test_calibration_metrics_run_on_probability_like_signal(records):
    table, _, labels, _, _, _ = run_screener.build_signal_table_and_claims(records)
    p_true_rag = table["p_true_rag"]
    y = labels["faithful"]
    b = metrics.brier_score(p_true_rag, y)
    e = metrics.ece(p_true_rag, y, n_bins=10)
    assert 0.0 <= b <= 1.0
    assert 0.0 <= e <= 1.0


def test_run_end_to_end_no_crash_and_reasonable_output(records):
    result = run_screener.run(records, n_boot=200, seed=0)
    assert result["n_records"] == N
    boot = result["paired_bootstrap_top2_factual"]
    assert boot is not None and "pair" in boot
    assert len(result["correlation"]["names"]) == len(registry.signal_polarity())


def test_leakage_suspects_flags_near_perfect_auroc():
    table = {"faithful": {"alignscore": {"auroc": 1.0}, "len_norm_rag": {"auroc": 0.62},
                          "inverted": {"auroc": 0.005}, "broken": {"auroc": float("nan")}}}
    flagged = {name for _, name, _ in run_screener.leakage_suspects(table)}
    assert flagged == {"alignscore", "inverted"}


def test_label_none_fails_with_clear_error(records):
    r = dict(records[0])
    r["label_faithful"] = None
    with pytest.raises(ValueError, match="relabel_faithfulness"):
        schema.label(r, "faithful")


def test_sidecar_attaches_semantic_clusters(tmp_path, records):
    import json
    stripped = [{k: v for k, v in r.items() if k != "derived"} for r in records[:5]]
    assert np.isnan(registry.compute_all_signals(stripped[0])["semantic_entropy_rag"])
    side = tmp_path / "sc.jsonl"
    side.write_text("".join(json.dumps({"qid": r["qid"], **r["derived"]["semantic_clusters"]}) + "\n"
                            for r in records[:5]))
    attached = list(run_screener.attach_sidecars(iter(stripped), run_screener.load_sidecars([side])))
    vals = registry.compute_all_signals(attached[0])
    assert not np.isnan(vals["semantic_entropy_rag"]) and not np.isnan(vals["semantic_entropy_discrete_cb"])


def test_leakage_by_group_catches_leak_hidden_in_overall():
    rng = np.random.default_rng(0)
    n = 100
    groups = np.array(["short"] * 60 + ["long"] * 40)
    labels = rng.integers(0, 2, n)
    sig = rng.normal(size=n)
    sig[:60] = -labels[:60] + 0.01 * rng.normal(size=60)     # на short метка = порог сигнала
    flagged = run_screener.leakage_suspects_by_group({"s": sig}, {"faithful": labels}, groups)
    assert [(t, name) for t, name, _ in flagged] == [("faithful@short", "s")]


def test_faithful_exclude_refusals_masks_only_faithful(records):
    recs = [dict(r, rag=dict(r["rag"])) for r in records[:40]]
    for r in recs[:10]:
        r["rag"]["answer"] = "The passages do not mention this."
    _, _, labels, _, _, _ = run_screener.build_signal_table_and_claims(recs, faithful_exclude_refusals=True)
    assert (labels["faithful"][:10] == run_screener.EXCLUDED).all()
    assert (labels["faithful"][10:] != run_screener.EXCLUDED).all()
    assert (labels["factual"] != run_screener.EXCLUDED).all()
    table = run_screener.compute_metrics_table({"s": np.arange(40, dtype=float)}, labels)
    assert table["faithful"]["s"]["n"] == 30 and table["factual"]["s"]["n"] == 40


def test_card_verdict_rule():
    from screener import card_report as cr
    assert cr.verdict(0.03, 0.95, 0.9) == "жива"
    assert cr.verdict(0.03, 0.80, 0.3) == "в пул"
    assert cr.verdict(-0.01, 0.40, 0.3) == "в пул"
    assert cr.verdict(-0.01, 0.40, 0.8) == "убита"
    assert cr.verdict(-0.05, 0.10, 0.1) == "убита"


def test_card_report_runs_on_synthetic(records, monkeypatch):
    from screener import card_report as cr
    monkeypatch.setattr(cr, "WITH_INCREMENTAL", False)
    result = run_screener.run(records, n_boot=50, seed=0, cards=True)
    rep = result["card_report"]
    rows = rep["selfcheckgpt-consistency"]["factual"]
    assert {r["signal"] for r in rows} == {"selfcheck_nli_rag", "selfcheck_nli_cb"}
    assert all(r["verdict"] in ("жива", "в пул", "убита") for r in rows)


def test_incremental_value_detects_orthogonal_signal():
    from screener import incremental
    rng = np.random.default_rng(0)
    n = 400
    a, b = rng.normal(size=n), rng.normal(size=n)
    labels = (a + b + 0.5 * rng.normal(size=n) < 0).astype(int)     # ошибка ~ a + b
    risk = {"len_norm_rag": a, "mean_sample_len_rag": rng.normal(size=n), "orth": b, "dup": a + 0.01 * rng.normal(size=n)}
    orth = incremental.incremental_value(risk, labels, "orth", n_boot=100)
    dup = incremental.incremental_value(risk, labels, "dup", n_boot=100)
    assert orth["NLL"]["delta"] > 0.08 and orth["NLL"]["ci"][0] > 0
    assert abs(dup["NLL"]["delta"]) < 0.02


def test_synthesis_beats_single_signal_when_signals_are_complementary():
    from screener import synthesis
    rng = np.random.default_rng(0)
    n = 300
    a, b = rng.normal(size=n), rng.normal(size=n)
    labels = (a + b + 0.5 * rng.normal(size=n) < 0).astype(int)
    aux = {"form": np.array(["short"] * n), "split": np.array(["dev"] * 120 + ["test"] * 180)}
    x = {"len_norm_rag": a, "orth": b}
    p_one, _ = synthesis.oof(np.column_stack([a]), 1 - labels, n_repeats=2)
    p_two, _ = synthesis.oof(np.column_stack([a, b]), 1 - labels, n_repeats=2)
    assert synthesis.summarize(p_two, 1 - labels)["auroc"] > synthesis.summarize(p_one, 1 - labels)["auroc"] + 0.08
    sel = synthesis.forward_select(["len_norm_rag", "orth", "noise"])
    cols = sel(np.column_stack([a, b, rng.normal(size=n)]), 1 - labels)
    assert cols[:2] == [0, 1] and 2 not in cols
