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
    table, _, _, _, _ = run_screener.build_signal_table_and_claims(records)
    risk = run_screener.to_risk_scores(table, registry.signal_polarity())
    names, matrix = correlation.spearman_matrix(risk)
    n = len(names)
    assert matrix.shape == (n, n)
    assert np.allclose(matrix, matrix.T, equal_nan=True)
    diag = np.diag(matrix)
    finite_diag = diag[~np.isnan(diag)]
    assert np.allclose(finite_diag, 1.0)


def test_paired_bootstrap_matches_point_estimate(records):
    table, _, labels, _, _ = run_screener.build_signal_table_and_claims(records)
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
    _, _, _, claim_data, _ = run_screener.build_signal_table_and_claims(records)
    tau = run_screener.claim_level_kendall(claim_data, target="factual")
    assert not math.isnan(tau)
    assert -1.0 <= tau <= 1.0


def test_calibration_metrics_run_on_probability_like_signal(records):
    table, _, labels, _, _ = run_screener.build_signal_table_and_claims(records)
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
