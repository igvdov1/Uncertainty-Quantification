"""mech_cards / mech_fit без модели: формулы эталонов и подгонка на синтетике."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from dump_assembly import mech_cards as mc
from dump_assembly import mech_fit as mf


def test_sink_rate_matches_reference_definition():
    n_total, resp = 6, 2
    attn = np.zeros((n_total, n_total))
    attn[4, 1] = attn[5, 1] = 0.9         # оба токена ответа смотрят в позицию 1
    attn[5, 5] = 0.1
    # столбец 1: 1.8 / 2 (видят обе строки ответа) = 0.9; столбец 5: 0.1 / 1 = 0.1
    assert mc.sink_rate(attn, resp) == pytest.approx(0.9)


def test_jsd_ref_zero_for_identical_and_symmetric():
    a, b = torch.randn(3, 10), torch.randn(3, 10)
    assert torch.allclose(mc.jsd_ref(a, a), torch.zeros(3), atol=0.05)   # шкала ×1e6 эталона
    assert torch.allclose(mc.jsd_ref(a, b), mc.jsd_ref(b, a), rtol=1e-4)


def test_lumina_ipr_formula():
    ent = torch.tensor([[1.0], [2.0]])            # 2 слоя, 1 токен
    gp = torch.tensor([[0.2], [0.4]])
    ipr = mc.lumina_ipr(ent, gp, torch.tensor([0.8]), torch.tensor([0.8]))
    ratios = torch.tensor([1 - 0.25, 1 - 0.5])
    expected = (ratios[0] * 1 + ratios[1] * 2) / (1 * 1.0 + 2 * 0.5)
    assert ipr.item() == pytest.approx(expected.item(), rel=1e-5)


def test_lumina_mmd_zero_for_same_distribution():
    emb = torch.nn.Embedding(50, 8)
    p = torch.softmax(torch.randn(4, 50), -1)
    assert torch.allclose(mc.lumina_mmd(p, p, emb, k=10), torch.zeros(4), atol=1e-5)
    q = torch.softmax(torch.randn(4, 50) * 5, -1)
    assert (mc.lumina_mmd(p, q, emb, k=10) >= -1e-6).all()


def test_conditional_entropy():
    assert mf.conditional_entropy([0, 0, 1, 1], [0, 0, 1, 1]) == 0.0      # источник определён смыслом
    assert mf.conditional_entropy([0, 1, 0, 1], [0, 0, 0, 0]) == pytest.approx(np.log(2))
    assert np.isnan(mf.conditional_entropy([-1, -1], [0, 0]))


def test_fit_redeep_and_intrygue_recover_signal():
    rng = np.random.default_rng(0)
    n = 200
    err = rng.integers(0, 2, n)
    ecs = rng.normal(size=(n, 6)); ecs[:, 3] -= 1.5 * err             # голова 3 информативна (ниже ECS при ошибке)
    pks = rng.normal(size=(n, 5)); pks[:, 2] += 1.5 * err             # слой 2 информативен
    p = mf.fit_redeep(ecs, pks, err)
    assert p["head_order"][0] == 3 and p["layer_order"][0] == 2 and p["dev_auroc"] > 0.8
    ent = np.abs(rng.normal(size=(n, 2))) + 1
    sinks = rng.random((n, 8)); sinks[:, 0] += err
    assert mf.fit_intrygue_n(ent, sinks, err, "mean") >= 1


def test_tad_two_stage_learns():
    rng = np.random.default_rng(1)
    feats, conf = [], []
    for i in range(80):
        c = i % 2
        T = rng.integers(3, 8)
        att = rng.normal(size=(T, 12)).astype(np.float16); att[:, 0] += 2 * c
        logp = np.log(rng.uniform(0.1, 1, size=(T, 4))); logp[0, 1:] = np.nan
        feats.append((att, logp.astype(np.float32))); conf.append(c)
    conf = np.array(conf)
    m1, m2 = mf.fit_tad(feats[:60], conf[:60])
    scores = np.array([mf.tad_score(f, m1, m2) for f in feats[60:]])
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(1 - conf[60:], scores) > 0.8


def test_run_end_to_end_on_fake_features():
    rng = np.random.default_rng(2)
    meta, feats = {}, {}
    for i in range(140):
        split = "dev" if i < 70 else "test"
        lab = i % 2
        q = f"nq_{i}"
        meta[q] = {"split": split, "source": "nq", "label_factual": lab, "sem_ids": [0, 0, 1, 1, 0],
                   "cb_correct": [bool(lab)] * 6}
        h = rng.normal(size=16).astype(np.float16); h[0] += 3 * (1 - lab)
        feats[q] = {"intrygue_entropy": np.array([2.0, 1.0]) + (1 - lab), "intrygue_sink": rng.random(18),
                    "redeep_ecs": rng.normal(size=32), "redeep_pks": rng.normal(size=32),
                    "lumina": np.array([0.3, 0.1]), "ats": np.array([1.0, 2.0, 0.5, 0.0]),
                    "tad_att": rng.normal(size=(4, 12)).astype(np.float16),
                    "tad_logp": np.full((4, 4), -0.5, np.float32), "hack_h15": h, "source_ids": np.array([0, 1, 0, 1, 2])}
    out, rep = mf.run(meta, feats)
    assert np.isfinite(out["nq_0"]["intrygue_minmax_k5"]) and "redeep" not in out["nq_0"]   # dev — без подогнанных
    assert all(k in out["nq_100"] for k in ("intrygue_tuned", "redeep", "tad", "hack_dontknow", "src_given_sem"))
    assert rep["hack_dev_counts"] == {"hk_plus": 35, "consistently_correct": 35}
