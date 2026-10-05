"""ATS: формула loss и применение головы — на маленьких тензорах."""
from __future__ import annotations

import math

import pytest
import torch

from dump_assembly import ats


def test_loss_matches_formula():
    logits = torch.tensor([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
    targets = torch.tensor([0, 2])          # 1-й токен угадан, 2-й нет
    tau = torch.zeros(2)
    loss = ats.selective_smoothing_loss(logits, tau, targets, alpha=0.5)
    lp0 = torch.log_softmax(logits[0], -1)
    lp1 = torch.log_softmax(logits[1], -1)
    expected = (-(0.5) * lp0[0] + -(0.5 / 3) * lp1.sum()) / 2
    assert loss.item() == pytest.approx(expected.item(), rel=1e-5)


def test_zero_head_is_identity_and_temperature_sharpens():
    logits = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    hidden = torch.randn(2, 4)
    targets = torch.tensor([0, 0])
    head = ats.make_head(4)
    s0 = ats.ats_token_stats(logits, hidden, targets, head)
    nll_plain = -torch.log_softmax(logits, -1)[[0, 1], [0, 0]].mean().item()
    assert s0["ats_nll"] == pytest.approx(nll_plain) and s0["ats_mean_tau"] == 0.0
    with torch.no_grad():
        head.bias.fill_(math.log(3.0))     # e^τ = 3 -> распределения острее
    s1 = ats.ats_token_stats(logits, hidden, targets, head)
    assert s1["ats_entropy"] < s0["ats_entropy"]


def test_training_reduces_loss_on_overconfident_model():
    torch.manual_seed(0)
    h = torch.randn(200, 8)
    targets = torch.randint(0, 5, (200,))
    logits = torch.randn(200, 5) * 6          # самоуверенная модель, часто не права
    head = ats.make_head(8)
    opt = torch.optim.AdamW(head.parameters(), lr=0.05)
    first = ats.selective_smoothing_loss(logits, head(h).squeeze(-1), targets).item()
    for _ in range(200):
        loss = ats.selective_smoothing_loss(logits, head(h).squeeze(-1), targets)
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < first and head(h).mean().item() < 0   # научилась «охлаждать» (τ<0)
