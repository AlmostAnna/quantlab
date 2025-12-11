# tests/ml/test_hedge_net.py
"""
Tests for hedging network.

This module contains tests to ensure basic correctness of hedge net implemetation.
"""

import torch

from quantlab.ml.models.hedge_net import HedgeNet


def test_hedge_net_output_range():
    """Test hedge net output range."""
    net = HedgeNet(hidden_dim=32)
    tau = torch.tensor([0.5, 1.0])
    moneyness = torch.tensor([0.8, 1.2])
    phi = net(tau, moneyness)
    assert torch.all(phi >= 0.0)
    assert torch.all(phi <= 1.0)


def test_hedge_net_gradients():
    """Test hedge net gradients."""
    net = HedgeNet()
    tau = torch.tensor([0.5], requires_grad=True)
    moneyness = torch.tensor([1.0], requires_grad=True)
    phi = net(tau, moneyness)
    loss = phi.sum()
    loss.backward()
    assert tau.grad is not None
    assert moneyness.grad is not None
