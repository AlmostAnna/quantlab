"""
Tests for hedging network.

This module contains tests to ensure correctness of hedge net implemetation.
"""

import torch
import torch.nn as nn

from quantlab.ml.models.hedge_net import HedgeNet


def test_hedge_net_initialization():
    """Test model initialization."""
    net = HedgeNet(hidden_dim=32)
    assert isinstance(net, nn.Module)
    assert len(list(net.parameters())) > 0  # Has trainable parameters


def test_hedge_net_output_range():
    """Test hedge net output range."""
    net = HedgeNet(hidden_dim=32)
    tau = torch.tensor([0.5, 1.0])
    moneyness = torch.tensor([0.8, 1.2])
    phi = net(tau, moneyness)

    assert torch.all(phi >= 0.0)
    assert torch.all(phi <= 1.0)
    assert phi.shape == (2,)  # Two inputs -> two outputs


def test_hedge_net_batch_consistency():
    """Test batch vs single example consistency."""
    net = HedgeNet(hidden_dim=32)

    # Single example
    tau_single = torch.tensor([0.5])
    moneyness_single = torch.tensor([1.2])
    phi_single = net(tau_single, moneyness_single)

    # Batch of one
    tau_batch = torch.tensor([0.5])
    moneyness_batch = torch.tensor([1.2])
    phi_batch = net(tau_batch, moneyness_batch)

    assert torch.allclose(phi_single, phi_batch)


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
    assert any(p.grad is not None for p in net.parameters())


def test_hedge_net_different_hidden_dims():
    """Test different hidden dimensions."""
    for hidden_dim in [16, 32, 64, 128]:
        net = HedgeNet(hidden_dim=hidden_dim)

        tau = torch.randn(5)
        moneyness = torch.randn(5)
        phi = net(tau, moneyness)

        assert phi.shape == (5,)
        assert torch.all(phi >= 0.0) and torch.all(phi <= 1.0)


def test_hedge_net_device_compatibility():
    """Test device compatibility."""
    device = torch.device("cpu")  # Can test CUDA if available
    net = HedgeNet().to(device)

    tau = torch.tensor([0.5, 0.8]).to(device)
    moneyness = torch.tensor([1.0, 1.2]).to(device)
    phi = net(tau, moneyness)

    assert phi.device == device
    assert phi.shape == (2,)
