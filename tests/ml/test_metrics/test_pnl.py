"""Tests for P&L computation functions."""

import pytest
import torch

from quantlab.ml.metrics.pnl import compute_pnl_with_tx, decompose_pnl


def test_compute_pnl_with_tx_basic():
    """Test basic P&L calculation."""
    S = torch.tensor([[100.0, 105.0, 110.0]])  # Single path, 3 time steps
    K = 105.0
    phi = torch.tensor([[0.5, 0.6]])  # Holdings at t0, t1
    lambda_tx = 0.01

    pnl = compute_pnl_with_tx(S, K, phi, lambda_tx)

    expected_payoff = max(110.0 - 105.0, 0)  # 5.0
    expected_hedging_gain = 0.5 * (105.0 - 100.0) + 0.6 * (
        110.0 - 105.0
    )  # 2.5 + 3.0 = 5.5
    expected_tx_cost = 0.01 * abs(0.6 - 0.5) * 105.0  # 0.01 * 0.1 * 105 = 0.105
    expected_pnl = expected_payoff - expected_hedging_gain - expected_tx_cost

    assert torch.isclose(pnl[0], torch.tensor(expected_pnl), atol=1e-5)


def test_compute_pnl_with_tx_no_transactions():
    """Test P&L with no trading."""
    S = torch.tensor([[100.0, 105.0, 110.0]])
    K = 105.0
    phi = torch.tensor([[0.5, 0.5]])  # No change in holdings
    lambda_tx = 0.01

    pnl = compute_pnl_with_tx(S, K, phi, lambda_tx)

    # Should have no transaction costs
    expected_payoff = 5.0
    expected_hedging_gain = 0.5 * (105.0 - 100.0) + 0.5 * (110.0 - 105.0)  # 5.0
    expected_pnl = expected_payoff - expected_hedging_gain  # 0.0

    assert torch.isclose(pnl[0], torch.tensor(expected_pnl), atol=1e-5)


def test_decompose_pnl():
    """Test P&L decomposition."""
    S = torch.tensor([[100.0, 105.0, 110.0]])
    K = 105.0
    phi = torch.tensor([[0.5, 0.6]])
    lambda_tx = 0.01

    total_pnl, hedging_gain, tx_cost = decompose_pnl(S, K, phi, lambda_tx)

    assert total_pnl.shape == (1,)
    assert hedging_gain.shape == (1,)
    assert tx_cost.shape == (1,)

    # Verify decomposition
    pnl_direct = compute_pnl_with_tx(S, K, phi, lambda_tx)
    assert torch.allclose(total_pnl, pnl_direct)


def test_pnl_edge_cases():
    """Test edge cases."""
    # Out-of-the-money option
    S = torch.tensor([[100.0, 95.0, 90.0]])
    K = 105.0
    phi = torch.tensor([[0.0, 0.0]])
    lambda_tx = 0.0

    pnl = compute_pnl_with_tx(S, K, phi, lambda_tx)
    assert torch.isclose(pnl[0], torch.tensor(0.0))  # Zero payoff, zero hedging gain

    # At-the-money
    S = torch.tensor([[100.0, 100.0, 100.0]])
    K = 100.0
    pnl = compute_pnl_with_tx(S, K, phi, lambda_tx)
    assert torch.isclose(pnl[0], torch.tensor(0.0))


@pytest.mark.parametrize(
    "shape",
    [
        ((1, 2), (1, 1)),  # Minimal case
        ((10, 5), (10, 4)),  # Multiple paths
        ((1, 100), (1, 99)),  # Long time series
    ],
)
def test_pnl_shapes(shape):
    """Test various input shapes."""
    S_shape, phi_shape = shape
    S = torch.randn(S_shape)
    K = 100.0
    phi = torch.rand(phi_shape)
    lambda_tx = 0.01

    pnl = compute_pnl_with_tx(S, K, phi, lambda_tx)
    assert pnl.shape[0] == S_shape[0]  # Same number of samples


def test_pnl_gradient_flow():
    """Test that gradients flow through P&L computation."""
    S = torch.tensor([[100.0, 105.0, 110.0]], requires_grad=True)
    K = torch.tensor(105.0, requires_grad=True)
    phi = torch.tensor([[0.5, 0.6]], requires_grad=False)
    lambda_tx = 0.01

    pnl = compute_pnl_with_tx(S, K, phi, lambda_tx)
    pnl.sum().backward()

    assert S.grad is not None
    assert K.grad is not None
