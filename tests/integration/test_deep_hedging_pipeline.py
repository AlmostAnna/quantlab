"""End-to-end integration tests for the deep hedging pipeline."""

import subprocess
import sys
from pathlib import Path

import torch


def test_full_training_evaluation_cycle():
    """Test a minimal training-evaluation cycle."""
    project_root = Path(__file__).parent.parent.parent

    # Create a minimal training test that doesn't actually train much
    train_code = """
import sys
sys.path.insert(0, '.')
sys.path.insert(0, './src')

import torch
import argparse
from unittest.mock import patch, MagicMock

# Mock the heavy components to run quickly
with patch('ml.train.simulate_gbm') as mock_simulate:
    # Mock small dataset
    mock_S = torch.tensor([[[100.0, 105.0, 110.0]]]).float().expand(10, -1, -1)  # 10 small paths # noqa: E501
    mock_simulate.return_value = mock_S.squeeze(1)  # Return (10, 3)

    with patch('ml.train.HedgingConfig') as mock_hedge_cfg:
        mock_cfg = MagicMock()
        mock_cfg.device = "cpu"
        mock_cfg.K = 105.0
        mock_cfg.lambda_tx = 0.01
        mock_cfg.hidden_dim = 16  # Small model
        mock_cfg.lr = 0.001
        mock_cfg.epochs = 1  # Just 1 epoch for speed
        mock_hedge_cfg.return_value = mock_cfg

        with patch('ml.train.GBMConfig') as mock_gbm_cfg:
            mock_gbm_cfg_inst = MagicMock()
            mock_gbm_cfg_inst.S0 = 100.0
            mock_gbm_cfg_inst.sigma = 0.2
            mock_gbm_cfg_inst.r = 0.05
            mock_gbm_cfg_inst.T = 1.0
            mock_gbm_cfg_inst.N = 10
            mock_gbm_cfg_inst.M = 2
            mock_gbm_cfg.return_value = mock_gbm_cfg_inst

            # Temporarily redirect model saving
            import os
            os.makedirs('temp_artifacts', exist_ok=True)

            with patch('torch.save') as mock_save:
                with patch('os.makedirs'):
                    # Create Args class like CLI would
                    class Args:
                        data_type_model = "gbm"

                    from ml.train import main
                    main(Args())

                    # Verify save was called
                    assert mock_save.called
                    print('Training completed successfully')

# Clean up temp files
import shutil
shutil.rmtree('temp_artifacts', ignore_errors=True)
"""

    result = subprocess.run(
        [sys.executable, "-c", train_code],
        capture_output=True,
        text=True,
        cwd=project_root,
    )

    assert result.returncode == 0, f"Training cycle failed: {result.stderr}"
    assert "Training completed successfully" in result.stdout


def test_pnl_computation_integration():
    """Test that P&L computation works correctly with HedgeNet outputs."""
    # Test the full pipeline: simulate paths -> get hedge ratios -> compute P&L
    from quantlab.ml.metrics.pnl import compute_pnl_with_tx, decompose_pnl
    from quantlab.ml.models.hedge_net import HedgeNet

    # Create a simple test case: 2 paths, 4 time steps (so M=3)
    S = torch.tensor(
        [[[100.0, 105.0, 110.0, 115.0], [100.0, 95.0, 90.0, 85.0]]]
    )  # Shape: (1, 2, 4) -> (N, M+1) where N=2, M+1=4, so M=3
    S = S.squeeze(0)  # Now shape: (2, 4) - 2 paths, 4 time steps
    N, M_plus_1 = S.shape  # N=2, M+1=4, so M=3

    # Create a simple model that returns constant hedge ratio
    net = HedgeNet(hidden_dim=16)

    # Create inputs following the exact logic from prepare_inputs
    K = 105.0
    T = 1.0
    M = M_plus_1 - 1  # M = 3

    t_grid = torch.linspace(0, T - T / M, M)  # M=3 time steps
    tau_grid = T - t_grid
    tau_batch = tau_grid.unsqueeze(0).expand(N, -1).reshape(-1)  # (N*M,) = (6,)
    moneyness_batch = (S[:, :-1] / K).reshape(-1)  # (N*M,) = (6,) - S[:, :-1] is (2, 3)

    # Verify shapes match before feeding to model
    assert (
        tau_batch.shape == moneyness_batch.shape
    ), f"Shape mismatch: {tau_batch.shape} vs {moneyness_batch.shape}"

    # Get hedge ratios from model
    with torch.no_grad():
        phi_flat = net(tau_batch, moneyness_batch)  # (N*M,) = (6,)
        phi = phi_flat.reshape(N, M)  # (N, M) = (2, 3)

    # Compute P&L
    lambda_tx = 0.01
    pnl = compute_pnl_with_tx(S, K, phi, lambda_tx)
    total_pnl, hedging_gain, tx_cost = decompose_pnl(S, K, phi, lambda_tx)

    # Basic sanity checks
    assert pnl.shape == (N,)  # One P&L per path
    assert torch.allclose(
        pnl, total_pnl
    )  # Decomposition should match direct calculation
    assert torch.all(tx_cost >= 0)  # Transaction costs should be non-negative

    # Verify decomposition
    payoff = torch.clamp(S[:, -1] - K, min=0.0)  # Final payoff
    calculated_pnl = payoff - hedging_gain - tx_cost
    assert torch.allclose(pnl, calculated_pnl, atol=1e-5)


def test_model_risk_sensitivity():
    """Test sensitivity to model risk factors."""
    from quantlab.ml.metrics.pnl import compute_pnl_with_tx
    from quantlab.ml.models.hedge_net import HedgeNet

    # Create test paths with different characteristics
    # Shape: (N, M+1) where N=2, M+1=4, so M=3
    S = torch.tensor(
        [
            [100.0, 105.0, 110.0, 115.0],  # Relatively stable path
            [100.0, 110.0, 90.0, 120.0],  # More volatile path
        ]
    )  # (2, 4)

    net = HedgeNet(hidden_dim=16)
    K = 105.0
    lambda_tx = 0.01

    # Generate inputs following the same logic as in training
    N, M_plus_1 = S.shape  # N=2, M+1=4, so M=3
    M = M_plus_1 - 1

    t_grid = (
        torch.linspace(0, 1.0 - 1.0 / M, M) if M > 1 else torch.tensor([0.0])
    )  # Handle edge case
    tau_grid = 1.0 - t_grid
    tau_batch = tau_grid.unsqueeze(0).expand(N, -1).reshape(-1)  # (N*M,) = (6,)
    moneyness_batch = (S[:, :-1] / K).reshape(-1)  # (N*M,) = (6,) - S[:, :-1] is (2, 3)

    # Verify shapes match
    assert (
        tau_batch.shape == moneyness_batch.shape
    ), f"Shape mismatch: {tau_batch.shape} vs {moneyness_batch.shape}"

    with torch.no_grad():
        phi_flat = net(tau_batch, moneyness_batch)
        phi = phi_flat.reshape(N, M)  # (2, 3)

    # Compute P&L for both paths
    pnl = compute_pnl_with_tx(S, K, phi, lambda_tx)

    # Both should have reasonable P&L values (not NaN or infinite)
    assert not torch.any(torch.isnan(pnl))
    assert not torch.any(torch.isinf(pnl))
    assert pnl.shape == (N,)  # One P&L per path
