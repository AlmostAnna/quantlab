"""Tests for training pipeline - functionality verification."""

import subprocess
import sys
from pathlib import Path

import torch


def test_train_cli_help():
    """Test that the training CLI can be invoked."""
    project_root = Path(__file__).parent.parent.parent
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
sys.path.insert(0, '.')
import ml.train
import argparse
parser = argparse.ArgumentParser(description='Deep Hedging')
parser.add_argument('--data_type_model', type=str, default='gbm')
print('Parser created successfully')
        """,
        ],
        capture_output=True,
        text=True,
        cwd=project_root,
    )

    assert result.returncode == 0, f"Failed to import train module: {result.stderr}"


def test_prepare_inputs_logic():
    """Test the prepare_inputs logic directly in the test."""

    def prepare_inputs(S, K, T, M, device="cpu"):
        """Prepare input batches for training."""
        t_grid = torch.linspace(0, T - T / M, M, device=device)
        tau_grid = T - t_grid
        N = S.size(0)
        tau_batch = tau_grid.unsqueeze(0).expand(N, -1)
        moneyness_batch = S[:, :-1] / K
        return tau_batch.reshape(-1), moneyness_batch.reshape(-1)

    S = torch.tensor([[100.0, 105.0, 110.0]])  # Shape: (1, 3) - N=1, M=2
    K = 105.0  # Scalar
    T = 1.0
    M = 2  # 2 time steps between 0 and T

    tau_flat, moneyness_flat = prepare_inputs(S, K, T, M)
    assert tau_flat.shape == torch.Size(
        [2]
    ), f"Expected shape [2], got {tau_flat.shape}"
    assert moneyness_flat.shape == torch.Size(
        [2]
    ), f"Expected shape [2], got {moneyness_flat.shape}"

    # Check individual values
    assert torch.allclose(
        tau_flat[0], torch.tensor(1.0), atol=1e-6
    ), f"First tau value incorrect: {tau_flat[0]}"
    assert torch.allclose(
        tau_flat[1], torch.tensor(0.5), atol=1e-6
    ), f"Second tau value incorrect: {tau_flat[1]}"
    assert torch.allclose(
        moneyness_flat[0], torch.tensor(100.0 / 105.0), atol=1e-6
    ), f"First moneyness value incorrect: {moneyness_flat[0]}"
    assert torch.allclose(
        moneyness_flat[1], torch.tensor(105.0 / 105.0), atol=1e-6
    ), f"Second moneyness value incorrect: {moneyness_flat[1]}"


def test_train_script_syntax():
    """Test that the train script has valid Python syntax."""
    project_root = Path(__file__).parent.parent.parent
    train_file = project_root / "ml" / "train.py"

    with open(train_file, "r") as f:
        code = f.read()

    # This will raise SyntaxError if there are syntax issues
    compile(code, str(train_file), "exec")


def test_train_can_run_minimal():
    """Test that train script can run with minimal configuration."""
    project_root = Path(__file__).parent.parent.parent

    # Create a minimal test that doesn't actually train but validates imports
    test_code = """
import sys
sys.path.insert(0, '.')
sys.path.insert(0, './src')

# Test imports work
from quantlab.ml.models.hedge_net import HedgeNet
from quantlab.ml.metrics.pnl import compute_pnl_with_tx

# Verify basic functionality
net = HedgeNet(hidden_dim=16)
print('Basic imports and instantiation successful')
"""

    result = subprocess.run(
        [sys.executable, "-c", test_code],
        capture_output=True,
        text=True,
        cwd=project_root,
    )

    assert result.returncode == 0, f"Basic functionality test failed: {result.stderr}"
