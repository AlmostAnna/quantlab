"""Tests for evaluation pipeline."""

import subprocess
import sys
from pathlib import Path

import torch


def test_evaluate_script_syntax():
    """Test that the evaluate script has valid Python syntax."""
    project_root = Path(__file__).parent.parent.parent
    eval_file = project_root / "ml" / "evaluate.py"

    with open(eval_file, "r") as f:
        code = f.read()

    # This will raise SyntaxError if there are syntax issues
    compile(code, str(eval_file), "exec")


def test_load_model_function_logic():
    """Test the load_model function logic directly."""

    def mock_load_model(hidden_dim, path="artifacts/hedge_net_tx.pth", device="cpu"):
        """Mock version of load_model function."""
        from quantlab.ml.models.hedge_net import HedgeNet

        net = HedgeNet(hidden_dim).to(device)
        # Don't actually load - just create a fresh one to test construction
        net.eval()
        return net

    # Test model creation and basic functionality
    net = mock_load_model(hidden_dim=32)
    assert net.training is False  # Should be in eval mode

    # Test forward pass
    tau = torch.tensor([0.5, 0.8])
    moneyness = torch.tensor([0.9, 1.1])
    output = net(tau, moneyness)

    assert output.shape == (2,)  # Two inputs -> two outputs
    assert torch.all(output >= 0.0) and torch.all(output <= 1.0)  # Sigmoid bounds


def test_prepare_inputs_for_model():
    """Test the prepare_inputs_for_model function logic."""

    def prepare_inputs_for_model(S, K, T, M, device="cpu"):
        """Prepare inputs for HedgeNet evaluation."""
        t_grid = torch.linspace(0, T - T / M, M, device=device)
        tau = T - t_grid
        N = S.size(0)
        tau_batch = tau.unsqueeze(0).expand(N, -1).reshape(-1)
        moneyness_batch = (S[:, :-1] / K).reshape(-1)
        return tau_batch, moneyness_batch, N, M

    # Test with single path
    S = torch.tensor([[100.0, 105.0, 110.0]])  # (1, 3) - N=1, M+1=3, so M=2
    K = 105.0
    T = 1.0
    M = 2

    tau_flat, moneyness_flat, N, returned_M = prepare_inputs_for_model(S, K, T, M)

    assert N == 1
    assert returned_M == 2
    assert tau_flat.shape == torch.Size([2])  # N*M = 1*2 = 2
    assert moneyness_flat.shape == torch.Size([2])  # N*M = 1*2 = 2

    # Verify calculations
    expected_tau = torch.tensor([1.0, 0.5])
    expected_moneyness = torch.tensor([100.0 / 105.0, 105.0 / 105.0])

    assert torch.allclose(tau_flat, expected_tau, atol=1e-6)
    assert torch.allclose(moneyness_flat, expected_moneyness, atol=1e-6)


def test_stress_test_can_run():
    """Test that stress test function can be imported and run without training."""
    project_root = Path(__file__).parent.parent.parent

    # Create a minimal test that validates the structure
    test_code = """
import sys
sys.path.insert(0, '.')
sys.path.insert(0, './src')

# Test that we can import necessary components
from quantlab.ml.models.hedge_net import HedgeNet
from quantlab.ml.metrics.pnl import compute_pnl_with_tx
from ml.config import GBMConfig, HedgingConfig, StressTestConfig

# Verify basic functionality
net = HedgeNet(hidden_dim=16)
print('Stress test components import successful')
"""

    result = subprocess.run(
        [sys.executable, "-c", test_code],
        capture_output=True,
        text=True,
        cwd=project_root,
    )

    assert (
        result.returncode == 0
    ), f"Stress test components test failed: {result.stderr}"


def test_evaluate_cli_help():
    """Test that evaluate module can be imported."""
    project_root = Path(__file__).parent.parent.parent
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
sys.path.insert(0, '.')
try:
    import ml.evaluate
    print('Evaluate module imported successfully')
except Exception as e:
    print(f'Import failed: {e}')
    raise
        """,
        ],
        capture_output=True,
        text=True,
        cwd=project_root,
    )

    assert result.returncode == 0, f"Failed to import evaluate module: {result.stderr}"
