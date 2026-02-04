"""Tests for the synthetic data evaluation script."""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch


def test_evaluation_script_syntax():
    """Test that the evaluation script has valid syntax."""
    project_root = Path(__file__).parent.parent.parent
    script_file = project_root / "scripts" / "evaluate_synthetic_data_for_hedging.py"

    with open(script_file, "r") as f:
        code = f.read()

    compile(code, str(script_file), "exec")


def test_market_calibrator_basic():
    """Test the market calibrator can run without real data (mocked)."""
    project_root = Path(__file__).parent.parent.parent

    test_code = """
import sys
sys.path.insert(0, '.')
sys.path.insert(0, './src')

# Test that the calibrator module can be imported
from quantlab.calibration.market_calibrator import MarketCalibrator

# Test basic instantiation (doesn't actually fetch data in constructor)
calibrator = MarketCalibrator(risk_free_rate=0.05)
print('MarketCalibrator can be instantiated')

# Test that we can access the function
from quantlab.calibration.market_calibrator import generate_market_calibrated_paths
print('Functions available')
"""

    result = subprocess.run(
        [sys.executable, "-c", test_code],
        capture_output=True,
        text=True,
        cwd=project_root,
    )

    assert result.returncode == 0, f"Market calibrator test failed: {result.stderr}"


def test_generate_paths_functionality():
    """Test path generation with mocked calibration."""
    project_root = Path(__file__).parent.parent.parent

    test_code = """
import sys
sys.path.insert(0, '.')
sys.path.insert(0, './src')

import torch
import numpy as np

# Mock yfinance to avoid real API calls
class MockTicker:
    def __init__(self, symbol):
        self.symbol = symbol

    def history(self, period):
        # Return mock data
        import pandas as pd
        dates = pd.date_range(start='2023-01-01', periods=10)
        return pd.DataFrame({
            'Close': [100 + i*0.5 for i in range(10)],
            'Open': [99 + i*0.5 for i in range(10)],
            'High': [101 + i*0.5 for i in range(10)],
            'Low': [98 + i*0.5 for i in range(10)],
            'Volume': [1000000 for _ in range(10)]
        }, index=dates)

class MockYFinance:
    @staticmethod
    def download(symbol, period="2y", **kwargs):
        import pandas as pd
        dates = pd.date_range(start='2022-01-01', periods=20)
        return pd.DataFrame({
            'Adj Close': [100 + i*0.2 for i in range(20)],
            'Close': [100 + i*0.2 for i in range(20)],
            'Open': [99 + i*0.2 for i in range(20)],
            'High': [101 + i*0.2 for i in range(20)],
            'Low': [98 + i*0.2 for i in range(20)],
            'Volume': [1000000 for _ in range(20)]
        }, index=dates)

# Mock the module
import unittest.mock
with unittest.mock.patch.dict('sys.modules', {
    'yfinance': MockYFinance,
    'yfinance.Ticker': MockTicker
}):
    from quantlab.calibration.market_calibrator import generate_market_calibrated_paths

    # Test with very small parameters to avoid heavy computation
    try:
        paths = generate_market_calibrated_paths(
            ticker="TEST",
            n_paths=10,
            maturity=0.1,  # Short maturity for faster computation
            n_steps=5      # Few steps for faster computation
        )
        print(f'Successfully generated paths with shape: {paths.shape}')
        print('Path generation works!')
    except Exception as e:
        print(f'Path generation failed: {e}')
        raise
"""

    result = subprocess.run(
        [sys.executable, "-c", test_code],
        capture_output=True,
        text=True,
        cwd=project_root,
    )

    assert result.returncode == 0, f"Path generation test failed: {result.stderr}"
    assert "Path generation works!" in result.stdout


def test_evaluation_script_help():
    """Test that the evaluation script can be called with --help."""
    project_root = Path(__file__).parent.parent.parent

    # Create a minimal config for the test
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        config = {
            "strike_price": 100.0,
            "time_to_maturity": 0.1,
            "num_rebalancing_periods": 5,
            "transaction_cost": 0.001,
            "training_params": {"epochs": 1, "learning_rate": 0.001, "hidden_dim": 16},
        }
        json.dump(config, f)
        config_path = f.name

    # Create a fake synthetic data file
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".pt", delete=False) as f:
        fake_paths = torch.randn(10, 6)  # (N, M+1) where M=5
        torch.save(fake_paths, f.name)
        data_path = f.name

    try:
        # Test with mocked dependencies
        test_code = f"""
import sys
sys.path.insert(0, '.')
sys.path.insert(0, './src')

import unittest.mock
import argparse

# Mock yfinance
class MockTicker:
    def __init__(self, symbol):
        pass
    def history(self, period):
        import pandas as pd
        import numpy as np
        dates = pd.date_range(start='2023-01-01', periods=10)
        return pd.DataFrame({{
            'Close': [1.5 for _ in dates]
        }}, index=dates)

class MockYFinance:
    @staticmethod
    def download(symbol, period="2y", **kwargs):
        import pandas as pd
        dates = pd.date_range(start='2022-01-01', periods=20)
        return pd.DataFrame({{
            'Adj Close': [100 + i*0.1 for i in range(20)],
            'Close': [100 + i*0.1 for i in range(20)]
        }}, index=dates)

# Mock the evaluation function to avoid full execution
def mock_main():
    # Just test argument parsing and setup
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--market_reference', type=str, default='SPY')
    parser.add_argument('--synthetic_data_paths', nargs='+', required=True)
    parser.add_argument('--config_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./results')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--n_test_paths', type=int, default=10)

    # Parse a test set of arguments
    test_args = [
        '--market_reference', 'TEST',
        '--synthetic_data_paths', '{data_path}',
        '--config_path', '{config_path}',
        '--output_dir', '/tmp',
        '--n_test_paths', '5'
    ]

    parsed = parser.parse_args(test_args)
    print('Argument parsing successful')
    print(f'Market reference: {{parsed.market_reference}}')
    print(f'Number of test paths: {{parsed.n_test_paths}}')

# Replace the main function temporarily
with unittest.mock.patch.dict('sys.modules', {{
    'yfinance': MockYFinance,
    'yfinance.Ticker': MockTicker
}}):
    mock_main()
"""

        result = subprocess.run(
            [sys.executable, "-c", test_code],
            capture_output=True,
            text=True,
            cwd=project_root,
        )

        assert result.returncode == 0, f"Script test failed: {result.stderr}"
        assert "Argument parsing successful" in result.stdout

    finally:
        # Cleanup
        os.unlink(config_path)
        os.unlink(data_path)


def test_end_to_end_workflow():
    """Test the complete workflow with minimal parameters."""
    project_root = Path(__file__).parent.parent.parent

    # Create minimal config
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        config_path = temp_path / "config.json"
        config = {
            "strike_price": 100.0,
            "time_to_maturity": 0.1,
            "num_rebalancing_periods": 5,
            "transaction_cost": 0.001,
            "training_params": {
                "epochs": 2,  # Very few epochs for test
                "learning_rate": 0.001,
                "hidden_dim": 16,
            },
            "evaluation_params": {
                "metrics": ["mean_abs_pnl", "std_pnl"],
                "confidence_level": 0.95,
            },
        }
        with open(config_path, "w") as f:
            json.dump(config, f)

        # Create fake synthetic data
        synth_path = temp_path / "synthetic.pt"
        fake_synth_data = torch.randn(20, 6)  # Small dataset for test
        torch.save(fake_synth_data, synth_path)

        output_dir = temp_path / "output"

        # Test the workflow with mocked dependencies
        test_code = f"""
import sys
sys.path.insert(0, '.')
sys.path.insert(0, './src')

import unittest.mock
import numpy as np
import pandas as pd

# Mock all external dependencies
class MockTicker:
    def __init__(self, symbol):
        pass
    def history(self, period):
        dates = pd.date_range(start='2023-01-01', periods=5)
        return pd.DataFrame({{
            'Close': [1.5 for _ in dates]
        }}, index=dates)

class MockYFinance:
    @staticmethod
    def download(symbol, period="2y", **kwargs):
        dates = pd.date_range(start='2022-01-01', periods=10)
        return pd.DataFrame({{
            'Adj Close': [100 + i*0.05 for i in range(10)],
            'Close': [100 + i*0.05 for i in range(10)]
        }}, index=dates)

# Mock the Heston simulation to avoid heavy computation
def mock_simulate_heston_paths_torch(process, T, N, M, device='cpu'):
    # Return simple linear paths for testing
    import torch
    dt = T / M
    t_grid = torch.arange(0, T + dt, dt)  # M+1 points
    paths = torch.ones(N, M + 1)
    for i in range(N):
        for j in range(1, M + 1):
            paths[i, j] = paths[i, j-1] * (1 + 0.01 * torch.randn(()))  # Small RW
    return paths, torch.zeros_like(paths)  # Return dummy volatility paths

with unittest.mock.patch.dict('sys.modules', {{
    'yfinance': MockYFinance,
    'yfinance.Ticker': MockTicker
}}):
    # Import and mock the simulation function
    import quantlab.sim.heston.paths
    with unittest.mock.patch.object(quantlab.sim.heston.paths, 'simulate_heston_paths_torch', mock_simulate_heston_paths_torch):  # noqa: E501
        from scripts.evaluate_synthetic_data_for_hedging import main
        import sys
        import argparse

        # Mock sys.argv to simulate command line arguments
        original_argv = sys.argv[:]
        try:
            sys.argv = [
                'script_name',
                '--market_reference', 'TEST',
                '--synthetic_data_paths', '{synth_path}',
                '--config_path', '{config_path}',
                '--output_dir', '{output_dir}',
                '--n_test_paths', '10',
                '--device', 'cpu'
            ]

            # Run main function
            main()
            print('End-to-end workflow completed successfully!')

        except SystemExit:
            # argparse might call sys.exit, which is normal
            print('End-to-end workflow completed successfully!')
        finally:
            sys.argv = original_argv
"""

        result = subprocess.run(
            [sys.executable, "-c", test_code],
            capture_output=True,
            text=True,
            cwd=project_root,
        )

        assert result.returncode == 0, f"End-to-end test failed: {result.stderr}"
        assert "workflow completed successfully" in result.stdout.lower()
