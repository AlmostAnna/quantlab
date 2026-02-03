"""Tests for market calibrator path generation functions."""

from unittest.mock import Mock, patch

import pytest
import torch

from quantlab.calibration.market_calibrator import generate_market_calibrated_paths


def test_generate_market_calibrated_paths_success():
    """Test successful path generation returns expected tensor structure."""
    # Mock the calibration and simulation
    with patch(
        "quantlab.calibration.market_calibrator.MarketCalibrator"
    ) as mock_calibrator_class:
        # Mock the calibrator instance
        mock_calibrator = Mock()
        mock_process = Mock()
        mock_calibrator.calibrate_to_market_data.return_value = mock_process

        mock_calibrator_class.return_value = mock_calibrator

        # Mock the simulation function
        with patch(
            "quantlab.calibration.market_calibrator.simulate_heston_paths_torch"
        ) as mock_sim:
            # Create mock tensor result
            mock_paths_tensor = torch.randn(100, 253)  # (n_paths, n_steps + 1)
            mock_sim.return_value = (mock_paths_tensor, Mock())  # (paths, other)

            # Call the function
            result = generate_market_calibrated_paths(
                ticker="TEST", n_paths=100, maturity=1.0, n_steps=252
            )

            # Verify calls
            mock_calibrator.calibrate_to_market_data.assert_called_once_with(
                "TEST", period="2y"
            )
            mock_sim.assert_called_once_with(
                mock_process, T=1.0, N=100, M=252, device="cpu"
            )

            # Verify result structure
            assert isinstance(result, torch.Tensor)
            assert result.shape == (100, 253)  # (n_paths, n_steps + 1)


def test_generate_market_calibrated_paths_different_parameters():
    """Test path generation with different parameters."""
    with patch(
        "quantlab.calibration.market_calibrator.MarketCalibrator"
    ) as mock_calibrator_class:
        mock_calibrator = Mock()
        mock_process = Mock()
        mock_calibrator.calibrate_to_market_data.return_value = mock_process
        mock_calibrator_class.return_value = mock_calibrator

        with patch(
            "quantlab.calibration.market_calibrator.simulate_heston_paths_torch"
        ) as mock_sim:
            # Test with different parameters
            mock_paths_tensor = torch.randn(50, 127)  # (50 paths, 126 steps + 1)
            mock_sim.return_value = (mock_paths_tensor, Mock())

            result = generate_market_calibrated_paths(
                ticker="SPY", n_paths=50, maturity=0.5, n_steps=126
            )

            # Verify correct parameters passed
            mock_sim.assert_called_once_with(
                mock_process, T=0.5, N=50, M=126, device="cpu"
            )
            assert result.shape == (50, 127)


def test_generate_market_calibrated_paths_tensor_properties():
    """Test that generated paths have expected tensor properties."""
    with patch(
        "quantlab.calibration.market_calibrator.MarketCalibrator"
    ) as mock_calibrator_class:
        mock_calibrator = Mock()
        mock_process = Mock()
        mock_calibrator.calibrate_to_market_data.return_value = mock_process
        mock_calibrator_class.return_value = mock_calibrator

        with patch(
            "quantlab.calibration.market_calibrator.simulate_heston_paths_torch"
        ) as mock_sim:
            # Create realistic path data (should be positive prices)
            mock_paths = torch.ones(10, 21) * 100.0  # Start at 100, all same initially
            # Add some variation
            mock_paths += torch.randn(10, 21) * 5.0  # Add some noise
            mock_paths = torch.clamp(mock_paths, min=0.1)  # Ensure positive
            mock_sim.return_value = (mock_paths, Mock())

            result = generate_market_calibrated_paths(
                ticker="TEST", n_paths=10, maturity=1.0, n_steps=20
            )

            # Verify tensor properties
            assert result.dtype == torch.float  # Should be float type
            assert not torch.isnan(result).any()  # Should not contain NaN
            assert not torch.isinf(result).any()  # Should not contain Inf
            assert (result >= 0).all()  # Prices should be non-negative


def test_generate_market_calibrated_paths_with_exception():
    """Test path generation handles calibration exceptions."""
    with patch(
        "quantlab.calibration.market_calibrator.MarketCalibrator"
    ) as mock_calibrator_class:
        # Make calibration raise an exception
        mock_calibrator = Mock()
        mock_calibrator.calibrate_to_market_data.side_effect = Exception(
            "Calibration failed"
        )
        mock_calibrator_class.return_value = mock_calibrator

        # This should propagate the exception or handle it appropriately
        # Based on your implementation, this would likely raise the exception
        with pytest.raises(Exception, match="Calibration failed"):
            generate_market_calibrated_paths(
                ticker="TEST", n_paths=10, maturity=1.0, n_steps=252
            )


def test_generate_market_calibrated_paths_device_handling():
    """Test that the function handles device parameter correctly."""
    with patch(
        "quantlab.calibration.market_calibrator.MarketCalibrator"
    ) as mock_calibrator_class:
        mock_calibrator = Mock()
        mock_process = Mock()
        mock_calibrator.calibrate_to_market_data.return_value = mock_process
        mock_calibrator_class.return_value = mock_calibrator

        with patch(
            "quantlab.calibration.market_calibrator.simulate_heston_paths_torch"
        ) as mock_sim:
            mock_paths_tensor = torch.randn(5, 11)
            mock_sim.return_value = (mock_paths_tensor, Mock())

            generate_market_calibrated_paths(
                ticker="TEST", n_paths=5, maturity=0.25, n_steps=10
            )

            # Verify that CPU device was specified
            mock_sim.assert_called_once_with(
                mock_process, T=0.25, N=5, M=10, device="cpu"
            )
