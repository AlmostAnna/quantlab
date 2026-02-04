"""Tests for market calibrator path generation functions."""

import tempfile
from unittest.mock import Mock, patch

import pytest
import torch

from quantlab.calibration.market_calibrator import generate_market_calibrated_paths


def test_generate_market_calibrated_paths_success():
    """Test successful path generation returns expected tensor structure."""
    # Use temporary directory to avoid caching issues
    with tempfile.TemporaryDirectory():
        # Mock the calibration and simulation
        with patch(
            "quantlab.calibration.market_calibrator.MarketCalibrator"
        ) as mock_calibrator_class:
            # Mock the calibrator instance
            mock_calibrator = Mock()

            mock_calibrator.load_cached_paths.return_value = (
                None,
                False,
            )  # No cached paths
            mock_calibrator.save_paths_to_cache.return_value = (
                None  # Don't actually save
            )

            # Mock the calibration
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

                # Call the function with caching enabled (default)
                result = generate_market_calibrated_paths(
                    ticker="TEST",
                    n_paths=100,
                    maturity=1.0,
                    n_steps=252,
                    use_cache=True,
                )

                # Verify calls: caching should be attempted
                mock_calibrator.load_cached_paths.assert_called_once_with(
                    "TEST", 100, 1.0, 252, True
                )
                mock_calibrator.calibrate_to_market_data.assert_called_once_with(
                    "TEST", period="2y", use_options_if_available=True
                )
                mock_sim.assert_called_once_with(
                    mock_process, T=1.0, N=100, M=252, device="cpu"
                )

                # Verify result structure
                assert isinstance(result, torch.Tensor)
                assert result.shape == (100, 253)  # (n_paths, n_steps + 1)


def test_generate_market_calibrated_paths_different_parameters():
    """Test path generation with different parameters."""
    with tempfile.TemporaryDirectory():
        with patch(
            "quantlab.calibration.market_calibrator.MarketCalibrator"
        ) as mock_calibrator_class:
            mock_calibrator = Mock()
            mock_calibrator.load_cached_paths.return_value = (
                None,
                False,
            )  # No cached paths
            mock_calibrator.save_paths_to_cache.return_value = (
                None  # Don't actually save
            )

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
                    ticker="SPY", n_paths=50, maturity=0.5, n_steps=126, use_cache=True
                )

                # Verify correct parameters passed
                mock_calibrator.load_cached_paths.assert_called_once_with(
                    "SPY", 50, 0.5, 126, True
                )
                mock_sim.assert_called_once_with(
                    mock_process, T=0.5, N=50, M=126, device="cpu"
                )
                assert result.shape == (50, 127)


def test_generate_market_calibrated_paths_tensor_properties():
    """Test that generated paths have expected tensor properties."""
    with tempfile.TemporaryDirectory():
        with patch(
            "quantlab.calibration.market_calibrator.MarketCalibrator"
        ) as mock_calibrator_class:
            mock_calibrator = Mock()
            mock_calibrator.load_cached_paths.return_value = (
                None,
                False,
            )  # No cached paths
            mock_calibrator.save_paths_to_cache.return_value = (
                None  # Don't actually save
            )

            mock_process = Mock()
            mock_calibrator.calibrate_to_market_data.return_value = mock_process
            mock_calibrator_class.return_value = mock_calibrator

            with patch(
                "quantlab.calibration.market_calibrator.simulate_heston_paths_torch"
            ) as mock_sim:
                # Create realistic path data (should be positive prices)
                mock_paths = (
                    torch.ones(10, 21) * 100.0
                )  # Start at 100, all same initially
                # Add some variation
                mock_paths += torch.randn(10, 21) * 5.0  # Add some noise
                mock_paths = torch.clamp(mock_paths, min=0.1)  # Ensure positive
                mock_sim.return_value = (mock_paths, Mock())

                result = generate_market_calibrated_paths(
                    ticker="TEST", n_paths=10, maturity=1.0, n_steps=20, use_cache=True
                )

                # Verify tensor properties
                assert result.dtype == torch.float  # Should be float type
                assert not torch.isnan(result).any()  # Should not contain NaN
                assert not torch.isinf(result).any()  # Should not contain Inf
                assert (result >= 0).all()  # Prices should be non-negative


def test_generate_market_calibrated_paths_with_exception():
    """Test path generation handles calibration exceptions."""
    with tempfile.TemporaryDirectory():
        with patch(
            "quantlab.calibration.market_calibrator.MarketCalibrator"
        ) as mock_calibrator_class:
            # Mock the calibrator instance with caching methods
            mock_calibrator = Mock()
            mock_calibrator.load_cached_paths.return_value = (
                None,
                False,
            )  # No cached paths
            mock_calibrator.save_paths_to_cache.return_value = (
                None  # Don't actually save
            )

            # Make calibration raise an exception
            mock_calibrator.calibrate_to_market_data.side_effect = Exception(
                "Calibration failed"
            )
            mock_calibrator_class.return_value = mock_calibrator

            # This should propagate the exception or handle it appropriately
            # Based on your implementation, this would likely raise the exception
            with pytest.raises(Exception, match="Calibration failed"):
                generate_market_calibrated_paths(
                    ticker="TEST", n_paths=10, maturity=1.0, n_steps=252, use_cache=True
                )


def test_generate_market_calibrated_paths_device_handling():
    """Test that the function handles device parameter correctly."""
    with tempfile.TemporaryDirectory():
        with patch(
            "quantlab.calibration.market_calibrator.MarketCalibrator"
        ) as mock_calibrator_class:
            mock_calibrator = Mock()
            mock_calibrator.load_cached_paths.return_value = (
                None,
                False,
            )  # No cached paths
            mock_calibrator.save_paths_to_cache.return_value = (
                None  # Don't actually save
            )

            mock_process = Mock()
            mock_calibrator.calibrate_to_market_data.return_value = mock_process
            mock_calibrator_class.return_value = mock_calibrator

            with patch(
                "quantlab.calibration.market_calibrator.simulate_heston_paths_torch"
            ) as mock_sim:
                mock_paths_tensor = torch.randn(5, 11)
                mock_sim.return_value = (mock_paths_tensor, Mock())

                generate_market_calibrated_paths(
                    ticker="TEST", n_paths=5, maturity=0.25, n_steps=10, use_cache=True
                )

                # Verify that CPU device was specified
                mock_sim.assert_called_once_with(
                    mock_process, T=0.25, N=5, M=10, device="cpu"
                )


def test_generate_market_calibrated_paths_with_caching():
    """Test path generation with caching functionality."""
    # Use temporary directory to test caching
    with tempfile.TemporaryDirectory():
        with patch(
            "quantlab.calibration.market_calibrator.MarketCalibrator"
        ) as mock_calibrator_class:
            mock_calibrator = Mock()
            # Simulate that cached paths are available
            cached_tensor = torch.randn(10, 21)
            mock_calibrator.load_cached_paths.return_value = (
                cached_tensor,
                True,
            )  # Cached paths found

            # Don't need to mock calibration since cached paths are returned
            mock_calibrator_class.return_value = mock_calibrator

            # Call the function WITH caching enabled
            result = generate_market_calibrated_paths(
                ticker="TEST", n_paths=10, maturity=1.0, n_steps=20, use_cache=True
            )

            # Should return cached paths without calling simulation
            assert torch.equal(result, cached_tensor)
            # Verify that calibration was not called since cached paths were used
            mock_calibrator.calibrate_to_market_data.assert_not_called()


def test_generate_market_calibrated_paths_bypass_caching():
    """Test path generation bypasses caching when disabled."""
    with tempfile.TemporaryDirectory():
        with patch(
            "quantlab.calibration.market_calibrator.MarketCalibrator"
        ) as mock_calibrator_class:
            mock_calibrator = Mock()
            # Even if cached paths exist,
            # they shouldn't be used when caching is disabled
            mock_calibrator.load_cached_paths.return_value = (torch.randn(10, 21), True)

            mock_process = Mock()
            mock_calibrator.calibrate_to_market_data.return_value = mock_process
            mock_calibrator_class.return_value = mock_calibrator

            with patch(
                "quantlab.calibration.market_calibrator.simulate_heston_paths_torch"
            ) as mock_sim:
                mock_paths_tensor = torch.randn(10, 21)
                mock_sim.return_value = (mock_paths_tensor, Mock())

                # Call with caching disabled
                result = generate_market_calibrated_paths(
                    ticker="TEST", n_paths=10, maturity=1.0, n_steps=20, use_cache=False
                )

                # Should call calibration and simulation despite cached paths existing
                mock_calibrator.calibrate_to_market_data.assert_called_once()
                mock_sim.assert_called_once()
                assert result.shape == (10, 21)

                # Verify that caching methods were NOT called when use_cache=False
                mock_calibrator.load_cached_paths.assert_not_called()


def test_generate_market_calibrated_paths_default_behavior():
    """Test that the function defaults to using caching."""
    with tempfile.TemporaryDirectory():
        with patch(
            "quantlab.calibration.market_calibrator.MarketCalibrator"
        ) as mock_calibrator_class:
            mock_calibrator = Mock()
            mock_calibrator.load_cached_paths.return_value = (
                None,
                False,
            )  # No cached paths
            mock_calibrator.save_paths_to_cache.return_value = (
                None  # Don't actually save
            )

            mock_process = Mock()
            mock_calibrator.calibrate_to_market_data.return_value = mock_process
            mock_calibrator_class.return_value = mock_calibrator

            with patch(
                "quantlab.calibration.market_calibrator.simulate_heston_paths_torch"
            ) as mock_sim:
                mock_paths_tensor = torch.randn(5, 11)
                mock_sim.return_value = (mock_paths_tensor, Mock())

                # Call without specifying use_cache (should default to True)
                generate_market_calibrated_paths(
                    ticker="TEST", n_paths=5, maturity=0.25, n_steps=10
                )

                # Should attempt to load cached paths
                mock_calibrator.load_cached_paths.assert_called_once()
                # Should call simulation after finding no cache
                mock_sim.assert_called_once()
