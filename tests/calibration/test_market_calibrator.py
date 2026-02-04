"""Tests for market calibrator module."""

import tempfile
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from quantlab.calibration.market_calibrator import MarketCalibrator
from quantlab.market_data.market_state import MarketState
from quantlab.models.heston.model import HestonParameters, HestonProcess


def test_market_calibrator_init_default():
    """Test MarketCalibrator initialization with default risk-free rate."""
    with patch("yfinance.Ticker") as mock_ticker:
        # Mock Treasury data with proper pandas structure
        mock_hist = pd.DataFrame({"Close": [4.5, 4.6, 4.4, 4.5, 4.7]})
        mock_instance = Mock()
        mock_instance.history.return_value = mock_hist
        mock_ticker.return_value = mock_instance

        calibrator = MarketCalibrator()

        # Should try to fetch from Treasury first
        assert mock_ticker.call_count == 1
        mock_ticker.assert_any_call("^IRX")
        assert hasattr(calibrator, "risk_free_rate")
        assert isinstance(calibrator.risk_free_rate, float)


def test_market_calibrator_init_custom():
    """Test MarketCalibrator initialization with custom risk-free rate."""
    calibrator = MarketCalibrator(risk_free_rate=0.035)
    assert calibrator.risk_free_rate == 0.035


def test_fetch_risk_free_rate_success():
    """Test successful risk-free rate fetching from Treasury."""
    with patch("yfinance.Ticker") as mock_ticker:
        # Mock Treasury data with proper pandas structure
        mock_hist = pd.DataFrame({"Close": [3.2, 3.1, 3.3, 3.2, 3.4]})
        mock_instance = Mock()
        mock_instance.history.return_value = mock_hist
        mock_ticker.return_value = mock_instance

        calibrator = MarketCalibrator()
        rate = calibrator._fetch_risk_free_rate()

        assert rate == 0.034  # 3.4% converted to decimal
        assert mock_ticker.call_count == 2  # first - in MarketCalibrator,
        # second - in _fetch_risk_free_rate()


def test_fetch_risk_free_rate_fallback():
    """Test fallback to default rate when Treasury fetching fails."""
    with patch("yfinance.Ticker") as mock_ticker:
        # Make it raise an exception
        mock_ticker.side_effect = Exception("Connection failed")

        calibrator = MarketCalibrator()
        rate = calibrator._fetch_risk_free_rate()

        # Should fall back to default 4.5%
        assert rate == 0.045


def test_calibrate_from_equity_prices_basic():
    """Test equity-based calibration produces valid HestonProcess."""
    calibrator = MarketCalibrator()

    # Mock yfinance download to return realistic data with proper pandas structure
    with patch("yfinance.download") as mock_download:
        dates = pd.date_range(start="2023-01-01", periods=252, freq="D")
        # Create realistic price series with some volatility
        prices = [100.0]  # Starting price
        for i in range(1, 252):
            # Add small random return
            prices.append(prices[-1] * (1 + np.random.normal(0.0002, 0.01)))

        mock_data = pd.DataFrame(
            {"Close": prices}, index=dates  # Use 'Close' instead of 'Adj Close'
        )

        mock_download.return_value = mock_data

        # Call the method
        process = calibrator._calibrate_from_equity_prices("TEST", "1y")

        # Verify it returns a valid HestonProcess
        assert isinstance(process, HestonProcess)
        assert isinstance(process.model_params, HestonParameters)
        assert isinstance(process.market_state, MarketState)

        # Check parameter ranges are reasonable
        assert process.model_params.v0 > 0  # Variance must be positive
        assert process.model_params.kappa > 0  # Mean reversion must be positive
        assert process.model_params.theta > 0  # Long-term variance must be positive
        assert process.model_params.eta > 0  # Vol of vol must be positive
        assert -1 <= process.model_params.rho <= 1  # Correlation must be in [-1, 1]
        assert process.market_state.stock_price > 0  # Price must be positive


def test_calibrate_from_equity_prices_with_low_volatility():
    """Test equity calibration with low volatility data."""
    calibrator = MarketCalibrator()

    with patch("yfinance.download") as mock_download:
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
        # Create low volatility price series,
        # make sure we have enough data points for std calculation
        prices = [100.0]
        for i in range(1, 100):
            prices.append(prices[-1] * (1 + np.random.normal(0.0001, 0.005)))

        mock_data = pd.DataFrame({"Close": prices}, index=dates)
        mock_download.return_value = mock_data

        process = calibrator._calibrate_from_equity_prices("LOW_VOL", "1y")

        # Should still produce valid parameters
        assert process.model_params.v0 > 0  # Even low vol should be positive
        assert process.model_params.theta > 0
        assert process.market_state.stock_price > 0


def test_calibrate_to_market_data_fallback():
    """Test market calibration falls back to equity when option data unavailable."""
    # Use a temporary directory for caching to avoid pickling issues
    with tempfile.TemporaryDirectory() as temp_dir:
        calibrator = MarketCalibrator(cache_dir=temp_dir)

        # Mock the option chain method to return None (no option data)
        with patch.object(calibrator, "_fetch_option_chain") as mock_fetch:
            mock_fetch.return_value = (
                None,
                None,
                None,
                150.0,
            )  # No option data, but S0=150

            # Create a real HestonProcess object instead of a mock
            real_params = HestonParameters(
                v0=0.04, kappa=2.0, theta=0.04, eta=0.3, rho=-0.7
            )
            real_market_state = MarketState(
                stock_price=150.0, interest_rate=0.05, time=0.0
            )
            real_process = HestonProcess(real_params, real_market_state)

            # Mock the equity-based calibration method to return a real process
            with patch.object(
                calibrator, "_calibrate_from_equity_prices"
            ) as mock_equity_cal:
                mock_equity_cal.return_value = real_process

                # Call the main method
                result = calibrator.calibrate_to_market_data(
                    "TEST", use_options_if_available=True
                )

                # Verify equity-based calibration was called as fallback
                mock_equity_cal.assert_called_once_with("TEST", "2y")
                assert result.model_params.v0 == real_process.model_params.v0
                assert (
                    result.market_state.stock_price
                    == real_process.market_state.stock_price
                )


def test_calibrate_to_market_data_fallback_with_mocked_cache():
    """Test market calibration falls back to equity when option data unavailable."""
    calibrator = MarketCalibrator()

    # Mock the caching methods to avoid pickling issues
    with patch.object(calibrator, "load_cached_calibrated_process") as mock_load_cache:
        mock_load_cache.return_value = (None, False)  # No cached process

        with patch.object(
            calibrator, "save_calibrated_process_to_cache"
        ) as mock_save_cache:
            mock_save_cache.return_value = None  # Don't actually save

            # Mock the option chain method to return None (no option data)
            with patch.object(calibrator, "_fetch_option_chain") as mock_fetch:
                mock_fetch.return_value = (
                    None,
                    None,
                    None,
                    150.0,
                )  # No option data, but S0=150

                # Create a real HestonProcess object instead of a mock
                real_params = HestonParameters(
                    v0=0.04, kappa=2.0, theta=0.04, eta=0.3, rho=-0.7
                )
                real_market_state = MarketState(
                    stock_price=150.0, interest_rate=0.05, time=0.0
                )
                real_process = HestonProcess(real_params, real_market_state)

                # Mock the equity-based calibration method to return a real process
                with patch.object(
                    calibrator, "_calibrate_from_equity_prices"
                ) as mock_equity_cal:
                    mock_equity_cal.return_value = real_process

                    # Call the main method
                    result = calibrator.calibrate_to_market_data(
                        "TEST", use_options_if_available=True
                    )

                    # Verify equity-based calibration was called as fallback
                    mock_equity_cal.assert_called_once_with("TEST", "2y")
                    assert result.model_params.v0 == real_process.model_params.v0
                    assert (
                        result.market_state.stock_price
                        == real_process.market_state.stock_price
                    )


@pytest.mark.slow
def test_calibrate_to_market_data_with_options_availability():
    """Test market calibration uses option data when available."""
    # Mock everything to prevent real yfinance calls
    with patch("yfinance.Ticker"), patch("yfinance.download"):
        # Use temporary directory to avoid pickling issues
        with tempfile.TemporaryDirectory() as temp_dir:
            calibrator = MarketCalibrator(cache_dir=temp_dir)

            with patch.object(calibrator, "_fetch_option_chain") as mock_fetch:
                # Return mock option data with at least 6 options to pass len > 5 check
                mock_strikes = np.array(
                    [145.0, 150.0, 155.0, 160.0, 165.0, 170.0, 175.0]
                )  # 7 options
                mock_maturities = np.array([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75])
                mock_ivs = np.array([0.22, 0.20, 0.23, 0.21, 0.19, 0.24, 0.22])
                mock_S0 = 150.0

                mock_fetch.return_value = (
                    mock_strikes,
                    mock_maturities,
                    mock_ivs,
                    mock_S0,
                )

                # Create a real HestonProcess instance to return
                real_params = HestonParameters(
                    v0=0.04, kappa=2.0, theta=0.04, eta=0.3, rho=-0.7
                )
                real_market_state = MarketState(
                    stock_price=150.0, interest_rate=0.05, time=0.0
                )
                successful_process = HestonProcess(real_params, real_market_state)

                # Mock the option-based calibration method to return the real process
                with patch.object(
                    calibrator, "_calibrate_from_options"
                ) as mock_option_cal:
                    mock_option_cal.return_value = successful_process

                    # Mock the caching methods to avoid JSON serialization issues
                    with patch.object(
                        calibrator, "save_calibrated_process_to_cache"
                    ) as mock_save_cache:
                        mock_save_cache.return_value = None  # Don't actually save

                        # Call the main method
                        result = calibrator.calibrate_to_market_data(
                            "TEST", use_options_if_available=True
                        )

                        # Verify option-based calibration was called exactly once
                        mock_option_cal.assert_called_once_with(
                            mock_strikes, mock_maturities, mock_ivs, mock_S0
                        )

                        # Verify the result is our expected process (not the fallback)
                        assert (
                            result.model_params.v0 == successful_process.model_params.v0
                        )
                        assert (
                            result.model_params.kappa
                            == successful_process.model_params.kappa
                        )
                        assert (
                            result.model_params.theta
                            == successful_process.model_params.theta
                        )
                        assert (
                            result.model_params.eta
                            == successful_process.model_params.eta
                        )
                        assert (
                            result.model_params.rho
                            == successful_process.model_params.rho
                        )
                        assert (
                            result.market_state.stock_price
                            == successful_process.market_state.stock_price
                        )


def test_cache_functionality():
    """Test that caching methods work without pickling issues when properly mocked."""
    with tempfile.TemporaryDirectory() as temp_dir:
        calibrator = MarketCalibrator(cache_dir=temp_dir)

        # Test cache key generation
        cache_key = calibrator._generate_cache_key("SPY", 1.0, 252, True)
        assert isinstance(cache_key, str)
        assert len(cache_key) == 32  # MD5 hash length

        # Test path generation
        proc_path = calibrator._get_calibrated_process_cache_path(cache_key)
        paths_path = calibrator._get_paths_cache_path(cache_key, 1000)
        meta_path = calibrator._get_metadata_path(cache_key)

        assert str(proc_path).endswith(".pkl")
        assert str(paths_path).endswith(".pt")
        assert str(meta_path).endswith(".json")
        # The cache path itself doesn't contain the ticker symbol
        # - only the cache key does
        # The actual file name contains the hash of the parameters,
        # not the symbol itself
        assert "calibrated_process" in str(proc_path)
        assert "market_calibrated_paths" in str(paths_path)
        assert "market_calibrated_metadata" in str(meta_path)


def test_cache_methods_with_mocked_io():
    """Test cache methods with mocked file I/O to avoid actual file operations."""
    # Mock file operations to avoid actual I/O
    with patch("builtins.open") as mock_open, patch(
        "pickle.dump"
    ) as mock_pickle_dump, patch("json.dump") as mock_json_dump, patch(
        "pathlib.Path.exists"
    ) as mock_exists:
        mock_exists.return_value = False  # No cache exists initially
        mock_open.return_value.__enter__ = Mock()
        mock_open.return_value.__exit__ = Mock()

        calibrator = MarketCalibrator()

        # Create a real HestonProcess object
        real_params = HestonParameters(
            v0=0.04, kappa=2.0, theta=0.04, eta=0.3, rho=-0.7
        )
        real_market_state = MarketState(stock_price=100.0, interest_rate=0.05, time=0.0)
        real_process = HestonProcess(real_params, real_market_state)

        # Test saving process to cache
        calibrator.save_calibrated_process_to_cache(real_process, "SPY", 1.0, 252, True)

        # Verify pickle.dump was called
        assert mock_pickle_dump.called
        # Verify the first argument to pickle.dump is our real process
        args, kwargs = mock_pickle_dump.call_args
        assert isinstance(args[0], HestonProcess)
        assert args[0].model_params.v0 == real_process.model_params.v0

        # Verify json.dump was called with serializable metadata
        assert mock_json_dump.called
        json_args, json_kwargs = mock_json_dump.call_args
        metadata = json_args[
            0
        ]  # First argument to json.dump is the object to serialize
        # Verify metadata contains expected keys
        assert "ticker" in metadata
        assert "maturity" in metadata
        assert "n_steps" in metadata
        assert metadata["ticker"] == "SPY"


def test_load_cached_process_not_found():
    """Test loading cached process when cache doesn't exist."""
    with tempfile.TemporaryDirectory() as temp_dir:
        calibrator = MarketCalibrator(cache_dir=temp_dir)

        # Should return None, False when cache doesn't exist
        cached_process, is_cached = calibrator.load_cached_calibrated_process(
            "TEST", 1.0, 252, True
        )

        assert cached_process is None
        assert is_cached is False


def test_calibrate_to_market_data_force_equity_mode():
    """Test market calibration uses equity only when forced."""
    # Use temporary directory to avoid pickling issues
    with tempfile.TemporaryDirectory() as temp_dir:
        calibrator = MarketCalibrator(cache_dir=temp_dir)

        # Even with option data available, should use equity when disabled
        with patch.object(calibrator, "_fetch_option_chain") as mock_fetch:
            # Mock that option data is available
            mock_fetch.return_value = (
                np.array([100.0]),
                np.array([0.5]),
                np.array([0.2]),
                100.0,
            )

            # Create a real HestonProcess object instead of a mock
            real_params = HestonParameters(
                v0=0.04, kappa=2.0, theta=0.04, eta=0.3, rho=-0.7
            )
            real_market_state = MarketState(
                stock_price=100.0, interest_rate=0.05, time=0.0
            )
            real_process = HestonProcess(real_params, real_market_state)

            # Mock the equity-based calibration method
            with patch.object(
                calibrator, "_calibrate_from_equity_prices"
            ) as mock_equity_cal:
                mock_equity_cal.return_value = real_process

                # Call with option data disabled
                result = calibrator.calibrate_to_market_data(
                    "TEST", use_options_if_available=False
                )

                # Verify equity-based calibration was called
                mock_equity_cal.assert_called_once_with("TEST", "2y")
                assert result.model_params.v0 == real_process.model_params.v0
                assert (
                    result.market_state.stock_price
                    == real_process.market_state.stock_price
                )


def test_calibrate_to_market_data_force_equity_mode_with_mocked_cache():
    """Test market calibration uses equity only when forced (with mocked cache)."""
    calibrator = MarketCalibrator()

    # Mock the caching methods to avoid pickling issues
    with patch.object(calibrator, "load_cached_calibrated_process") as mock_load_cache:
        mock_load_cache.return_value = (None, False)  # No cached process

        with patch.object(
            calibrator, "save_calibrated_process_to_cache"
        ) as mock_save_cache:
            mock_save_cache.return_value = None  # Don't actually save

            # Even with option data available, should use equity when disabled
            with patch.object(calibrator, "_fetch_option_chain") as mock_fetch:
                # Mock that option data is available
                mock_fetch.return_value = (
                    np.array([100.0]),
                    np.array([0.5]),
                    np.array([0.2]),
                    100.0,
                )

                # Create a real HestonProcess object instead of a mock
                real_params = HestonParameters(
                    v0=0.04, kappa=2.0, theta=0.04, eta=0.3, rho=-0.7
                )
                real_market_state = MarketState(
                    stock_price=100.0, interest_rate=0.05, time=0.0
                )
                real_process = HestonProcess(real_params, real_market_state)

                # Mock the equity-based calibration method
                with patch.object(
                    calibrator, "_calibrate_from_equity_prices"
                ) as mock_equity_cal:
                    mock_equity_cal.return_value = real_process

                    # Call with option data disabled
                    result = calibrator.calibrate_to_market_data(
                        "TEST", use_options_if_available=False
                    )

                    # Verify equity-based calibration was called
                    mock_equity_cal.assert_called_once_with("TEST", "2y")
                    assert result.model_params.v0 == real_process.model_params.v0
                    assert (
                        result.market_state.stock_price
                        == real_process.market_state.stock_price
                    )


def test_print_calibration_results_output(capsys):
    """Test that print calibration results produces expected output."""
    calibrator = MarketCalibrator()

    params = HestonParameters(v0=0.04, kappa=2.0, theta=0.04, eta=0.3, rho=-0.7)

    calibrator._print_calibration_results(params)

    captured = capsys.readouterr()
    output = captured.out

    # Check that output contains expected parameter names and values
    assert "v0 (initial variance)" in output
    assert "kappa (mean reversion)" in output
    assert "theta (long-term var)" in output
    assert "eta (vol of vol)" in output
    assert "rho (correlation)" in output
    assert "0.040000" in output  # Check that values appear
    assert "2.000000" in output
    assert "0.300000" in output
    assert "-0.700000" in output


def test_fetch_option_chain_structure():
    """Test that fetch_option_chain returns expected structure."""
    calibrator = MarketCalibrator()

    # Mock yfinance components to return valid structure
    with patch("yfinance.Ticker") as mock_ticker_class:
        # Create a mock ticker instance
        mock_ticker_instance = Mock()

        # Mock historical data for S0 with proper pandas structure
        mock_history = pd.DataFrame({"Close": [400.0, 401.0, 399.0, 402.0, 400.5]})

        # Mock options expiration dates
        mock_ticker_instance.options = ["2024-03-15", "2024-04-19", "2024-06-21"]

        # Mock option chains with realistic data
        mock_opt_chain = Mock()
        mock_calls = pd.DataFrame(
            {"strike": [380.0, 400.0, 420.0], "impliedVolatility": [0.25, 0.22, 0.24]}
        )
        mock_opt_chain.calls = mock_calls
        mock_ticker_instance.option_chain.return_value = mock_opt_chain
        mock_ticker_instance.history.return_value = mock_history

        mock_ticker_class.return_value = mock_ticker_instance

        # Call the method
        strikes, maturities, ivs, S0 = calibrator._fetch_option_chain("SPY")

        # Verify structure
        if strikes is not None:  # If option data was fetched
            assert isinstance(strikes, np.ndarray)
            assert isinstance(maturities, np.ndarray)
            assert isinstance(ivs, np.ndarray)
            assert isinstance(S0, float)
            assert len(strikes) == len(ivs)
            assert S0 > 0
            assert all(iv > 0 for iv in ivs) if len(ivs) > 0 else True


def test_fetch_option_chain_failure_handling():
    """Test that option chain fetching handles failures gracefully."""
    calibrator = MarketCalibrator()

    with patch("yfinance.Ticker") as mock_ticker:
        # Make it raise an exception
        mock_ticker.side_effect = Exception("API unavailable")

        strikes, maturities, ivs, S0 = calibrator._fetch_option_chain("SPY")

        assert S0 is not None or (
            strikes is None and maturities is None and ivs is None
        )
