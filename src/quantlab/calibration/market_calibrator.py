"""
Market-calibrated data generator for deep hedging evaluation.

This module calibrates stochastic models to publicly available market data
and generates synthetic paths for model evaluation.
"""
import hashlib
import json
import pickle
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from quantlab.calibration.inverse import recover_heston_params_from_implied_vols
from quantlab.calibration.utils import make_heston_object_wrapper
from quantlab.market_data.market_state import MarketState
from quantlab.models.heston.model import HestonParameters, HestonProcess
from quantlab.pricing.heston.cos import price as cos_price
from quantlab.sim.heston.paths import simulate_heston_paths_torch

warnings.filterwarnings("ignore")


class MarketCalibrator:
    """Calibrates Heston model to market data for realistic synthetic data generation."""  # noqa: E501

    def __init__(self, risk_free_rate=None, cache_dir="./cache"):
        """
        Initialize with optional risk-free rate and cache directory.

        Args:
            risk_free_rate: Risk free rate.
                        If None, will fetch from Treasury data.
            cache_dir: Directory to store cached calibrated processes and paths
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        if risk_free_rate is None:
            self.risk_free_rate = self._fetch_risk_free_rate()
        else:
            self.risk_free_rate = risk_free_rate

    def _fetch_risk_free_rate(self, maturity_years=1.0):
        """
        Fetch appropriate risk-free rate for the given maturity.

        Args:
            maturity_years: Maturity of the derivatives being hedged
        """
        try:
            # Map maturity to appropriate Treasury security
            if maturity_years <= 1.0:
                ticker = "^IRX"  # 3-month Treasury
            elif maturity_years <= 5.0:
                ticker = "^FVX"  # 5-year Treasury
            elif maturity_years <= 10.0:
                ticker = "^TNX"  # 10-year Treasury
            else:  # Longer maturity
                ticker = "^TYX"  # 30-year Treasury

            treasury = yf.Ticker(ticker)
            hist = treasury.history(period="5d")
            rate_percent = hist["Close"].iloc[-1]  # Annual percentage rate
            return rate_percent / 100  # Convert to decimal

        except Exception as e:
            print(f"Warning: Could not fetch Treasury data: {e}")
            # Fallback: reasonable estimate based on maturity
            if maturity_years <= 1.0:
                return 0.045  # Short-term rate
            else:
                return 0.050  # Long-term rate

    def _generate_cache_key(self, ticker, maturity, n_steps, use_options_if_available):
        """Generate a unique cache key for the given parameters."""
        params_str = f"{ticker}_{maturity}_{n_steps}_{use_options_if_available}_{self.risk_free_rate}"  # noqa: E501
        return hashlib.md5(params_str.encode()).hexdigest()

    def _get_calibrated_process_cache_path(self, cache_key):
        """Get the path for cached calibrated process."""
        return self.cache_dir / f"calibrated_process_{cache_key}.pkl"

    def _get_paths_cache_path(self, cache_key, n_paths):
        """Get the path for cached paths."""
        return self.cache_dir / f"market_calibrated_paths_{cache_key}_n{n_paths}.pt"

    def _get_metadata_path(self, cache_key):
        """Get the path for cache metadata."""
        return self.cache_dir / f"market_calibrated_metadata_{cache_key}.json"

    def load_cached_calibrated_process(
        self, ticker="SPY", maturity=1.0, n_steps=252, use_options_if_available=True
    ):
        """
        Load cached calibrated process if available.

        Args:
            ticker: Equity symbol used for calibration
            maturity: Time to maturity in years
            n_steps: Number of time steps per path
            use_options_if_available: Whether to use options data for calibration

        Returns:
            Calibrated HestonProcess and bool indicating if cached
        """
        cache_key = self._generate_cache_key(
            ticker, maturity, n_steps, use_options_if_available
        )
        cache_path = self._get_calibrated_process_cache_path(cache_key)
        metadata_path = self._get_metadata_path(cache_key)

        if cache_path.exists() and metadata_path.exists():
            # Verify metadata matches
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            expected_metadata = {
                "ticker": ticker,
                "maturity": maturity,
                "n_steps": n_steps,
                "use_options_if_available": use_options_if_available,
                "risk_free_rate": self.risk_free_rate,
            }

            if metadata.get("calibration_parameters") == expected_metadata:
                print(f"Loading cached calibrated process from {cache_path}")
                with open(cache_path, "rb") as f:
                    return pickle.load(f), True

        print(f"No cached calibrated process found for {ticker}, {maturity}y maturity")
        return None, False

    def save_calibrated_process_to_cache(
        self,
        process,
        ticker="SPY",
        maturity=1.0,
        n_steps=252,
        use_options_if_available=True,
    ):
        """
        Save calibrated process to cache.

        Args:
            process: Calibrated HestonProcess object
            ticker: Equity symbol used for calibration
            maturity: Time to maturity in years
            n_steps: Number of time steps per path
            use_options_if_available: Whether options data was used
        """
        cache_key = self._generate_cache_key(
            ticker, maturity, n_steps, use_options_if_available
        )
        cache_path = self._get_calibrated_process_cache_path(cache_key)
        metadata_path = self._get_metadata_path(cache_key)

        # Save process
        with open(cache_path, "wb") as f:
            pickle.dump(process, f)

        # Save metadata
        metadata = {
            "ticker": ticker,
            "maturity": maturity,
            "n_steps": n_steps,
            "use_options_if_available": use_options_if_available,
            "risk_free_rate": self.risk_free_rate,
            "generated_at": datetime.now().isoformat(),
            "calibration_method": "market_calibrated_heston",
            "calibration_parameters": {
                "ticker": ticker,
                "maturity": maturity,
                "n_steps": n_steps,
                "use_options_if_available": use_options_if_available,
                "risk_free_rate": self.risk_free_rate,
            },
        }

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved calibrated process to cache: {cache_path}")

    def load_cached_paths(
        self,
        ticker="SPY",
        n_paths=10000,
        maturity=1.0,
        n_steps=252,
        use_options_if_available=True,
    ):
        """
        Load cached market-calibrated paths if available.

        Args:
            ticker: Equity symbol used for calibration
            n_paths: Number of paths to generate/load
            maturity: Time to maturity in years
            n_steps: Number of time steps per path
            use_options_if_available: Whether options data was used for calibration

        Returns:
            torch.Tensor of paths and bool indicating if cached
        """
        cache_key = self._generate_cache_key(
            ticker, maturity, n_steps, use_options_if_available
        )
        cache_path = self._get_paths_cache_path(cache_key, n_paths)

        if cache_path.exists():
            print(f"Loading cached paths from {cache_path}")
            import torch

            return torch.load(cache_path), True

        print(
            f"No cached paths found for {ticker}, {n_paths} paths, {maturity}y maturity"
        )
        return None, False

    def save_paths_to_cache(
        self,
        paths,
        ticker="SPY",
        n_paths=10000,
        maturity=1.0,
        n_steps=252,
        use_options_if_available=True,
    ):
        """
        Save generated paths to cache.

        Args:
            paths: torch.Tensor of generated paths
            ticker: Equity symbol used for calibration
            n_paths: Number of paths
            maturity: Time to maturity in years
            n_steps: Number of time steps
            use_options_if_available: Whether options data was used
        """
        cache_key = self._generate_cache_key(
            ticker, maturity, n_steps, use_options_if_available
        )
        cache_path = self._get_paths_cache_path(cache_key, n_paths)

        import torch

        # Save paths
        torch.save(paths, cache_path)
        print(f"Saved {n_paths} paths to cache: {cache_path}")

    def _fetch_option_chain(self, ticker="SPY"):
        """Fetch option chain data for calibration."""
        try:
            stock = yf.Ticker(ticker)

            # Get current stock price from historical data
            hist = stock.history(period="5d")
            S0 = hist["Close"].iloc[-1]

            # Get available expiration dates (using near-term options for calibration)
            exp_dates = stock.options[:3]  # Use first 3 expiration dates

            if not exp_dates:
                return None, None, None, S0

            strikes = []
            maturities = []
            implied_vols = []

            today = datetime.today()

            for exp_date in exp_dates:
                try:
                    # Check if expiration date is in the future
                    expiry = datetime.strptime(exp_date, "%Y-%m-%d")
                    if expiry <= today:
                        print(f"Skipping expired option date: {exp_date}")
                        continue

                    # Get options for this expiration
                    opt = stock.option_chain(exp_date)

                    # Filter to reasonable strikes around current price
                    atm_strike = round(S0, -1)  # Round to nearest 10
                    strike_range = [
                        atm_strike - 40,
                        atm_strike + 40,
                    ]  # 80-strike range around ATM

                    # Use calls with valid implied volatility and reasonable volume
                    calls = opt.calls[
                        (opt.calls["strike"] >= strike_range[0])
                        & (opt.calls["strike"] <= strike_range[1])
                        & (opt.calls["impliedVolatility"] > 0)
                        & (opt.calls["impliedVolatility"] < 1.0)  # valid IVs
                        & (opt.calls["volume"] > 10)  # Exclude extremely high IVs
                        & (pd.notna(opt.calls["lastPrice"]))  # At least some volume
                        & (opt.calls["lastPrice"] > 0)
                    ].copy()

                    if len(calls) == 0:
                        continue

                    # Calculate time to maturity in years
                    T = (expiry - today).days / 365.25
                    if T <= 0:
                        continue

                    for _, row in calls.iterrows():
                        if row["impliedVolatility"] > 0 and row["lastPrice"] > 0:
                            strikes.append(row["strike"])
                            maturities.append(T)
                            implied_vols.append(row["impliedVolatility"])

                except Exception as e:
                    print(f"Warning: Could not get options for {exp_date}: {e}")
                    continue

            if len(strikes) > 5:  # Only return if we have enough data points
                return (
                    np.array(strikes),
                    np.array(maturities),
                    np.array(implied_vols),
                    S0,
                )
            else:
                return None, None, None, S0  # Not enough options data

        except Exception as e:
            print(f"Warning: Could not fetch option chain for {ticker}: {e}")
            print("Falling back to equity-based calibration...")

            # Still try to get S0 from equity data
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period="5d")
                S0 = hist["Close"].iloc[-1]
                return None, None, None, S0
            except Exception as e:
                print(f"Could not extract S0: {e}")
                return None, None, None, None

    def _calibrate_from_options(self, strikes, maturities, ivs, S0):
        """Calibrate using option market data."""
        # Create market state
        market_state = MarketState(
            stock_price=S0, interest_rate=self.risk_free_rate, time=0.0
        )

        # Initial guess based on market conditions
        initial_guess = {
            "kappa": 2.0,
            "theta": np.mean(ivs) ** 2,  # Rough estimate from average IV
            "eta": 0.3,
            "rho": -0.7,
            "v0": np.mean(ivs) ** 2,  # Start with average IV squared
        }

        # Calibration bounds
        bounds = {
            "kappa": (0.1, 10.0),
            "theta": (0.001, 0.5),
            "eta": (0.01, 2.0),
            "rho": (-0.99, 0.99),
            "v0": (0.001, 0.5),
        }

        # Create wrapper
        cos_wrapper = make_heston_object_wrapper(
            pricer_func=cos_price,
            market_state_for_calibration=market_state,
            pricer_kwargs={"n_points": 2048},
        )

        # Perform calibration
        calibrated_params = recover_heston_params_from_implied_vols(
            strikes=strikes,
            maturities=maturities,
            target_implied_vols=ivs,
            market_state=market_state,
            initial_guess=initial_guess,
            pricing_func=cos_wrapper,
            pricing_kwargs={},
            bounds=bounds,
            weights=None,  # Equal weighting
            method="differential_evolution",
            optimizer_options={
                "maxiter": 100,
                "seed": 42,
                "polish": True,
                "disp": True,
            },
            verbose=False,
        )

        calibrated_heston_params = HestonParameters(
            v0=calibrated_params["v0"],
            kappa=calibrated_params["kappa"],
            theta=calibrated_params["theta"],
            eta=calibrated_params["eta"],
            rho=calibrated_params["rho"],
        )

        print("Option-based calibration successful!")
        self._print_calibration_results(calibrated_heston_params)

        return HestonProcess(calibrated_heston_params, market_state)

    def _calibrate_from_equity_prices(self, ticker, period):
        """Calibrate using equity price returns."""
        # Fetch historical data
        data = yf.download(ticker, period=period)
        prices = data["Close"].values
        returns = np.diff(np.log(prices))

        # Only proceed if we have enough data points for meaningful statistics
        if len(returns) < 2:
            print(f"Warning: Insufficient data for {ticker}, using default parameters")
            S0 = prices[-1] if len(prices) > 0 else 100.0
        else:
            # Calculate target statistics
            target_vol = np.std(returns) * np.sqrt(252)
            target_drift = np.mean(returns) * 252

            print(f"Target volatility: {target_vol:.4f}")
            print(f"Target drift: {target_drift:.4f}")

            S0 = prices[-1]

        # Create market state and parameters
        market_state = MarketState(
            stock_price=S0,
            interest_rate=self.risk_free_rate,
            time=0.0,
        )

        initial_params = HestonParameters(
            v0=target_vol**2,  # Square of target volatility
            kappa=2.0,  # Mean reversion speed (typical value)
            theta=target_vol**2,  # Long-term variance (matches target)
            eta=0.3,  # Vol of vol (typical value)
            rho=-0.7,  # Leverage effect (typical for equities)
        )

        print("Equity-based calibration successful!")
        self._print_calibration_results(initial_params)

        return HestonProcess(initial_params, market_state)

    def _print_calibration_results(self, params: HestonParameters):
        """Print calibration results for debugging."""
        print("Calibrated parameters:")
        print(f"  v0 (initial variance): {params.v0:.6f}")
        print(f"  kappa (mean reversion): {params.kappa:.6f}")
        print(f"  theta (long-term var): {params.theta:.6f}")
        print(f"  eta (vol of vol): {params.eta:.6f}")
        print(f"  rho (correlation): {params.rho:.6f}")

    def calibrate_to_market_data(
        self, ticker="SPY", period="2y", use_options_if_available=True
    ):
        """
        Calibrate Heston model to market data.

        Args:
            ticker: Stock/ETF symbol (e.g., 'SPY', 'QQQ')
            period: Historical period ('1y', '2y', '5y')
            use_options_if_available: Whether to try option data first

        Returns:
            Calibrated HestonProcess object
        """
        print(f"Calibrating Heston model to {ticker} market data...")

        # Check if we have cached calibrated process
        # Use 1y maturity, 252 steps as proxy for calibration cache
        cached_process, is_cached = self.load_cached_calibrated_process(
            ticker, 1.0, 252, use_options_if_available
        )

        if is_cached:
            return cached_process

        if use_options_if_available:
            strikes, maturities, ivs, S0 = self._fetch_option_chain(ticker)

            if strikes is not None and len(strikes) > 5:  # Have enough option data
                print(f"Found {len(strikes)} option quotes, using for calibration...")
                calibrated_process = self._calibrate_from_options(
                    strikes, maturities, ivs, S0
                )
            else:
                print(
                    "Insufficient option data, falling back to equity-based calibration..."  # noqa: E501
                )
                calibrated_process = self._calibrate_from_equity_prices(ticker, period)
        else:
            print("Using equity price data for calibration...")
            calibrated_process = self._calibrate_from_equity_prices(ticker, period)

        # Cache the calibrated process
        self.save_calibrated_process_to_cache(
            calibrated_process, ticker, 1.0, 252, use_options_if_available
        )

        return calibrated_process


def generate_market_calibrated_paths(
    ticker="SPY",
    n_paths=10000,
    maturity=1.0,
    n_steps=252,
    use_cache=True,
    use_options_if_available=True,
):
    """
    Generate market-calibrated synthetic paths for evaluation.

    Optionally load from cache if available.

    Args:
        ticker: Equity symbol to calibrate to
        n_paths: Number of paths to generate
        maturity: Time to maturity in years
        n_steps: Number of time steps per path
        use_cache: Whether to use cached paths if available
        use_options_if_available: Whether to try option data for calibration

    Returns:
        torch.Tensor of shape (n_paths, n_steps + 1) containing asset paths
    """
    calibrator = MarketCalibrator()

    # Check if we have cached paths
    if use_cache:
        cached_paths, is_cached = calibrator.load_cached_paths(
            ticker, n_paths, maturity, n_steps, use_options_if_available
        )
        if is_cached:
            return cached_paths

    # Calibrate model
    calibrated_process = calibrator.calibrate_to_market_data(
        ticker, period="2y", use_options_if_available=use_options_if_available
    )

    # Generate paths
    print(f"Generating {n_paths} paths with {n_steps} steps...")
    paths, _ = simulate_heston_paths_torch(
        calibrated_process, T=maturity, N=n_paths, M=n_steps, device="cpu"
    )

    if use_cache:
        calibrator.save_paths_to_cache(
            paths, ticker, n_paths, maturity, n_steps, use_options_if_available
        )

    return paths.float()


if __name__ == "__main__":
    try:
        # Example usage with caching
        print("Generating paths with caching enabled...")
        paths = generate_market_calibrated_paths(
            "SPY", n_paths=100, maturity=1.0, n_steps=252
        )
        print(f"Generated paths shape: {paths.shape}")
        print(f"Sample path: {paths[0, :10]}")  # First 10 points of first path

        # Try loading the same paths (should use cache)
        import torch

        print("\nTrying to load cached paths...")
        cached_paths = generate_market_calibrated_paths(
            "SPY", n_paths=100, maturity=1.0, n_steps=252
        )
        print(f"Cached paths shape: {cached_paths.shape}")
        print(f"Are they the same? {torch.allclose(paths, cached_paths)}")

    except Exception as e:
        print(f"Error in generation: {e}")
