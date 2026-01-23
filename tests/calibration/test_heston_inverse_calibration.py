"""
Tests for Heston inverse calibration.

This module contains generate-recover-compare parameters tests for Heston model.
"""
import numpy as np
import pytest

from quantlab.calibration.inverse import recover_heston_params_from_prices
from quantlab.calibration.utils import make_heston_object_wrapper
from quantlab.instruments.base import StockOption
from quantlab.market_data.market_state import MarketState
from quantlab.models.heston.model import HestonParameters, HestonProcess
from quantlab.pricing.heston.cos import price as cos_price_function


@pytest.mark.slow
def test_recover_heston_params_cos():
    """Test the inverse calibration pipeline using COS pricer."""
    # 1. Define True Parameters
    true_params = HestonParameters(v0=0.04, kappa=2.0, theta=0.04, eta=0.3, rho=-0.7)
    S0, r, t = 100.0, 0.05, 0.0
    market_state = MarketState(stock_price=S0, interest_rate=r, time=t)

    # 2. Generate Synthetic Prices
    strikes = np.linspace(80, 120, 5)  # Few points for a quick test
    maturities = np.array([0.5, 1.0, 1.5])
    # Use meshgrid or nested loops to get all (K, T) pairs
    K_mesh, T_mesh = np.meshgrid(strikes, maturities)
    K_flat, T_flat = K_mesh.flatten(), T_mesh.flatten()

    prices_synthetic = []
    heston_process_fixed = HestonProcess(
        true_params, market_state
    )  # Process with true params
    for K, T in zip(K_flat, T_flat):
        option = StockOption(strike_price=K, expiration_time=T, is_call=True)
        # Price using the true process
        price = cos_price_function(
            option, heston_process_fixed, n_points=512
        )  # Use sufficient points for accuracy
        prices_synthetic.append(float(price))
    prices_synthetic = np.array(prices_synthetic)

    # 3. Set up calibration
    forward = S0 * np.exp(r * T_flat)
    discount_factors = np.exp(-r * T_flat)

    initial_guess = {"v0": 0.05, "kappa": 1.5, "theta": 0.05, "eta": 0.25, "rho": -0.5}
    bounds = {
        "v0": (0.01, 0.2),
        "kappa": (0.1, 10.0),
        "theta": (0.01, 0.2),
        "eta": (0.01, 1.0),
        "rho": (-0.99, 0.99),
    }

    # 4. Create Wrapper and Recover
    cos_wrapper = make_heston_object_wrapper(
        pricer_func=cos_price_function,
        market_state_for_calibration=market_state,  # Same market state as generation
        pricer_kwargs={"n_points": 512},  # Use same settings as generation for fairness
    )

    recovered_params = recover_heston_params_from_prices(
        strikes=K_flat,
        maturities=T_flat,
        prices=prices_synthetic,
        forward=forward,
        discount_factors=discount_factors,
        initial_guess=initial_guess,
        pricing_func=cos_wrapper,
        pricing_kwargs={},
        bounds=bounds,
        method="differential_evolution",
        optimizer_options={"maxiter": 1000, "seed": 42, "polish": True, "disp": True},
        verbose=True,
    )

    # 5. Assertions
    tolerance = 0.05  # 5% tolerance
    for param_name in ["v0", "kappa", "theta", "eta", "rho"]:
        true_val = getattr(true_params, param_name)
        recovered_val = recovered_params[param_name]
        assert (
            abs(recovered_val - true_val) / true_val < tolerance
        ), f"Parameter {param_name} mismatch: true={true_val}, recovered={recovered_val}"  # noqa: E501


# Add similar tests for other Heston pricers
# def test_recover_heston_params_analytic():
#    ...
