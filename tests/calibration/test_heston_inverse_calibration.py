"""
Tests for Heston inverse calibration.

This module contains generate-recover-compare parameters tests for Heston model.
"""
import numpy as np
import pytest

from quantlab.calibration.inverse import (
    recover_heston_params_from_implied_vols as recover_heston_params_from_ivs,
)
from quantlab.calibration.inverse import recover_heston_params_from_prices
from quantlab.calibration.utils import make_heston_object_wrapper
from quantlab.data.synthetic import generate_heston_vol_surface
from quantlab.market_data.market_state import MarketState
from quantlab.models.heston.model import HestonParameters
from quantlab.pricing.heston.cos import price as cos_price


@pytest.mark.slow
def test_recover_heston_params_cos():
    """Test the inverse calibration pipeline using COS pricer."""
    # 1. Generate synthetic *prices* using known true params
    true_params_input = {
        "v0": 0.04,
        "kappa": 2.0,
        "theta": 0.04,
        "eta": 0.3,
        "rho": -0.7,
    }
    true_params = HestonParameters(**true_params_input)
    market_state_input = {"stock_price": 100.0, "interest_rate": 0.05, "time": 0.0}
    strikes_grid = np.linspace(80, 120, 5)  # Few points for a quick test
    maturities_grid = np.array([0.5, 1.0, 1.5])

    strikes_syn, maturities_syn, prices_synthetic = generate_heston_vol_surface(
        market_state=MarketState(**market_state_input),
        heston_params=HestonParameters(**true_params_input),
        strikes=strikes_grid,
        maturities=maturities_grid,  # Absolute maturities
        output_format="prices",
        pricing_method="cos",
    )

    # 2. Calculate corresponding forwards and discount factors for the inverse function
    S0, r = market_state_input["stock_price"], market_state_input["interest_rate"]
    discount_factors_synthetic = np.exp(-r * maturities_syn)
    forwards_synthetic = S0 * np.exp(r * maturities_syn)

    # 3. Set up calibration
    initial_guess = {"v0": 0.02, "kappa": 1.5, "theta": 0.1, "eta": 0.2, "rho": -0.4}
    bounds = {
        "v0": (1e-4, 1.0),
        "kappa": (0.1, 20.0),
        "theta": (1e-4, 1.0),
        "eta": (0.01, 2.0),
        "rho": (-0.999, 0.999),
    }
    # 4. Create Wrapper and Recover
    cos_wrapper = make_heston_object_wrapper(
        pricer_func=cos_price,
        market_state_for_calibration=MarketState(**market_state_input),
        pricer_kwargs={"n_points": 4096},  # Use same settings as generation
    )

    recovered_params = recover_heston_params_from_prices(
        strikes=strikes_syn,
        maturities=maturities_syn,
        prices=prices_synthetic,
        forward=forwards_synthetic,
        discount_factors=discount_factors_synthetic,
        initial_guess=initial_guess,
        pricing_func=cos_wrapper,
        pricing_kwargs={},
        bounds=bounds,
        method="differential_evolution",
        optimizer_options={"maxiter": 200, "seed": 42, "polish": True, "disp": True},
        verbose=False,
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


@pytest.mark.slow
def test_recover_heston_params_cos_ivs():
    """Test the inverse calibration pipeline using COS pricer and IVs recovery."""
    # 1. Generate synthetic *prices* using known true params
    true_params_input = {
        "v0": 0.04,
        "kappa": 2.0,
        "theta": 0.04,
        "eta": 0.3,
        "rho": -0.7,
    }
    true_params = HestonParameters(**true_params_input)
    market_state_input = {"stock_price": 100.0, "interest_rate": 0.05, "time": 0.0}
    strikes_grid = np.linspace(80, 120, 5)  # Few points for a quick test
    maturities_grid = np.array([0.5, 1.0, 1.5])

    strikes_syn, maturities_syn, ivs_synthetic = generate_heston_vol_surface(
        market_state=MarketState(**market_state_input),
        heston_params=HestonParameters(**true_params_input),
        strikes=strikes_grid,
        maturities=maturities_grid,  # Absolute maturities
        output_format="implied_vols",
        pricing_method="cos",
    )

    # 2. Set up calibration
    initial_guess = {"v0": 0.02, "kappa": 1.5, "theta": 0.1, "eta": 0.2, "rho": -0.4}
    bounds = {
        "v0": (1e-4, 1.0),
        "kappa": (0.1, 20.0),
        "theta": (1e-4, 1.0),
        "eta": (0.01, 2.0),
        "rho": (-0.999, 0.999),
    }
    # 3. Create Wrapper and Recover
    cos_wrapper = make_heston_object_wrapper(
        pricer_func=cos_price,
        market_state_for_calibration=MarketState(**market_state_input),
        pricer_kwargs={"n_points": 4096},  # Use same settings as generation
    )

    recovered_params = recover_heston_params_from_ivs(
        strikes=strikes_syn,
        maturities=maturities_syn,
        target_implied_vols=ivs_synthetic,
        market_state=MarketState(**market_state_input),
        initial_guess=initial_guess,
        pricing_func=cos_wrapper,
        pricing_kwargs={},
        bounds=bounds,
        method="differential_evolution",
        optimizer_options={"maxiter": 200, "seed": 42, "polish": True, "disp": True},
        verbose=False,
    )

    # 4. Assertions
    tolerance = 0.05  # 5% tolerance
    for param_name in ["v0", "kappa", "theta", "eta", "rho"]:
        true_val = getattr(true_params, param_name)
        recovered_val = recovered_params[param_name]
        assert (
            abs(recovered_val - true_val) / true_val < tolerance
        ), f"Parameter {param_name} mismatch: true={true_val}, recovered={recovered_val}"  # noqa: E501
