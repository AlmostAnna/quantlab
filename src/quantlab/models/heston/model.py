"""
Heston model.

This module contains basic classes and functions for Heston model implementation.
"""
from dataclasses import dataclass

import numpy as np
import torch

from quantlab.market_data.market_state import MarketState


@dataclass
class HestonParameters:
    """
    Represents the parameters of the Heston model.

    The model is defined by the system of stochastic differential equations
    under risk-neutral measure:
    dS_t = (r-q)S_t dt + √v_t S_t dW_t^S,
    dv_t = κ(θ - v_t)dt + η√v_t dW_t^v,
    dW_t^S dW_t^v = ρ dt.

    Attributes:
        v0: The initial variance of the asset price at time t=0.
        kappa: The speed at which the variance reverts to its long-term mean.
        theta: The long-term mean variance.
        eta: The volatility of the variance process (also known as "vol of vol").
        rho: The instantaneous correlation between the two Brownian motions.
    """

    v0: float  # Initial variance
    kappa: float  # Speed of mean-reversion
    theta: float  # Long-term mean variance
    eta: float  # Volatility of the variance (vol of vol)
    rho: float  # Correlation between Brownian motions


def _evolve_numpy(
    s_t: float,
    v_t: float,
    t: float,
    dt: float,
    dw_s: float,
    dw_v: float,
    params: HestonParameters,
    market_state: MarketState,
) -> tuple[float, float]:
    """
    Evolve the Heston process by one time step using Numpy.

    Current implementation uses Euler evolution for both S
    and v.

    Args:
        s_t: Current stock price S_t.
        v_t: Current variance v_t.
        t: Current time t.
        dt: Time increment dt.
        dw_s: Increment of the stock price Brownian motion (dW_t^S).
        dw_v: Increment of the variance Brownian motion (dW_t^v).
              These should be independent normals.
        params (HestonParameters): The model parameters.
        market_state (MarketState): The current market state.

    Returns:
        A tuple (s_tpdt, v_tpdt) representing the stock price S_{t+dt}
        and variance v_{t+dt} after the time step.
    """
    # Use NumPy operations
    r = market_state.interest_rate
    kappa, theta, eta, rho = params.kappa, params.theta, params.eta, params.rho

    # 1. Evolve S using the log-Euler step
    # This assumes dw_s and dw_v are independent N(0,1) increments.
    # The correlated S increment is rho * dw_v + sqrt(1 - rho**2) * dw_s
    s_tpdt = s_t * np.exp(
        (r - 0.5 * v_t) * dt
        + np.sqrt(v_t * dt) * (rho * dw_v + np.sqrt(1 - rho**2) * dw_s)
    )

    # 2. Evolve v using the Euler step
    dv = (
        kappa * (theta - v_t) * dt
        + eta * np.sqrt(np.maximum(v_t, 0.0)) * np.sqrt(dt) * dw_v
    )
    v_tpdt = v_t + dv
    # Ensure variance stays non-negative (standard fix for Euler)
    v_tpdt = np.maximum(v_tpdt, 0.0)

    return s_tpdt, v_tpdt


def _evolve_torch(
    s_t: float,
    v_t: float,
    t: float,
    dt: float,
    dw_s: float,
    dw_v: float,
    params: HestonParameters,
    market_state: MarketState,
) -> tuple[float, float]:
    """
    Evolve the Heston process by one time step using Torch.

    Args:
        s_t: Current stock price S_t.
        v_t: Current variance v_t.
        t: Current time t.
        dt: Time increment dt.
        dw_s: Increment of the stock price Brownian motion (dW_t^S).
        dw_v: Increment of the variance Brownian motion (dW_t^v).
        params (HestonParameters): The model parameters.
        market_state (MarketState): The current market state.

    Returns:
        A tuple (s_tpdt, v_tpdt) representing the stock price S_{t+dt}
        and variance v_{t+dt} after the time step.
    """
    # Use PyTorch operations
    r = market_state.interest_rate
    kappa, theta, eta, rho = params.kappa, params.theta, params.eta, params.rho

    # 1. Evolve S using the log-Euler step
    # This assumes dw_s and dw_v are independent N(0,1) increments.
    # The correlated S increment is rho * dw_v + sqrt(1 - rho**2) * dw_s

    s_tpdt = s_t * torch.exp(
        (r - 0.5 * v_t) * dt
        + torch.sqrt(torch.clamp_min(v_t, 0.0))
        * torch.sqrt(dt)
        * (rho * dw_v + torch.sqrt(1 - rho**2) * dw_s)
    )

    # Euler for v
    dv = kappa * (theta - v_t) * dt + eta * torch.sqrt(torch.clamp_min(v_t, 0.0)) * dw_v
    v_tpdt = v_t + dv
    v_tpdt = torch.clamp_min(v_tpdt, 0.0)  # Ensure non-negativity

    return s_tpdt, v_tpdt


# Registry mapping backend names to their evolution functions
_EVOLVE_FUNCTIONS = {
    "numpy": _evolve_numpy,
    "torch": _evolve_torch,
}


class HestonProcess:
    """
    Represents the dynamics of the Heston stochastic volatility model.

    This class encapsulates the evolution of the underlying asset price (S_t) and
    its instantaneous variance (v_t) over time according to the Heston model SDEs,
    using the parameters defined in HestonParameters. It provides methods for
    simulating paths (e.g., via Monte Carlo) or calculating model-specific
    quantities like the characteristic function for analytical pricing.

    The model parameters (kappa, theta, eta, rho, v0) are typically held within
    a HestonParameters instance passed during initialization.

    The risk-neutral dynamics are:
    dS_t = (r - q) S_t dt + sqrt(v_t) S_t dW_t^S,
    dv_t = κ(θ - v_t)dt + η sqrt(v_t) dW_t^v,
    dW_t^S dW_t^V = ρ dt.

    Attributes:
        model_params: The HestonParameters instance
        market_state: The MarketState instance providing the current risk-free rate (r)
                      and initial stock price (S0).
    """

    def __init__(
        self,
        model_params: HestonParameters,
        market_state: MarketState,
        backend: str = "numpy",
    ):
        """
        Initialize the Heston process.

        Args:
            model_params (HestonParameters): The model parameters.
            market_state (MarketState): The current market state.
            backend (str): The computational backend ('numpy' or 'torch').
        """
        self.model_params = model_params
        self.market_state = market_state
        self.backend = backend.lower()

        # Fetch the appropriate evolution function based on the backend
        if self.backend not in _EVOLVE_FUNCTIONS:
            raise ValueError(
                f"Unsupported backend: {backend}. Supported: {list(_EVOLVE_FUNCTIONS.keys())}"  # noqa: E501
            )
        self._evolve_func = _EVOLVE_FUNCTIONS[self.backend]

    def evolve(self, s_t, v_t, t, dt, dw_s, dw_v):
        """
        Evolve the process by one step using the specified backend's logic.

        Args:
          s_t: Current stock price S_t.
          v_t: Current variance v_t.
          t: Current time t.
          dt: Time increment dt.
          dw_s: Independent standard normal increment for S (perpendicular to v's BM).
          dw_v: Independent standard normal increment for v.
                The correlated increment for S will be
                rho * dw_v + sqrt(1-rho^2) * dw_s.

        Returns:
            A tuple (s_tpdt, v_tpdt) representing the stock price S_{t+dt}
            and variance v_{t+dt} after the time step.
        """
        return self._evolve_func(
            s_t, v_t, t, dt, dw_s, dw_v, self.model_params, self.market_state
        )

    # def characteristic_function(self, u, t):
    # TODO: lives in a separate module now, should it be moved here?
    # TODO: Uses self.params and potentially self.market_state
    #    pass
