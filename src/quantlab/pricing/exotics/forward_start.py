"""
Forward-start option pricing module.

Implements industry-standard forward-start options (cash-settled, strike = strike_ratio * S_T1).
Supports analytical pricing under Black–Scholes and Monte Carlo under arbitrary models.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.stats import norm


@dataclass(frozen=True)
class ForwardStartOption:
    """
    Represents a forward-start European option.

    At time T1 (start_date), the strike is fixed as K = strike_ratio * S_T1.
    Payoff at T2 (maturity) is cash-settled.

    Attributes
    ----------
    start_date : float
        Time (in years) when the option becomes active and strike is set (T1).
    maturity : float
        Time (in years) when the option expires (T2 > T1).
    option_type : str
        'call' or 'put'.
    strike_ratio : float, default=1.0
        Multiplier for ATM strike: K = strike_ratio * S_T1.
        strike_ratio=1.0 → ATM.
    """
    start_date: float
    maturity: float
    option_type: str
    strike_ratio: float = 1.0

    def __post_init__(self) -> None:
        if self.maturity <= self.start_date:
            raise ValueError("maturity must be > start_date")
        if self.option_type not in {"call", "put"}:
            raise ValueError("option_type must be 'call' or 'put'")
        if self.strike_ratio <= 0:
            raise ValueError("strike_ratio must be positive")


def price_forward_start_bs(
    option: ForwardStartOption,
    spot: float,
    r: float,
    q: float,
    vol: float,
    today: float = 0.0,
) -> float:
    """
    Price a forward-start option under Black–Scholes (analytical).

    Formula derived from scaling invariance: the option is equivalent to
    a vanilla option with S0=1, K=strike_ratio, maturity = T2 - T1,
    discounted back to today and scaled by spot * exp(-q * T1).

    Parameters
    ----------
    option : ForwardStartOption
        The forward-start contract.
    spot : float
        Spot price at time `today`.
    r : float
        Risk-free rate (continuously compounded).
    q : float
        Dividend yield (continuously compounded).
    vol : float
        Constant volatility (annualized).
    today : float, default=0.0
        Valuation date (in same time units as option dates).

    Returns
    -------
    float
        Present value of the forward-start option.
    """
    tau = option.maturity - option.start_date  # time between T1 and T2
    T1_from_today = option.start_date - today

    if tau <= 0:
        raise ValueError("Option maturity must be after start date.")

    # Vanilla BS formula with S0=1, K=strike_ratio
    sqrt_tau = np.sqrt(tau)
    d1 = (np.log(1.0 / option.strike_ratio) + (r - q + 0.5 * vol**2) * tau) / (vol * sqrt_tau)
    d2 = d1 - vol * sqrt_tau

    if option.option_type == "call":
        vanilla_price = norm.cdf(d1) - option.strike_ratio * np.exp(-r * tau) * norm.cdf(d2)
    else:
        vanilla_price = option.strike_ratio * np.exp(-r * tau) * norm.cdf(-d2) - norm.cdf(-d1)

    # Scale by spot * exp(-q * T1) to account for forward start
    present_value = spot * np.exp(-q * T1_from_today) * vanilla_price
    return present_value


def mc_price_forward_start(
    option: ForwardStartOption,
    paths_T1_T2: np.ndarray,
    discount_factor_T2: float,
) -> float:
    """
    Monte Carlo pricer for forward-start options.

    Parameters
    ----------
    option : ForwardStartOption
        The forward-start contract.
    paths_T1_T2 : np.ndarray, shape (N, 2)
        Simulated asset prices at [T1, T2] for N paths.
        Column 0: S_T1, Column 1: S_T2.
    discount_factor_T2 : float
        Discount factor from T2 to today: exp(-r * T2).

    Returns
    -------
    float
        Monte Carlo estimate of present value.
    """
    if paths_T1_T2.shape[1] != 2:
        raise ValueError("paths_T1_T2 must have shape (N, 2)")

    S_T1 = paths_T1_T2[:, 0]
    S_T2 = paths_T1_T2[:, 1]
    strike = option.strike_ratio * S_T1

    if option.option_type == "call":
        payoff = np.maximum(S_T2 - strike, 0.0)
    else:
        payoff = np.maximum(strike - S_T2, 0.0)

    return discount_factor_T2 * np.mean(payoff)


# Usage (to be moved to tests)
if __name__ == "__main__":
    # ATM forward-start call: start in 1Y, expire in 2Y
    opt = ForwardStartOption(start_date=1.0, maturity=2.0, option_type="call", strike_ratio=1.0)

    # Black–Scholes price
    price_bs = price_forward_start_bs(
        option=opt,
        spot=100.0,
        r=0.02,
        q=0.01,
        vol=0.25,
        today=0.0,
    )
    print(f"BS Price: {price_bs:.4f}")


