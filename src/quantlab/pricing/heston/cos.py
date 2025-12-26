"""
The COS(Cosine Series Expansion) method for Heston pricing.

The COS (Cosine Series Expansion) method is a fast, Fourier-based numerical technique
for pricing European options, especially efficient for models with known
characteristic functions by approximating the log-return density with a cosine series,
offering error control and handling multiple strikes/dimensions,
though requiring careful selection of truncation range.

This module contains COS method implementation for Heston model.
"""
import numpy as np

from quantlab.instruments.base import StockOption
from quantlab.models.heston.model import HestonParameters, HestonProcess
from quantlab.utils.types import Floats


def heston_log_char_func(omega: Floats, dt: Floats, p: HestonParameters) -> Floats:
    """Compute the logarithm of the Heston CF assuming S=1 and no drift.

    To recover the CF if the stock trades at s multiply by the factor
    exp(i omega (log(s) + mu dt)) the exponent of the result.

    Args:
        omega: frequency dual to the price domain
        dt: time to evaluate CF at
        p: model parameters

    Returns:
        float: log characteristic function value
    """
    D = np.sqrt(
        (p.kappa - 1j * p.rho * p.eta * omega) ** 2
        + (omega**2 + 1j * omega) * p.eta**2
    )
    F = p.kappa - 1j * p.rho * p.eta * omega - D
    G = F / (F + 2 * D)
    df = np.exp(-D * dt)  # discount factor
    return p.v0 / p.eta**2 * (1 - df) / (
        1 - G * df
    ) * F + p.kappa * p.theta / p.eta**2 * (
        dt * F - 2 * np.log((1 - G * df) / (1 - G))
    )


def vanilla_put_cos_expansion_coeffs(a: Floats, b: Floats, k: int) -> Floats:
    """
    Compute the cosine expansion coefficients of the vanilla put payoff.

    Args:
        a, b: integration bounds for the payoff function
        k: number of coefficients

    Returns:
        vector of coefficients
    """
    ks = np.arange(k)
    a, b = a[..., None], b[..., None]  # new axis for coefficients
    c, d = a, np.zeros_like(a)

    c_frac = ks * np.pi * (c - a) / (b - a)
    d_frac = ks * np.pi * (d - a) / (b - a)

    chi = (
        1
        / (1 + (ks * np.pi / (b - a)) ** 2)
        * (
            np.cos(d_frac) * np.exp(d)
            - np.cos(c_frac) * np.exp(c)
            + ks
            * np.pi
            / (b - a)
            * (np.sin(d_frac) * np.exp(d) - np.sin(c_frac) * np.exp(c))
        )
    )

    psi = (b - a) / ks[1:] / np.pi * (np.sin(d_frac[..., 1:]) - np.sin(c_frac[..., 1:]))
    psi = np.insert(psi, 0, np.squeeze(d - c, axis=-1), axis=-1)

    return 2 / (b - a) * (psi - chi)


def heston_cumulants(dt, params: HestonParameters):
    """
    Compute 1st, 2nd, and 4th cumulants of X = log(S_T/S_0).

    Parameters:
        dt: array-like, time to maturity
        params: HestonParameters(v, kappa, theta, eta, rho)

    Returns:
        c1, c2, c4: cumulants (same shape as dt)
    """
    dt = np.asarray(dt)
    v0 = params.v0
    kappa = params.kappa
    theta = params.theta
    eta = params.eta
    # rho = params.rho

    # Precompute common terms
    kdt = kappa * dt

    exp_kdt = np.exp(-kdt)

    # ---- c1 = E[X] ----
    # c1 = (theta / kappa) * (dt - (1 - exp(-kdt)) / kappa)
    # - 0.5 * v0 * (1 - exp(-kdt)) / kappa
    if np.any(kdt < 1e-6):
        # Taylor: (1 - exp(-kdt))/kappa ≈ dt - 0.5*kappa*dt^2 + ...
        int1 = dt - 0.5 * kappa * dt**2 + (kappa**2 * dt**3) / 6
    else:
        int1 = (1.0 - exp_kdt) / kappa
    c1 = (theta / kappa) * (dt - int1) - 0.5 * v0 * int1

    # ---- c2 = Var[X] ----
    term1 = v0 * int1
    term2 = theta * (
        dt / kappa
        - 2.0 * (1.0 - exp_kdt) / kappa**2
        + (1.0 - exp_kdt**2) / (2.0 * kappa**2)
    )
    term3 = (eta**2 / (2.0 * kappa**3)) * (
        (1.0 - exp_kdt) ** 2 / kappa
        - 2.0 * dt * (1.0 - exp_kdt) / kappa
        + 3.0 * dt
        - 4.0 * (1.0 - exp_kdt) / kappa
        + (1.0 - exp_kdt**2) / (2.0 * kappa)
    )
    c2 = term1 + term2 - term3

    # Enforce non-negative variance
    c2 = np.maximum(c2, 1e-12)

    # ---- c4 (fourth cumulant) ----
    xi = eta / kappa
    # zeta = v0 / theta
    T = dt

    e = exp_kdt
    e2 = e * e
    e3 = e2 * e

    # Coefficients
    # A = xi**2 * zeta * (1 - e) ** 2 * (1 + e - 2 * kappa * T)
    B = xi**2 * (1 - e) ** 2 * (2 * kappa * T - 3 + 4 * e - e2)
    # C = xi**3 * rho * zeta * (1 - e) ** 2 * (1 + e - 2 * kappa * T)
    # D = xi**3 * rho * (1 - e) ** 2 * (6 * kappa * T - 11 + 18 * e - 9 * e2 + 2 * e3)
    E = (
        xi**4
        * (1 - e) ** 4
        * (8 * kappa * T - 25 + 40 * e - 24 * e2 + 8 * e3 - e**4)
    )

    # c3 = 0.5 * C + (1.0 / 12.0) * D
    c4 = B + (1.0 / 8.0) * E

    c4 = np.maximum(c4, -10.0 * c2**2)  # allow negative kurtosis but not extreme

    return c1, c2, c4


def trunc_range_cumulant(dt, params: HestonParameters, method="c4", L=None):
    """Cumulant based truncation half-width L."""
    c1, c2, c4 = heston_cumulants(dt, params)

    if method == "c2":
        L = L or 12
        width = L * np.sqrt(np.maximum(c2, 1e-12))
    elif method == "c4":
        L = L or 10
        width = L * np.sqrt(np.maximum(c2 + np.sqrt(np.maximum(c4, 0)), 1e-12))
    else:
        raise ValueError("Unknown method")
    return width  # half-width L * ...


def log_price_moments(dt, params: HestonParameters, n: int):
    """Compute central moment mu_n = E[|X - E[X]|^n] for X = log(S_T/S0)."""
    c1, c2, c4 = heston_cumulants(dt, params)
    if n == 2:
        return c2
    elif n == 4:
        # mu4 = c4 + 3*c2^2
        return c4 + 3 * c2**2
    elif n == 6:
        # c6 is needed; if unavailable, raise
        raise NotImplementedError("c6 not implemented")
    else:
        raise ValueError("Only n=2,4 supported")


def trunc_range_jp(dt, params: HestonParameters, epsilon=1e-6, n=4, K_bound=1.0):
    """
    Junike & Pankrashkin truncation half-width L.

    X = log S_T ~ log S0 + log-return
    We use Markov: P(|X - E[X]| > L) <= mu_n / L^n <= epsilon / (2 * K_bound)
    => L = (2 * K_bound * mu_n / epsilon)^(1/n)
    """
    mu_n = log_price_moments(dt, params, n)
    mu_n = np.maximum(mu_n, 1e-16)  # prevent negative moments
    L = (2 * K_bound * mu_n / epsilon) ** (1.0 / n)
    return L


def calc_trunc_range(params, dt, method="jp", **kwargs):
    """General truncation range calculation."""
    if method == "cumulant":
        # Default to c4 rule
        return trunc_range_cumulant(dt, params, method="c4")
    elif method == "jp":
        epsilon = kwargs.get("epsilon", 1e-6)
        n = kwargs.get("n", 4)
        K_bound = kwargs.get("K_bound", 1.0)
        return trunc_range_jp(dt, params, epsilon=epsilon, n=n, K_bound=K_bound)
    else:
        raise ValueError("Unknown method")


def price(
    option: StockOption,
    process: HestonProcess,
    n_points: int = 4096,
    method="jp",
):
    """
    Compute the prices of given options.

    Args:
        option: Stock option to price.
        process: Underlying dynamics.
        n_points: number of terms in Fourier expansion.
        method: algorithm for truncation range calculation.

    Returns:
        A price(s) for the StockOption option(s).
    """
    # Extract market parameters
    ms = process.market_state

    # Extract model parameters
    params = process.model_params

    dt = np.asarray(option.expiration_time - ms.time)
    mean_log_s_t = ms.interest_rate * dt
    trunc_range = calc_trunc_range(params, dt, method=method)

    x = np.log(ms.stock_price / option.strike_price)
    trunc_lb = mean_log_s_t - trunc_range
    trunc_ub = mean_log_s_t + trunc_range
    x_within_bounds = (x >= trunc_lb) & (x <= trunc_ub)

    # COS method
    cos_cf = vanilla_put_cos_expansion_coeffs(trunc_lb, trunc_ub, n_points)
    omega = (np.pi / (trunc_ub - trunc_lb))[..., None] * np.arange(n_points)

    df = np.exp(-ms.interest_rate * dt)  # discount factor
    lcf = (
        heston_log_char_func(omega, dt[..., None], params)
        + 1j * (x + mean_log_s_t - trunc_lb)[..., None] * omega
    )
    prod = np.exp(lcf) * cos_cf
    prod[..., 0] /= 2

    put_price = option.strike_price * df * np.real(prod.sum(-1))
    put_price_lb = np.maximum(option.strike_price * df - ms.stock_price, 0.0)

    call_price = put_price + ms.stock_price - option.strike_price * df
    call_price_lb = np.maximum(ms.stock_price - option.strike_price * df, 0.0)

    raw_price = np.where(option.is_call, call_price, put_price)
    price_lb = np.where(option.is_call, call_price_lb, put_price_lb)

    # ensure price fits the lower bound
    return np.maximum(
        np.where(x_within_bounds, raw_price, price_lb),
        price_lb,
    )
