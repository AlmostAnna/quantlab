import numpy as np
from quantlab.sim.heston.mc_pricer import heston_exact_mc_price_with_paths


def heston_delta_bump_revalue(S0, K, T, r, kappa, theta, sigma, rho, v0,
                              n_paths=100000, n_steps=500, bump_size=0.01, seed=42):
    """
    Compute delta using bump-and-revalue with common random numbers.
    bump_size: e.g., 0.01 for 1% bump (S0 = 1.0 → bump = 0.01)
    """
    np.random.seed(seed)
    dt = T / n_steps

    # Pre-generate ALL random numbers (CRN)
    Z1_base = np.random.randn(n_paths, n_steps)
    Z2_base = np.random.randn(n_paths, n_steps)

    # Base price
    C0 = heston_exact_mc_price_with_paths(
        S0, K, T, r, kappa, theta, sigma, rho, v0, Z1_base, Z2_base, n_steps
    )

    # Up bump
    C_up = heston_exact_mc_price_with_paths(
        S0 + bump_size, K, T, r, kappa, theta, sigma, rho, v0, Z1_base, Z2_base, n_steps
    )

    # Down bump
    C_down = heston_exact_mc_price_with_paths(
        S0 - bump_size, K, T, r, kappa, theta, sigma, rho, v0, Z1_base, Z2_base, n_steps
    )

    delta = (C_up - C_down) / (2 * bump_size)
    return delta, C0, C_up, C_down

