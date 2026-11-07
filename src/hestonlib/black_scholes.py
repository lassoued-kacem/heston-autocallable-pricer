"""Black-Scholes pricing with a continuous dividend yield."""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr


def bs_price(S0, K, T, r, q, sigma, kind: str = "call"):
    """Black-Scholes European price. Vectorized over S0, K, sigma."""
    S0 = np.asarray(S0, dtype=float)
    K = np.asarray(K, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    fwd_disc_S = S0 * np.exp(-q * T)
    disc_K = K * np.exp(-r * T)
    sd = sigma * np.sqrt(T)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(S0 / K) + (r - q) * T) / sd + 0.5 * sd
        d2 = d1 - sd
        call = fwd_disc_S * ndtr(d1) - disc_K * ndtr(d2)
    call = np.where(sd > 0, call, np.maximum(fwd_disc_S - disc_K, 0.0))
    if kind == "call":
        return call
    if kind == "put":
        return call - fwd_disc_S + disc_K
    raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")
