"""Black-Scholes pricing with a continuous dividend yield."""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr


def bs_price(S0, K, T, r, q, sigma, kind: str = "call"):
    """Black-Scholes European price. Vectorized over S0, K, sigma."""
    S0 = np.asarray(S0, dtype=float)
    K = np.asarray(K, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    if T < 0 or np.any(sigma < 0) or np.any(S0 <= 0) or np.any(K <= 0):
        raise ValueError("need T >= 0, sigma >= 0, S0 > 0, K > 0")
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


def bs_vega(S0, K, T, r, q, sigma):
    """dC/dsigma = S0 e^{-qT} phi(d1) sqrt(T) (same for calls and puts). Vectorized."""
    S0, K, sigma = (np.asarray(a, dtype=float) for a in (S0, K, sigma))
    if T <= 0 or np.any(sigma <= 0):
        raise ValueError("need T > 0 and sigma > 0")
    sd = sigma * np.sqrt(T)
    d1 = (np.log(S0 / K) + (r - q) * T) / sd + 0.5 * sd
    return S0 * np.exp(-q * T) * np.exp(-0.5 * d1 * d1) / np.sqrt(2.0 * np.pi) * np.sqrt(T)
