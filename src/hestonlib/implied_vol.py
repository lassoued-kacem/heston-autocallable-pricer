"""Black-Scholes implied volatility with explicit no-arbitrage and resolution checks.

For a call with forward-discounted spot S0 e^{-qT} and discounted strike K e^{-rT}:

    lower = max(S0 e^{-qT} - K e^{-rT}, 0)  <  C  <  upper = S0 e^{-qT}

(puts: lower = max(K e^{-rT} - S0 e^{-qT}, 0), upper = K e^{-rT}). BS prices are strictly
increasing in sigma between these bounds, so the inverse is unique, and ``brentq`` is used
on a bracket [SIGMA_MIN, SIGMA_MAX] after checking that the bracket contains the price.

A price that carries a numerical error ``price_err`` (for example the quadrature error of
the Heston pricer) only determines sigma if its time value C - lower exceeds that error.
Prices within ``price_err`` of a bound, including slightly negative ones, are reported as
NaN ("not resolvable") instead of being clipped to a meaningless volatility. Prices outside
the bounds by more than ``price_err`` are arbitrage and raise ``ValueError``.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from hestonlib.black_scholes import bs_price

SIGMA_MIN, SIGMA_MAX = 1e-6, 5.0


def _bounds(S0, K, T, r, q, kind):
    fs, dk = S0 * np.exp(-q * T), K * np.exp(-r * T)
    if kind == "call":
        return max(fs - dk, 0.0), fs
    if kind == "put":
        return max(dk - fs, 0.0), dk
    raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")


def implied_vol(price, S0, K, T, r, q, kind="call", price_err=0.0, xtol=1e-14):
    """BS implied volatility of one option; NaN if the price does not resolve sigma."""
    if T <= 0 or S0 <= 0 or K <= 0 or price_err < 0:
        raise ValueError("need T, S0, K > 0 and price_err >= 0")
    price = float(price)
    lo, hi = _bounds(S0, K, T, r, q, kind)
    if price < lo - price_err or price > hi + price_err:
        raise ValueError(f"price {price} violates no-arbitrage bounds [{lo}, {hi}]")
    if price - lo <= price_err or hi - price <= price_err:
        return np.nan
    f = lambda s: float(bs_price(S0, K, T, r, q, s, kind)) - price
    f_lo, f_hi = f(SIGMA_MIN), f(SIGMA_MAX)
    if f_lo >= 0 or f_hi <= 0:
        return np.nan  # sigma outside the search range [SIGMA_MIN, SIGMA_MAX]
    return brentq(f, SIGMA_MIN, SIGMA_MAX, xtol=xtol, rtol=4 * np.finfo(float).eps, maxiter=200)


def implied_vols(prices, S0, K, T, r, q, kind="call", price_err=0.0):
    """Vector version over strikes (``prices`` and ``K`` broadcast; ``price_err`` scalar or array)."""
    prices, K = np.broadcast_arrays(np.asarray(prices, float), np.asarray(K, float))
    err = np.broadcast_to(np.asarray(price_err, float), prices.shape)
    return np.array([implied_vol(p, S0, k, T, r, q, kind, e)
                     for p, k, e in zip(prices.ravel(), K.ravel(), err.ravel())]).reshape(prices.shape)


def heston_implied_vols(K, T: float, p, kind: str = "call"):
    """Implied vols of Heston prices, using the pricer's own error bound.

    Returns (iv, price, price_err). Strikes whose time value is within the quadrature error
    (deep out of the money at short T, where the computed price can even be about -1e-10)
    get NaN: their implied volatility is not determined by the computed price.
    """
    from hestonlib.analytic import heston_call, heston_put

    pricer = heston_call if kind == "call" else heston_put
    price, err = pricer(np.asarray(K, float), T, p, return_error=True)
    iv = implied_vols(price, p.S0, K, T, p.r, p.q, kind, err)
    return iv, price, err
