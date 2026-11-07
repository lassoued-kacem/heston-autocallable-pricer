"""Semi-analytic Heston European prices.

Characteristic function
-----------------------
Let X_T = ln(S_T / F_T) with forward F_T = S0 exp((r - q) T), so E[e^{X_T}] = 1.
In Gatheral's notation (The Volatility Surface, ch. 2) with the Albrecher et al. (2007)
"little Heston trap" choice of root:

    alpha = -(u^2 + i u)/2,  beta = kappa - rho xi i u,  gamma = xi^2 / 2
    d     = sqrt(beta^2 - 4 alpha gamma)
    r_pm  = (beta +- d) / xi^2,      g = r_- / r_+
    D(T)  = r_- (1 - e^{-dT}) / (1 - g e^{-dT})
    C(T)  = kappa [ r_- T - (2 / xi^2) ln((1 - g e^{-dT}) / (1 - g)) ]
    phi(u) = E[e^{i u X_T}] = exp(C(T) theta + D(T) v0)

Using e^{-dT} with Re(d) > 0 (the principal sqrt) gives |g e^{-dT}| < 1 along the integration
path, so the principal complex log stays on one branch (no discontinuity in the integrand).

Lewis (2001) single-integral call price
---------------------------------------
With x = ln(F_T / K):

    C = e^{-rT} [ F_T - sqrt(F_T K)/pi * int_0^inf Re[e^{i u x} phi(u - i/2)] / (u^2 + 1/4) du ]

(the contour Im(z) = -1/2 lies inside the strip of regularity of the call transform).
Puts follow from put-call parity  P = C - S0 e^{-qT} + K e^{-rT}.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad, quad_vec

from hestonlib.params import HestonParams


def heston_cf(u, T: float, p: HestonParams):
    """E[exp(i u ln(S_T / F_T))], vectorized over complex u."""
    # Numerically stable evaluation (no cancellation as xi -> 0). With w = u^2 + i u,
    # d^2 = beta^2 + xi^2 w, so beta - d = -xi^2 w / (beta + d), and therefore
    #   r_-  = -w / (beta + d),        g = -xi^2 w / (beta + d)^2.
    # The log term is ln((1 - g e)/(1 - g)) = ln(1 + y) with y = g (1 - e)/(1 - g) = O(xi^2).
    # It is evaluated as 2 artanh(y / (2 + y)), which is accurate for small complex y
    # (numpy's complex log1p is not). Re(beta + d) > 0 on the integration paths.
    u = np.asarray(u, dtype=complex)
    iu = 1j * u
    xi2 = p.xi * p.xi
    w = u * u + iu  # = -2 alpha
    beta = p.kappa - p.rho * p.xi * iu
    d = np.sqrt(beta * beta + xi2 * w)
    bpd = beta + d
    r_minus = -w / bpd
    g = -xi2 * w / (bpd * bpd)
    edt = np.exp(-d * T)
    one_minus_edt = -np.expm1(-d * T)
    D = r_minus * one_minus_edt / (1.0 - g * edt)
    y = g * one_minus_edt / (1.0 - g)
    log_term = 2.0 * np.arctanh(y / (2.0 + y))
    C = p.kappa * (r_minus * T - (2.0 / xi2) * log_term)
    return np.exp(C * p.theta + D * p.v0)


_QUAD_KW = dict(epsabs=1e-12, epsrel=1e-10)


def heston_call(K, T: float, p: HestonParams):
    """Heston European call price(s) by the Lewis integral. K may be scalar or 1-D array."""
    if T <= 0:
        raise ValueError("T must be positive")
    K_arr = np.asarray(K, dtype=float)
    F = p.S0 * np.exp((p.r - p.q) * T)
    x = np.log(F / K_arr)

    if K_arr.ndim == 0:
        xs = float(x)

        def f(u):
            z = heston_cf(u - 0.5j, T, p)
            return float(np.real(np.exp(1j * u * xs) * z)) / (u * u + 0.25)

        integral, _ = quad(f, 0.0, np.inf, limit=400, **_QUAD_KW)
    else:
        xv = x.ravel()

        def f(u):
            z = heston_cf(u - 0.5j, T, p)
            return np.real(np.exp(1j * u * xv) * z) / (u * u + 0.25)

        integral, _ = quad_vec(f, 0.0, np.inf, limit=400, **_QUAD_KW)
        integral = integral.reshape(K_arr.shape)

    return np.exp(-p.r * T) * (F - np.sqrt(F * K_arr) / np.pi * integral)


def heston_put(K, T: float, p: HestonParams):
    """Heston European put via put-call parity."""
    K_arr = np.asarray(K, dtype=float)
    return heston_call(K_arr, T, p) - p.S0 * np.exp(-p.q * T) + K_arr * np.exp(-p.r * T)


def heston_price(K, T: float, p: HestonParams, kind: str = "call"):
    if kind == "call":
        return heston_call(K, T, p)
    if kind == "put":
        return heston_put(K, T, p)
    raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")
