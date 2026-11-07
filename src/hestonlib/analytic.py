"""Semi-analytic Heston European prices.

Characteristic function
-----------------------
Let X_T = ln(S_T / F_T) with forward F_T = S0 exp((r - q) T), so E[e^{X_T}] = 1.
In Gatheral's notation (The Volatility Surface, ch. 2) with the Albrecher et al. (2007)
"little Heston trap" choice of root:

    w     = u^2 + i u,   beta = kappa - rho xi i u,   d = sqrt(beta^2 + xi^2 w)
    r_-   = (beta - d) / xi^2 = -w / (beta + d)
    g     = (beta - d) / (beta + d) = -xi^2 w / (beta + d)^2
    D(T)  = r_- (1 - e^{-dT}) / (1 - g e^{-dT})
    C(T)  = kappa [ r_- T - (2 / xi^2) ln(1 + y) ],   y = g (1 - e^{-dT}) / (1 - g)
    phi(u) = E[e^{i u X_T}] = exp(C(T) theta + D(T) v0)

The right-hand (conjugate) forms are the ones evaluated: they avoid the cancellation in
beta - d as xi -> 0. ln(1 + y) is computed as 2 artanh(y / (2 + y)), which has the same
principal branch cut (y real < -1) and is accurate for small complex y.
Using e^{-dT} with Re(d) > 0 keeps 1 + y away from the negative real axis in the regimes
tested (rho < 0, and generally kappa > rho xi / 2, where |g| < 1), so the integrand is
continuous in u. This is verified numerically by the continuity and stress tests.

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


def heston_call(K, T: float, p: HestonParams, return_error: bool = False):
    """Heston European call price(s) by the Lewis integral. K may be scalar or 1-D array.

    With ``return_error``, also returns an absolute error estimate in price units (QUADPACK's
    estimate, not a rigorous bound; the tests show it covers the true error): the
    quadrature's own error estimate scaled by e^{-rT} sqrt(F K) / pi, plus a rounding term
    16 eps e^{-rT} max(F, K) for the cancellation F - sqrt(F K)/pi * integral. For a vector
    of strikes ``quad_vec`` reports one max-norm estimate, which is applied to every strike.
    Prices can be slightly negative (about -1e-10) at very short T deep out of the money;
    they lie within this bound, so callers should treat |price| <= error as zero time value.
    """
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

        integral, abserr = quad(f, 0.0, np.inf, limit=400, **_QUAD_KW)
    else:
        xv = x.ravel()

        def f(u):
            z = heston_cf(u - 0.5j, T, p)
            return np.real(np.exp(1j * u * xv) * z) / (u * u + 0.25)

        integral, abserr = quad_vec(f, 0.0, np.inf, limit=400, norm="max", **_QUAD_KW)
        integral = integral.reshape(K_arr.shape)

    disc = np.exp(-p.r * T)
    price = disc * (F - np.sqrt(F * K_arr) / np.pi * integral)
    if not return_error:
        return price
    eps = np.finfo(float).eps
    err = disc * (np.sqrt(F * K_arr) / np.pi * abserr + 16 * eps * np.maximum(F, K_arr))
    return price, err


def heston_put(K, T: float, p: HestonParams, return_error: bool = False):
    """Heston European put via put-call parity (same error bound as the call, plus the
    rounding of the parity terms)."""
    K_arr = np.asarray(K, dtype=float)
    call, err = heston_call(K_arr, T, p, return_error=True)
    fs, dk = p.S0 * np.exp(-p.q * T), K_arr * np.exp(-p.r * T)
    put = call - fs + dk
    if not return_error:
        return put
    return put, err + 4 * np.finfo(float).eps * np.maximum(fs, dk)


def heston_price(K, T: float, p: HestonParams, kind: str = "call"):
    if kind == "call":
        return heston_call(K, T, p)
    if kind == "put":
        return heston_put(K, T, p)
    raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")
