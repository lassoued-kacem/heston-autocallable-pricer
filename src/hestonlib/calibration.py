"""Heston calibration to a surface of implied volatilities.

Fits theta_vec = (v0, kappa, theta, xi, rho) for given (S0, r, q) by bounded nonlinear least
squares (``scipy.optimize.least_squares``, trust-region reflective). The residual for option j
is the vega-weighted price error

    e_j = (C_model,j - C_market,j) / vega_market,j,

which to first order equals the implied-volatility error IV_model,j - IV_market,j, so the
loss is the sum of squared IV errors without an IV inversion inside every evaluation. The
final IV errors are computed exactly by inversion (with the pricer's error bound) and
reported separately, together with the number of model IVs that are not resolvable.

Identifiability: with J the residual Jacobian at the solution and s^2 = RSS / (m - 5) the
residual variance, Cov(theta_hat) ~ s^2 (J^T J)^{-1} when the IV errors are iid. The standard
errors, the parameter correlation matrix and the condition number of J (columns scaled by the
parameter magnitudes, floored at 1% of the bound range) are reported, as is the Feller ratio 2 kappa theta / xi^2 (the
calibration does not enforce Feller: it is reported, not imposed).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from hestonlib.analytic import heston_call
from hestonlib.black_scholes import bs_vega
from hestonlib.implied_vol import heston_implied_vols, implied_vols
from hestonlib.params import HestonParams

NAMES = ("v0", "kappa", "theta", "xi", "rho")
LOWER = np.array([1e-4, 1e-2, 1e-4, 1e-2, -0.999])
UPPER = np.array([1.0, 10.0, 1.0, 3.0, 0.999])


@dataclass(frozen=True)
class Surface:
    """Market call prices on a grid: ``prices[i, j]`` for ``maturities[i]``, ``strikes[j]``."""

    S0: float
    r: float
    q: float
    maturities: np.ndarray
    strikes: np.ndarray
    prices: np.ndarray

    def ivs(self, price_err=0.0):
        return np.array([implied_vols(c, self.S0, self.strikes, T, self.r, self.q, "call", price_err)
                         for T, c in zip(self.maturities, self.prices)])

    def vegas(self, ivs=None):
        ivs = self.ivs() if ivs is None else ivs
        return np.array([bs_vega(self.S0, self.strikes, T, self.r, self.q, iv)
                         for T, iv in zip(self.maturities, ivs)])


@dataclass(frozen=True)
class CalibrationResult:
    params: HestonParams
    x: np.ndarray
    se: np.ndarray  # s * sqrt(diag((J^T J)^-1)), s^2 = RSS / (m - 5)
    corr: np.ndarray
    cond: float
    jac: np.ndarray  # residual Jacobian at the solution, shape (m, 5)
    residuals: np.ndarray  # vega-weighted residuals at the solution
    iv_errors: np.ndarray  # model IV - market IV; NaN where the model IV is unresolvable
    iv_rmse: float  # over the resolved quotes only
    n_iv_unresolved: int  # model IVs not resolvable from the model price (reported, not hidden)
    feller_ratio: float
    at_bound: tuple[str, ...]
    nfev: int
    success: bool
    message: str


def _params(x, S0, r, q):
    v0, kappa, theta, xi, rho = x
    return HestonParams(S0=S0, r=r, q=q, v0=v0, kappa=kappa, theta=theta, xi=xi, rho=rho)


def model_prices(x, surface: Surface):
    p = _params(x, surface.S0, surface.r, surface.q)
    return np.array([heston_call(surface.strikes, T, p) for T in surface.maturities])


def calibrate(surface: Surface, x0=(0.06, 1.0, 0.06, 0.5, -0.4), xtol=1e-12, ftol=1e-12,
              gtol=1e-12, max_nfev=400) -> CalibrationResult:
    mkt = np.asarray(surface.prices, float)
    iv_mkt = surface.ivs()
    if not np.all(np.isfinite(iv_mkt)):
        raise ValueError("every market quote must have a resolvable implied volatility")
    vega = surface.vegas(iv_mkt)

    def resid(x):
        return ((model_prices(x, surface) - mkt) / vega).ravel()

    x0 = np.clip(np.asarray(x0, float), LOWER, UPPER)
    sol = least_squares(resid, x0, bounds=(LOWER, UPPER), method="trf", x_scale="jac",
                        xtol=xtol, ftol=ftol, gtol=gtol, max_nfev=max_nfev)
    x = sol.x
    J = sol.jac
    m, k = J.shape
    dof = max(m - k, 1)
    s2 = float(sol.fun @ sol.fun) / dof
    JtJ_inv = np.linalg.pinv(J.T @ J)
    cov = s2 * JtJ_inv
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    d = np.sqrt(np.clip(np.diag(JtJ_inv), 1e-300, None))
    corr = JtJ_inv / np.outer(d, d)
    # Scale columns by the parameter size, floored at 1% of the box so that a parameter
    # near 0 (e.g. rho) does not make the scaled Jacobian degenerate.
    scale = np.maximum(np.abs(x), 0.01 * (UPPER - LOWER))
    cond = float(np.linalg.cond(J * scale[None, :]))

    p = _params(x, surface.S0, surface.r, surface.q)
    # Model IVs use the pricer's error bound: a model price whose time value is within its
    # quadrature error (short-dated wings, possibly slightly negative) has no resolvable IV.
    iv_model = np.array([heston_implied_vols(surface.strikes, T, p)[0]
                         for T in surface.maturities])
    iv_err = iv_model - iv_mkt
    n_unres = int(np.sum(np.isnan(iv_err)))
    rmse = float(np.sqrt(np.mean(iv_err[~np.isnan(iv_err)] ** 2))) if n_unres < iv_err.size else np.nan
    tol_b = 1e-6 * (UPPER - LOWER)
    at_bound = tuple(n for n, xi_, lo, hi, tb in zip(NAMES, x, LOWER, UPPER, tol_b)
                     if xi_ - lo < tb or hi - xi_ < tb)
    return CalibrationResult(
        params=p, x=x, se=se, corr=corr, cond=cond, jac=J, residuals=sol.fun, iv_errors=iv_err,
        iv_rmse=rmse, n_iv_unresolved=n_unres, feller_ratio=p.feller_ratio,
        at_bound=at_bound, nfev=sol.nfev, success=bool(sol.success), message=sol.message,
    )
