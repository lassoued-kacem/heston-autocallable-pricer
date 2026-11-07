"""Self-checks of the independent oracle in tests/reference.py.

The oracle is the ground truth for the analytic and MC tests, so it is validated against
things that do not depend on it: a numerical ODE solution of the Riccati system, the
Black-Scholes closed form, and its own convergence under (N, L) doubling.
"""

import numpy as np
import pytest

import reference as R
from conftest import REF, STRESS, model_args


@pytest.mark.parametrize("d", [REF, STRESS], ids=["ref", "stress"])
@pytest.mark.parametrize("T", [0.25, 1.0, 5.0])
def test_closed_form_cf_matches_riccati_ode(d, T):
    """Closed-form CF vs DOP853 integration of the Riccati ODEs (rtol 1e-12).

    Tolerance 1e-9: the ODE solver is set to rtol=1e-12/atol=1e-13 and |phi| <= 1, so
    the agreement should be ~1e-11. 1e-9 leaves margin for error accumulation over long T.
    Points include the Lewis contour (u - i/2) and the real axis used by COS.
    """
    for u in [0.0, 0.3, 1.0, 4.0, 15.0, 0.3 - 0.5j, 4.0 - 0.5j, 15.0 - 0.5j]:
        a = R.heston_cf_logret(u, T, *model_args(d))
        b = R.heston_cf_ode(u, T, *model_args(d))
        assert abs(a - b) < 1e-9, (u, a, b)


def test_cf_is_continuous_along_real_axis_in_stress_regime():
    """No branch-cut jumps: on a fine u grid, |phi(u_{j+1}) - phi(u_j)| is bounded by
    max|phi'| * du. For a CF, |phi'(u)| <= E|X_T|, and E|X_T| < 3 for these parameters at
    T = 10 (sd of X is about sqrt(theta * T) ~ 0.95 even with heavy tails). A jump from a
    wrong branch is of order |phi|, which is much larger than 3 * du = 3e-3."""
    u = np.linspace(0, 60, 60001)
    phi = R.heston_cf_logret(u, 10.0, *model_args(STRESS))
    assert np.max(np.abs(np.diff(phi))) < 3.0 * (u[1] - u[0])


@pytest.mark.parametrize("kind", ["call", "put"])
def test_bs_closed_form_matches_lognormal_quadrature(kind):
    """Oracle BS formula vs direct numerical integration of the payoff against the
    lognormal density (trapezoid on 200k points, error < 1e-8)."""
    from scipy.stats import norm

    S0, K, T, r, q, sig = 100.0, 95.0, 1.5, 0.03, 0.01, 0.25
    z = np.linspace(-12, 12, 200001)
    ST = S0 * np.exp((r - q - 0.5 * sig**2) * T + sig * np.sqrt(T) * z)
    pay = np.maximum(ST - K, 0) if kind == "call" else np.maximum(K - ST, 0)
    num = np.exp(-r * T) * np.trapezoid(pay * norm.pdf(z), z)
    assert abs(R.bs_price(S0, K, T, r, q, sig, kind) - num) < 1e-8


def test_cos_converged_under_doubling():
    """Doubling N and L changes the default-converged result by < 1e-9 (reference params)."""
    K = np.array([70.0, 100.0, 140.0])
    args = (100.0, K, 1.0, *model_args(REF))
    a = R.cos_put(*args)
    b = R.cos_put(*args, N=8192, L=48.0, adaptive=False)
    assert np.max(np.abs(a - b)) < 1e-9


def test_cos_rho0_small_xi_matches_bs():
    """With rho = 0 the first-order vol-of-vol correction vanishes, so
    |Heston - BS(sqrt(v0))| = O(xi^2) when v0 = theta. At xi = 1e-2 the measured term is
    ~2e-4, so the 1e-3 tolerance is about 5x the expected O(xi^2) size."""
    K = np.array([80.0, 100.0, 120.0])
    d = dict(REF, xi=1e-2, rho=0.0)
    cos = R.cos_call(100.0, K, 1.0, *model_args(d))
    bs = R.bs_price(100.0, K, 1.0, d["r"], d["q"], np.sqrt(d["v0"]))
    assert np.max(np.abs(cos - bs)) < 1e-3


def test_cos_digital_matches_bs_digital_in_limit():
    """Digital put e^{-rT} N(-d2) in the BS limit (rho = 0, xi = 1e-2): O(xi^2) gap < 1e-4."""
    from scipy.stats import norm

    K = np.array([80.0, 100.0, 120.0])
    d = dict(REF, xi=1e-2, rho=0.0)
    sig, T = 0.2, 1.0
    d2 = (np.log(100.0 / K) + (d["r"] - 0.5 * sig**2) * T) / (sig * np.sqrt(T))
    expected = np.exp(-d["r"] * T) * norm.cdf(-d2)
    got = R.cos_digital_put(100.0, K, T, *model_args(d))
    assert np.max(np.abs(got - expected)) < 1e-4
