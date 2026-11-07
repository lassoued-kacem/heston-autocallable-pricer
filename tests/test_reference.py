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


# --------------------------------------------------------------------------- BS autocallable
from scipy.integrate import quad  # noqa: E402
from scipy.stats import norm  # noqa: E402

AC = dict(S0=100.0, r=0.03, q=0.01, sigma=0.2, B=1.0, KI=0.6, c=0.08)


def _bs_continuation(x, dt, mu, s, b, k, c_next, r, N):
    """Value at t_i, given X_i = x < b, of the last date at t_i + dt (closed form):
    N (1 + c_next) P(call) + N P(KI <= S < B) + N E[e^{X} ; X < k], discounted over dt."""
    m, sd = x + mu * dt, s * np.sqrt(dt)
    p_call = 1 - norm.cdf((b - m) / sd)
    p_mid = norm.cdf((b - m) / sd) - norm.cdf((k - m) / sd)
    e_ki = np.exp(m + 0.5 * sd * sd) * norm.cdf((k - m - sd * sd) / sd)
    return np.exp(-r * dt) * N * ((1 + c_next) * p_call + p_mid + e_ki)


def test_bs_autocallable_one_date_closed_form():
    """n = 1 is European: N (1+c) e^{-rT} N(d2(B)) + N e^{-rT} [N(-d2(B)) - N(-d2(KI))]
    + N e^{-qT} N(-d1(KI)), with d1, d2 the BS terms at strike K S0. Tolerance 1e-12."""
    S0, r, q, s, B, KI, c = AC.values()
    T, N = 1.5, 100.0

    def d12(K):
        d1 = (np.log(1 / K) + (r - q + 0.5 * s * s) * T) / (s * np.sqrt(T))
        return d1, d1 - s * np.sqrt(T)

    _, d2b = d12(B)
    d1k, d2k = d12(KI)
    expect = N * (1 + c) * np.exp(-r * T) * norm.cdf(d2b) + N * np.exp(-r * T) * (
        norm.cdf(-d2b) - norm.cdf(-d2k)) + N * np.exp(-q * T) * norm.cdf(-d1k)
    assert abs(R.bs_autocallable(S0, r, q, s, [T], B, KI, c, N) - expect) < 1e-12


def test_bs_autocallable_two_dates_vs_quadrature():
    """n = 2 by 1-D quadrature over X_1 with the closed-form last date. quad runs to
    1e-12; the bivariate normal CDF is accurate to about 1e-10. Tolerance 1e-8 on a price
    of about 100."""
    S0, r, q, s, B, KI, c = AC.values()
    mu, b, k, N = r - q - 0.5 * s * s, np.log(B), np.log(KI), 100.0

    def f(x1):
        return norm.pdf(x1, mu, s) * _bs_continuation(x1, 1.0, mu, s, b, k, 2 * c, r, N)

    cont = quad(f, mu - 12 * s, b, epsabs=1e-12, epsrel=1e-12, limit=200)[0]
    expect = np.exp(-r) * (N * (1 + c) * (1 - norm.cdf((b - mu) / s)) + cont)
    assert abs(R.bs_autocallable(S0, r, q, s, [1.0, 2.0], B, KI, c, N) - expect) < 1e-8


def test_bs_autocallable_three_dates_vs_nested_quadrature():
    """n = 3 by nested 1-D quadratures. The trivariate CDF comes from Genz's randomized
    integration (requested abseps 1e-10); a 5e-8 price difference was seen when this was
    built. Tolerance 1e-6 on a price of about 100 (a probability error of 1e-8)."""
    S0, r, q, s, B, KI, c = AC.values()
    mu, b, k, N = r - q - 0.5 * s * s, np.log(B), np.log(KI), 100.0

    def at_t1(x1):
        g = lambda x2: norm.pdf(x2, x1 + mu, s) * _bs_continuation(x2, 1.0, mu, s, b, k, 3 * c, r, N)
        cont = quad(g, x1 + mu - 12 * s, b, epsabs=1e-12, epsrel=1e-12, limit=200)[0]
        return np.exp(-r) * (N * (1 + 2 * c) * (1 - norm.cdf((b - x1 - mu) / s)) + cont)

    cont = quad(lambda x1: norm.pdf(x1, mu, s) * at_t1(x1), mu - 12 * s, b,
                epsabs=1e-11, epsrel=1e-11, limit=200)[0]
    expect = np.exp(-r) * (N * (1 + c) * (1 - norm.cdf((b - mu) / s)) + cont)
    got = R.bs_autocallable(S0, r, q, s, [1.0, 2.0, 3.0], B, KI, c, N)
    assert abs(got - expect) < 1e-6, (got, expect)


def test_bs_orthant_probability_exact():
    """For a zero-mean trivariate normal, P(all < 0) = 1/8 + sum_{i<j} asin(rho_ij)/(4 pi).
    With Brownian correlations rho_ij = sqrt(t_i/t_j) and b = 0, mu = 0 (r - q = sigma^2/2),
    F_3 = 1 - (sum of call probabilities). Tolerance 1e-8 (Genz abseps 1e-10)."""
    t = np.array([1.0, 2.0, 3.0])
    s = 0.2
    probs = R.bs_autocall_probabilities(100.0, 0.5 * s * s, 0.0, s, t, 1.0)
    rho = np.sqrt(np.minimum.outer(t, t) / np.maximum.outer(t, t))
    exact = 1 / 8 + (np.arcsin(rho[0, 1]) + np.arcsin(rho[0, 2]) + np.arcsin(rho[1, 2])) / (4 * np.pi)
    assert abs((1 - probs.sum()) - exact) < 1e-8
    assert probs[0] == pytest.approx(0.5, abs=1e-14)  # P(X_1 >= 0) = 1/2 with zero drift


def test_bs_autocallable_ki_above_barrier_one_date():
    """KI > B: every uncalled path is below KI, so for one date the payoff is N (1 + c) if
    S_T >= B S0 and N S_T / S0 otherwise: N (1+c) e^{-rT} N(d2(B)) + N e^{-qT} N(-d1(B)).
    This exercises the min(KI, B) branch. Tolerance 1e-12."""
    S0, r, q, s, _, _, c = AC.values()
    T, B, KI, N = 2.0, 0.9, 1.1, 100.0
    d1 = (np.log(1 / B) + (r - q + 0.5 * s * s) * T) / (s * np.sqrt(T))
    d2 = d1 - s * np.sqrt(T)
    expect = N * (1 + c) * np.exp(-r * T) * norm.cdf(d2) + N * np.exp(-q * T) * norm.cdf(-d1)
    assert abs(R.bs_autocallable(S0, r, q, s, [T], B, KI, c, N) - expect) < 1e-12
