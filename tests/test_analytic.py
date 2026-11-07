"""Heston semi-analytic (Lewis) pricer vs the independent COS oracle and exact limits."""

import numpy as np
import pytest

import reference as R
from conftest import REF, STRESS, model_args
from hestonlib.analytic import heston_call, heston_cf, heston_put
from hestonlib.params import HestonParams

STRIKES = np.array([60.0, 80.0, 90.0, 100.0, 110.0, 120.0, 150.0])
MATURITIES = [0.05, 0.25, 1.0, 3.0, 10.0]


@pytest.mark.parametrize("T", MATURITIES)
def test_call_matches_cos_oracle_reference(ref_params, T):
    """Lewis/quad vs COS: two independent quadratures of the same model. Both are accurate
    to ~1e-10 (quad epsabs 1e-12, COS self-converged to 1e-9). Tolerance 1e-8 absolute on a
    notional of S0 = 100."""
    cos = R.cos_call(REF["S0"], STRIKES, T, *model_args(REF))
    assert np.max(np.abs(heston_call(STRIKES, T, ref_params) - cos)) < 1e-8


@pytest.mark.parametrize("T", [0.5, 5.0, 15.0])
def test_call_matches_cos_oracle_stress(stress_params, T):
    """Same comparison in the trap-prone regime (xi = 1, rho = -0.95, Feller ratio 0.09,
    T up to 15). A branch-cut error would show up as an O(1) price error. Tolerance 1e-8."""
    cos = R.cos_call(STRESS["S0"], STRIKES, T, *model_args(STRESS))
    assert np.max(np.abs(heston_call(STRIKES, T, stress_params) - cos)) < 1e-8


@pytest.mark.parametrize("T", [0.25, 3.0])
def test_put_matches_cos_native_put(ref_params, stress_params, T):
    """The oracle prices puts natively (calls via parity), and the package prices calls
    natively (puts via parity). Agreement checks put-call parity across two independent
    pricers. Tolerance 1e-8."""
    for p, d in [(ref_params, REF), (stress_params, STRESS)]:
        cos_put = R.cos_put(d["S0"], STRIKES, T, *model_args(d))
        assert np.max(np.abs(heston_put(STRIKES, T, p) - cos_put)) < 1e-8


def test_put_call_parity_with_dividends():
    """C - P = S0 e^{-qT} - K e^{-rT}, with C from the package and P from the oracle.
    Tolerance 1e-8 (sum of both quadrature errors)."""
    d = dict(REF, q=0.025)
    p = HestonParams(**d)
    T = 2.0
    C = heston_call(STRIKES, T, p)
    P = R.cos_put(d["S0"], STRIKES, T, *model_args(d))
    rhs = d["S0"] * np.exp(-d["q"] * T) - STRIKES * np.exp(-d["r"] * T)
    assert np.max(np.abs(C - P - rhs)) < 1e-8


def test_scalar_and_vector_paths_agree(ref_params):
    """quad (scalar K) and quad_vec (array K) integrate the same integrand: 1e-9."""
    vec = heston_call(STRIKES, 1.0, ref_params)
    sca = np.array([heston_call(k, 1.0, ref_params) for k in STRIKES])
    assert np.max(np.abs(vec - sca)) < 1e-9


@pytest.mark.parametrize("p_dict", [REF, STRESS], ids=["ref", "stress"])
def test_package_cf_matches_riccati_ode(p_dict):
    """Package CF (for ln(S_T/F_T)) times e^{i u (r-q) T} is the CF of ln(S_T/S0), which the
    oracle computes by ODE integration. Tolerance 1e-9 (ODE rtol 1e-12)."""
    p = HestonParams(**p_dict)
    T = 2.0
    for u in [0.1, 2.0, 10.0, 0.1 - 0.5j, 10.0 - 0.5j]:
        pkg = heston_cf(u, T, p) * np.exp(1j * u * (p.r - p.q) * T)
        ode = R.heston_cf_ode(u, T, *model_args(p_dict))
        assert abs(pkg - ode) < 1e-9


def _bs_gap(xi, rho):
    p = HestonParams(**dict(REF, xi=xi, rho=rho))
    K = np.array([80.0, 100.0, 120.0])
    bs = R.bs_price(REF["S0"], K, 1.0, REF["r"], REF["q"], np.sqrt(REF["v0"]))
    return np.max(np.abs(heston_call(K, 1.0, p) - bs))


def test_bs_limit_first_order_in_xi():
    """v0 = theta, xi -> 0: Heston -> BS(sigma = sqrt(v0)). The leading correction of the
    vol-of-vol expansion is linear in rho*xi, so each decade of xi should cut the gap ~10x.
    Asserted: ratio > 8 per decade (allowing for the O(xi^2) term), and the gap is < 5e-3 at
    xi = 1e-3."""
    g = [_bs_gap(x, -0.7) for x in (1e-1, 1e-2, 1e-3)]
    assert g[0] / g[1] > 8 and g[1] / g[2] > 8, g
    assert g[2] < 5e-3


def test_bs_limit_second_order_when_rho_zero():
    """With rho = 0 the linear term vanishes, so the gap is O(xi^2): ratio > 80 per decade
    and the gap is < 1e-5 at xi = 1e-3."""
    g = [_bs_gap(x, 0.0) for x in (1e-1, 1e-2, 1e-3)]
    assert g[0] / g[1] > 80 and g[1] / g[2] > 80, g
    assert g[2] < 1e-5


def test_no_arbitrage_bounds_and_shape(ref_params):
    """Model-free: max(S e^{-qT} - K e^{-rT}, 0) <= C <= S e^{-qT}; C is decreasing and convex
    in K (strict inequalities hold on this grid, with no tolerance needed beyond 1e-12)."""
    K = np.linspace(50, 200, 61)
    T = 1.0
    C = heston_call(K, T, ref_params)
    lower = np.maximum(100.0 - K * np.exp(-0.03 * T), 0.0)
    assert np.all(C >= lower - 1e-12) and np.all(C <= 100.0 + 1e-12)
    assert np.all(np.diff(C) < 0)
    assert np.all(np.diff(C, 2) > -1e-12)


def test_invalid_maturity_raises(ref_params):
    with pytest.raises(ValueError):
        heston_call(100.0, 0.0, ref_params)
