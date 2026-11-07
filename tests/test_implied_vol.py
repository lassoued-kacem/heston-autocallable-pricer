"""Implied volatility, BS vega and the Heston pricer's error bound.

Ground truth: the oracle's Black-Scholes formula and COS prices (tests/reference.py), and
the no-arbitrage bounds. This also covers the two items deferred from stage 1: the discarded
quadrature error estimates, and slightly negative Heston prices at short T deep OTM.
"""

import numpy as np
import pytest

import reference as R
from conftest import REF, STRESS, model_args
from hestonlib.analytic import heston_call, heston_put
from hestonlib.black_scholes import bs_vega
from hestonlib.implied_vol import heston_implied_vols, implied_vol, implied_vols
from hestonlib.params import HestonParams

EPS = np.finfo(float).eps


# --------------------------------------------------------------------------- BS vega
def test_bs_vega_matches_central_difference_of_oracle():
    """Central difference of the oracle BS price with h = 1e-5. The truncation error is
    h^2/6 |d^3C/dsigma^3|. On this grid |d^3C/dsigma^3| <= about 8.4e3 (measured by a second
    difference of the vega, worst at T = 5, sigma = 0.1), giving <= 1.4e-7. Rounding is about
    eps * C / h ~ 1e-9. Tolerance 1e-6."""
    h = 1e-5
    for T in (0.1, 1.0, 5.0):
        for s in (0.1, 0.3, 0.8):
            K = np.array([60.0, 100.0, 150.0])
            fd = (R.bs_price(100.0, K, T, 0.03, 0.01, s + h) - R.bs_price(100.0, K, T, 0.03, 0.01, s - h)) / (2 * h)
            assert np.max(np.abs(bs_vega(100.0, K, T, 0.03, 0.01, s) - fd)) < 1e-6, (T, s)


# --------------------------------------------------------------------------- inversion
@pytest.mark.parametrize("kind", ["call", "put"])
def test_round_trip_recovers_sigma(kind):
    """Prices from the oracle's BS formula, inverted. The inverse is exact up to the price
    rounding (a few eps * S0, the two BS implementations differ only in rounding) divided by
    vega, plus brentq's xtol of 1e-14: tolerance 64 eps S0 / vega + 1e-13. Only quotes with
    vega > 1e-4 are asserted: below that, the price does not pin sigma to useful accuracy."""
    S0, r, q = 100.0, 0.03, 0.01
    n_checked = 0
    for T in (0.1, 1.0, 5.0):
        for s in (0.05, 0.2, 0.8):
            for K in (60.0, 100.0, 150.0):
                price = float(R.bs_price(S0, K, T, r, q, s, kind))
                vega = float(bs_vega(S0, K, T, r, q, s))
                if vega < 1e-4:
                    continue
                iv = implied_vol(price, S0, K, T, r, q, kind)
                assert abs(iv - s) < 64 * EPS * S0 / vega + 1e-13, (T, s, K, iv)
                n_checked += 1
    assert n_checked >= 20


def test_arbitrage_violations_raise():
    """Call bounds: max(S0 e^{-qT} - K e^{-rT}, 0) < C < S0 e^{-qT}."""
    S0, K, T, r, q = 100.0, 90.0, 1.0, 0.03, 0.01
    lo = S0 * np.exp(-q * T) - K * np.exp(-r * T)
    with pytest.raises(ValueError):
        implied_vol(lo - 1e-6, S0, K, T, r, q)
    with pytest.raises(ValueError):
        implied_vol(S0 * np.exp(-q * T) + 1e-6, S0, K, T, r, q)
    with pytest.raises(ValueError):
        implied_vol(-1e-6, S0, 150.0, T, r, q, "call")  # negative price, no error allowance


def test_prices_within_error_of_a_bound_are_unresolvable():
    """A time value at or below the stated price error does not determine sigma: NaN, not a
    clipped guess. A slightly negative price (-7e-11, as seen at T = 1e-3 deep OTM) is
    within an error bound of 1e-9 of the lower bound 0, so it gives NaN without raising."""
    S0, K, T, r, q = 100.0, 300.0, 1e-3, 0.03, 0.0
    assert np.isnan(implied_vol(-7e-11, S0, K, T, r, q, price_err=1e-9))
    assert np.isnan(implied_vol(5e-10, S0, K, T, r, q, price_err=1e-9))
    with pytest.raises(ValueError):
        implied_vol(-2e-9, S0, K, T, r, q, price_err=1e-9)  # beyond the error: arbitrage


# --------------------------------------------------------------------------- Heston pricer error
@pytest.mark.parametrize("d", [REF, STRESS], ids=["ref", "stress"])
@pytest.mark.parametrize("T", [0.05, 1.0, 10.0])
def test_quadrature_error_bound_covers_true_error(d, T):
    """|Lewis - COS| <= reported bound + COS accuracy. The COS oracle is converged to
    1e-11 * max(1, max K) under (N, L) doubling, so that is its allowance."""
    p = HestonParams(**d)
    K = np.array([60.0, 80.0, 100.0, 120.0, 150.0])
    for pricer, oracle in ((heston_call, R.cos_call), (heston_put, R.cos_put)):
        price, err = pricer(K, T, p, return_error=True)
        cos = oracle(d["S0"], K, T, *model_args(d))
        assert np.all(np.abs(price - cos) <= err + 1e-11 * K.max()), (pricer.__name__, price - cos, err)
    # and the scalar path reports a bound too
    c1, e1 = heston_call(100.0, T, p, return_error=True)
    assert abs(c1 - R.cos_call(d["S0"], 100.0, T, *model_args(d))[0]) <= e1 + 1e-9


def test_short_maturity_deep_otm_heston_ivs():
    """Deferred from stage 1. At T in {1e-3, 1e-2, 0.05}, deep OTM Heston prices can be
    slightly negative. heston_implied_vols must (1) never raise, (2) report NaN exactly
    where the time value is within the error bound, (3) resolve ATM, and (4) give IVs that
    reprice to within the error bound (each finite IV is consistent with the price)."""
    p = HestonParams(**REF)
    K = np.array([30.0, 50.0, 70.0, 100.0, 150.0, 200.0, 300.0])
    saw_negative = False
    for T in (1e-3, 1e-2, 0.05):
        for kind in ("call", "put"):
            iv, price, err = heston_implied_vols(K, T, p, kind)
            saw_negative |= bool(np.any(price < 0))
            fs, dk = p.S0 * np.exp(-p.q * T), K * np.exp(-p.r * T)
            lo = np.maximum(fs - dk, 0) if kind == "call" else np.maximum(dk - fs, 0)
            unresolved = price - lo <= err
            assert np.array_equal(np.isnan(iv), unresolved), (T, kind, iv, price - lo, err)
            assert not np.isnan(iv[K == 100.0]).any()
            ok = ~np.isnan(iv)
            rep = R.bs_price(p.S0, K[ok], T, p.r, p.q, iv[ok], kind)
            vega = bs_vega(p.S0, K[ok], T, p.r, p.q, iv[ok])
            assert np.all(np.abs(rep - price[ok]) <= err[ok] + 64 * EPS * p.S0 + vega * 1e-13)
    assert saw_negative  # the case the deferred item is about actually occurs


def test_heston_iv_in_bs_limit():
    """xi = 1e-8, v0 = theta: test_analytic bounds the price gap to BS(0.2) by 1e-5 at T = 1
    for K in {80, 100, 120}, so the IV gap is below 1e-5 / vega (vega >= 15 here)."""
    p = HestonParams(**dict(REF, xi=1e-8))
    K = np.array([80.0, 100.0, 120.0])
    iv, _, _ = heston_implied_vols(K, 1.0, p)
    vega = bs_vega(100.0, K, 1.0, REF["r"], REF["q"], 0.2)
    assert np.all(vega >= 15)
    assert np.all(np.abs(iv - 0.2) < 1e-5 / vega)


def test_vector_inversion_matches_scalar():
    K = np.array([80.0, 100.0, 120.0])
    prices = R.bs_price(100.0, K, 1.0, 0.03, 0.0, 0.25)
    v = implied_vols(prices, 100.0, K, 1.0, 0.03, 0.0)
    assert np.all(np.abs(v - 0.25) < 1e-12)
