"""Autocallable Greeks: bump-and-revalue with common random numbers.

The truth for an MC finite difference is the same finite difference of an exact price, at
the same bump: the MVN oracle in the BS limit, and the COS static replication for one date
under Heston. Comparing like with like means the FD truncation error does not enter the
tolerance. That error is bounded separately, to show the default bumps are fine.
"""

import functools

import numpy as np
import pytest
from scipy.stats import chi2

import reference as R
from conftest import REF, model_args
from hestonlib.autocall import Autocallable, autocallable_pv, price_autocallable
from hestonlib.greeks import _vol_bump, autocallable_greeks
from hestonlib.mc import mc_estimate, simulate
from hestonlib.params import HestonParams

PRODUCT = Autocallable(obs_times=(1.0, 2.0, 3.0), autocall_barrier=1.0, coupon=0.08,
                       ki_barrier=0.6, notional=100.0)
H_SPOT, H_VOL, H_RATE = 0.02, 0.01, 1e-3  # the defaults of autocallable_greeks
SIG = np.sqrt(REF["v0"])


def _oracle(S0=100.0, sig=SIG, r=REF["r"]):
    return R.bs_autocallable(S0, r, REF["q"], sig, PRODUCT.obs_times, 1.0, 0.6, 0.08, 100.0,
                             S_ref=100.0)


@functools.lru_cache(maxsize=None)
def _oracle_fd(h_spot=H_SPOT, h_vol=H_VOL, h_rate=H_RATE):
    hs = 100.0 * h_spot
    up, mid, dn = _oracle(100.0 + hs), _oracle(), _oracle(100.0 - hs)
    return {
        "price": mid,
        "delta": (up - dn) / (2 * hs),
        "gamma": (up - 2 * mid + dn) / hs**2,
        "vega": (_oracle(sig=SIG + h_vol) - _oracle(sig=SIG - h_vol)) / (2 * h_vol),
        "rho": (_oracle(r=REF["r"] + h_rate) - _oracle(r=REF["r"] - h_rate)) / (2 * h_rate),
    }


def test_fixing_is_needed_for_spot_risk(ref_params):
    """The payoff is scale invariant: scaling spot and fixing by 1.3 leaves every path's PV
    unchanged (the log-paths shift by ln 1.3), so the price agrees to rounding (1e-10).
    This is why delta must bump spot with the fixing held at ref_spot."""
    a = price_autocallable(PRODUCT, ref_params, 6, 20_000, seed=3)
    b = price_autocallable(PRODUCT, ref_params.replace(S0=130.0), 6, 20_000, seed=3, ref_spot=130.0)
    assert abs(a.price - b.price) < 1e-10


def test_greeks_are_paired_differences_with_crn(ref_params):
    """Delta is the mean of per-path (PV(S0 + h) - PV(S0 - h)) / (2h) on the same seed, and its
    SE is that of the pair means of those per-path values (checked against a manual run)."""
    g = autocallable_greeks(PRODUCT, ref_params, 6, 20_000, seed=4)
    run = lambda S0: autocallable_pv(
        PRODUCT, simulate(ref_params.replace(S0=S0), 3.0, 6, 20_000, 4, obs_times=PRODUCT.obs_times).S,
        100.0, REF["r"])[0]
    manual = mc_estimate((run(102.0) - run(98.0)) / 4.0, True)
    assert g["delta"].price == pytest.approx(manual.price, abs=1e-12)
    assert g["delta"].se == pytest.approx(manual.se, abs=1e-12)


def test_bs_limit_greeks_match_oracle_fd():
    """xi = 1e-8, v0 = theta: every bumped scenario is an unbiased MC estimate of its exact BS
    price (log-Euler is exact GBM at the dates; the vol bump keeps v0 = theta), so each MC
    finite difference is unbiased for the oracle's finite difference at the same bump.
    Tolerance 4 SE for price, delta, gamma, vega and rho.

    Bump sizes vs the barrier discontinuities: the central-difference truncation error of the
    exact price is c h^2, so FD(h) - FD(h/2) = (3/4) c h^2 and the bias at h is
    (4/3) |FD(h) - FD(h/2)|. It must be below a quarter of this run's SE for each Greek, so
    the default bumps barely distort the Greek. The oracle's Genz error (seeded, abseps
    1e-10) is below 1e-8 in price, at most 4e-8 / h^2 = 1e-8 in gamma, which is negligible."""
    p = HestonParams(**dict(REF, xi=1e-8))
    g = autocallable_greeks(PRODUCT, p, 3, 400_000, seed=5)
    ora = _oracle_fd()
    for k in ("price", "delta", "gamma", "vega", "rho"):
        assert abs(g[k].price - ora[k]) < 4 * g[k].se, (k, g[k].price, ora[k], g[k].se)
    half = _oracle_fd(H_SPOT / 2, H_VOL / 2, H_RATE / 2)
    for k in ("delta", "gamma", "vega", "rho"):
        bias = 4 / 3 * abs(ora[k] - half[k])
        assert bias < g[k].se / 4, (k, bias, g[k].se)


def test_greek_se_matches_spread_across_seeds(ref_params):
    """Each Greek's SE (from paired per-path differences, on pair means) must equal the true
    sd of the estimator: over 200 seeds, var(estimates) / mean(SE^2) lies within the
    chi2_199/199 0.05% and 99.95% quantiles. Each estimate is a mean over 10,000 pairs; the
    digital jumps at the barrier make per-path values heavy tailed, but their mean is close to
    normal at this size (seen over two seed bases while building this test)."""
    est = {k: [] for k in ("delta", "gamma", "vega", "rho")}
    se2 = {k: [] for k in est}
    for s in range(200):
        g = autocallable_greeks(PRODUCT, ref_params, 3, 20_000, seed=9000 + s)
        for k in est:
            est[k].append(g[k].price)
            se2[k].append(g[k].se ** 2)
    lo, hi = chi2.ppf([0.0005, 0.9995], 199) / 199
    for k in est:
        ratio = np.var(est[k], ddof=1) / np.mean(se2[k])
        assert lo < ratio < hi, (k, ratio, lo, hi)


# --------------------------------------------------------------------------- Heston, one date
T1, B1, KI1, C1 = 1.0, 1.05, 0.7, 0.10
ONE_DATE = Autocallable(obs_times=(T1,), autocall_barrier=B1, coupon=C1, ki_barrier=KI1)


def _replication(d, S_ref=100.0, N=100.0):
    """One date, fixing S_ref, spot d["S0"], levels H = B S_ref, L = KI S_ref (L <= H):
    V = N e^{-rT} + N c (e^{-rT} - DP(H)) - (N / S_ref) [P(L) + (S_ref - L) DP(L)]."""
    S0, r = d["S0"], d["r"]
    a = model_args(d)
    H, L = B1 * S_ref, KI1 * S_ref
    dph = R.cos_digital_put(S0, H, T1, *a)[0]
    pl, dpl = R.cos_put(S0, L, T1, *a)[0], R.cos_digital_put(S0, L, T1, *a)[0]
    return N * np.exp(-r * T1) + N * C1 * (np.exp(-r * T1) - dph) - (N / S_ref) * (pl + (S_ref - L) * dpl)


def _vb(d, h):
    return dict(d, v0=(np.sqrt(d["v0"]) + h) ** 2, theta=(np.sqrt(d["theta"]) + h) ** 2)


def _replication_fd(d):
    hs = 100.0 * H_SPOT
    return {
        "delta": (_replication(dict(d, S0=100.0 + hs)) - _replication(dict(d, S0=100.0 - hs))) / (2 * hs),
        "vega": (_replication(_vb(d, H_VOL)) - _replication(_vb(d, -H_VOL))) / (2 * H_VOL),
    }


def test_heston_one_date_delta_and_vega_match_cos_fd():
    """REF parameters. MC delta and vega on 64 steps (200k paths) vs the COS replication at
    the same bumps. The Euler bias allowance on each FD estimator is the stage 2 one: the
    larger of |D| + 4 SE(D) with D = FD(32) - FD(64), and sqrt(8/64)(|FD(8) - truth| +
    4 SE(FD(8))); tolerance 4 SE + allowance. Power: the tolerance must be below half the
    change in the true Greek from flipping the sign of rho. _vol_bump is the bump used by
    autocallable_greeks, so the MC scenarios are those of the production code.

    What this does and does not show: for these one-date Greeks the Euler bias is below MC
    resolution at 200k paths (D was within 2 SE(D) of 0 on every seed tried in review), so
    the allowance is in effect statistical and the test does not resolve the Greek's
    step-size dependence. Bias shrinkage with dt is shown for prices in test_mc and
    test_autocall; under CRN a Greek's bias is a difference of those price biases. The power
    alternative is a sign flip of rho: a nearby rho (-0.5) moves the one-date delta by only
    0.0016, below what 200k paths can resolve. Gamma and rho are tested only in the BS limit;
    they use the same scenario code as delta and vega."""
    p = HestonParams(**REF)
    assert _vol_bump(p, H_VOL).v0 == pytest.approx(_vb(REF, H_VOL)["v0"], abs=1e-15)
    truth, flipped = _replication_fd(REF), _replication_fd(dict(REF, rho=-REF["rho"]))
    hs = 100.0 * H_SPOT
    fd = {}
    for n in (8, 32, 64):
        run = lambda q: autocallable_pv(
            ONE_DATE, simulate(q, T1, n, 200_000, 71, obs_times=[T1], rng_steps=64).S, 100.0, q.r)[0]
        fd[n] = {
            "delta": (run(p.replace(S0=100.0 + hs)) - run(p.replace(S0=100.0 - hs))) / (2 * hs),
            "vega": (run(_vol_bump(p, H_VOL)) - run(_vol_bump(p, -H_VOL))) / (2 * H_VOL),
        }
    for k in ("delta", "vega"):
        fine = mc_estimate(fd[64][k], True)
        D = mc_estimate(fd[32][k] - fd[64][k], True)
        coarse = mc_estimate(fd[8][k], True)
        allowance = max(abs(D.price) + 4 * D.se,
                        np.sqrt(8 / 64) * (abs(coarse.price - truth[k]) + 4 * coarse.se))
        tol = 4 * fine.se + allowance
        assert abs(fine.price - truth[k]) < tol, (k, fine.price, truth[k], tol)
        assert tol < 0.5 * abs(flipped[k] - truth[k]), (k, tol, flipped[k] - truth[k])


def test_live_note_greeks_use_the_fixing():
    """A live note: fixing 100, spot 95, BS limit. The price must equal price_autocallable
    with ref_spot = 100 (same seed), and delta must match the oracle's central difference at
    spot 95 +- 2 with the fixing at 100, within 4 SE (unbiased, as in the BS-limit test).
    A fixing that followed spot would give the note at spot 95 struck at 95 instead."""
    p = HestonParams(**dict(REF, xi=1e-8, S0=95.0))
    g = autocallable_greeks(PRODUCT, p, 3, 400_000, seed=6, ref_spot=100.0)
    direct = price_autocallable(PRODUCT, p, 3, 400_000, seed=6, ref_spot=100.0)
    assert g["price"].price == pytest.approx(direct.price, abs=1e-12)
    hs = 100.0 * H_SPOT
    ora = (_oracle(95.0 + hs) - _oracle(95.0 - hs)) / (2 * hs)
    assert abs(g["delta"].price - ora) < 4 * g["delta"].se, (g["delta"].price, ora, g["delta"].se)
