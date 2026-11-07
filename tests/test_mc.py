"""Monte Carlo (full-truncation Euler) tests.

Ground truth: the COS oracle (tests/reference.py), the Black-Scholes limit, and exact
properties of the discrete scheme (martingale forward, one-step correlation, the
truncation rule). Every MC comparison uses 4 SE (two-sided false-failure probability about
6e-5 per assertion for a normal estimate) plus, for Euler prices, an explicit bias allowance.
"""

import numpy as np
import pytest
from scipy.stats import chi2

import reference as R
from conftest import REF, STRESS, model_args
from hestonlib.mc import euler_step, mc_estimate, mc_european, simulate
from hestonlib.params import HestonParams


# --------------------------------------------------------------------------- estimator
def test_antithetic_se_uses_pair_means():
    """Rows i and i + n/2 are pairs. For x = [1, 2, 3, 3, 4, 5] the pair means are 2, 3, 4:
    mean 3, sample sd 1, so SE = 1/sqrt(3) from 3 samples (not 6)."""
    r = mc_estimate(np.array([1.0, 2.0, 3.0, 3.0, 4.0, 5.0]), antithetic=True)
    assert r.price == pytest.approx(3.0, abs=1e-15)
    assert r.se == pytest.approx(1.0 / np.sqrt(3.0), rel=1e-14)
    assert r.n_samples == 3


def test_antithetic_odd_payoff_has_zero_se():
    """An odd function of the normals, c + a, c - a, has constant pair means: SE exactly 0."""
    a = np.random.default_rng(0).standard_normal(50)
    r = mc_estimate(np.concatenate([2.0 + a, 2.0 - a]), antithetic=True)
    assert r.price == pytest.approx(2.0, abs=1e-14) and r.se < 1e-14


def test_reported_se_matches_spread_across_seeds(ref_params):
    """The SE must equal the true sd of the estimator. Over R = 400 independent seeds the
    sample variance of the estimates, divided by the mean reported SE^2, is
    chi2_{R-1}/(R-1) for normal estimates (each is a mean of 1000 pair means, so close to
    normal). The window is its 0.05% and 99.95% quantiles. The noise in the mean of the
    reported SE^2 is about sqrt(2/1000)/sqrt(400) ~ 0.2% and is ignored.
    The naive SE over the 2n halves must fall outside the window: for an ATM call, antithetic
    pairs are negatively correlated, so it overstates the error. This shows the test has power."""
    n_rep, n_paths = 400, 2000
    est, se2, naive2 = [], [], []
    for s in range(n_rep):
        P = simulate(ref_params, 1.0, 8, n_paths, seed=1000 + s)
        x = np.exp(-ref_params.r) * np.maximum(P.S[:, -1] - 100.0, 0.0)
        r = P.estimate(x)
        est.append(r.price)
        se2.append(r.se**2)
        naive2.append(x.var(ddof=1) / n_paths)
    lo, hi = chi2.ppf([0.0005, 0.9995], n_rep - 1) / (n_rep - 1)
    ratio = np.var(est, ddof=1) / np.mean(se2)
    assert lo < ratio < hi, (lo, ratio, hi)
    naive_ratio = np.var(est, ddof=1) / np.mean(naive2)
    assert not lo < naive_ratio < hi, naive_ratio


# --------------------------------------------------------------------------- scheme
def test_full_truncation_step_with_negative_variance():
    """With v < 0, v+ = 0: the asset gets drift (r - q) dt and no diffusion, and the variance
    gets kappa theta dt and no diffusion. The raw v is carried forward (v + kappa theta dt),
    not reset to 0. Values derived by hand from the scheme definition."""
    p = HestonParams(**REF)
    dt = 0.01
    lnS, v = euler_step(np.log(100.0), -0.01, dt, 1.7, -0.4, p)
    assert lnS == pytest.approx(np.log(100.0) + REF["r"] * dt, abs=1e-15)
    assert v == pytest.approx(-0.01 + REF["kappa"] * REF["theta"] * dt, abs=1e-15)


def test_euler_step_positive_variance_by_hand():
    """One step from v = 0.04 with z_v = 0.5, z_perp = -1.2, dt = 0.01 (REF parameters):
    z_S = -0.7 * 0.5 + sqrt(0.51) * (-1.2), sd = sqrt(0.04 * 0.01) = 0.02,
    ln S' = ln 100 + (0.03 - 0.02) * 0.01 + 0.02 z_S,
    v'    = 0.04 + 2 (0.04 - 0.04) 0.01 + 0.3 * 0.02 * 0.5 = 0.043."""
    p = HestonParams(**REF)
    lnS, v = euler_step(np.log(100.0), 0.04, 0.01, 0.5, -1.2, p)
    z_s = -0.35 - 1.2 * np.sqrt(0.51)
    assert lnS == pytest.approx(np.log(100.0) + 0.0001 + 0.02 * z_s, abs=1e-14)
    assert v == pytest.approx(0.043, abs=1e-15)


def test_one_step_correlation_equals_rho():
    """After one step, ln S and v are affine in (Z_S, Z_v) with deterministic coefficients
    of the same sign, so corr(ln S_dt, v_dt) = rho exactly. The SE of a sample correlation
    is (1 - rho^2)/sqrt(n) for bivariate normal data (iid paths, no antithetics).
    Tolerance 4 SE. A sign error in rho would be off by 1.4."""
    for rho in (-0.7, 0.4):
        p = HestonParams(**dict(REF, rho=rho))
        P = simulate(p, 0.01, 1, 100_000, seed=7, antithetic=False)
        c = np.corrcoef(np.log(P.S[:, 0]), P.v[:, 0])[0, 1]
        assert abs(c - rho) < 4 * (1 - rho**2) / np.sqrt(100_000), (rho, c)


@pytest.mark.parametrize("d", [REF, STRESS], ids=["ref", "stress"])
def test_discounted_asset_is_exact_martingale(d):
    """E[e^{-(r-q) t} S_t] = S0 holds exactly for the discrete log-Euler scheme (see the
    mc.py docstring), so there is no bias term: tolerance 4 SE at every observation time.
    The stress set (Feller ratio 0.09) exercises the truncation."""
    p = HestonParams(**d)
    t = np.array([0.25, 0.5, 1.0, 2.0])
    P = simulate(p, 2.0, 32, 200_000, seed=3, obs_times=t)
    r = P.estimate(np.exp(-(p.r - p.q) * t)[None, :] * P.S)
    assert np.all(np.abs(r.price - p.S0) < 4 * r.se), (r.price, r.se)


def test_bs_limit_is_unbiased():
    """With v0 = theta and xi = 1e-8, v stays at theta (to O(xi)), and log-Euler is exact for
    geometric Brownian motion, so even 4 steps give an unbiased BS price. The Heston-BS gap
    is O(xi) ~ 1e-8, far below the SE. Tolerance 4 SE."""
    p = HestonParams(**dict(REF, xi=1e-8))
    K = np.array([80.0, 100.0, 125.0])
    for kind in ("call", "put"):
        r = mc_european(K, 1.0, p, kind, n_steps=4, n_paths=200_000, seed=11)
        bs = R.bs_price(100.0, K, 1.0, REF["r"], REF["q"], np.sqrt(REF["v0"]), kind)
        assert np.all(np.abs(r.price - bs) < 4 * r.se), (kind, r.price - bs, r.se)


def test_levels_share_the_brownian_path():
    """With the same seed and rng_steps, a 4-step and a 64-step run use the same Brownian
    path. In the near-BS limit (v = theta to O(xi)), ln S_T = ln S0 + (r - q - theta/2) T
    + sqrt(theta) W^S_T does not depend on the step count, so the two runs agree to O(xi)
    plus rounding. Tolerance 1e-6, well above xi = 1e-8 times O(1) path factors and far below
    the O(0.1) differences that uncoupled paths give."""
    p = HestonParams(**dict(REF, xi=1e-8))
    a = simulate(p, 1.0, 4, 2000, seed=5, rng_steps=64).S[:, -1]
    b = simulate(p, 1.0, 64, 2000, seed=5, rng_steps=64).S[:, -1]
    assert np.max(np.abs(np.log(a / b))) < 1e-6


def test_same_seed_reproduces_and_obs_grid_is_checked(ref_params):
    a = simulate(ref_params, 1.0, 8, 1000, seed=9).S
    b = simulate(ref_params, 1.0, 8, 1000, seed=9).S
    assert np.array_equal(a, b)
    with pytest.raises(ValueError):
        simulate(ref_params, 1.0, 8, 1000, obs_times=[0.3])  # 0.3 is not a multiple of 1/8
    with pytest.raises(ValueError):
        simulate(ref_params, 1.0, 8, 1001)  # antithetic needs an even count
    with pytest.raises(ValueError):
        simulate(ref_params, 1.0, 8, 4, obs_times=[1e-10, 1.0])  # rounds to grid index 0


# --------------------------------------------------------------------------- Euler bias
def test_euler_bias_shrinks_with_step_size():
    """Stress set (Feller ratio 0.09), T = 1, K = 100. Levels n = 8, 16, 32, 64 share one
    Brownian path (rng_steps = 128), so the change in price between n and 2n,
    D(n) = MC(n) - MC(2n), is a paired difference with a small SE.
    (1) The bias at n = 8 is resolved: MC(8) - COS > 4 SE.
    (2) It shrinks at every halving of dt: D(n) > 4 SE(D(n)) for n = 8, 16, 32, i.e. the
        error decreases monotonically (the bias is positive here, as in (1)).
    (3) The error at n = 128 is smaller than at n = 8 by more than 4 combined SE, and
        consistent with weak order 1: below 8/128 of the n = 8 error plus 4 SE."""
    p = HestonParams(**STRESS)
    cos = R.cos_call(STRESS["S0"], 100.0, 1.0, *model_args(STRESS))[0]
    pay = {}
    for n in (8, 16, 32, 64, 128):
        P = simulate(p, 1.0, n, 200_000, seed=21, rng_steps=128)
        pay[n] = np.exp(-p.r) * np.maximum(P.S[:, -1] - 100.0, 0.0)
    e8 = mc_estimate(pay[8], True)
    assert e8.price - cos > 4 * e8.se
    for n in (8, 16, 32):
        D = mc_estimate(pay[n] - pay[2 * n], True)
        assert D.price > 4 * D.se, (n, D)
    e128 = mc_estimate(pay[128], True)
    gap = mc_estimate(pay[8] - pay[128], True)
    assert gap.price > 4 * gap.se
    assert abs(e128.price - cos) < (8 / 128) * (e8.price - cos) + 4 * (e128.se + e8.se)


def test_call_prices_match_cos_within_se_and_bias():
    """REF parameters, T = 1, n = 64 steps; tolerance 4 SE(MC(64)) + a bias allowance.

    At REF the bias at n >= 32 is below the MC noise, so its rate cannot be measured there.
    The allowance is therefore the larger of two estimates of bias(64):
    * Richardson, under weak order alpha >= 1: E[D] = bias(32) - bias(64)
      = bias(64) (2^alpha - 1) >= bias(64), with D = MC(32) - MC(64); allowance |D| + 4 SE(D).
    * A conservative order-1/2 extrapolation from n = 8, where the REF bias is resolved:
      |bias(64)| <= sqrt(8/64) (|MC(8) - COS| + 4 SE(MC(8))). Order 1/2 is half the rate
      seen on the stress set, so this covers a much slower convergence than observed.
    All levels share one Brownian path (rng_steps = 64)."""
    p = HestonParams(**REF)
    K = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    cos = R.cos_call(REF["S0"], K, 1.0, *model_args(REF))
    pay = {}
    for n in (8, 32, 64):
        P = simulate(p, 1.0, n, 200_000, seed=31, rng_steps=64)
        pay[n] = np.exp(-p.r) * np.maximum(P.S[:, -1:] - K[None, :], 0.0)
    fine = mc_estimate(pay[64], True)
    D = mc_estimate(pay[32] - pay[64], True)
    coarse = mc_estimate(pay[8], True)
    richardson = np.abs(D.price) + 4 * D.se
    half_order = np.sqrt(8 / 64) * (np.abs(coarse.price - cos) + 4 * coarse.se)
    tol = 4 * fine.se + np.maximum(richardson, half_order)
    assert np.all(np.abs(fine.price - cos) < tol), (fine.price - cos, tol)
    # The tolerance must stay meaningful: well below the prices themselves.
    assert np.all(tol < 0.1)
