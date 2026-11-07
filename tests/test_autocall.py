"""Autocallable tests.

Ground truth:
(a) Single observation date under Heston: the payoff is a static portfolio of a zero-coupon
    bond, a digital and a put, priced by the COS oracle.
(b) Black-Scholes limit, several dates: the exact discrete-monitoring price from
    multivariate normal CDFs (tests/reference.py, bs_autocallable).
(c) Exact limits of the payoff (barrier 0, barrier infinite, KI 0, KI infinite).
"""

import numpy as np
import pytest

import reference as R
from conftest import REF, STRESS, model_args
from hestonlib.autocall import Autocallable, autocallable_pv, price_autocallable
from hestonlib.mc import mc_estimate, simulate
from hestonlib.params import HestonParams

PRODUCT = Autocallable(obs_times=(1.0, 2.0, 3.0), autocall_barrier=1.0, coupon=0.08,
                       ki_barrier=0.6, notional=100.0)


# --------------------------------------------------------------------------- payoff logic
def test_payoff_by_hand():
    """Four hand-built paths, S0 = 100, r = 0, barriers 100% / 60%, c = 8%:
    called at t1 (S = 100 is >= B, so it calls) -> 108; called at t3 -> 124;
    never called, S_T = 70 >= 60 -> 100; never called, S_T = 50 -> 50.
    A path above the barrier after an earlier call still pays only the first call."""
    S = np.array([
        [100.0, 150.0, 150.0],
        [90.0, 99.99, 101.0],
        [90.0, 80.0, 70.0],
        [90.0, 80.0, 50.0],
    ])
    pv, idx = autocallable_pv(PRODUCT, S, 100.0, 0.0)
    assert np.allclose(pv, [108.0, 124.0, 100.0, 50.0], atol=1e-12)
    assert list(idx) == [1, 3, 0, 0]


def test_payoff_discounts_each_cashflow_at_its_own_date():
    """r = 5%: a call at t2 pays 116 e^{-0.1}; a redemption at T = 3 pays 100 e^{-0.15}."""
    S = np.array([[90.0, 120.0, 130.0], [90.0, 80.0, 70.0]])
    pv, _ = autocallable_pv(PRODUCT, S, 100.0, 0.05)
    assert pv == pytest.approx([116.0 * np.exp(-0.10), 100.0 * np.exp(-0.15)], abs=1e-12)


def test_ki_boundary_is_inclusive():
    """S_T = KI S0 exactly redeems at par (S_T >= KI)."""
    S = np.array([[90.0, 80.0, 60.0]])
    pv, _ = autocallable_pv(PRODUCT, S, 100.0, 0.0)
    assert pv[0] == pytest.approx(100.0, abs=1e-12)


# --------------------------------------------------------------------------- exact limits
def test_zero_barrier_calls_at_first_date(ref_params):
    """B = 0: every path is called at t1, so PV = N (1 + c) e^{-r t1} with zero variance."""
    pr = Autocallable(obs_times=(0.5, 1.0), autocall_barrier=0.0, coupon=0.05)
    r = price_autocallable(pr, ref_params, 4, 2000, seed=1)
    assert r.price == pytest.approx(105.0 * np.exp(-REF["r"] * 0.5), abs=1e-10)
    assert r.se < 1e-12
    assert r.call_prob[0] == 1.0


def test_infinite_barrier_zero_ki_is_a_bond(ref_params):
    """B = inf, KI = 0: never called, always redeems at par: N e^{-rT}, zero variance."""
    pr = Autocallable(obs_times=(1.0, 2.0), autocall_barrier=np.inf, ki_barrier=0.0)
    r = price_autocallable(pr, ref_params, 8, 2000, seed=1)
    assert r.price == pytest.approx(100.0 * np.exp(-2 * REF["r"]), abs=1e-10)
    assert r.se < 1e-12


@pytest.mark.parametrize("d", [REF, dict(STRESS)], ids=["ref", "stress"])
def test_infinite_barriers_is_the_forward(d):
    """B = inf, KI = inf: pays N S_T / S0 at T, worth N e^{-qT}. The discounted asset is an
    exact martingale of the scheme (test_mc), so there is no bias: tolerance 4 SE."""
    p = HestonParams(**d)
    pr = Autocallable(obs_times=(1.0, 2.0), autocall_barrier=np.inf, ki_barrier=np.inf)
    r = price_autocallable(pr, p, 16, 200_000, seed=2)
    assert abs(r.price - 100.0 * np.exp(-p.q * 2.0)) < 4 * r.se


def test_probabilities_sum_to_one(ref_params):
    r = price_autocallable(PRODUCT, ref_params, 12, 20_000, seed=3)
    assert r.call_prob.sum() + r.no_call_prob.price == pytest.approx(1.0, abs=1e-12)
    assert 0.0 <= r.loss_prob.price <= r.no_call_prob.price


# --------------------------------------------------------------------------- Heston, 1 date
def _single_date_truth(d, T, B, KI, c, N=100.0):
    """Payoff N + N c 1{S >= B S0} - (N/S0) (S0 - S) 1{S < KI S0}, with KI <= B, and
    (S0 - S) 1{S < K} = (K - S)^+ + (S0 - K) 1{S < K} for K = KI S0. Hence
    V = N e^{-rT} + N c (e^{-rT} - DP(B S0)) - (N/S0) [P(KI S0) + (S0 - KI S0) DP(KI S0)]
    where P is the put and DP the cash-or-nothing digital put, both from the COS oracle."""
    assert KI <= B, "the replication double counts the digital when KI > B"
    S0, r = d["S0"], d["r"]
    a = model_args(d)
    dp_b = R.cos_digital_put(S0, B * S0, T, *a)[0]
    put_k = R.cos_put(S0, KI * S0, T, *a)[0]
    dp_k = R.cos_digital_put(S0, KI * S0, T, *a)[0]
    return N * np.exp(-r * T) + N * c * (np.exp(-r * T) - dp_b) - (N / S0) * (
        put_k + (S0 - KI * S0) * dp_k
    )


def test_single_date_matches_cos_decomposition():
    """REF parameters, T = 1, B = 105%, KI = 70%, c = 10%: MC on 64 steps (400k paths)
    against the COS static-replication price. As in test_mc, the bias allowance is the
    larger of the Richardson estimate |D| + 4 SE(D), D = MC(32) - MC(64) (valid for weak
    order >= 1), and an order-1/2 extrapolation from n = 8,
    sqrt(8/64) (|MC(8) - truth| + 4 SE(MC(8))). Tolerance 4 SE(MC(64)) + allowance.
    Power: the tolerance must be below half the price change from flipping the sign of rho
    (computed from COS), so a sign error cannot pass.
    For this payoff the Euler bias is below the noise even at n = 8, so the allowance is in
    effect statistical; bias shrinkage is shown by test_multi_date_heston_bias_shrinks."""
    T, B, KI, c = 1.0, 1.05, 0.7, 0.10
    truth = _single_date_truth(REF, T, B, KI, c)
    flipped = _single_date_truth(dict(REF, rho=-REF["rho"]), T, B, KI, c)
    pr = Autocallable(obs_times=(T,), autocall_barrier=B, coupon=c, ki_barrier=KI)
    p = HestonParams(**REF)
    pv = {}
    for n in (8, 32, 64):
        P = simulate(p, T, n, 400_000, seed=41, obs_times=[T], rng_steps=64)
        pv[n] = autocallable_pv(pr, P.S, p.S0, p.r)[0]
    fine = mc_estimate(pv[64], True)
    D = mc_estimate(pv[32] - pv[64], True)
    coarse = mc_estimate(pv[8], True)
    allowance = max(abs(D.price) + 4 * D.se,
                    np.sqrt(8 / 64) * (abs(coarse.price - truth) + 4 * coarse.se))
    tol = 4 * fine.se + allowance
    assert abs(fine.price - truth) < tol, (fine.price, truth, tol)
    assert tol < 0.5 * abs(flipped - truth), (tol, flipped - truth)


# --------------------------------------------------------------------------- BS limit
@pytest.mark.parametrize("q", [0.0, 0.02])
def test_bs_limit_matches_mvn_oracle(q):
    """xi = 1e-8, v0 = theta: v stays at theta to O(xi) and log-Euler is exact for GBM at
    the observation dates, so the MC is unbiased for the discrete-monitoring BS price.
    Four annual dates, 95% barrier, KI 65%, 7% coupon. Tolerance 4 SE (the oracle's
    own error is about 1e-6 for four dates, from Genz integration, against 4 SE ~ 0.06). Call probabilities are checked too,
    each within 4 of its own SE."""
    d = dict(REF, xi=1e-8, q=q)
    p = HestonParams(**d)
    pr = Autocallable(obs_times=(1.0, 2.0, 3.0, 4.0), autocall_barrier=0.95, coupon=0.07,
                      ki_barrier=0.65)
    r = price_autocallable(pr, p, 4, 400_000, seed=51)
    sig = np.sqrt(d["v0"])
    truth = R.bs_autocallable(100.0, d["r"], q, sig, pr.obs_times, 0.95, 0.65, 0.07, 100.0)
    probs = R.bs_autocall_probabilities(100.0, d["r"], q, sig, pr.obs_times, 0.95)
    assert abs(r.price - truth) < 4 * r.se, (r.price, truth, r.se)
    assert np.all(np.abs(r.call_prob - probs) < 4 * r.call_prob_se), (r.call_prob, probs)


# --------------------------------------------------------------------------- Heston bias
def test_multi_date_heston_bias_shrinks():
    """Three annual dates under the stress set, coupled levels n = 6, 12, 24, 48 on one
    Brownian path (rng_steps = 96). There is no closed form, so this checks that the
    discretization changes shrink: D(n) = MC(n) - MC(2n) is resolved (> 4 SE) for
    n = 6, 12, and |D(48)| is smaller than |D(6)| by more than 4 combined SE. The accuracy
    of the scheme itself is established by the one-date and BS-limit tests."""
    p = HestonParams(**STRESS)
    pv = {}
    for n in (6, 12, 24, 48, 96):
        P = simulate(p, 3.0, n, 200_000, seed=61, obs_times=PRODUCT.obs_times, rng_steps=96)
        pv[n] = autocallable_pv(PRODUCT, P.S, p.S0, p.r)[0]
    D = {n: mc_estimate(pv[n] - pv[2 * n], True) for n in (6, 12, 24, 48)}
    for n in (6, 12):
        assert abs(D[n].price) > 4 * D[n].se, (n, D[n])
    assert abs(D[6].price) - abs(D[48].price) > 4 * (D[6].se + D[48].se)
