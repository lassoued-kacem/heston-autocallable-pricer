"""Autocallable Greeks by bump-and-revalue with common random numbers.

Every bumped revaluation reuses the same seed, so the same normals drive every scenario
(CRN). A Greek is estimated per path, for example for delta
g_i = (PV_i(S0 + h) - PV_i(S0 - h)) / (2 h), and its SE comes from these paired per-path
differences (on antithetic pair means). Differencing two independent price estimates would
give an SE about sqrt(2) * SE(price) / (2 h) instead.

The barriers and the KI redemption are fixed relative to the note's initial fixing
``ref_spot`` (default S0), which does not move when spot is bumped. Bumping S0 with the
fixing would leave the price unchanged, since the payoff is scale invariant.

Greeks and bumps (central differences, O(h^2) truncation error):
* delta = dV/dS0 and gamma = d2V/dS0^2, bump h_spot * ref_spot;
* vega = dV/dsigma for a parallel shift of sqrt(v0) and sqrt(theta) by h_vol (in the BS
  limit, v0 = theta = sigma^2, this is the BS vega);
* rho = dV/dr, bump h_rate.

Bump sizes and discontinuities: the payoff jumps at every autocall date, where the call
amount N(1 + c i) exceeds the value of continuing (at maturity the jump is N(1 + c n) - N),
and at the KI level at maturity, where it drops from N to N KI (a jump of N(1 - KI), the
largest one for the reference note). For a jump of size J, the per-path delta difference is
J/(2h) on a set of probability O(h), so Var(delta estimate) = O(1/h) and
Var(gamma) = O(1/h^3). The default h_spot = 2% keeps the truncation bias (O(h^2) times the
third derivative) below a quarter of the SE at 400k paths in the BS-limit test. Smaller
bumps quickly blow up the gamma SE.
"""

from __future__ import annotations

import numpy as np

from hestonlib.autocall import Autocallable, autocallable_pv
from hestonlib.mc import MCResult, mc_estimate, simulate
from hestonlib.params import HestonParams


def _pv(product, p, ref_spot, n_steps, n_paths, seed, rng_steps):
    P = simulate(p, product.maturity, n_steps, n_paths, seed, obs_times=product.obs_times,
                 antithetic=True, rng_steps=rng_steps)
    return autocallable_pv(product, P.S, ref_spot, p.r)[0]


def _vol_bump(p: HestonParams, h: float) -> HestonParams:
    return p.replace(v0=(np.sqrt(p.v0) + h) ** 2, theta=(np.sqrt(p.theta) + h) ** 2)


def autocallable_greeks(
    product: Autocallable,
    p: HestonParams,
    n_steps: int,
    n_paths: int = 200_000,
    seed: int = 0,
    ref_spot: float | None = None,
    h_spot: float = 0.02,
    h_vol: float = 0.01,
    h_rate: float = 1e-3,
    rng_steps: int | None = None,
) -> dict[str, MCResult]:
    """Price, delta, gamma, vega and rho, each an MCResult with its own paired SE."""
    ref = p.S0 if ref_spot is None else float(ref_spot)
    hs = h_spot * ref
    if p.S0 - hs <= 0 or np.sqrt(p.v0) - h_vol < 0 or np.sqrt(p.theta) - h_vol < 0:
        raise ValueError("bumps too large for the current state")
    run = lambda scen: _pv(product, scen, ref, n_steps, n_paths, seed, rng_steps)
    base = run(p)
    up, dn = run(p.replace(S0=p.S0 + hs)), run(p.replace(S0=p.S0 - hs))
    vu, vd = run(_vol_bump(p, h_vol)), run(_vol_bump(p, -h_vol))
    ru, rd = run(p.replace(r=p.r + h_rate)), run(p.replace(r=p.r - h_rate))
    est = lambda x: mc_estimate(x, True)
    return {
        "price": est(base),
        "delta": est((up - dn) / (2 * hs)),
        "gamma": est((up - 2 * base + dn) / hs**2),
        "vega": est((vu - vd) / (2 * h_vol)),
        "rho": est((ru - rd) / (2 * h_rate)),
    }
