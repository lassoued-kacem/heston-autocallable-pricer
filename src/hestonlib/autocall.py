"""Autocallable note: payoff and Monte Carlo pricing under Heston.

Product (notional N, observation times t_1 < ... < t_n = T, levels relative to S0):

* At each t_i, if S_{t_i} >= B * S0 the note is called: it pays N (1 + c i) at t_i and
  nothing afterwards. The maturity date t_n is an autocall date.
* If it is never called, at T it pays N if S_T >= KI * S0 (European knock-in, observed at
  maturity only), and N * S_T / S0 otherwise.

Every cashflow is discounted at exp(-r t) for its own payment date. The dividend yield q
enters only through the simulated drift.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hestonlib.mc import MCResult, mc_estimate, simulate
from hestonlib.params import HestonParams


@dataclass(frozen=True)
class Autocallable:
    obs_times: tuple[float, ...] = (1.0, 2.0, 3.0)
    autocall_barrier: float = 1.0
    coupon: float = 0.08
    ki_barrier: float = 0.6
    notional: float = 100.0

    def __post_init__(self):
        t = np.asarray(self.obs_times, dtype=float)
        if t.ndim != 1 or len(t) < 1 or np.any(t <= 0) or np.any(np.diff(t) <= 0):
            raise ValueError("obs_times must be positive and strictly increasing")
        if self.autocall_barrier < 0 or self.ki_barrier < 0 or self.notional <= 0:
            raise ValueError("barriers must be >= 0 and the notional > 0")
        object.__setattr__(self, "obs_times", tuple(float(x) for x in t))

    @property
    def maturity(self) -> float:
        return self.obs_times[-1]


def autocallable_pv(product: Autocallable, S_obs, S0: float, r: float):
    """Discounted payoff per path and the 1-based call index (0 = never called).

    ``S_obs`` has shape (n_paths, n_obs) with columns at ``product.obs_times``.
    """
    S_obs = np.asarray(S_obs, dtype=float)
    t = np.asarray(product.obs_times)
    n_obs = len(t)
    if S_obs.ndim != 2 or S_obs.shape[1] != n_obs:
        raise ValueError("S_obs must have one column per observation time")
    N = product.notional
    hit = S_obs >= product.autocall_barrier * S0
    called = hit.any(axis=1)
    first = np.where(called, hit.argmax(axis=1), n_obs - 1)  # 0-based first call index
    call_idx = np.where(called, first + 1, 0)

    disc = np.exp(-r * t)
    pv_called = N * (1.0 + product.coupon * (first + 1)) * disc[first]
    ST = S_obs[:, -1]
    redemption = np.where(ST >= product.ki_barrier * S0, N, N * ST / S0)
    pv = np.where(called, pv_called, redemption * disc[-1])
    return pv, call_idx


@dataclass(frozen=True)
class AutocallResult:
    """Price with SE, plus per-date call probabilities (each with its SE)."""

    price: float
    se: float
    n_samples: int
    call_prob: np.ndarray  # P(called at t_i), i = 1..n
    call_prob_se: np.ndarray
    no_call_prob: MCResult  # P(never called)
    loss_prob: MCResult  # P(never called and S_T < KI S0)
    expected_life: MCResult  # E[payment time]


def price_autocallable(
    product: Autocallable,
    p: HestonParams,
    n_steps: int,
    n_paths: int = 200_000,
    seed: int = 0,
    antithetic: bool = True,
    rng_steps: int | None = None,
) -> AutocallResult:
    """MC price of ``product`` under ``p`` with full-truncation Euler on ``n_steps`` steps.

    The observation dates must lie on the uniform grid of ``n_steps`` steps over [0, T].
    """
    paths = simulate(
        p, product.maturity, n_steps, n_paths, seed,
        obs_times=product.obs_times, antithetic=antithetic, rng_steps=rng_steps,
    )
    pv, idx = autocallable_pv(product, paths.S, p.S0, p.r)
    res = paths.estimate(pv)
    n_obs = len(product.obs_times)
    ind = (idx[:, None] == np.arange(1, n_obs + 1)[None, :]).astype(float)
    probs = paths.estimate(ind)
    t = np.asarray(product.obs_times)
    life = np.where(idx > 0, t[np.maximum(idx, 1) - 1], t[-1])
    ST = paths.S[:, -1]
    loss = (idx == 0) & (ST < product.ki_barrier * p.S0)
    return AutocallResult(
        price=res.price, se=res.se, n_samples=res.n_samples,
        call_prob=probs.price, call_prob_se=probs.se,
        no_call_prob=paths.estimate((idx == 0).astype(float)),
        loss_prob=paths.estimate(loss.astype(float)),
        expected_life=paths.estimate(life),
    )
