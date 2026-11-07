"""Monte Carlo simulation of the Heston model: full-truncation Euler on the log-asset.

Scheme (Lord, Koekkoek and van Dijk, 2010, "full truncation"), with v+ = max(v, 0):

    ln S_{k+1} = ln S_k + (r - q - v+_k / 2) dt + sqrt(v+_k dt) Z_S
    v_{k+1}    = v_k + kappa (theta - v+_k) dt + xi sqrt(v+_k dt) Z_v
    Z_S        = rho Z_v + sqrt(1 - rho^2) Z_perp,   Z_v, Z_perp iid N(0, 1)

The raw v (which may be negative) is carried forward; only v+ enters drift and diffusion.
Because v+_k is known at the start of step k and Z_S is independent of it,
E[exp(sqrt(v+ dt) Z_S - v+ dt / 2) | F_k] = 1, so e^{-(r-q) t_k} S_k is an exact
martingale of the discrete scheme: the forward carries no discretization bias.

Random numbers
--------------
Normals are drawn on a grid of ``rng_steps`` steps (default ``n_steps``), one block of shape
(f, n_half, 2) per simulation step, where f = rng_steps / n_steps. A simulation step uses the
sum of its f fine normals divided by sqrt(f), which is exactly the Brownian increment over the
coarse step. With the same ``seed``, ``n_paths`` and ``rng_steps``, runs with different
``n_steps`` therefore see the same Brownian path (coupled levels), and runs with different
model parameters see the same normals (common random numbers).

Antithetic variates negate both independent drivers (Z_v, Z_perp) together. Path i and path
i + n_paths/2 form a pair, and the standard error is computed from the n_paths/2 pair means,
which are independent; the 2n halves are not.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hestonlib.params import HestonParams


@dataclass(frozen=True)
class MCResult:
    """An MC estimate with its standard error. ``n_samples`` counts independent samples
    (pairs when antithetic)."""

    price: np.ndarray | float
    se: np.ndarray | float
    n_samples: int


def mc_estimate(x, antithetic: bool) -> MCResult:
    """Mean and SE of samples ``x`` (shape (n_paths, ...)).

    With ``antithetic``, rows i and i + n/2 are a pair; the SE uses the pair means.
    """
    x = np.asarray(x, dtype=float)
    if antithetic:
        n = x.shape[0]
        if n % 2:
            raise ValueError("antithetic samples need an even number of paths")
        x = 0.5 * (x[: n // 2] + x[n // 2 :])
    m = x.shape[0]
    if m < 2:
        raise ValueError("need at least two independent samples for an SE")
    mean = x.mean(axis=0)
    se = x.std(axis=0, ddof=1) / np.sqrt(m)
    if mean.ndim == 0:
        return MCResult(float(mean), float(se), m)
    return MCResult(mean, se, m)


@dataclass(frozen=True)
class HestonPaths:
    """Simulated values at the observation times. Rows are paths, columns are times."""

    times: np.ndarray
    S: np.ndarray
    v: np.ndarray
    antithetic: bool

    def estimate(self, x) -> MCResult:
        return mc_estimate(x, self.antithetic)


def euler_step(lnS, v, dt: float, z_v, z_perp, p: HestonParams):
    """One full-truncation Euler step. Returns (lnS_next, v_next)."""
    vp = np.maximum(v, 0.0)
    sd = np.sqrt(vp * dt)
    z_s = p.rho * z_v + np.sqrt(1.0 - p.rho * p.rho) * z_perp
    lnS_next = lnS + (p.r - p.q - 0.5 * vp) * dt + sd * z_s
    v_next = v + p.kappa * (p.theta - vp) * dt + p.xi * sd * z_v
    return lnS_next, v_next


def _obs_indices(obs_times, T: float, n_steps: int) -> np.ndarray:
    t = np.atleast_1d(np.asarray(obs_times, dtype=float))
    if np.any(t <= 0) or np.any(t > T * (1 + 1e-12)) or np.any(np.diff(t) <= 0):
        raise ValueError("observation times must be increasing and lie in (0, T]")
    j = np.rint(t * n_steps / T).astype(int)
    if np.any(np.abs(j * T / n_steps - t) > 1e-9 * max(1.0, T)):
        raise ValueError("observation times must lie on the simulation grid")
    return j


def simulate(
    p: HestonParams,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int = 0,
    obs_times=None,
    antithetic: bool = True,
    rng_steps: int | None = None,
) -> HestonPaths:
    """Simulate S and v at ``obs_times`` (default [T]) with full-truncation Euler."""
    if T <= 0 or n_steps < 1 or n_paths < 2:
        raise ValueError("need T > 0, n_steps >= 1, n_paths >= 2")
    if antithetic and n_paths % 2:
        raise ValueError("antithetic simulation needs an even n_paths")
    rng_steps = n_steps if rng_steps is None else int(rng_steps)
    if rng_steps % n_steps:
        raise ValueError("rng_steps must be a multiple of n_steps")
    f = rng_steps // n_steps
    obs_idx = _obs_indices([T] if obs_times is None else obs_times, T, n_steps)
    times = obs_idx * T / n_steps

    dt = T / n_steps
    n_draw = n_paths // 2 if antithetic else n_paths
    rng = np.random.default_rng(seed)
    lnS = np.full(n_paths, np.log(p.S0))
    v = np.full(n_paths, p.v0)
    S_out = np.empty((n_paths, len(obs_idx)))
    v_out = np.empty((n_paths, len(obs_idx)))
    col = 0
    scale = 1.0 / np.sqrt(f)
    for k in range(1, n_steps + 1):
        z = rng.standard_normal((f, n_draw, 2)).sum(axis=0) * scale
        if antithetic:
            z = np.concatenate([z, -z])
        lnS, v = euler_step(lnS, v, dt, z[:, 0], z[:, 1], p)
        while col < len(obs_idx) and obs_idx[col] == k:
            S_out[:, col] = np.exp(lnS)
            v_out[:, col] = v
            col += 1
    return HestonPaths(times=times, S=S_out, v=v_out, antithetic=antithetic)


def mc_european(
    K,
    T: float,
    p: HestonParams,
    kind: str = "call",
    n_steps: int = 64,
    n_paths: int = 100_000,
    seed: int = 0,
    antithetic: bool = True,
    rng_steps: int | None = None,
) -> MCResult:
    """European call/put price(s) by MC, with SE. K may be scalar or 1-D."""
    paths = simulate(p, T, n_steps, n_paths, seed, antithetic=antithetic, rng_steps=rng_steps)
    ST = paths.S[:, -1:]
    K_arr = np.atleast_1d(np.asarray(K, dtype=float))[None, :]
    if kind == "call":
        pay = np.maximum(ST - K_arr, 0.0)
    elif kind == "put":
        pay = np.maximum(K_arr - ST, 0.0)
    else:
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")
    res = paths.estimate(np.exp(-p.r * T) * pay)
    if np.ndim(K) == 0:
        return MCResult(float(res.price[0]), float(res.se[0]), res.n_samples)
    return res
