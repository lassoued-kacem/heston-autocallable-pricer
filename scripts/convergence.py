"""Convergence study: Euler bias vs time step and MC standard error vs path count.

1. Bias. ATM European call, T = 1, against the COS oracle, on coupled levels (one Brownian
   path on a 256-step grid), for the stress set (Feller ratio 0.09) and the reference set.
   Prints each level's error with its SE and the paired change D(n) = MC(n) - MC(2n).
2. Standard error. ATM call under REF (64 steps): the SE at 4x more paths should halve
   (SE ~ 1/sqrt(N)); antithetic vs plain sampling at equal path counts.

Writes docs/figures/bias_vs_steps.png and docs/figures/se_vs_paths.png. The README quotes
this script's output.

    .venv/bin/python scripts/convergence.py
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import reference as R  # noqa: E402
from hestonlib import REFERENCE, HestonParams  # noqa: E402
from hestonlib.mc import mc_estimate, simulate  # noqa: E402

STRESS = HestonParams(S0=100.0, r=0.02, q=0.01, v0=0.09, kappa=0.5, theta=0.09, xi=1.0, rho=-0.95)
LEVELS = (4, 8, 16, 32, 64, 128, 256)
N_BIAS, SEED, K, T = 400_000, 2026, 100.0, 1.0
PATHS = (12_500, 50_000, 200_000, 800_000)

# Chart tokens (reference palette, light mode): surface, ink, and categorical slots 1-2.
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def _args(p):
    return p.r, p.q, p.v0, p.kappa, p.theta, p.xi, p.rho


def bias_table(p, name):
    cos = R.cos_call(p.S0, K, T, *_args(p))[0]
    pay = {}
    for n in LEVELS:
        P = simulate(p, T, n, N_BIAS, SEED, rng_steps=LEVELS[-1])
        pay[n] = np.exp(-p.r * T) * np.maximum(P.S[:, -1] - K, 0.0)
    print(f"\n[{name}] ATM call T = 1, COS = {cos:.4f}, {N_BIAS:,} paths, seed {SEED}, "
          f"Feller ratio {p.feller_ratio:.2f}")
    print(f"{'steps':>6} {'MC - COS':>10} {'SE':>8} {'D(n)=MC(n)-MC(2n)':>19} {'SE(D)':>8}")
    rows = []
    for n in LEVELS:
        e = mc_estimate(pay[n], True)
        if 2 * n in pay:
            D = mc_estimate(pay[n] - pay[2 * n], True)
            d_txt = f"{D.price:>+19.4f} {D.se:>8.4f}"
        else:
            d_txt = f"{'':>19} {'':>8}"
        print(f"{n:>6} {e.price - cos:>+10.4f} {e.se:>8.4f} {d_txt}")
        rows.append((n, e.price - cos, e.se))
    return np.array(rows)


def se_table():
    p = REFERENCE
    print(f"\n[SE] ATM call under REF, T = 1, 64 steps, seed {SEED}")
    print(f"{'paths':>9} {'SE antithetic':>14} {'SE plain':>10} {'variance ratio plain/anti':>26}")
    rows = []
    for n in PATHS:
        se = {}
        for anti in (True, False):
            P = simulate(p, T, 64, n, SEED, antithetic=anti)
            se[anti] = mc_estimate(np.exp(-p.r * T) * np.maximum(P.S[:, -1] - K, 0.0), anti).se
        print(f"{n:>9,} {se[True]:>14.5f} {se[False]:>10.5f} {(se[False] / se[True]) ** 2:>26.2f}")
        rows.append((n, se[True], se[False]))
    rows = np.array(rows)
    slope = np.polyfit(np.log(rows[:, 0]), np.log(rows[:, 1]), 1)[0]
    print(f"log-log slope of SE (antithetic) vs paths: {slope:.3f} (theory -0.5)")
    return rows


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _ticks(ax, values, fmt):
    """Label the x-axis at the sampled values only (no crowded minor labels)."""
    from matplotlib.ticker import FixedLocator, NullLocator

    ax.xaxis.set_major_locator(FixedLocator(values))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([fmt(v) for v in values])


def plot_bias(stress_rows):
    fig, ax = plt.subplots(figsize=(6.4, 4.2), facecolor=SURFACE)
    _style(ax)
    n, err, se = stress_rows.T
    ax.errorbar(n, err, yerr=2 * se, color=BLUE, lw=2, marker="o", ms=6, capsize=3,
                label="MC - COS (stress set), +-2 SE")
    guide = err[1] * n[1] / n
    ax.plot(n, guide, color=INK_2, lw=1.2, ls="--", label="slope -1 (weak order 1)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    _ticks(ax, n, lambda v: f"{int(v)}")
    ax.set_xlabel("time steps per year", color=INK)
    ax.set_ylabel("price error vs COS", color=INK)
    ax.set_title("Euler bias, ATM call, Feller ratio 0.09", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(ROOT / "docs/figures/bias_vs_steps.png", dpi=150, facecolor=SURFACE)


def plot_se(se_rows):
    fig, ax = plt.subplots(figsize=(6.4, 4.2), facecolor=SURFACE)
    _style(ax)
    n, anti, plain = se_rows.T
    ax.plot(n, plain, color=ORANGE, lw=2, marker="o", ms=6, label="plain")
    ax.plot(n, anti, color=BLUE, lw=2, marker="o", ms=6, label="antithetic (SE on pair means)")
    ax.plot(n, anti[0] * np.sqrt(n[0] / n), color=INK_2, lw=1.2, ls="--", label="slope -1/2")
    ax.set_xscale("log")
    ax.set_yscale("log")
    _ticks(ax, n, lambda v: f"{int(v):,}")
    ax.set_xlabel("paths", color=INK)
    ax.set_ylabel("standard error", color=INK)
    ax.set_title("MC standard error, ATM call, reference set", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(ROOT / "docs/figures/se_vs_paths.png", dpi=150, facecolor=SURFACE)


def main():
    stress = bias_table(STRESS, "stress")
    bias_table(REFERENCE, "reference")
    se = se_table()
    (ROOT / "docs/figures").mkdir(parents=True, exist_ok=True)
    plot_bias(stress)
    plot_se(se)
    print("\nwrote docs/figures/bias_vs_steps.png, docs/figures/se_vs_paths.png")


if __name__ == "__main__":
    main()
