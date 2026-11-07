"""Calibrate Heston to a synthetic implied-vol surface and report identifiability.

The surface comes from the reference parameters (priced by the independent COS oracle in
tests/reference.py), with iid N(0, 0.002^2) noise added to the implied vols (seed 2026).
The README quotes this script's output.

    .venv/bin/python scripts/calibrate.py
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]

import reference as R  # noqa: E402
from hestonlib import REFERENCE  # noqa: E402
from hestonlib.calibration import NAMES, Surface, calibrate  # noqa: E402

MATS = np.array([0.25, 0.5, 1.0, 2.0])
STRIKES = np.array([80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0])
NOISE, SEED = 0.002, 2026


def main():
    p = REFERENCE
    args = (p.r, p.q, p.v0, p.kappa, p.theta, p.xi, p.rho)
    prices = np.array([R.cos_call(p.S0, STRIKES, T, *args) for T in MATS])
    clean = Surface(p.S0, p.r, p.q, MATS, STRIKES, prices)
    iv = clean.ivs() + NOISE * np.random.default_rng(SEED).standard_normal(prices.shape)
    noisy = np.array([R.bs_price(p.S0, STRIKES, T, p.r, p.q, v) for T, v in zip(MATS, iv)])
    surf = Surface(p.S0, p.r, p.q, MATS, STRIKES, noisy)

    res = calibrate(surf)
    truth = np.array([getattr(p, n) for n in NAMES])
    print(f"Surface: {len(MATS)} maturities x {len(STRIKES)} strikes, IV noise sd {NOISE}, seed {SEED}")
    print(f"{'param':>6} {'truth':>8} {'fit':>9} {'SE':>8} {'z':>6}")
    for n, t, x, s in zip(NAMES, truth, res.x, res.se):
        print(f"{n:>6} {t:>8.4f} {x:>9.4f} {s:>8.4f} {(x - t) / s:>6.2f}")
    print(f"IV RMSE {res.iv_rmse:.5f} (noise sd {NOISE})")
    print(f"Feller ratio 2 kappa theta / xi^2 = {res.feller_ratio:.3f} "
          f"({'satisfied' if res.feller_ratio >= 1 else 'violated'}; reported, not imposed)")
    print(f"Scaled Jacobian condition number {res.cond:.1f}; parameters at a bound: {res.at_bound or 'none'}")
    print("Parameter correlation matrix (" + ", ".join(NAMES) + "):")
    print(np.array2string(res.corr, precision=2, suppress_small=True))
    print(f"Function evaluations {res.nfev}; {res.message}")


if __name__ == "__main__":
    main()
