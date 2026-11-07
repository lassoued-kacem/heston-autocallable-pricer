"""Price the reference autocallable under the reference Heston parameters.

Prints the MC price with its SE at several step counts (coupled levels on one Brownian
path), plus call probabilities, P(loss) and expected life at the finest level. The README
quotes this script's output.

    .venv/bin/python scripts/price_autocallable.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hestonlib import REFERENCE
from hestonlib.autocall import Autocallable, autocallable_pv, price_autocallable
from hestonlib.mc import mc_estimate, simulate

PRODUCT = Autocallable(obs_times=(1.0, 2.0, 3.0), autocall_barrier=1.0, coupon=0.08,
                       ki_barrier=0.6, notional=100.0)
N_PATHS, SEED, FINE = 1_000_000, 2026, 192


def main():
    print(f"Product: {PRODUCT}")
    print(f"Model:   {REFERENCE}")
    print(f"Paths: {N_PATHS:,} (antithetic), seed {SEED}, Brownian grid {FINE} steps\n")
    print(f"{'steps':>6} {'price':>10} {'SE':>8}")
    pv = {}
    for n in (12, 48, 192):
        P = simulate(REFERENCE, PRODUCT.maturity, n, N_PATHS, SEED,
                     obs_times=PRODUCT.obs_times, rng_steps=FINE)
        pv[n] = autocallable_pv(PRODUCT, P.S, REFERENCE.S0, REFERENCE.r)[0]
        e = mc_estimate(pv[n], True)
        print(f"{n:>6} {e.price:>10.4f} {e.se:>8.4f}")
    for a, b in ((12, 48), (48, 192)):
        d = mc_estimate(pv[a] - pv[b], True)
        print(f"paired difference MC({a}) - MC({b}) = {d.price:+.4f} +- {d.se:.4f}")
    r = price_autocallable(PRODUCT, REFERENCE, FINE, N_PATHS, seed=SEED, rng_steps=FINE)
    print()
    for i, (t, pc, se) in enumerate(zip(PRODUCT.obs_times, r.call_prob, r.call_prob_se), 1):
        print(f"P(called at t{i} = {t:g}y) = {pc:.4f} +- {se:.4f}")
    print(f"P(never called)             = {r.no_call_prob.price:.4f} +- {r.no_call_prob.se:.4f}")
    print(f"P(loss: S_T < KI)           = {r.loss_prob.price:.4f} +- {r.loss_prob.se:.4f}")
    print(f"Expected life (years)       = {r.expected_life.price:.4f} +- {r.expected_life.se:.4f}")


if __name__ == "__main__":
    np.set_printoptions(precision=4)
    main()
