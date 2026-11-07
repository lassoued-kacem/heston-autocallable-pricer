"""Greeks of the reference autocallable under the reference Heston parameters.

Bump-and-revalue with common random numbers (one seed for every scenario); each Greek's SE
comes from its paired per-path differences. The README quotes this script's output.

    .venv/bin/python scripts/greeks.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hestonlib import REFERENCE  # noqa: E402
from hestonlib.autocall import Autocallable  # noqa: E402
from hestonlib.greeks import autocallable_greeks  # noqa: E402

PRODUCT = Autocallable(obs_times=(1.0, 2.0, 3.0), autocall_barrier=1.0, coupon=0.08,
                       ki_barrier=0.6, notional=100.0)
N_STEPS, N_PATHS, SEED = 48, 1_000_000, 2026

UNITS = {
    "price": "per 100 notional",
    "delta": "per unit of spot (fixing held at 100)",
    "gamma": "per unit of spot, squared (2% bump)",
    "vega": "per unit of vol (parallel shift of sqrt(v0), sqrt(theta); 1-point bump)",
    "rho": "per unit of rate (1e-3 bump)",
}


def main():
    print(f"Product: {PRODUCT}")
    print(f"Model:   {REFERENCE}")
    print(f"{N_STEPS} steps, {N_PATHS:,} antithetic paths, seed {SEED}\n")
    g = autocallable_greeks(PRODUCT, REFERENCE, N_STEPS, N_PATHS, seed=SEED)
    for k, r in g.items():
        print(f"{k:>6} {r.price:>11.4f} +- {r.se:.4f}   {UNITS[k]}")


if __name__ == "__main__":
    main()
