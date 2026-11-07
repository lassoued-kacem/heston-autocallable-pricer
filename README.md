# heston-autocallable

Pricing and risk for an autocallable structured product under the Heston stochastic-volatility
model, in pure NumPy/SciPy (vectorized; no C++, no Numba).

> Status: stage 1 of 7 (semi-analytic European pricer). Later sections are added as the
> stages land.

## Model

    dS = (r - q) S dt + sqrt(v) S dW1
    dv = kappa (theta - v) dt + xi sqrt(v) dW2,      corr(dW1, dW2) = rho

Reference parameters: S0 = 100, r = 0.03, q = 0, v0 = 0.04, kappa = 2, theta = 0.04,
xi = 0.3, rho = -0.7 (`hestonlib.REFERENCE`).

## Semi-analytic European prices (`hestonlib.analytic`)

* **Characteristic function** of ln(S_T/F_T) in the Albrecher et al. (2007) "little Heston
  trap" form (root choice with `exp(-dT)`), which keeps the complex logarithm on a single
  branch even for long maturities, high vol-of-vol and rho near -1.
* **Lewis (2001) single integral** along Im(z) = -1/2, evaluated with `scipy.integrate.quad`
  (scalar strike) or `quad_vec` (vector of strikes, one adaptive pass).
* Puts come from put-call parity.

```python
from hestonlib import REFERENCE
from hestonlib.analytic import heston_call, heston_put
heston_call([90, 100, 110], 1.0, REFERENCE)
```

## How correctness is established

The tests never use the package's own output as an expected value. The ground truth is:

1. **An independent oracle** (`tests/reference.py`, which does not import `hestonlib`):
   Fang–Oosterlee COS method for puts and digitals, and its own Black-Scholes formula.
   The oracle's closed-form CF is itself checked against a numerical integration of the
   Heston Riccati ODEs, and its truncation range is widened until prices stop changing.
   This was needed: a fixed cumulant-based range under-truncates when Feller is badly violated.
2. **The Black-Scholes limit**: with v0 = theta and xi -> 0, the gap to BS(sqrt(v0)) shrinks
   linearly in xi, and quadratically when rho = 0, as the vol-of-vol expansion predicts.
3. **Model-free identities**: put-call parity (package call vs oracle native put),
   no-arbitrage bounds, and monotonicity and convexity in strike.

Run the suite with `.venv/bin/python -m pytest -q`.

## Development

See `CLAUDE.md` for the project rules. A Claude Code Stop hook (`scripts/check.sh`) runs the
full suite whenever an agent turn ends, and blocks completion on failure.
