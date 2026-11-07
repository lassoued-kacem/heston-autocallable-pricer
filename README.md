# heston-autocallable

Pricing and risk for an autocallable structured product under the Heston stochastic-volatility
model, in pure NumPy/SciPy (vectorized; no C++, no Numba).

> Status: stage 5 of 7 (semi-analytic European pricer, Monte Carlo, autocallable,
> implied vols and calibration, Greeks). Later sections are added as the
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

## Monte Carlo (`hestonlib.mc`)

* **Full-truncation Euler** on ln S (Lord, Koekkoek and van Dijk, 2010): v+ = max(v, 0) enters
  the drift and diffusion of both processes, and the raw v is carried forward. On the log
  scale, the discounted forward is an exact martingale of the discrete scheme.
* **Antithetic variates** flip both independent drivers together. The standard error is
  computed from the n/2 pair means, which are independent.
* **Coupled levels and common random numbers**: normals are drawn on a fine grid
  (`rng_steps`) and summed per step, so runs with different step counts share one Brownian
  path, and runs with different parameters share the same normals.
* Every estimate is an `MCResult(price, se, n_samples)`.

```python
from hestonlib import REFERENCE
from hestonlib.mc import mc_european
mc_european([90, 100, 110], 1.0, REFERENCE, n_steps=64, n_paths=200_000, seed=1)
```

The Euler bias is tested in two ways. On coupled levels with Feller strongly violated, the
price falls by a statistically significant amount at every halving of dt. At the reference
parameters, the 64-step price matches the COS oracle within 4 SE plus a bias allowance
|MC(32) - MC(64)| + 4 SE (the weak-order-1 estimate of the remaining bias).

## Autocallable (`hestonlib.autocall`)

Notional N, observation dates t_1 < ... < t_n = T, and levels relative to S0:

* **Autocall:** at each t_i, if S_{t_i} >= B S0, the note pays N (1 + c i) at t_i and
  ends. The maturity date is an autocall date.
* **Redemption if never called (European knock-in, observed at T only):** N if
  S_T >= KI S0, and N S_T / S0 otherwise.
* **Discounting:** every cashflow is discounted at exp(-r t) for its own date.

`price_autocallable` returns the price and SE. It also reports the per-date call
probabilities, P(never called), P(loss) and the expected life, each with its SE.

Reference note: annual dates over 3 years, B = 100%, KI = 60%, c = 8%, under the reference
Heston parameters (`scripts/price_autocallable.py`, 1,000,000 antithetic paths, seed 2026):

| steps | price | SE |
|---|---|---|
| 12 | 99.4262 | 0.0151 |
| 48 | 99.5063 | 0.0150 |
| 192 | 99.5144 | 0.0150 |

On one shared Brownian path:
* MC(12) - MC(48) = -0.0801 +- 0.0088.
* MC(48) - MC(192) = -0.0081 +- 0.0059.

At 192 steps:
* P(called at year 1, 2, 3) = 0.5799, 0.1200, 0.0561.
* P(never called) = 0.2439.
* P(loss) = 0.0629.
* Expected life = 1.7201 +- 0.0006 years.

The probability SEs are 0.0002 to 0.0004.

How the pricer is validated:
* **One date:** the note is a bond plus a digital minus a put, priced by the COS oracle.
* **Several dates:** in the BS limit, against an exact discrete-monitoring price built from
  multivariate normal CDFs (Girsanov for the knock-in leg). That oracle is itself checked
  against quadrature and the trivariate orthant formula.
* **Exact payoff limits:** barrier 0, infinite barriers, KI 0 and KI infinite.

## Implied volatility and calibration

**Pricer error estimate.** `heston_call(..., return_error=True)` returns the Lewis
quadrature's error estimate (QUADPACK's estimate, not a rigorous bound) in price units,
plus a rounding term. `tests/test_implied_vol.py` checks that it covers the true error
against a COS oracle converged to < 1e-12.

**Guarded implied volatility** (`hestonlib.implied_vol`):
* `brentq` runs inside checked no-arbitrage bounds.
* A price whose time value is within its error bound gives NaN rather than a clipped
  volatility. This covers deep-OTM short-maturity Heston prices, which can be slightly
  negative.
* A price that violates the bounds by more than its error raises.

**Calibration** (`hestonlib.calibration`):
* Bounded least squares on vega-weighted price errors, which to first order are IV errors.
* Reported with each fit: parameter SEs from s^2 (J^T J)^-1, the correlation matrix, the
  scaled condition number, any parameter at a bound, and the Feller ratio (reported, not
  imposed).

Example (`scripts/calibrate.py`): the reference parameters priced by the COS oracle on
4 maturities x 7 strikes, with N(0, 0.002^2) noise added to the IVs (seed 2026).

| param | truth | fit | SE |
|---|---|---|---|
| v0 | 0.0400 | 0.0398 | 0.0004 |
| kappa | 2.0000 | 1.9112 | 0.1951 |
| theta | 0.0400 | 0.0403 | 0.0005 |
| xi | 0.3000 | 0.2871 | 0.0165 |
| rho | -0.7000 | -0.7160 | 0.0269 |

* Every parameter is within 0.8 SE of the truth.
* IV RMSE is 0.00153 and the Feller ratio is 1.868.
* The scaled condition number is 24.4.
* kappa is the least identified parameter (relative SE about 10%). Its largest
  correlations are with xi (0.59), and xi in turn correlates with rho (0.78).

Without noise, the same fit recovers both the reference set and a Feller-violating stress
set to within a bound derived from sigma_min of the Jacobian.

## Greeks (`hestonlib.greeks`)

Bump-and-revalue with **common random numbers**: every scenario uses the same seed. Each
Greek's SE comes from its own per-path differences, taken on antithetic pair means. The
note's fixing (`ref_spot`) stays put while spot is bumped. The payoff is scale invariant,
so bumping both would give zero delta.

| Greek | bump |
|---|---|
| delta, gamma | 2% of the fixing |
| vega | parallel 1-vol-point shift of sqrt(v0) and sqrt(theta) |
| rho | 1e-3 in r |

The bumps are sized against the barrier discontinuities. An exact BS oracle puts the
finite-difference truncation bias below a quarter of the Greek's SE at 400k paths. Smaller
spot bumps inflate the gamma variance, which grows like 1/h^3 for a digital-type jump.

Reference note (`scripts/greeks.py`, 48 steps, 1,000,000 antithetic paths, seed 2026):

| Greek | value | SE |
|---|---|---|
| price | 99.5159 | 0.0150 |
| delta | 0.2757 | 0.0016 |
| gamma | -0.0160 | 0.0016 |
| vega (per unit vol) | -49.9824 | 0.2765 |
| rho (per unit rate) | -94.5984 | 1.0356 |

This run's 48-step price (99.5159) is not identical to the pricing script's 48-step price
(99.5063). The pricing script draws its normals on a 192-step grid, so the two runs share
only part of their random stream. Each SE is 0.0150.

How the Greeks are tested:
* In the BS limit, against the same finite difference of the MVN oracle.
* For one Heston date, against the same finite difference of the COS replication, with an
  Euler bias allowance and a power cap.
* Over 200 seeds, the reported SEs match the observed spread.

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
