# Progress

One entry per stage. Numbers come from running the code, never from memory.

## Stage 1: analytic Heston pricer and independent oracle (merged, 8c88605)

### What was built
- `HestonParams` and a Black-Scholes pricer (`src/hestonlib/params.py`,
  `src/hestonlib/black_scholes.py`).
- Heston "little trap" characteristic function and a Lewis-integral call/put pricer
  using `scipy.integrate.quad` (`src/hestonlib/analytic.py`).
- An independent oracle in `tests/reference.py`, which doesn't import `hestonlib`. It
  has a COS-method pricer for puts, calls and digital puts, and a closed-form
  characteristic function checked against a DOP853 solution of the Riccati ODEs.

### Tests
38 passing (`.venv/bin/python -m pytest -q`, 1.2 s).

- The Lewis pricer matches the COS oracle on the reference parameters and on a stress set
  (xi=1, rho=-0.95, Feller ratio 0.09, T up to 15). The package's own CF matches the
  Riccati ODE solution.
- Put-call parity holds with dividends.
- The gap to Black-Scholes (xi -> 0, v0 = theta) shrinks linearly in xi, and
  quadratically when rho = 0.

### Tolerances and why
| Check | Tolerance | Justification |
|---|---|---|
| Lewis vs COS, calls and puts | 1e-8 abs | Both methods converge to about 1e-10 (quad epsabs 1e-12; COS self-converged to 1e-9, per the test docstring). 1e-8 covers both errors, and a branch-cut error would be O(1). |
| Put-call parity | 1e-8 abs | The sum of the two quadrature errors. |
| COS self-convergence | 1e-11 * max(1, K) | Can't be tighter: at xi ~ 1e-3 the CF has a roundoff floor of about 1e-9 from the kappa*theta/xi^2 cancellation. |
| BS limit, rho = -0.7 | ratio > 50 per two decades of xi (theory 100); gap < 1e-5 at xi = 1e-8 | Leaves room for the O(xi^2) term and quadrature noise, but still catches partial cancellation (the pre-fix bug gave 0.39 at xi = 1e-8). |
| BS limit, rho = 0 | ratio > 80 per decade (theory 100) | The first-order term vanishes, so the gap is O(xi^2). |

### Reviewer findings and resolutions
- **The COS truncation range was too narrow in the stress set.** A fixed cumulant range
  underestimates Var(X_T) when xi is large and Feller is violated. On the test strikes,
  the fixed-range error against the adaptive oracle is 1.3e-3 at T=1, 5.0e-3 at T=2, and
  2.1e-2 to 2.6e-2 for T=5 to 15 (re-measured while writing this file). Resolved in
  68f65c8: L and N are doubled until prices change by less than the tolerance, and the
  oracle raises if they never settle.
- **The CF lost precision catastrophically as xi -> 0** (94e61db). (beta - d)/xi^2 and
  ln((1 - g e^{-dT})/(1 - g))/xi^2 cancelled. The price error against BS at the
  pre-fix commit was 1.4e-3 at xi = 1e-6 and 0.39 at xi = 1e-8 (re-measured), which
  broke the BS-limit ground truth. Resolved with the conjugate forms
  r_- = -w/(beta + d) and g = -xi^2 w/(beta + d)^2, plus expm1, and ln(1 + y) computed
  as 2 artanh(y/(2 + y)) because numpy's complex log1p is inaccurate for small y. A
  regression test covers xi down to 1e-8.
- **bs_price silently returned intrinsic value on invalid input.** It now raises on
  negative maturity, volatility or prices (0d309a1).
- **No-arbitrage test bounds hard-coded S0 = 100 and r = 0.03.** They are now built
  from the reference parameters (c325da8).
- **A docstring claimed |g e^{-dT}| < 1 unconditionally.** It is corrected to say this
  holds when kappa > rho*xi/2, and now documents the conjugate forms (d736399).
- **The tiny-xi rate check was too loose** (ratio > 8). It is tightened to > 50 per two
  decades against a theory value of 100 (a4f0218).

### Known limits
- Outside the project's parameter range, the COS oracle doesn't converge for kappa = 0.1,
  rho = +0.95, xi = 1, T >= 5. It raises `RuntimeError` and never returns a wrong price.
  The T = 5 case was re-checked while writing this file.

## Stage 2: Monte Carlo (merged)

### What was built
- `src/hestonlib/mc.py`:
  - full-truncation Euler on ln S (`euler_step`, `simulate`);
  - antithetic variates that negate both independent drivers;
  - `MCResult(price, se, n_samples)`, with the antithetic SE computed from the n/2 pair means;
  - `mc_european`.
- How the random numbers are shared:
  - Normals are drawn on a fine grid (`rng_steps`) and summed per step, so runs with
    different step counts share one Brownian path (coupled levels).
  - With the same seed, runs with different parameters share the same normals (CRN).
- On the log scale, the discounted asset is an exact martingale of the discrete scheme.

### Tests
52 passing in total (14 new in `tests/test_mc.py`); full suite 16 s.

### Tolerances and why
| Check | Tolerance | Justification |
|---|---|---|
| Discounted forward, REF and STRESS, 4 dates | 4 SE | Exact martingale of the scheme: no bias term. |
| BS limit (xi = 1e-8), calls and puts | 4 SE | v stays at theta to O(xi), and log-Euler is exact for GBM: unbiased. |
| One-step corr(ln S, v) = rho | 4 (1 - rho^2)/sqrt(n) | Exact after one step; this is the SE of a sample correlation (iid paths). |
| Reported SE vs spread over 400 seeds | chi2_{399} 0.05% to 99.95% quantiles (0.78 to 1.25) | Variance ratio of normal estimates. The naive SE over the 2n halves fails it (ratio about 0.5). |
| Euler bias shrinks (STRESS, coupled n = 8..128) | Each D(n) = MC(n) - MC(2n) > 4 SE; error(128) below the order-1 envelope 8/128 * error(8) + 4 SE | Shows a resolved bias that shrinks with dt. Observed per-halving ratio 2.1 to 2.5. |
| MC(64) vs COS, REF, 5 strikes, 1.6M paths | 4 SE + max(\|D\| + 4 SE(D), sqrt(8/64)(\|MC(8) - COS\| + 4 SE)) | Richardson bias estimate (valid for weak order >= 1), or an order-1/2 extrapolation from the resolved n = 8 bias, whichever is larger. |
| Same test: power cap | tol < half the COS shift from flipping the sign of rho, per strike | A rho sign error cannot pass. The at-the-money cap of 0.061 sets the path count. |
| Exact scheme identities (hand-computed step, antithetic pair sums, level coupling) | 1e-15 to 1e-6 | Rounding, or O(xi) with xi = 1e-8. |

A mutation check was run before review. Each of these planted bugs was caught by at
least one test:
- drift uses raw v;
- raw v not carried forward;
- rho sign flipped;
- antithetic flips one driver only;
- drift ignores q;
- diffusion uses |v|.

### Reviewer findings and resolutions
1. **Minor:** an observation time with 0 < t < dt/2 rounded to grid index 0 and left the
   outputs uninitialised. Fixed (b5dfb39): such times are rejected, and the code raises
   if any column is left unwritten.
2. **Minor:** the Richardson allowance assumed weak order 1, which cannot be checked at REF
   because the bias is below the noise for n >= 32. Fixed (716bf67): the allowance is now
   the larger of the Richardson estimate and an order-1/2 extrapolation from n = 8. This
   widened the tolerance by up to 1.54 times at 200k paths.
3. **Major, introduced by fix 2:** the wider tolerance pushed against the fixed `tol < 0.1`
   cap, so the test passed only for some seeds. Fixed (858669e):
   - The cap is now derived from power (half the shift from flipping the sign of rho).
   - Paths were raised to 1.6M, so the tolerance is about 2.5 times tighter than the
     original.
   - Over 12 of my seeds and 12 of the reviewer's, the largest tol/cap was 0.73 and the
     largest |err|/tol was 0.38.
4. **Nits:**
   - The bias-shrink docstring overstated what D(n) > 0 shows (it shows the price
     falling); reworded.
   - Added a direct test that antithetic pairs negate both drivers (f60ec2e).
   - Open, not acted on: duplicate observation times within 1e-9 are recorded twice,
     which is harmless. The 1.6M-path test holds about 190 MB.

Final re-review: 0 unresolved issues.

## Stage 3: autocallable (merged, tagged v0.1.0)

### What was built
- `src/hestonlib/autocall.py`:
  - the `Autocallable` product;
  - `autocallable_pv`, the per-path discounted payoff and call index;
  - `price_autocallable`, which returns the price and SE plus per-date call probabilities,
    P(never called), P(loss) and the expected life, each with its SE.
- Payoff conventions: S >= B calls, the first call ends the note, and the maturity date
  is an autocall date. The coupon is N(1 + c i) with i 1-based. The knock-in is European
  (observed at T only) and inclusive (S_T >= KI redeems at par). Each cashflow is
  discounted at its own date.
- `tests/reference.py`: `bs_autocallable` and `bs_autocall_probabilities`, an exact
  discrete-monitoring BS price from multivariate normal CDFs, using Girsanov
  (drift mu + sigma^2) for the knock-in leg. The Genz integration is seeded, so the
  oracle is deterministic.
- `scripts/price_autocallable.py`, whose output the README quotes.

### Tests
69 passing in total (12 in `tests/test_autocall.py`, 5 new oracle self-checks); full
suite about 32 s.

### Tolerances and why
| Check | Tolerance | Justification |
|---|---|---|
| Payoff by hand, discounting, KI boundary | 1e-12 | Exact arithmetic. |
| B = 0; B = inf with KI = 0 | 1e-10, SE < 1e-12 | Zero-variance payoffs with closed-form PV. |
| B = KI = inf (pays N S_T/S0) | 4 SE | Exact martingale of the scheme: no bias. |
| One date, Heston, vs COS replication (bond + digital - put) | 4 SE + max(Richardson, order-1/2 from n = 8); cap: half the rho-flip shift (0.354) | Same construction as stage 2. Here the bias is below the noise even at n = 8, so the allowance is in effect statistical. |
| Four dates, BS limit, vs MVN oracle (price and each call probability) | 4 SE | Log-Euler is exact GBM at the dates. The oracle's error (about 1e-6) is far below 4 SE (about 0.06). |
| Multi-date Heston bias shrinks (STRESS, coupled) | D(6), D(12) > 4 SE; \|D(6)\| - \|D(48)\| > 4 combined SE | No closed form. Shows the discretization changes shrinking (24-to-48 ratio about 0.5). |
| Oracle self-checks | 1e-12 (closed forms), 1e-8 (1-D quadrature, orthant formula), 1e-6 (nested quadrature) | Genz integration error with abseps 1e-10; 5e-8 was observed. |

A mutation check was run before review. Each of these planted bugs was caught:
- strict barrier;
- 0-based coupon;
- call discounted at maturity;
- strict KI;
- KI observed at all dates;
- flat coupon.

### Reviewer findings and resolutions
- First review: 0 unresolved, with five nits. All were addressed and confirmed in a second
  review (0 unresolved):
  1. The Genz error was understated (1e-7 against an observed scatter of about 1e-6), and
     the oracle was not seeded. The docstring is corrected and the integration seeded.
     I had suspected that the old frozen `multivariate_normal(...)` call dropped
     abseps/releps. A second review showed that scipy forwards them to the frozen
     instance, so they were honoured. Only the seeding changed.
  2. `_single_date_truth` now asserts KI <= B.
  3. The docstring now says the one-date Euler bias is below the noise.
  4. A KI > B closed-form test covers the `min(KI, B)` branch. The reviewer also confirmed
     it with exact-GBM MC.
  5. The README expected life now carries its SE.
- README numbers: the reviewer re-ran the script and found every number matched exactly.

## Deferred to later stages
- **Stage 4:** the quad error estimates are currently discarded
  (`integral, _ = quad(...)` in `analytic.py`), and Heston prices can be slightly
  negative at very short T deep out of the money. The minimum seen was -7.2e-11 over
  T in {1e-3, 1e-2, 0.05} with deep-OTM calls and puts, on the reference and stress
  parameters. Handle both in the implied-vol inversion: clip or guard the prices, and
  use the error estimates.
