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

## Stage 4: implied volatility and calibration (merged)

### What was built
- `heston_call` and `heston_put` take `return_error=True`. This returns QUADPACK's error
  estimate in price units, plus a rounding term for the F - sqrt(FK)/pi * I cancellation.
- `bs_vega`.
- `src/hestonlib/implied_vol.py`:
  - `brentq` inside checked no-arbitrage bounds;
  - a price within its error of a bound gives NaN;
  - a price beyond a bound by more than its error raises;
  - `heston_implied_vols` applies the pricer's error estimate.
- `src/hestonlib/calibration.py`:
  - bounded TRF least squares on (C_model - C_mkt)/vega;
  - SEs from s^2 (J^T J)^-1 with m - 5 dof;
  - the correlation matrix and the scaled condition number;
  - parameters at a bound;
  - the Feller ratio (reported, not imposed);
  - exact model IVs, with unresolvable ones counted (`n_iv_unresolved`).
- `scripts/calibrate.py`, whose output the README quotes.

### Tests
89 passing in total (14 in `test_implied_vol.py`, 6 in `test_calibration.py`); full suite
about 50 s.

### Tolerances and why
| Check | Tolerance | Justification |
|---|---|---|
| bs_vega vs central difference of the oracle (h = 1e-5) | 1e-6 | h^2/6 max\|d3C/dsigma3\| = 1.4e-7 (max about 8.4e3 on the grid); rounding about 1e-9. |
| IV round trip (oracle BS prices) | 64 eps S0 / vega + 1e-13, asserted only where vega > 1e-4 | Price rounding divided by vega, plus brentq xtol. The worst case uses 1.4% of it. |
| Pricer error estimate vs COS at N = 2^17, L = 192 | \|Lewis - COS\| <= err + 1e-12, and err < 1e-8 | The COS is checked to < 1e-12 by doubling, so the estimate is binding: with err = 0, 3 of 6 cases fail. The true error is at most 3.4% of the estimate. |
| Short-T deep-OTM IVs (deferred item) | NaN exactly where time value <= err; finite IVs reprice within err | Consistency of the guard. Negative prices do occur in this test. |
| Heston IV in the BS limit | 1e-5 / vega | The stage 1 price-gap bound of 1e-5, divided by vega. |
| Noise-free recovery, REF and STRESS | 2 (\|dr\| + \|r\|) / sigma_min(J), with dr_j = (err_j + 1e-12)/vega_j | First-order perturbation bound with a factor 2 for nonlinearity. The tolerance is about 5e-7; actual errors are 3.5e-12 (REF) and 1.6e-10 (STRESS). |
| Noisy recovery (IV noise 0.002) | \|z\| < 4 per parameter; Mahalanobis in 5 F(5,23) 0.05% to 99.95% (0.149, 34.0); s^2/sigma^2 in chi2_23/23 quantiles | Linear-Gaussian model with s^2 estimated. False-failure probability is at most 2.8e-3 (z) and 1e-3 (each window). |
| IV errors vs vega-weighted residuals | 2 (64 eps S0 / vega + 1e-13) | Two inversions; the Taylor term is O(r^2). |

### Reviewer findings and resolutions
The review agent described itself as "a general-purpose agent running the math-reviewer
brief". The Agent call used `subagent_type: math-reviewer`.

1. **Major:** the calibration's post-fit IV step ignored the error estimate. A short-dated
   wing price of -4.6e-12 made `calibrate` raise after converging, and `nanmean` hid any
   NaNs. Fixed (1289a6f):
   - Model IVs now go through `heston_implied_vols`, and unresolved IVs are counted.
   - A regression test reproduces the crash on the old code, and now also checks where
     the NaNs occur.
2. **Major:** the error-bound test could not fail, because its COS allowance (1.5e-9)
   exceeded the bound under test. Fixed (4eee4cc): the test now uses a converged COS and a
   1e-12 allowance. The noise-free calibration bound now uses per-quote error estimates.
3. **Minor:** the stated false-failure rates assumed a known sigma. Fixed: rates are now
   given for t_23. The Mahalanobis window uses the exact 5 F(5,23) law, whose upper
   quantile (34.0) replaces the chi2_5 value (22.1). This corrects the distribution and is
   not a loosening: the old bound had a true false-failure rate of 0.57%. The seed-7 value
   of 20.0 passed both.
4. **Minor:** the README "3%" claim came from no script. Dropped.

Nits:
- The conditioning scale is floored at 1% of the box.
- The docstrings say "estimate".
- An unused binding was removed.
- Not done: returning an err/vega uncertainty with each IV.

Final re-review: 0 unresolved.

## Stage 5: Greeks (merged, tagged v0.2.0)

### What was built
- `price_autocallable(..., ref_spot=...)`: the note's fixing, separate from spot. The
  payoff is scale invariant, so bumping spot together with the fixing gives zero delta.
- `src/hestonlib/greeks.py`: `autocallable_greeks`, bump-and-revalue with common random
  numbers (one seed for every scenario).

  | Greek | bump |
  |---|---|
  | delta, gamma | 2% of the fixing |
  | vega | parallel shift of sqrt(v0) and sqrt(theta) by 0.01 |
  | rho | 1e-3 |

  Each Greek's SE comes from its per-path paired differences, on antithetic pair means.
- `tests/reference.py`: `bs_autocallable(..., S_ref=...)`, with a closed-form self-check.
- `scripts/greeks.py`, whose output the README quotes.

### Tests
96 passing in total (6 in `test_greeks.py`; an earlier version of this entry said 7); full
suite about 68 s.

### Tolerances and why
| Check | Tolerance | Justification |
|---|---|---|
| Scale invariance (spot and fixing x 1.3) | 1e-10 | Paths scale exactly, so the per-path PVs are identical. |
| Delta equals a manual CRN paired difference | 1e-12 | Structural identity. |
| BS limit: price, delta, gamma, vega, rho vs the oracle FD at the same bumps | 4 SE | Every scenario is unbiased (exact GBM at the dates), so the MC FD is unbiased for the oracle FD. FD truncation does not enter. |
| Bump sizes | Richardson bias 4/3 \|FD(h) - FD(h/2)\| < SE/4 for each Greek | The default bumps barely distort the Greek. Vega is the tightest (0.077 vs about 0.11). |
| Live note (spot 95, fixing 100): delta vs oracle | 4 SE | Exercises `ref_spot`. A planted "fixing follows spot" bug fails it. |
| Greek SEs vs spread over 200 seeds | chi2_199/199 0.05% to 99.95% (0.70 to 1.36) | Ratios were 0.85 to 1.20 over several seed bases. The mean over 10k pairs is near normal despite heavy-tailed per-path values. |
| Heston one-date delta and vega vs COS replication FD | 4 SE + max(Richardson, order-1/2 from n = 8); cap: half the rho-flip change | Stage 2 construction. The Greeks' Euler bias is below MC resolution here, so the allowance is in effect statistical. Shrinkage with dt is shown at price level (test_mc, test_autocall). A nearby rho alternative is not resolvable at 200k paths. |

A mutation check was run before review. These planted bugs were caught:
- vol bump on v0 only;
- no CRN;
- wrong gamma scale;
- fixing recomputed from the bumped spot (caught after the live-note test was added).

### Reviewer findings and resolutions
The first report called itself "a general-purpose agent". The harness metadata
(`subagents/agent-*.meta.json`) shows agentType "math-reviewer", isBuiltIn false, for every
stage's review, including stage 4's, which made the same claim. The reviewer corrected
itself in the re-review.

1. **Minor:** the Greeks docstring misdescribed the discontinuities (the KI is a jump of
   N(1 - KI), not a kink) and the default bump (2%, not 1%). Fixed (aa11bee).
2. **Minor:** the README quoted an SE (0.021) for the difference between the two 48-step
   prices. It came from no script and was wrong, because the runs share part of their
   random stream. Fixed: the SE is dropped, and only script outputs are quoted.
3. **Minor:** the Heston one-date Greek test's allowance is noise-dominated and does not
   show bias shrinkage. Resolved by documentation (the reviewer's judgement: an assertion
   would be vacuous or flaky at feasible path counts). The docstring states the limits and
   cites the price-level shrinkage tests.
4. **Nits:** shadowed name `q`, implicit antithetic flag, an overstated Genz error, the
   sign of the autocall jump, and the 4/h^2 Genz weight in gamma. All fixed.

Final re-review: 0 unresolved.

## Stage 6: convergence study and README integrity (merged)

### What was built
- `scripts/convergence.py`:
  - Euler bias of an ATM call against COS, for the stress and reference sets, on coupled
    levels from 4 to 256 steps (one Brownian path), with paired changes D(n) and their SEs;
  - the SE against path count, antithetic vs plain;
  - two figures in `docs/figures/` (palette validated: blue and orange, all-pairs CVD
    check passed). `--figures-dir` redirects them.
- `scripts/check_readme.py`:
  - Every decimal or thousands-separated number inside README blocks marked
    `<!-- numbers-from: scripts/X.py [...] -->` must appear, sign included, in a fresh run
    of the named script(s).
  - Figures are rendered to a temporary directory and compared byte for byte with the
    committed ones; `--strict-figures` makes a mismatch fail.
  - Its limits are documented: set membership, so a number in the wrong row can pass;
    integers and qualitative wording are not checked.
- `scripts/calibrate.py` prints max |z|.
- README: a convergence section. Every script-derived block, including its configuration
  lines, is now marked.

### Tests
100 passing in total (4 in `test_check_readme.py`, which test the checker's parsing);
full suite about 68 s. The checker itself runs the scripts (about 90 s), so it is not in
the Stop hook. It was run with `--strict-figures` before merging and passed.

### Results (from the script output)
- **Stress set:** the error falls from +2.9462 (4 steps) to +0.0333 (256 steps). The paired
  change halves from 16 to 32 steps, as order 1 predicts, and falls by more than half per
  halving from 32 steps on.
- **Reference set:** the bias is resolved only up to about 32 steps.
- **SE:** the slope against paths is -0.502 (near -1/2 by construction). Antithetic
  sampling cuts the variance by 2.42 to 2.43 times at equal path counts.

### Reviewer findings and resolutions
1. **Minor:** script-derived numbers (the 48-step price comparison and the configuration
   lines) sat outside the markers. Fixed: they are marked, and blocks may name several
   scripts.
2. **Minor, introduced by fix 1:** the README said every stress change from 16 steps on was
   half the previous one or less, but 16-to-32 is 0.52. Fixed: the wording follows the
   numbers.
3. **Nits:**
   - all D rows are now listed;
   - the stress decay is called pre-asymptotic, then faster than order 1;
   - the -0.0021 reference change is "consistent with noise" rather than asserted to be a
     fluctuation;
   - the SE slope is said to hold by construction;
   - the checker no longer rewrites committed figures, and gained `--strict-figures`, a
     separate stale-figure message and a missing-figure failure;
   - the Greeks table is checked against greeks.py alone;
   - max |z| is printed;
   - one long line is rewrapped.

Final re-review: 0 unresolved.

## Stage 7: Streamlit dashboard (merged, tagged v1.0.0)

### What was built
- `app/dashboard.py` (`.venv/bin/streamlit run app/dashboard.py`):
  - Sidebar: the model, spot and fixing, the product (maturity, observation frequency,
    barrier, KI, coupon) and the MC settings.
  - Tabs: the price with its SE, 95% interval and call probabilities; CRN Greeks on
    demand, with units and per-vol-point and per-bp values; and the Heston smile, with
    unresolvable quotes left blank.
  - The Feller ratio is shown.
  - The computations are pure functions with bounded caches.
  - If the default bumps cannot be applied, the Greeks tab says why.
- pyproject is at version 1.0.0, with the dashboard test dependencies declared and altair
  pinned.

### Tests
106 passing in total (6 in `test_dashboard.py`); full suite about 74 s. Streamlit was also
launched as a server (`--server.headless true`), and `/_stcore/health` returned "ok".

### Tolerances and why
| Check | Tolerance | Justification |
|---|---|---|
| Wiring in the BS limit (spot 97, fixing 100, semi-annual, B 95%, KI 65%): price and each call probability vs the MVN oracle | 4 SE | Log-Euler is exact at the dates. A barrier/KI swap or a fixing that follows spot moves the price by 181 and 47 SE (measured in review). |
| Smile in the BS limit | 1e-5 / vega | Stage 1 price-gap bound. |
| Headless app's price vs a direct call with the defaults as fractions | exact string to 4 decimals | Same seed, same numbers. A dropped /100 on the coupon fails it. |

A mutation check was run. These planted bugs were caught by the wiring tests:
- barrier set to KI;
- KI set to the barrier;
- the fixing ignored;
- the coupon scaled by /10.

### Reviewer findings and resolutions
1. **Minor:** the Greeks button crashed when spot was within the 2% bump of zero. Fixed:
   a feasibility check plus a warning, an error backstop, and a regression test.
2. **Minor:** the Greeks had no units, and vega per 1.00 of vol could be misread by 100x.
   Fixed: a units column, plus per-vol-point and per-bp values.
3. **Minor:** the wiring test's docstring claimed probability and percent-scaling checks it
   did not make. Fixed: both are now asserted.
4. **Nits:**
   - the SE is visible under the price;
   - the caches are bounded;
   - the fixing default comes from the defaults table;
   - the README status line is fixed;
   - the packaging metadata is updated;
   - the bump defaults come from `autocallable_greeks`.

Final re-review: 0 unresolved.

## Deferred to later stages
- **Stage 4 (resolved in stage 4):** the quad error estimates are currently discarded
  (`integral, _ = quad(...)` in `analytic.py`), and Heston prices can be slightly
  negative at very short T deep out of the money. The minimum seen was -7.2e-11 over
  T in {1e-3, 1e-2, 0.05} with deep-OTM calls and puts, on the reference and stress
  parameters. Handle both in the implied-vol inversion: clip or guard the prices, and
  use the error estimates.
  Resolution: the prices are guarded, not clipped. The pricer returns its error estimate;
  IV inversion reports NaN for a time value within that error and raises beyond it; the
  calibration's model IVs use the same rule and count unresolved quotes. See the stage 4
  entry.
