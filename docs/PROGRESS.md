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

## Deferred to later stages
- **Stage 4:** the quad error estimates are currently discarded
  (`integral, _ = quad(...)` in `analytic.py`), and Heston prices can be slightly
  negative at very short T deep out of the money. The minimum seen was -7.2e-11 over
  T in {1e-3, 1e-2, 0.05} with deep-OTM calls and puts, on the reference and stress
  parameters. Handle both in the implied-vol inversion: clip or guard the prices, and
  use the error estimates.
