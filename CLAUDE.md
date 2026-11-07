# CLAUDE.md: heston-autocallable

Pricing and risk for an autocallable structured product under Heston stochastic volatility,
with calibration and a Streamlit dashboard. Pure Python: NumPy/SciPy, vectorized.
**No C++, no Numba** (Python 3.14 compatibility is uncertain).

## Layout
- Package: `/src/hestonlib`
- Tests: `/tests` (independent oracle in `tests/reference.py`, which must not import `hestonlib`)
- Scripts: `/scripts` (`check.sh` is the Stop-hook test runner)
- Dashboard: `/app/dashboard.py`
- README figures: `docs/figures/` (committed). Scratch output: `plots/` (ignored).
- Python: always `.venv/bin/python`. Tests: `.venv/bin/python -m pytest -q`.

## Model
    dS = (r - q) S dt + sqrt(v) S dW1
    dv = kappa (theta - v) dt + xi sqrt(v) dW2
    corr(dW1, dW2) = rho

Reference parameters: S0=100, r=0.03, q=0.0, v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7.

## Ground-truth hierarchy
(a) Heston semi-analytic European prices from an independent COS-method oracle in
    `tests/reference.py`.
(b) Black-Scholes in the degenerate limit (xi -> 0, v0 = theta gives sigma = sqrt(v0)).
(c) Exact analytic limits of the autocallable payoff.

## Testing rules
- NEVER set expected test values from the code's own output. Derive them independently.
- Do not quote literature benchmark values from memory. Use a published number only if you
  can verify it from a source, and cite it.
- Every MC price reports its standard error.
- Each test states its tolerance and why.
- For Euler-discretized MC the tolerance must cover both statistical error (4*SE) and
  discretization bias, and the bias must be shown to shrink as the time step shrinks.
- Antithetic SE is computed on pair means, not on the individual halves.
- Greeks use common random numbers; each Greek reports its own SE from paired differences.
- Keep the full suite fast (target < 3 min): the Stop hook runs it.

## Workflow
- Work in stages. After each stage: run the full test suite, run the math-reviewer subagent,
  then merge (see git rules).
- If `math-reviewer` is not invocable, run a general-purpose agent with the body of
  `.claude/agents/math-reviewer.md` and say explicitly which agent ran.
- Show the user the test output and reviewer findings before merging to main.
- Any number in the README must come from a script's actual output.

## Version control rules (follow throughout)

- One feature branch per stage, named `feat/stage-N-<topic>`, branched from main.
- Small atomic commits, one logical change each. No "wip" or "update" messages.
- Conventional Commits: feat:, test:, fix:, docs:, perf:, refactor:, chore:.
  Imperative subject under 72 chars; add a body when the "why" is not obvious
  (e.g. why a tolerance was chosen).
- A stage branch merges into main only when the full suite is green and the
  math-reviewer reports no unresolved issues. Merge with --no-ff, then delete
  the branch.
- Annotated tags: v0.1.0 after stage 3, v0.2.0 after stage 5, v1.0.0 at the end.
- Reviewer-found bugs get their own `fix:` commit on the stage branch.
- Never rewrite or backdate history, never force-push, never commit failing
  tests to main. No remote exists, so no push.
- Keep the Co-Authored-By trailer on every commit.
- Keep the README current as part of the stage that changes behavior.
