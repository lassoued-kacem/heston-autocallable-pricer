"""Independent ground-truth oracle for the test suite.

This module deliberately does NOT import ``hestonlib``. Everything here is written separately
from the package so that tests compare two independent implementations.

Contents
--------
* ``bs_price``: Black-Scholes closed form (call/put) with a continuous dividend yield.
* ``heston_cf_logret``: characteristic function of X_T = ln(S_T / S0) under Heston, in the
  Albrecher et al. (2007) "little Heston trap" form, written from the Riccati solution.
* ``heston_cf_ode``: the same characteristic function obtained by numerically integrating
  the Riccati ODEs with ``scipy.integrate.solve_ivp``. It is used to validate the closed form
  independently of any algebra.
* ``cos_put`` / ``cos_call`` / ``cos_digital_put``: Fang & Oosterlee (2008) COS method.
* ``bs_autocallable`` / ``bs_autocall_probabilities``: exact discrete-monitoring autocallable
  price under Black-Scholes from multivariate normal CDFs (derivation at the function).

COS method summary (Fang & Oosterlee, SIAM J. Sci. Comput. 31(2), 2008)
----------------------------------------------------------------------
With y = ln(S_T / K) and x = ln(S0 / K), y = x + X_T. For a payoff g(y) supported on [a, b]:

    V0 = e^{-rT} sum'_{k=0}^{N-1} Re[ phi_X(u_k) e^{i u_k (x - a)} ] * G_k,
    u_k = k pi / (b - a),   G_k = 2/(b-a) int_a^b g(y) cos(u_k (y - a)) dy,

where sum' halves the k = 0 term. With

    chi_k(c,d) = int_c^d e^y cos(u_k (y-a)) dy,   psi_k(c,d) = int_c^d cos(u_k (y-a)) dy,

the put  (g = K (1 - e^y)^+) gives G_k = 2/(b-a) K (psi_k(a,0) - chi_k(a,0)), and
the digital put (g = 1{y < 0}) gives G_k = 2/(b-a) psi_k(a,0).
Puts are used because they are bounded on the truncated left tail; calls come from parity.

Truncation range: [c1 - L sqrt(c2), c1 + L sqrt(c2)] in X-space (shifted by x), where
c1 = (r-q)T - E[int v dt]/2 and c2 = E[int_0^T v dt] = theta T + (v0 - theta)(1 - e^{-kappa T})/kappa.
c2 is the expected integrated variance, the leading term of Var(X_T). It under-estimates the
spread when xi is large, so by default the oracle doubles (L, N) until prices change < 1e-11 * max(1, max K).
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import ndtr


# --------------------------------------------------------------------------- Black-Scholes
def bs_price(S0, K, T, r, q, sigma, kind="call"):
    """Black-Scholes price with continuous dividend yield q."""
    S0, K = np.asarray(S0, float), np.asarray(K, float)
    sd = sigma * np.sqrt(T)
    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / sd
    d2 = d1 - sd
    call = S0 * np.exp(-q * T) * ndtr(d1) - K * np.exp(-r * T) * ndtr(d2)
    if kind == "call":
        return call
    return call - S0 * np.exp(-q * T) + K * np.exp(-r * T)


# --------------------------------------------------------------------------- Heston CF
def heston_cf_logret(u, T, r, q, v0, kappa, theta, xi, rho):
    """E[exp(i u ln(S_T/S0))] under Heston, little-trap form.

    Riccati system for f = exp(A(tau) + B(tau) v0 + i u x):
        B' = -(u^2 + i u)/2 - (kappa - rho xi i u) B + xi^2 B^2 / 2,   B(0) = 0
        A' = i u (r - q) + kappa theta B,                                A(0) = 0
    Roots of xi^2 B^2/2 - beta B - (u^2+iu)/2 = 0 with beta = kappa - rho xi i u are
    (beta +- D)/xi^2, D = sqrt(beta^2 + xi^2 (u^2 + i u)). Taking G = (beta - D)/(beta + D),
    the solution uses exp(-D tau), which keeps the complex log continuous (Albrecher et al.).
    """
    u = np.asarray(u, dtype=complex)
    iu = 1j * u
    beta = kappa - rho * xi * iu
    D = np.sqrt(beta * beta + xi * xi * (u * u + iu))
    G = (beta - D) / (beta + D)
    eDT = np.exp(-D * T)
    B = (beta - D) / (xi * xi) * (1.0 - eDT) / (1.0 - G * eDT)
    A = iu * (r - q) * T + kappa * theta / (xi * xi) * (
        (beta - D) * T - 2.0 * np.log((1.0 - G * eDT) / (1.0 - G))
    )
    return np.exp(A + B * v0)


def heston_cf_ode(u, T, r, q, v0, kappa, theta, xi, rho):
    """Same CF from numerically integrating the Riccati ODEs (scalar u)."""
    u = complex(u)
    iu = 1j * u

    def rhs(_t, y):
        A, B = y
        dB = -0.5 * (u * u + iu) - (kappa - rho * xi * iu) * B + 0.5 * xi * xi * B * B
        dA = iu * (r - q) + kappa * theta * B
        return [dA, dB]

    sol = solve_ivp(rhs, (0.0, T), [0j, 0j], method="DOP853", rtol=1e-12, atol=1e-13)
    A, B = sol.y[:, -1]
    return np.exp(A + B * v0)


# --------------------------------------------------------------------------- COS method
def _range(T, r, q, v0, kappa, theta, L):
    mean_int_var = theta * T + (v0 - theta) * (1.0 - np.exp(-kappa * T)) / kappa
    c1 = (r - q) * T - 0.5 * mean_int_var
    half = L * np.sqrt(mean_int_var)
    return c1 - half, c1 + half


def _chi_psi(k, a, b, c, d):
    w = k * np.pi / (b - a)
    chi = (
        np.cos(w * (d - a)) * np.exp(d)
        - np.cos(w * (c - a)) * np.exp(c)
        + w * np.sin(w * (d - a)) * np.exp(d)
        - w * np.sin(w * (c - a)) * np.exp(c)
    ) / (1.0 + w * w)
    psi = np.empty_like(w)
    psi[0] = d - c
    psi[1:] = (np.sin(w[1:] * (d - a)) - np.sin(w[1:] * (c - a))) / w[1:]
    return chi, psi


def _cos_sum(S0, K, T, r, q, model, coeff_fn, N, L):
    v0, kappa, theta, xi, rho = model
    K = np.atleast_1d(np.asarray(K, float))
    lo, hi = _range(T, r, q, v0, kappa, theta, L)
    x = np.log(S0 / K)  # shape (nK,)
    # y = x + X_T, so the y-range is [x + lo, x + hi] per strike. Use a common [a, b] that covers all.
    a = float(np.min(x) + lo)
    b = float(np.max(x) + hi)
    a, b = min(a, -1e-8), max(b, 1e-8)  # 0 must lie inside [a, b]
    k = np.arange(N)
    u = k * np.pi / (b - a)
    phi = heston_cf_logret(u, T, r, q, v0, kappa, theta, xi, rho)
    chi, psi = _chi_psi(k, a, b, a, 0.0)
    Gk = coeff_fn(chi, psi) * 2.0 / (b - a)
    terms = np.real(phi[None, :] * np.exp(1j * u[None, :] * (x[:, None] - a))) * Gk[None, :]
    terms[:, 0] *= 0.5
    return np.exp(-r * T) * terms.sum(axis=1), K


def _cos_converged(raw, S0, K, T, r, q, model, N, L, tol=1e-11, max_doublings=4):
    """Evaluate a COS price, then double (L, N) until the max change is below ``tol * max(1, max K)``.

    The cumulant range above under-estimates Var(X_T) when xi is large and the Feller ratio
    is small (fat left tail), so a fixed L can truncate too much mass. Doubling L (and N so the
    resolution u_k is preserved) until prices stop moving is a self-contained convergence check.
    Explicitly passing ``adaptive=False`` evaluates once at the given (N, L).
    """
    scale = max(1.0, float(np.max(K)))  # absolute tolerance relative to the price scale
    prev = raw(S0, K, T, r, q, model, N, L)
    for _ in range(max_doublings):
        N, L = 2 * N, 2 * L
        cur = raw(S0, K, T, r, q, model, N, L)
        if np.max(np.abs(cur - prev)) < tol * scale:
            return cur
        prev = cur
    raise RuntimeError("COS oracle did not converge; widen the range or increase N")


def _raw_put(S0, K, T, r, q, model, N, L):
    s, K = _cos_sum(S0, K, T, r, q, model, lambda chi, psi: psi - chi, N, L)
    return K * s


def _raw_digital_put(S0, K, T, r, q, model, N, L):
    s, _ = _cos_sum(S0, K, T, r, q, model, lambda chi, psi: psi, N, L)
    return s


def cos_put(S0, K, T, r, q, v0, kappa, theta, xi, rho, N=1024, L=12.0, adaptive=True):
    model = (v0, kappa, theta, xi, rho)
    if not adaptive:
        return _raw_put(S0, K, T, r, q, model, N, L)
    return _cos_converged(_raw_put, S0, K, T, r, q, model, N, L)


def cos_call(S0, K, T, r, q, v0, kappa, theta, xi, rho, N=1024, L=12.0, adaptive=True):
    K = np.atleast_1d(np.asarray(K, float))
    put = cos_put(S0, K, T, r, q, v0, kappa, theta, xi, rho, N, L, adaptive)
    return put + S0 * np.exp(-q * T) - K * np.exp(-r * T)


def cos_digital_put(S0, K, T, r, q, v0, kappa, theta, xi, rho, N=1024, L=12.0, adaptive=True):
    """Price of a cash-or-nothing put paying 1 if S_T < K: e^{-rT} Q(S_T < K)."""
    model = (v0, kappa, theta, xi, rho)
    if not adaptive:
        return _raw_digital_put(S0, K, T, r, q, model, N, L)
    return _cos_converged(_raw_digital_put, S0, K, T, r, q, model, N, L)


# --------------------------------------------------------------------------- BS autocallable
# Exact price of a discretely observed autocallable under Black-Scholes, from multivariate
# normal CDFs. X_i = ln(S_{t_i}/S0) = mu t_i + sigma W_{t_i}, mu = r - q - sigma^2/2, so
# X ~ N(mu t, sigma^2 min(t_i, t_j)). With b = ln B, k = ln min(KI, B) and
# F_i = P(X_1 < b, ..., X_i < b) (F_0 = 1):
#   P(called at t_i)                        = F_{i-1} - F_i
#   P(never called, S_T >= KI S0)           = F_n - G,     G = P(X_1..X_{n-1} < b, X_n < k)
#   E[S_T/S0 ; never called, S_T < KI S0]   = e^{(r-q)T} G*
# where G* is G with drift mu + sigma^2 (Girsanov: E[e^{X_n} 1_A] = E[e^{X_n}] Q*(A), and
# under Q* the Brownian motion has drift sigma). The price is
#   sum_i N (1 + c i) e^{-r t_i} (F_{i-1} - F_i) + N e^{-rT} (F_n - G) + N e^{-qT} G*.
def _mvn_below(upper, mean, cov):
    from scipy.stats import multivariate_normal

    upper, mean = np.asarray(upper, float), np.asarray(mean, float)
    if len(upper) == 1:
        return float(ndtr((upper[0] - mean[0]) / np.sqrt(cov[0, 0])))
    # Genz integration is randomized: a fixed rng makes the oracle deterministic.
    return float(multivariate_normal.cdf(upper, mean, cov, maxpts=2_000_000, abseps=1e-10,
                                         releps=1e-10, rng=np.random.default_rng(12345)))


def bs_autocall_probabilities(S0, r, q, sigma, obs_times, B):
    """P(first autocall at t_i), i = 1..n, under Black-Scholes."""
    t = np.asarray(obs_times, float)
    mu, b = r - q - 0.5 * sigma**2, np.log(B)
    cov = sigma**2 * np.minimum.outer(t, t)
    F = [1.0] + [_mvn_below(np.full(i, b), mu * t[:i], cov[:i, :i]) for i in range(1, len(t) + 1)]
    return np.array([F[i] - F[i + 1] for i in range(len(t))])


def bs_autocallable(S0, r, q, sigma, obs_times, B, KI, c, N=100.0, S_ref=None):
    """Exact BS price of the autocallable (barriers as fractions of the fixing S_ref, coupon
    c per date). S_ref defaults to S0. With S_ref != S0 the barrier levels B S_ref and
    KI S_ref are fixed, so b = ln(B S_ref / S0), and the KI redemption N S_T / S_ref is
    (S0 / S_ref) times N S_T / S0."""
    ref = S0 if S_ref is None else float(S_ref)
    t = np.asarray(obs_times, float)
    n = len(t)
    mu = r - q - 0.5 * sigma**2
    b, k = np.log(B * ref / S0), np.log(min(KI, B) * ref / S0)
    cov = sigma**2 * np.minimum.outer(t, t)
    F = [1.0] + [_mvn_below(np.full(i, b), mu * t[:i], cov[:i, :i]) for i in range(1, n + 1)]
    up = np.full(n, b)
    up[-1] = k
    G = _mvn_below(up, mu * t, cov)
    G_star = _mvn_below(up, (mu + sigma**2) * t, cov)
    price = sum(N * (1 + c * (i + 1)) * np.exp(-r * t[i]) * (F[i] - F[i + 1]) for i in range(n))
    return (price + N * np.exp(-r * t[-1]) * (F[n] - G)
            + N * (S0 / ref) * np.exp(-q * t[-1]) * G_star)
