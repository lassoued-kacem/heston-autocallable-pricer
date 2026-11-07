"""Streamlit dashboard: Heston autocallable pricing, risk and smile.

    .venv/bin/streamlit run app/dashboard.py

Every MC number is shown with its standard error. The computations live in pure functions
(``price_view``, ``greeks_view``, ``smile_view``) so they can be tested without a browser.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hestonlib.autocall import Autocallable, price_autocallable  # noqa: E402
from hestonlib.greeks import autocallable_greeks  # noqa: E402
from hestonlib.implied_vol import heston_implied_vols  # noqa: E402
from hestonlib.params import HestonParams  # noqa: E402

DEFAULTS = dict(S0=100.0, r=0.03, q=0.0, v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)


def make_product(years: int, per_year: int, barrier: float, ki: float, coupon: float,
                 notional: float = 100.0) -> Autocallable:
    times = tuple(k / per_year for k in range(1, years * per_year + 1))
    return Autocallable(obs_times=times, autocall_barrier=barrier, coupon=coupon,
                        ki_barrier=ki, notional=notional)


def n_steps_for(product: Autocallable, steps_per_year: int) -> int:
    """Total Euler steps; observation dates must fall on the grid."""
    n = int(round(product.maturity * steps_per_year))
    grid = np.arange(1, n + 1) * product.maturity / n
    if not all(np.min(np.abs(grid - t)) < 1e-9 for t in product.obs_times):
        raise ValueError("steps per year must be a multiple of the observation frequency")
    return n


def price_view(product, p, steps_per_year, n_paths, seed, ref_spot):
    r = price_autocallable(product, p, n_steps_for(product, steps_per_year), n_paths, seed,
                           ref_spot=ref_spot)
    probs = pd.DataFrame({
        "date (years)": list(product.obs_times),
        "P(called)": r.call_prob,
        "SE": r.call_prob_se,
    })
    summary = pd.DataFrame({
        "quantity": ["price", "P(never called)", "P(loss at maturity)", "expected life (years)"],
        "value": [r.price, r.no_call_prob.price, r.loss_prob.price, r.expected_life.price],
        "SE": [r.se, r.no_call_prob.se, r.loss_prob.se, r.expected_life.se],
    })
    return summary, probs


def greeks_problem(p: HestonParams, ref_spot: float, h_spot=0.02, h_vol=0.01):
    """Why the default bumps cannot be applied at this state (None if they can)."""
    if p.S0 - h_spot * ref_spot <= 0:
        return f"spot {p.S0:g} is within the {h_spot:.0%} spot bump ({h_spot * ref_spot:g}) of zero"
    if np.sqrt(p.v0) < h_vol or np.sqrt(p.theta) < h_vol:
        return f"sqrt(v0) and sqrt(theta) must be at least the vol bump {h_vol}"
    return None


def greeks_view(product, p, steps_per_year, n_paths, seed, ref_spot):
    g = autocallable_greeks(product, p, n_steps_for(product, steps_per_year), n_paths, seed,
                            ref_spot=ref_spot)
    return pd.DataFrame({"Greek": list(g), "value": [v.price for v in g.values()],
                         "SE": [v.se for v in g.values()]})


def smile_view(p, maturities=(0.25, 0.5, 1.0, 2.0), moneyness=np.linspace(0.7, 1.3, 13)):
    rows = []
    for T in maturities:
        K = p.S0 * np.asarray(moneyness)
        iv, _, _ = heston_implied_vols(K, T, p)
        rows += [{"maturity": f"{T:g}y", "strike": k, "implied vol": v} for k, v in zip(K, iv)]
    return pd.DataFrame(rows)


def main():  # pragma: no cover - exercised through streamlit's AppTest
    import altair as alt
    import streamlit as st

    st.set_page_config(page_title="Heston autocallable", layout="wide")
    st.title("Heston autocallable")
    st.caption("Full-truncation Euler Monte Carlo with antithetic variates. "
               "Every MC figure carries its standard error.")

    with st.sidebar:
        st.header("Model")
        c = DEFAULTS
        S0 = st.number_input("spot S0", 1.0, 1000.0, c["S0"])
        ref = st.number_input("fixing (initial level)", 1.0, 1000.0, 100.0)
        r = st.number_input("rate r", -0.05, 0.2, c["r"], format="%.4f")
        q = st.number_input("dividend yield q", 0.0, 0.2, c["q"], format="%.4f")
        v0 = st.number_input("v0", 1e-4, 1.0, c["v0"], format="%.4f")
        kappa = st.number_input("kappa", 0.01, 10.0, c["kappa"])
        theta = st.number_input("theta", 1e-4, 1.0, c["theta"], format="%.4f")
        xi = st.number_input("xi (vol of vol)", 0.01, 3.0, c["xi"])
        rho = st.slider("rho", -0.99, 0.99, c["rho"])
        st.header("Product")
        years = st.slider("maturity (years)", 1, 5, 3)
        per_year = st.selectbox("observations per year", [1, 2, 4], index=0)
        barrier = st.slider("autocall barrier (% of fixing)", 50, 150, 100) / 100
        ki = st.slider("knock-in (% of fixing)", 10, 100, 60) / 100
        coupon = st.slider("coupon per observation (%)", 0.0, 20.0, 8.0) / 100
        st.header("Monte Carlo")
        spy = st.selectbox("steps per year", [12, 24, 48, 96], index=2)
        n_paths = st.select_slider("paths", [20_000, 100_000, 200_000, 500_000], 100_000)
        seed = st.number_input("seed", 0, 2**31 - 1, 2026)

    try:
        p = HestonParams(S0=S0, r=r, q=q, v0=v0, kappa=kappa, theta=theta, xi=xi, rho=rho)
        product = make_product(years, per_year, barrier, ki, coupon)
        n_steps_for(product, spy)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    st.caption(f"Feller ratio 2 kappa theta / xi^2 = {p.feller_ratio:.2f} "
               f"({'satisfied' if p.feller_satisfied else 'violated: variance can hit 0'})")
    tab_price, tab_greeks, tab_smile = st.tabs(["Price", "Greeks", "Smile"])

    with tab_price:
        summary, probs = st.cache_data(price_view)(product, p, spy, n_paths, int(seed), ref)
        price, se = summary.loc[0, "value"], summary.loc[0, "SE"]
        st.metric("price (per 100 notional)", f"{price:.4f}", help=f"SE {se:.4f}")
        st.write(f"95% interval: {price - 1.96 * se:.4f} to {price + 1.96 * se:.4f}")
        st.dataframe(summary, hide_index=True)
        chart = alt.Chart(probs).mark_bar(color="#2a78d6", cornerRadiusEnd=4).encode(
            x=alt.X("date (years):O"), y=alt.Y("P(called):Q"),
            tooltip=["date (years)", alt.Tooltip("P(called):Q", format=".4f"),
                     alt.Tooltip("SE:Q", format=".4f")])
        st.altair_chart(chart, width="stretch")
        st.dataframe(probs, hide_index=True)

    with tab_greeks:
        st.write("Bump-and-revalue with common random numbers. Delta and gamma bump spot by "
                 "2% of the fixing, vega shifts sqrt(v0) and sqrt(theta) by 0.01, rho bumps r "
                 "by 1e-3. Each SE comes from paired per-path differences.")
        problem = greeks_problem(p, ref)
        if problem:
            st.warning(f"Greeks not available: {problem}.")
        elif st.button("compute Greeks"):
            try:
                st.dataframe(st.cache_data(greeks_view)(product, p, spy, n_paths, int(seed), ref),
                             hide_index=True)
            except ValueError as e:  # e.g. spot within the 2% bump of zero
                st.error(f"Greeks not available: {e}")

    with tab_smile:
        df = st.cache_data(smile_view)(p)
        st.write("Black-Scholes implied vols of semi-analytic Heston call prices. Quotes "
                 "whose time value is within the pricer's error estimate have no "
                 "resolvable implied vol and are left blank.")
        chart = alt.Chart(df.dropna()).mark_line(point=True, strokeWidth=2).encode(
            x="strike:Q", y=alt.Y("implied vol:Q", scale=alt.Scale(zero=False)),
            color=alt.Color("maturity:N", scale=alt.Scale(
                range=["#2a78d6", "#eb6834", "#1baf7a", "#eda100"])),
            tooltip=["maturity", "strike", alt.Tooltip("implied vol:Q", format=".4f")])
        st.altair_chart(chart, width="stretch")
        st.dataframe(df.pivot(index="strike", columns="maturity", values="implied vol"))


if __name__ == "__main__":
    main()
