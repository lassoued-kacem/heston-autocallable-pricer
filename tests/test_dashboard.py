"""Dashboard tests: the pure view functions, and the app run headless by Streamlit's AppTest.

The views are thin wrappers, so the risk is in the wiring (barriers in percent of the
fixing, spot vs fixing, the step grid). That is checked against the exact BS oracle.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

import reference as R
from conftest import REF
from hestonlib.black_scholes import bs_vega
from hestonlib.params import HestonParams

APP = Path(__file__).resolve().parents[1] / "app" / "dashboard.py"
_spec = importlib.util.spec_from_file_location("dashboard", APP)
D = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(D)


def test_product_dates_and_step_grid():
    pr = D.make_product(2, 4, 1.0, 0.6, 0.05)
    assert pr.obs_times == pytest.approx((0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0))
    assert D.n_steps_for(pr, 12) == 24
    with pytest.raises(ValueError):
        D.n_steps_for(pr, 6)  # 6 steps a year cannot hit quarterly dates


def test_price_view_wiring_matches_bs_oracle():
    """BS limit (xi = 1e-8), spot 97, fixing 100, semi-annual dates over 2 years, barrier
    95%, KI 65%, coupon 4% per date. log-Euler is exact at the dates, so the price, and each
    call probability, is within 4 SE of the exact MVN oracle with S_ref = 100. A wiring
    slip (barrier vs KI, spot vs fixing, percent scaling) moves the price by far more."""
    p = HestonParams(**dict(REF, xi=1e-8, S0=97.0))
    pr = D.make_product(2, 2, 0.95, 0.65, 0.04)
    summary, probs = D.price_view(pr, p, 12, 400_000, 8, 100.0)
    price, se = summary.loc[0, "value"], summary.loc[0, "SE"]
    truth = R.bs_autocallable(97.0, REF["r"], REF["q"], 0.2, pr.obs_times, 0.95, 0.65, 0.04,
                              100.0, S_ref=100.0)
    assert abs(price - truth) < 4 * se, (price, truth, se)
    assert len(probs) == 4 and np.all(probs["SE"] > 0)
    assert summary["SE"].notna().all()  # every MC figure carries an SE


def test_greeks_view_reports_each_greek_with_se(ref_params):
    pr = D.make_product(1, 1, 1.0, 0.6, 0.08)
    df = D.greeks_view(pr, ref_params, 12, 20_000, 1, 100.0)
    assert list(df["Greek"]) == ["price", "delta", "gamma", "vega", "rho"]
    assert np.all(df["SE"] > 0)


def test_smile_view_is_flat_in_bs_limit():
    """xi = 1e-8, v0 = theta: the Heston smile is flat at 0.2. At T = 1 the price gap is below
    1e-5 (test_analytic), so the IV gap is below 1e-5 / vega on strikes 80..120."""
    p = HestonParams(**dict(REF, xi=1e-8))
    df = D.smile_view(p, maturities=(1.0,), moneyness=np.array([0.8, 1.0, 1.2]))
    vega = bs_vega(100.0, df["strike"].to_numpy(), 1.0, REF["r"], REF["q"], 0.2)
    assert np.all(np.abs(df["implied vol"].to_numpy() - 0.2) < 1e-5 / vega)


def test_app_runs_headless():
    """The whole script under Streamlit's test harness: no exception, a price metric, three
    tabs, and the Greeks button produces a table."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=180)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    assert at.metric[0].label.startswith("price")
    assert len(at.tabs) == 3
    n_tables = len(at.dataframe)
    at.button[0].click().run()
    assert not at.exception
    assert len(at.dataframe) == n_tables + 1


def test_greeks_unavailable_state_warns_instead_of_crashing():
    """Regression (stage 7 review): spot 1 with fixing 100 is within the 2% spot bump of zero.
    The Greeks tab must explain that instead of raising from autocallable_greeks."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=180)
    at.run()
    at.sidebar.number_input[0].set_value(1.0).run()  # spot
    assert not at.exception, [e.value for e in at.exception]
    assert any("Greeks not available" in w.value for w in at.warning)
    assert D.greeks_problem(HestonParams(**dict(REF, S0=1.0)), 100.0) is not None
    assert D.greeks_problem(HestonParams(**REF), 100.0) is None
