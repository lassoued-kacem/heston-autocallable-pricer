"""hestonlib.black_scholes vs the independent oracle formula and model-free identities."""

import numpy as np
import pytest

import reference as R
from hestonlib.black_scholes import bs_price


@pytest.mark.parametrize("kind", ["call", "put"])
def test_matches_independent_formula(kind):
    """Two independent implementations of the same closed form: agree to 1e-12."""
    K = np.linspace(50, 150, 21)
    for T, sig in [(0.1, 0.1), (1.0, 0.2), (5.0, 0.6)]:
        a = bs_price(100.0, K, T, 0.03, 0.01, sig, kind)
        b = R.bs_price(100.0, K, T, 0.03, 0.01, sig, kind)
        assert np.max(np.abs(a - b)) < 1e-12


def test_zero_vol_is_discounted_intrinsic():
    """sigma = 0: the price is the discounted forward intrinsic value, exactly."""
    K = np.array([90.0, 110.0])
    T, r, q = 1.0, 0.03, 0.01
    fwd_intr = np.maximum(100 * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
    assert np.allclose(bs_price(100.0, K, T, r, q, 0.0, "call"), fwd_intr, atol=1e-14)


def test_bad_kind_raises():
    with pytest.raises(ValueError):
        bs_price(100.0, 100.0, 1.0, 0.0, 0.0, 0.2, "digital")


@pytest.mark.parametrize("bad", [dict(T=-1.0), dict(sigma=-0.1), dict(K=0.0)])
def test_invalid_inputs_raise(bad):
    """Negative T or sigma used to fall through to intrinsic value silently."""
    kw = dict(S0=100.0, K=100.0, T=1.0, r=0.0, q=0.0, sigma=0.2) | bad
    with pytest.raises(ValueError):
        bs_price(**kw)
