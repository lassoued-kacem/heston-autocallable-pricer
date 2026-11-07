"""Shared fixtures. The reference parameter set is restated here from CLAUDE.md."""

import pytest

from hestonlib.params import HestonParams

# Reference parameters (CLAUDE.md): restated literally rather than imported from the package.
REF = dict(S0=100.0, r=0.03, q=0.0, v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)

# A harsh regime: Feller strongly violated, rho near -1, high vol-of-vol. This is where the
# original Heston CF formulation has branch-cut problems.
STRESS = dict(S0=100.0, r=0.02, q=0.01, v0=0.09, kappa=0.5, theta=0.09, xi=1.0, rho=-0.95)


def model_args(d):
    """Positional args (r, q, v0, kappa, theta, xi, rho) for tests/reference.py functions."""
    return d["r"], d["q"], d["v0"], d["kappa"], d["theta"], d["xi"], d["rho"]


@pytest.fixture
def ref_params():
    return HestonParams(**REF)


@pytest.fixture
def stress_params():
    return HestonParams(**STRESS)
