"""Heston model parameters."""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class HestonParams:
    """Market and Heston parameters.

    dS = (r - q) S dt + sqrt(v) S dW1
    dv = kappa (theta - v) dt + xi sqrt(v) dW2,   corr(dW1, dW2) = rho
    """

    S0: float = 100.0
    r: float = 0.03
    q: float = 0.0
    v0: float = 0.04
    kappa: float = 2.0
    theta: float = 0.04
    xi: float = 0.3
    rho: float = -0.7

    def __post_init__(self):
        if self.S0 <= 0 or self.v0 < 0 or self.kappa <= 0 or self.theta < 0 or self.xi <= 0:
            raise ValueError(f"invalid Heston parameters: {self}")
        if not -1.0 < self.rho < 1.0:
            raise ValueError("rho must lie in (-1, 1)")

    def replace(self, **kw) -> "HestonParams":
        return replace(self, **kw)

    @property
    def feller_ratio(self) -> float:
        """2 kappa theta / xi^2; the Feller condition holds (v stays > 0) iff ratio >= 1."""
        return 2.0 * self.kappa * self.theta / self.xi**2

    @property
    def feller_satisfied(self) -> bool:
        return self.feller_ratio >= 1.0


REFERENCE = HestonParams()
