from __future__ import annotations

import math
from dataclasses import dataclass

from .exceptions import ValidationError


def _positive(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{name} must be a number")
    if not math.isfinite(value) or value <= 0:
        raise ValidationError(f"{name} must be finite and positive")
    return float(value)


@dataclass(frozen=True, slots=True)
class BetaPrior:
    """Positive Beta-distribution parameters for one action's cold start."""

    alpha: float
    beta: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "alpha", _positive(self.alpha, "alpha"))
        object.__setattr__(self, "beta", _positive(self.beta, "beta"))

    @classmethod
    def from_mean_strength(cls, mean: float, strength: float) -> BetaPrior:
        if isinstance(mean, bool) or not isinstance(mean, (int, float)):
            raise ValidationError("mean must be a number")
        if not math.isfinite(mean) or not 0 < mean < 1:
            raise ValidationError("mean must be finite and within (0, 1)")
        normalized_strength = _positive(strength, "strength")
        return cls(float(mean) * normalized_strength, (1 - float(mean)) * normalized_strength)

    def state(self) -> dict[str, float]:
        return {"alpha": self.alpha, "beta": self.beta}
