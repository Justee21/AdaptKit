from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from .exceptions import ValidationError


class ObservationStatus(str, Enum):
    UPDATED = "updated"
    NO_FEEDBACK = "no_feedback"
    BELOW_THRESHOLD = "below_threshold"
    EVALUATOR_ERROR = "evaluator_error"


@dataclass(frozen=True, slots=True)
class PreferenceEvent:
    has_feedback: bool
    reward: float | None = None
    confidence: float = 1.0
    source: str = "implicit"
    reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.has_feedback, bool):
            raise ValidationError("has_feedback must be a boolean")
        if isinstance(self.confidence, bool) or not isinstance(self.confidence, (int, float)):
            raise ValidationError("confidence must be a number")
        if not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValidationError("confidence must be finite and within [0, 1]")
        object.__setattr__(self, "confidence", float(self.confidence))

        if not self.has_feedback:
            if self.reward is not None:
                raise ValidationError("reward must be omitted when has_feedback is false")
        else:
            if isinstance(self.reward, bool) or not isinstance(self.reward, (int, float)):
                raise ValidationError("reward must be a number when has_feedback is true")
            if not math.isfinite(self.reward) or not -1 <= self.reward <= 1:
                raise ValidationError("reward must be finite and within [-1, 1]")
            object.__setattr__(self, "reward", float(self.reward))

        if not isinstance(self.source, str) or not self.source.strip():
            raise ValidationError("source must be a non-empty string")
        if self.reason is not None and not isinstance(self.reason, str):
            raise ValidationError("reason must be a string or None")


@dataclass(frozen=True, slots=True)
class ObservationResult:
    status: ObservationStatus
    updated: bool
    event: PreferenceEvent | None = None
    error: str | None = None

    @property
    def effective_reward(self) -> float | None:
        if self.event is None or not self.event.has_feedback:
            return None
        assert self.event.reward is not None
        return self.event.reward * self.event.confidence
