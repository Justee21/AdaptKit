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


class FeedbackTarget(str, Enum):
    BEHAVIOR = "behavior"
    ANSWER_CONTENT = "answer_content"
    TASK_CONTINUATION = "task_continuation"
    QUOTED_OR_META = "quoted_or_meta"
    UNRELATED = "unrelated"


class FeedbackSentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NONE = "none"


@dataclass(frozen=True, slots=True)
class PreferenceEvent:
    target: FeedbackTarget
    sentiment: FeedbackSentiment
    confidence: float = 1.0
    source: str = "implicit"
    reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.target, FeedbackTarget):
            raise ValidationError("target must be a FeedbackTarget")
        if not isinstance(self.sentiment, FeedbackSentiment):
            raise ValidationError("sentiment must be a FeedbackSentiment")
        if self.target is FeedbackTarget.BEHAVIOR:
            if self.sentiment is FeedbackSentiment.NONE:
                raise ValidationError("behavior feedback requires positive or negative sentiment")
        elif self.sentiment is not FeedbackSentiment.NONE:
            raise ValidationError("non-behavior targets require none sentiment")

        if isinstance(self.confidence, bool) or not isinstance(self.confidence, (int, float)):
            raise ValidationError("confidence must be a number")
        if not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValidationError("confidence must be finite and within [0, 1]")
        object.__setattr__(self, "confidence", float(self.confidence))

        if not isinstance(self.source, str) or not self.source.strip():
            raise ValidationError("source must be a non-empty string")
        if self.reason is not None and not isinstance(self.reason, str):
            raise ValidationError("reason must be a string or None")

    @property
    def has_feedback(self) -> bool:
        return self.target is FeedbackTarget.BEHAVIOR

    @property
    def reward(self) -> float | None:
        if self.target is not FeedbackTarget.BEHAVIOR:
            return None
        return 1.0 if self.sentiment is FeedbackSentiment.POSITIVE else -1.0


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
