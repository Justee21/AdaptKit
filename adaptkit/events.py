from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

from .exceptions import ValidationError


class ObservationStatus(str, Enum):
    UPDATED = "updated"
    NO_PREFERENCE_SIGNAL = "no_preference_signal"
    BELOW_THRESHOLD = "below_threshold"
    SHADOW = "shadow"
    DUPLICATE = "duplicate"
    LEARNER_IGNORED = "learner_ignored"
    EVALUATOR_ERROR = "evaluator_error"


class LearningMode(str, Enum):
    ACTIVE = "active"
    SHADOW = "shadow"


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
class Decision:
    decision_id: str
    user_id: str
    context: str
    action: str
    policy_version: int
    created_at: datetime

    def __post_init__(self) -> None:
        for name in ("decision_id", "user_id", "context", "action"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValidationError(f"{name} must be a non-empty string")
        if isinstance(self.policy_version, bool) or not isinstance(self.policy_version, int):
            raise ValidationError("policy_version must be an integer")
        if self.policy_version < 0:
            raise ValidationError("policy_version must be non-negative")
        if not isinstance(self.created_at, datetime) or self.created_at.tzinfo is None:
            raise ValidationError("created_at must be a timezone-aware datetime")
        object.__setattr__(self, "created_at", self.created_at.astimezone(timezone.utc))


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
    def has_preference_signal(self) -> bool:
        return self.target is FeedbackTarget.BEHAVIOR

    @property
    def reward(self) -> int | None:
        if not self.has_preference_signal:
            return None
        return 1 if self.sentiment is FeedbackSentiment.POSITIVE else -1


@dataclass(frozen=True, slots=True)
class ObservationResult:
    status: ObservationStatus
    updated: bool
    recorded: bool = False
    duplicate: bool = False
    event: PreferenceEvent | None = None
    observation_id: str | None = None
    policy_version_before: int | None = None
    policy_version_after: int | None = None
    error: str | None = None

    @property
    def effective_reward(self) -> float | None:
        if self.event is None or not self.event.has_preference_signal:
            return None
        assert self.event.reward is not None
        return self.event.reward * self.event.confidence


@dataclass(frozen=True, slots=True)
class LifecycleEvent:
    name: str
    user_id: str
    context: str
    action: str | None
    decision_id: str | None
    status: str
    policy_version_before: int | None
    policy_version_after: int | None
    created_at: datetime
