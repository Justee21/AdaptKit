from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from dataclasses import field
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


class SelectionSource(str, Enum):
    POLICY = "policy"
    PROMPT_OVERRIDE = "prompt_override"


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


_SIGNAL_TYPE = re.compile(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$")
MAX_METADATA_BYTES = 8 * 1024
MAX_METADATA_DEPTH = 4


def _validated_json(value: object, *, depth: int = 0) -> object:
    if depth > MAX_METADATA_DEPTH:
        raise ValidationError(f"metadata nesting must not exceed {MAX_METADATA_DEPTH}")
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValidationError("metadata numbers must be finite")
        return value
    if isinstance(value, list):
        return [_validated_json(item, depth=depth + 1) for item in value]
    if isinstance(value, dict):
        normalized: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ValidationError("metadata keys must be non-empty strings")
            normalized[key] = _validated_json(item, depth=depth + 1)
        return normalized
    raise ValidationError("metadata must contain only JSON-compatible values")


def validated_metadata(value: object) -> dict[str, object]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValidationError("metadata must be a JSON object")
    normalized = _validated_json(value)
    assert isinstance(normalized, dict)
    encoded = json.dumps(normalized, allow_nan=False, separators=(",", ":"), sort_keys=True)
    if len(encoded.encode("utf-8")) > MAX_METADATA_BYTES:
        raise ValidationError(f"metadata must not exceed {MAX_METADATA_BYTES} encoded bytes")
    return normalized


@dataclass(frozen=True, slots=True)
class Decision:
    """Immutable record of the policy action selected for one response."""

    decision_id: str
    user_id: str
    context: str
    action: str
    policy_version: int
    created_at: datetime
    selection_source: SelectionSource = SelectionSource.POLICY
    selection_confidence: float | None = None

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
        if not isinstance(self.selection_source, SelectionSource):
            raise ValidationError("selection_source must be a SelectionSource")
        if self.selection_confidence is not None:
            if (
                isinstance(self.selection_confidence, bool)
                or not isinstance(self.selection_confidence, (int, float))
                or not math.isfinite(self.selection_confidence)
                or not 0 <= self.selection_confidence <= 1
            ):
                raise ValidationError("selection_confidence must be finite and within [0, 1]")
            object.__setattr__(self, "selection_confidence", float(self.selection_confidence))
        if (
            self.selection_source is SelectionSource.PROMPT_OVERRIDE
            and self.selection_confidence is None
        ):
            raise ValidationError("prompt overrides require selection_confidence")


@dataclass(frozen=True, slots=True)
class PromptPreference:
    """Transient current-prompt preference for an available action."""

    action: str | None
    confidence: float
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.action is not None and (
            not isinstance(self.action, str) or not self.action.strip()
        ):
            raise ValidationError("action must be a non-empty string or None")
        if (
            isinstance(self.confidence, bool)
            or not isinstance(self.confidence, (int, float))
            or not math.isfinite(self.confidence)
            or not 0 <= self.confidence <= 1
        ):
            raise ValidationError("confidence must be finite and within [0, 1]")
        object.__setattr__(self, "confidence", float(self.confidence))
        if self.reason is not None and not isinstance(self.reason, str):
            raise ValidationError("reason must be a string or None")


@dataclass(frozen=True, slots=True)
class PreferenceEvent:
    """Validated selected-action-relative feedback classification."""

    target: FeedbackTarget
    sentiment: FeedbackSentiment
    confidence: float = 1.0
    source: str = "implicit"
    reason: str | None = None
    signal_type: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

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
        if self.signal_type is not None:
            if (
                not isinstance(self.signal_type, str)
                or len(self.signal_type) > 128
                or _SIGNAL_TYPE.fullmatch(self.signal_type) is None
            ):
                raise ValidationError(
                    "signal_type must be an open namespaced lowercase identifier"
                )
        if self.source == "passive" and self.signal_type is None:
            raise ValidationError("passive feedback requires signal_type")
        object.__setattr__(self, "metadata", validated_metadata(self.metadata))

    @property
    def has_preference_signal(self) -> bool:
        return self.target is FeedbackTarget.BEHAVIOR

    @property
    def reward(self) -> int | None:
        if not self.has_preference_signal:
            return None
        return 1 if self.sentiment is FeedbackSentiment.POSITIVE else 0


@dataclass(frozen=True, slots=True)
class ObservationResult:
    """Outcome of recording and optionally applying a feedback observation."""

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
        """Return the binary reward applied by this call, never confidence-weighted."""
        if not self.updated or self.event is None or not self.event.has_preference_signal:
            return None
        assert self.event.reward is not None
        return float(self.event.reward)


@dataclass(frozen=True, slots=True)
class LifecycleEvent:
    """Privacy-safe structured event emitted to an optional application hook."""

    name: str
    user_id: str
    context: str
    action: str | None
    decision_id: str | None
    status: str
    policy_version_before: int | None
    policy_version_after: int | None
    created_at: datetime
