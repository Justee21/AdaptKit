from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from adaptkit.events import Decision, LearningMode, ObservationStatus, PreferenceEvent
from adaptkit.exceptions import ValidationError

StateUpdater = Callable[[dict[str, float]], None]


def action_set_identity(actions: Sequence[str]) -> tuple[tuple[str, ...], str, str]:
    canonical = tuple(sorted(actions))
    serialized = json.dumps(canonical, separators=(",", ":"))
    fingerprint = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return canonical, serialized, fingerprint


def same_persisted_event(first: PreferenceEvent, second: PreferenceEvent) -> bool:
    return (
        first.target == second.target
        and first.sentiment == second.sentiment
        and first.confidence == second.confidence
        and first.source == second.source
        and first.signal_type == second.signal_type
        and first.metadata == second.metadata
    )


def same_observation_request(
    observation: StoredObservation,
    event: PreferenceEvent,
    learning_mode: LearningMode,
    status: ObservationStatus,
) -> bool:
    return (
        same_persisted_event(observation.event, event)
        and observation.learning_mode is learning_mode
        and observation.status is status
    )


def canonical_operation_identity(identity: Mapping[str, str]) -> str:
    if not identity or any(
        not isinstance(key, str)
        or not key.strip()
        or not isinstance(value, str)
        or not value.strip()
        for key, value in identity.items()
    ):
        raise ValidationError(
            "operation_identity must map non-empty strings to non-empty strings"
        )
    return json.dumps(dict(identity), separators=(",", ":"), sort_keys=True)


@dataclass(frozen=True, slots=True)
class PolicySnapshot:
    version: int
    states: dict[str, dict[str, float]]


@dataclass(frozen=True, slots=True)
class StoredObservation:
    observation_id: str
    decision_id: str
    idempotency_key: str
    event: PreferenceEvent
    learning_mode: LearningMode
    status: ObservationStatus
    applied: bool
    policy_version_before: int
    policy_version_after: int
    created_at: datetime
    duplicate: bool = False


@dataclass(frozen=True, slots=True)
class PolicyUpdateResult:
    updated: bool
    duplicate: bool
    policy_version_before: int
    policy_version_after: int


class StateStore(ABC):
    schema_version = 2

    @abstractmethod
    def policy_snapshot(
        self,
        user_id: str,
        context: str,
        actions: Sequence[str],
        initial_states: Mapping[str, Mapping[str, float]],
    ) -> PolicySnapshot:
        """Return a consistent policy version and detached state for every action."""

    @abstractmethod
    def create_decision(self, decision: Decision) -> None:
        """Persist structured decision metadata without prompt content."""

    @abstractmethod
    def get_decision(self, decision_id: str) -> Decision | None:
        """Return one persisted decision."""

    @abstractmethod
    def find_observation(
        self, decision_id: str, idempotency_key: str
    ) -> StoredObservation | None:
        """Find a previously committed observation."""

    @abstractmethod
    def record_observation(
        self,
        *,
        decision: Decision,
        observation_id: str,
        idempotency_key: str,
        event: PreferenceEvent,
        learning_mode: LearningMode,
        status: ObservationStatus,
        initial_state: Mapping[str, float],
        updater: StateUpdater | None,
    ) -> StoredObservation:
        """Atomically insert an observation and apply at most one policy update."""

    @abstractmethod
    def atomic_policy_update(
        self,
        *,
        user_id: str,
        context: str,
        idempotency_key: str,
        operation_identity: Mapping[str, str],
        initial_states: Mapping[str, Mapping[str, float]],
        updaters: Mapping[str, StateUpdater],
    ) -> PolicyUpdateResult:
        """Atomically apply one identified update across one or more actions."""

    @abstractmethod
    def snapshot(self, user_id: str) -> dict[str, dict[str, dict[str, float]]]:
        """Return detached numeric learner state for a user."""

    @abstractmethod
    def export_user(self, user_id: str) -> dict[str, Any]:
        """Export structured state without conversation or prompt content."""

    @abstractmethod
    def delete_user(self, user_id: str) -> None:
        """Delete all persisted state belonging to a user."""
