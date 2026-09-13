from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from adaptkit.events import Decision, LearningMode, ObservationStatus, PreferenceEvent

StateUpdater = Callable[[dict[str, float]], None]


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
    schema_version = 1

    @abstractmethod
    def policy_snapshot(
        self,
        user_id: str,
        context: str,
        actions: Sequence[str],
        initial_state: Mapping[str, float],
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
        initial_state: Mapping[str, float],
        updaters: Mapping[str, StateUpdater],
    ) -> PolicyUpdateResult:
        """Atomically apply one logical explicit update across one or more actions."""

    @abstractmethod
    def snapshot(self, user_id: str) -> dict[str, dict[str, dict[str, float]]]:
        """Return detached numeric learner state for a user."""

    @abstractmethod
    def export_user(self, user_id: str) -> dict[str, Any]:
        """Export structured state without conversation or prompt content."""

    @abstractmethod
    def delete_user(self, user_id: str) -> None:
        """Delete all persisted state belonging to a user."""
