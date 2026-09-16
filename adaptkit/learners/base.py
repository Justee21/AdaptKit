from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence

from adaptkit.storage import PolicySnapshot, StateStore
from adaptkit.storage.base import StateUpdater


class BaseLearner(ABC):
    def __init__(self, store: StateStore) -> None:
        self.store = store

    @abstractmethod
    def initial_state(self, action: str) -> Mapping[str, float]:
        """Return detached default state for one unseen action."""

    def initial_states(self, actions: Sequence[str]) -> dict[str, Mapping[str, float]]:
        return {action: self.initial_state(action) for action in actions}

    @abstractmethod
    def select(self, states: Mapping[str, Mapping[str, float]], actions: Sequence[str]) -> str:
        """Select one action from a consistent policy snapshot."""

    def choose(self, user_id: str, context: str, actions: Sequence[str]) -> tuple[str, int]:
        snapshot = self.snapshot(user_id, context, actions)
        return self.select(snapshot.states, actions), snapshot.version

    def snapshot(
        self, user_id: str, context: str, actions: Sequence[str]
    ) -> PolicySnapshot:
        return self.store.policy_snapshot(
            user_id, context, actions, self.initial_states(actions)
        )

    @abstractmethod
    def updater(self, reward: int) -> StateUpdater | None:
        """Return a state mutation for binary reward 0 or 1, or None for a baseline."""
