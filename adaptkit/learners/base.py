from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence

from adaptkit.storage import StateStore
from adaptkit.storage.base import StateUpdater


class BaseLearner(ABC):
    def __init__(self, store: StateStore) -> None:
        self.store = store

    @property
    @abstractmethod
    def initial_state(self) -> Mapping[str, float]:
        """Return detached default state for an unseen action."""

    @abstractmethod
    def select(self, states: Mapping[str, Mapping[str, float]], actions: Sequence[str]) -> str:
        """Select one action from a consistent policy snapshot."""

    def choose(self, user_id: str, context: str, actions: Sequence[str]) -> tuple[str, int]:
        snapshot = self.store.policy_snapshot(user_id, context, actions, self.initial_state)
        return self.select(snapshot.states, actions), snapshot.version

    @abstractmethod
    def updater(self, reward: int) -> StateUpdater | None:
        """Return a state mutation for a reward, or None for a non-learning baseline."""
