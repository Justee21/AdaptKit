from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from adaptkit.storage import StateStore


class BaseLearner(ABC):
    def __init__(self, store: StateStore) -> None:
        self.store = store

    @abstractmethod
    def choose(self, user_id: str, context: str, actions: Sequence[str]) -> str:
        """Choose one action from actions."""

    @abstractmethod
    def update(self, user_id: str, context: str, action: str, reward: int) -> None:
        """Apply a Bernoulli success (1) or failure (-1)."""
