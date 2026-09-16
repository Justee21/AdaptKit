from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from threading import Lock

from adaptkit.storage.base import StateUpdater
from adaptkit.storage import StateStore

from .base import BaseLearner


class RandomLearner(BaseLearner):
    def __init__(self, store: StateStore, *, seed: int | None = None) -> None:
        super().__init__(store)
        self._rng = random.Random(seed)
        self._rng_lock = Lock()

    def initial_state(self, action: str) -> Mapping[str, float]:
        del action
        return {"alpha": 1.0, "beta": 1.0}

    def select(
        self, states: Mapping[str, Mapping[str, float]], actions: Sequence[str]
    ) -> str:
        del states
        with self._rng_lock:
            return self._rng.choice(actions)

    def updater(self, reward: int) -> StateUpdater | None:
        if isinstance(reward, bool) or reward not in (0, 1):
            raise ValueError("reward must be 0 or 1")
        return None
