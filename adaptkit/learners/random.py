from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from threading import Lock

from adaptkit.storage.base import StateUpdater

from .base import BaseLearner


class RandomLearner(BaseLearner):
    def __init__(self, store, *, seed: int | None = None) -> None:
        super().__init__(store)
        self._rng = random.Random(seed)
        self._rng_lock = Lock()

    @property
    def initial_state(self) -> Mapping[str, float]:
        return {"alpha": 1.0, "beta": 1.0}

    def select(
        self, states: Mapping[str, Mapping[str, float]], actions: Sequence[str]
    ) -> str:
        del states
        with self._rng_lock:
            return self._rng.choice(actions)

    def updater(self, reward: int) -> StateUpdater | None:
        if reward not in (-1, 1):
            raise ValueError("reward must be 1 or -1")
        return None
