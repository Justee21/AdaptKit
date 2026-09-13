from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from threading import Lock

from adaptkit.storage.base import StateUpdater

from .base import BaseLearner


class ThompsonLearner(BaseLearner):
    def __init__(
        self,
        store,
        *,
        seed: int | None = None,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
    ) -> None:
        super().__init__(store)
        self._rng = random.Random(seed)
        self._rng_lock = Lock()
        self._initial_state = {"alpha": prior_alpha, "beta": prior_beta}

    @property
    def initial_state(self) -> Mapping[str, float]:
        return dict(self._initial_state)

    def select(
        self, states: Mapping[str, Mapping[str, float]], actions: Sequence[str]
    ) -> str:
        with self._rng_lock:
            samples = {
                action: self._rng.betavariate(
                    states[action]["alpha"], states[action]["beta"]
                )
                for action in actions
            }
        return max(actions, key=samples.__getitem__)

    def updater(self, reward: int) -> StateUpdater:
        if reward not in (-1, 1):
            raise ValueError("reward must be 1 or -1")

        def apply(state: dict[str, float]) -> None:
            if reward == 1:
                state["alpha"] += 1
            else:
                state["beta"] += 1

        return apply
