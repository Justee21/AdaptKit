from __future__ import annotations

import asyncio
from abc import ABC
from typing import Any

from adaptkit.events import PreferenceEvent
from adaptkit.exceptions import ConfigurationError


class FeedbackExtractor(ABC):
    """Pluggable interface for interpreting a user's reaction."""

    def extract(self, **interaction: Any) -> PreferenceEvent:
        raise ConfigurationError("this extractor does not support synchronous extraction")

    async def aextract(self, **interaction: Any) -> PreferenceEvent:
        return await asyncio.to_thread(self.extract, **interaction)
