"""Optional Mem0-to-AdaptKit cold-start example.

Install only this provider integration with ``pip install 'adaptkit[memory-mem0]'``.
"""

from __future__ import annotations

import importlib
import os
from typing import Any

from adaptkit import Profile, cold_start_priors

ACTIONS = ("direct_solution", "guided_learning")


def build_mem0_client() -> Any:
    """Create the optional hosted Mem0 client without making a network call."""
    try:
        module = importlib.import_module("mem0")
        client_type = module.MemoryClient
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "Mem0 support requires: pip install 'adaptkit[memory-mem0]'"
        ) from exc
    api_key = os.getenv("MEM0_API_KEY")
    if not api_key or api_key.startswith("insert-"):
        raise RuntimeError("set MEM0_API_KEY before creating the Mem0 client")
    return client_type(api_key=api_key)


def retrieve_durable_preferences(client: Any, *, user_id: str) -> list[str]:
    """Retrieve candidate durable preferences; callers should avoid logging them."""
    response = client.search(
        query="durable response-style and learning preferences",
        filters={"user_id": user_id},
        top_k=5,
    )
    if isinstance(response, dict):
        rows = response.get("results", [])
    elif isinstance(response, list):
        rows = response
    else:
        rows = []
    if not isinstance(rows, list):
        rows = []
    return [
        row["memory"]
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("memory"), str)
    ]


def map_memory_to_action(memories: list[str]) -> str | None:
    """Application-owned allowlist mapping, not arbitrary text-to-numeric trust."""
    durable_phrases = {
        "the user generally prefers guided explanations": "guided_learning",
        "the user generally prefers direct solutions": "direct_solution",
    }
    normalized = {memory.strip().casefold() for memory in memories}
    matches = {action for phrase, action in durable_phrases.items() if phrase in normalized}
    return matches.pop() if len(matches) == 1 else None


def profile_from_mem0(client: Any, *, user_id: str, store: Any = None) -> Profile:
    """Cold-start once; already-materialized state overrides the candidate priors."""
    action = map_memory_to_action(
        retrieve_durable_preferences(client, user_id=user_id)
    )
    return Profile(
        user_id=user_id,
        actions=ACTIONS,
        action_priors=cold_start_priors(actions=ACTIONS, preferred_action=action),
        store=store,
    )
