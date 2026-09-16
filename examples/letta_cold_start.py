"""Optional Letta-to-AdaptKit cold-start example.

Install only this provider integration with ``pip install 'adaptkit[memory-letta]'``.
"""

from __future__ import annotations

import importlib
import os
from typing import Any

from adaptkit import Profile, cold_start_priors

ACTIONS = ("direct_solution", "guided_learning")


def build_letta_client() -> Any:
    """Create the optional Letta client without making a network call."""
    try:
        module = importlib.import_module("letta_client")
        client_type = module.Letta
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "Letta support requires: pip install 'adaptkit[memory-letta]'"
        ) from exc
    api_key = os.getenv("LETTA_API_KEY")
    if not api_key or api_key.startswith("insert-"):
        raise RuntimeError("set LETTA_API_KEY before creating the Letta client")
    return client_type(api_key=api_key)


def retrieve_durable_preference(client: Any, *, agent_id: str) -> str:
    """Read the agent's human memory block; callers should avoid logging it."""
    block = client.agents.blocks.retrieve(agent_id=agent_id, block_label="human")
    value = getattr(block, "value", "")
    return value if isinstance(value, str) else ""


def map_memory_to_action(memory: str) -> str | None:
    """Application-owned allowlist mapping, not arbitrary text-to-numeric trust."""
    durable_phrases = {
        "the user generally prefers guided explanations": "guided_learning",
        "the user generally prefers direct solutions": "direct_solution",
    }
    normalized_lines = {line.strip().casefold() for line in memory.splitlines()}
    matches = {
        action for phrase, action in durable_phrases.items() if phrase in normalized_lines
    }
    return matches.pop() if len(matches) == 1 else None


def profile_from_letta(
    client: Any, *, agent_id: str, user_id: str, store: Any = None
) -> Profile:
    """Cold-start once; already-materialized state overrides the candidate priors."""
    action = map_memory_to_action(
        retrieve_durable_preference(client, agent_id=agent_id)
    )
    return Profile(
        user_id=user_id,
        actions=ACTIONS,
        action_priors=cold_start_priors(actions=ACTIONS, preferred_action=action),
        store=store,
    )
