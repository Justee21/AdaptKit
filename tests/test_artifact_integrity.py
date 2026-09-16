import json
from pathlib import Path
from typing import Any

import pytest

from adaptkit.prompting import PROMPT_PREFERENCE_SYSTEM_PROMPT
from benchmarks.integrity import canonical_sha256
from benchmarks.openai_evaluator_benchmark import (
    BATCH_INSTRUCTIONS,
    INSTRUCTIONS,
    load_examples,
    release_gate as feedback_release_gate,
)
from benchmarks.openai_prompt_routing_benchmark import (
    release_gate as prompt_release_gate,
)
from benchmarks.prompt_routing_cases import prompt_routing_examples

ROOT = Path(__file__).parents[1]
FORBIDDEN_CONTENT_FIELDS = {
    "api_key",
    "messages",
    "next_message",
    "previous_prompt",
    "previous_response",
    "prompt",
    "reasoning",
    "response",
    "user_message",
}


def _load(name: str) -> dict[str, Any]:
    return json.loads((ROOT / "artifacts" / name).read_text(encoding="utf-8"))


def _walk_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {
            nested
            for item in value.values()
            for nested in _walk_keys(item)
        }
    if isinstance(value, list):
        return {nested for item in value for nested in _walk_keys(item)}
    return set()


def _assert_intervals_match_point_metrics(run: dict[str, Any]) -> None:
    for name, interval in run["confidence_intervals_95"].items():
        total = interval["total"]
        metric = run["metrics"][name]
        if total == 0:
            assert metric is None
            assert interval["lower"] is None
            assert interval["upper"] is None
        else:
            assert metric == pytest.approx(interval["successes"] / total)


def test_feedback_artifact_matches_frozen_inputs_and_gates() -> None:
    summary = _load("real_evaluator_summary.json")
    rows = load_examples()
    metadata = summary["reproducibility"]
    assert metadata["dataset_sha256"] == canonical_sha256(rows)
    assert metadata["evaluator_prompt_sha256"] == canonical_sha256(
        {"single": INSTRUCTIONS, "batch": BATCH_INSTRUCTIONS}
    )
    assert summary["example_count"] == len(rows) == 120
    assert summary["calibration_example_count"] == sum(
        row["split"] == "calibration" for row in rows
    )
    assert summary["holdout_example_count"] == sum(
        row["split"] == "holdout" for row in rows
    )
    assert summary["frozen_learning_threshold"] == 0.95
    assert summary["runs_completed"] == len(summary["run_results"]) == 3
    assert summary["release_gate"]["passed"]
    for run in summary["run_results"]:
        assert run["release_gate"] == feedback_release_gate(run["metrics"])
        _assert_intervals_match_point_metrics(run)
    assert summary["release_gate"]["passed"] == all(
        run["release_gate"]["passed"] for run in summary["run_results"]
    )
    assert not (_walk_keys(summary) & FORBIDDEN_CONTENT_FIELDS)


def test_prompt_router_artifact_matches_frozen_inputs_and_gates() -> None:
    summary = _load("real_prompt_router_summary.json")
    rows = prompt_routing_examples()
    metadata = summary["reproducibility"]
    assert metadata["dataset_sha256"] == canonical_sha256(rows)
    assert metadata["evaluator_prompt_sha256"] == canonical_sha256(
        PROMPT_PREFERENCE_SYSTEM_PROMPT
    )
    assert summary["example_count"] == len(rows) == 80
    assert summary["calibration_example_count"] == sum(
        row["split"] == "calibration" for row in rows
    )
    assert summary["holdout_example_count"] == sum(
        row["split"] == "holdout" for row in rows
    )
    assert summary["frozen_routing_threshold"] == 0.70
    assert summary["runs_completed"] == len(summary["run_results"]) == 3
    assert summary["release_gate"]["passed"]
    for run in summary["run_results"]:
        assert run["release_gate"] == prompt_release_gate(run["metrics"])
        _assert_intervals_match_point_metrics(run)
    assert summary["release_gate"]["passed"] == all(
        run["release_gate"]["passed"] for run in summary["run_results"]
    )
    assert not (_walk_keys(summary) & FORBIDDEN_CONTENT_FIELDS)
