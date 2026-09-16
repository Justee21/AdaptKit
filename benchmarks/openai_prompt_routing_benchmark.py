from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, cast

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field

from adaptkit.prompting import PROMPT_PREFERENCE_SYSTEM_PROMPT
from benchmarks.openai_evaluator_benchmark import (
    LEARNING_THRESHOLDS,
    _add_usage,
    aggregate_metrics,
    safe_error_metadata,
)
from benchmarks.integrity import canonical_sha256, generated_at
from benchmarks.prompt_routing_cases import (
    ACTION_DESCRIPTIONS,
    prompt_routing_examples,
)
from benchmarks.statistics import wilson_interval

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts/real_prompt_router_summary.json"


class PromptPrediction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: str | None = Field(
        description="An available action key, or null when there is no clear cue."
    )
    confidence: float = Field(ge=0, le=1)
    reason: str | None


def benchmark_input(row: dict[str, Any]) -> str:
    return "Interaction data:\n" + json.dumps(
        {
            "context": row["context"],
            "current_user_prompt": row["prompt"],
            "available_actions": list(ACTION_DESCRIPTIONS),
            "action_descriptions": ACTION_DESCRIPTIONS,
        }
    )


def score_prompt_routing(
    rows: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    *,
    threshold: float,
) -> dict[str, float | int | None]:
    explicit = no_cue = correct_routes = applied = correct_applied = 0
    wrong_action = false_no_cue = invalid = 0
    for row, prediction in zip(rows, predictions):
        expected = row["expected_action"]
        explicit += expected is not None
        no_cue += expected is None
        if prediction.get("invalid"):
            invalid += 1
            continue
        routed = (
            prediction["action"] is not None
            and prediction["confidence"] >= threshold
        )
        applied += routed
        correct = routed and prediction["action"] == expected
        correct_applied += correct
        correct_routes += expected is not None and correct
        wrong_action += routed and expected is not None and prediction["action"] != expected
        false_no_cue += routed and expected is None
    return {
        "examples": len(rows),
        "routing_threshold": threshold,
        "explicit_cue_examples": explicit,
        "no_cue_examples": no_cue,
        "prompt_cue_routing_accuracy": (
            correct_routes / explicit if explicit else None
        ),
        "routed_action_precision": (
            correct_applied / applied if applied else None
        ),
        "wrong_action_overrides": wrong_action,
        "wrong_action_override_rate": wrong_action / explicit if explicit else None,
        "no_cue_false_override_rate": false_no_cue / no_cue if no_cue else None,
        "invalid_outputs": invalid,
    }


def confidence_intervals(
    rows: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    *,
    threshold: float,
) -> dict[str, dict[str, float | int | None]]:
    explicit = no_cue = applied = correct_routes = correct_applied = wrong = false_no_cue = 0
    for row, prediction in zip(rows, predictions):
        expected = row["expected_action"]
        explicit += expected is not None
        no_cue += expected is None
        if prediction.get("invalid"):
            continue
        routed = prediction["action"] is not None and prediction["confidence"] >= threshold
        applied += routed
        correct = routed and prediction["action"] == expected
        correct_applied += correct
        correct_routes += expected is not None and correct
        wrong += routed and expected is not None and prediction["action"] != expected
        false_no_cue += routed and expected is None
    return {
        "prompt_cue_routing_accuracy": wilson_interval(correct_routes, explicit),
        "routed_action_precision": wilson_interval(correct_applied, applied),
        "wrong_action_override_rate": wilson_interval(wrong, explicit),
        "no_cue_false_override_rate": wilson_interval(false_no_cue, no_cue),
    }


def choose_threshold(
    rows: list[dict[str, Any]], runs: list[list[dict[str, Any]]]
) -> tuple[float, dict[str, dict[str, float | int | None]]]:
    calibration_rows = [row for row in rows if row["split"] == "calibration"]
    predictions = [
        prediction
        for run in runs
        for row, prediction in zip(rows, run)
        if row["split"] == "calibration"
    ]
    repeated_rows = calibration_rows * len(runs)
    scores = {
        f"{threshold:.2f}": score_prompt_routing(
            repeated_rows, predictions, threshold=threshold
        )
        for threshold in LEARNING_THRESHOLDS
    }
    for threshold in LEARNING_THRESHOLDS:
        metrics = scores[f"{threshold:.2f}"]
        if (
            _at_least(metrics["prompt_cue_routing_accuracy"], 0.95)
            and _at_least(metrics["routed_action_precision"], 0.95)
            and metrics["wrong_action_overrides"] == 0
            and _at_most(metrics["no_cue_false_override_rate"], 0.025)
        ):
            return threshold, scores
    return LEARNING_THRESHOLDS[-1], scores


def _at_least(value: float | int | None, minimum: float) -> bool:
    return value is not None and value >= minimum


def _at_most(value: float | int | None, maximum: float) -> bool:
    return value is not None and value <= maximum


def release_gate(metrics: dict[str, float | int | None]) -> dict[str, bool]:
    checks = {
        "prompt_cue_routing_accuracy": _at_least(
            metrics["prompt_cue_routing_accuracy"], 0.95
        ),
        "zero_wrong_action_overrides": metrics["wrong_action_overrides"] == 0,
        "no_cue_false_override_rate": _at_most(
            metrics["no_cue_false_override_rate"], 0.025
        ),
        "invalid_output_rate": (
            metrics["invalid_outputs"] is not None
            and metrics["examples"] is not None
            and metrics["examples"] > 0
            and metrics["invalid_outputs"] / metrics["examples"] <= 0.01
        ),
    }
    return {"passed": all(checks.values()), **checks}


def run_once(
    client: OpenAI, model: str, rows: list[dict[str, Any]], *, workers: int = 1
) -> tuple[list[dict[str, Any]], dict[str, int | None], str, list[dict[str, Any]]]:
    predictions: list[dict[str, Any]] = []
    usage: dict[str, int | None] = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }
    errors: list[dict[str, Any]] = []
    response_model = model

    def evaluate(job: tuple[int, dict[str, Any]]) -> dict[str, Any]:
        index, row = job
        try:
            response = client.responses.parse(
                model=model,
                instructions=PROMPT_PREFERENCE_SYSTEM_PROMPT,
                input=benchmark_input(row),
                text_format=PromptPrediction,
                store=False,
                max_output_tokens=800,
                reasoning={"effort": "low"},
            )
            parsed = response.output_parsed
            if parsed is None or (
                parsed.action is not None
                and parsed.action not in ACTION_DESCRIPTIONS
            ):
                return {
                    "prediction": {"invalid": True},
                    "error": {
                        "example_id": row["example_id"],
                        "example_index": index,
                        "error_type": "InvalidAction",
                    },
                    "usage": response.usage,
                    "model": response.model,
                }
            return {
                "prediction": {
                    "action": parsed.action,
                    "confidence": parsed.confidence,
                },
                "error": None,
                "usage": response.usage,
                "model": response.model,
            }
        except Exception as exc:
            return {
                "prediction": {"invalid": True},
                "error": {
                    "example_id": row["example_id"],
                    "example_index": index,
                    **safe_error_metadata(exc),
                },
                "usage": None,
                "model": model,
            }

    jobs = list(enumerate(rows))
    if workers == 1:
        results = [evaluate(job) for job in jobs]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(evaluate, jobs))
    for result in results:
        predictions.append(result["prediction"])
        response_model = result["model"]
        _add_usage(usage, result["usage"])
        if result["error"] is not None:
            errors.append(result["error"])
    return predictions, usage, response_model, errors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run deployment-shaped initial-prompt routing evaluation."
    )
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not 1 <= args.runs <= 5:
        raise SystemExit("--runs must be between 1 and 5")
    if not 1 <= args.workers <= 16:
        raise SystemExit("--workers must be between 1 and 16")

    load_dotenv(ROOT / ".env")
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_JUDGE_MODEL")
    if not api_key or api_key.startswith("insert-"):
        raise SystemExit("OPENAI_API_KEY is not configured in .env")
    if not model or model.startswith("insert-"):
        raise SystemExit("OPENAI_JUDGE_MODEL is not configured in .env")

    rows = prompt_routing_examples()
    client = OpenAI(api_key=api_key, max_retries=0, timeout=120.0)
    prediction_runs: list[list[dict[str, Any]]] = []
    raw_runs: list[tuple[dict[str, int | None], str, list[dict[str, Any]]]] = []
    for _ in range(args.runs):
        predictions, usage, response_model, errors = run_once(
            client, model, rows, workers=args.workers
        )
        prediction_runs.append(predictions)
        raw_runs.append((usage, response_model, errors))

    threshold, calibration_scores = choose_threshold(rows, prediction_runs)
    holdout_rows = [row for row in rows if row["split"] == "holdout"]
    run_results: list[dict[str, Any]] = []
    pooled_predictions: list[dict[str, Any]] = []
    for predictions, (usage, response_model, errors) in zip(
        prediction_runs, raw_runs
    ):
        holdout_predictions = [
            prediction
            for row, prediction in zip(rows, predictions)
            if row["split"] == "holdout"
        ]
        pooled_predictions.extend(holdout_predictions)
        metrics = score_prompt_routing(
            holdout_rows, holdout_predictions, threshold=threshold
        )
        failures = [
            {
                "example_id": row["example_id"],
                "expected_action": row["expected_action"],
                "predicted_action": prediction.get("action"),
                "confidence": prediction.get("confidence"),
                "invalid": bool(prediction.get("invalid")),
            }
            for row, prediction in zip(holdout_rows, holdout_predictions)
            if prediction.get("invalid")
            or (
                prediction.get("action") is not None
                and prediction.get("confidence", 0) >= threshold
                and prediction.get("action") != row["expected_action"]
            )
            or (
                row["expected_action"] is not None
                and not (
                    prediction.get("action") == row["expected_action"]
                    and prediction.get("confidence", 0) >= threshold
                )
            )
        ]
        run_results.append(
            {
                "model": response_model,
                "metrics": metrics,
                "confidence_intervals_95": confidence_intervals(
                    holdout_rows, holdout_predictions, threshold=threshold
                ),
                "release_gate": release_gate(metrics),
                "failure_diagnostics": failures,
                "request_errors": errors,
                "usage": usage,
                "requests": len(rows),
            }
        )

    pooled_rows = holdout_rows * len(run_results)
    aggregate, ranges = aggregate_metrics([run["metrics"] for run in run_results])
    total_usage = {
        name: sum((run["usage"][name] or 0) for run in run_results)
        for name in ("input_tokens", "output_tokens", "total_tokens")
    }
    summary = {
        "benchmark_type": "real_prompt_routing",
        "status": "completed",
        "reproducibility": {
            "generated_at": generated_at(),
            "dataset_sha256": canonical_sha256(rows),
            "evaluator_prompt_sha256": canonical_sha256(
                PROMPT_PREFERENCE_SYSTEM_PROMPT
            ),
            "split_strategy": "the fixed split field in prompt_routing_examples()",
        },
        "model": run_results[0]["model"],
        "example_count": len(rows),
        "calibration_example_count": len(rows) - len(holdout_rows),
        "holdout_example_count": len(holdout_rows),
        "frozen_routing_threshold": threshold,
        "threshold_selection": {
            "source": "pooled_calibration_partitions_only",
            "candidate_scores": calibration_scores,
        },
        "pooled_holdout_confidence_intervals_95": confidence_intervals(
            pooled_rows, pooled_predictions, threshold=threshold
        ),
        "metrics": aggregate,
        "metric_ranges": ranges,
        "release_gate": {
            "basis": "every_run_on_holdout_at_frozen_routing_threshold",
            "passed": all(run["release_gate"]["passed"] for run in run_results),
        },
        "runs_requested": args.runs,
        "runs_completed": len(run_results),
        "run_results": run_results,
        "usage": total_usage,
        "requests": len(rows) * len(run_results),
        "store": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
