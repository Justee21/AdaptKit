from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError as PydanticValidationError

from benchmarks.evaluator_benchmark import score
from adaptkit.feedback.llm import (
    DEFAULT_FEEDBACK_SYSTEM_PROMPT,
    FEEDBACK_CLASSIFICATION_GUIDANCE,
)

ROOT = Path(__file__).resolve().parents[1]
DATASETS = (
    ROOT / "benchmarks/data/evaluator_examples.json",
    ROOT / "benchmarks/data/evaluator_adversarial_examples.json",
)
DEFAULT_OUTPUT = ROOT / "artifacts/real_evaluator_summary.json"

INSTRUCTIONS = DEFAULT_FEEDBACK_SYSTEM_PROMPT
BATCH_INSTRUCTIONS = FEEDBACK_CLASSIFICATION_GUIDANCE + """

Classify every interaction independently. Return one target, sentiment, and confidence prediction for each input example in unchanged order. Target must be behavior, answer_content, task_continuation, quoted_or_meta, or unrelated. Use positive or negative sentiment only for behavior; every other target must use none."""


class Prediction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target: Literal[
        "behavior", "answer_content", "task_continuation", "quoted_or_meta", "unrelated"
    ] = Field(description="What the latest user message is primarily doing.")
    sentiment: Literal["positive", "negative", "none"] = Field(
        description=(
            "Sentiment toward selected_action: positive reinforces it, negative requests "
            "moving away from it, and none is required for non-behavior targets."
        )
    )
    confidence: float = Field(
        ge=0,
        le=1,
        description="Confidence that both target and selected-action-relative sentiment are correct.",
    )


class PredictionBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    predictions: list[Prediction] = Field(
        description="One prediction per input example, in unchanged input order.",
    )


def load_examples() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        source = dataset.stem
        for index, row in enumerate(json.loads(dataset.read_text(encoding="utf-8"))):
            rows.append({"example_id": f"{source}-{index:02d}", **row})
    return rows


def benchmark_input(rows: list[dict[str, Any]]) -> str:
    examples = [
        {
            "available_actions": [row["action"], "alternative"],
            "action_descriptions": {
                row["action"]: row["action"].replace("_", " "),
                "alternative": "an alternative response behavior",
            },
            "context": "benchmark",
            "previous_user_prompt": row["previous_prompt"],
            "selected_action": row["action"],
            "selected_action_description": row["action"].replace("_", " "),
            "agent_response": row["previous_response"],
            "latest_user_message": row["next_message"],
        }
        for row in rows
    ]
    if len(examples) == 1:
        return "Interaction data:\n" + json.dumps(examples[0])
    return (
        "Interaction data batch. Evaluate each independently and return predictions in "
        "the same order:\n" + json.dumps(examples)
    )


def aligned_predictions(
    rows: list[dict[str, Any]], batch: PredictionBatch
) -> tuple[list[dict[str, Any]], int]:
    aligned: list[dict[str, Any]] = []
    for prediction in batch.predictions[: len(rows)]:
        valid = (prediction.target == "behavior") == (prediction.sentiment != "none")
        if not valid:
            aligned.append(
                {
                    "invalid": True,
                    "target": prediction.target,
                    "sentiment": prediction.sentiment,
                    "confidence": prediction.confidence,
                }
            )
            continue
        aligned.append(
            {
                "target": prediction.target,
                "sentiment": prediction.sentiment,
                "confidence": prediction.confidence,
            }
        )
    aligned.extend({"invalid": True} for _ in range(len(rows) - len(aligned)))
    return aligned, max(0, len(batch.predictions) - len(rows))


def metrics_by_category(
    rows: list[dict[str, Any]], predictions: list[dict[str, Any]]
) -> dict[str, dict[str, float | int | None]]:
    categories: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        categories.setdefault(row["category"], []).append(index)

    result: dict[str, dict[str, float | int | None]] = {}
    for category, indices in categories.items():
        detection_correct = target_correct = direction_correct = false_positives = 0
        false_positive_learning_updates = 0
        behavior_predictions = true_behavior_predictions = false_direction_updates = 0
        feedback_total = non_feedback_total = invalid = 0
        for index in indices:
            row = rows[index]
            prediction = predictions[index]
            if prediction.get("invalid"):
                invalid += 1
                if row["expected_has_feedback"]:
                    feedback_total += 1
                else:
                    non_feedback_total += 1
                continue
            target = prediction["target"]
            predicted_feedback = target == "behavior"
            behavior_predictions += predicted_feedback
            expected_feedback = row["expected_has_feedback"]
            detection_correct += predicted_feedback == expected_feedback
            target_correct += target == row["expected_target"]
            if expected_feedback:
                feedback_total += 1
                true_behavior_predictions += predicted_feedback
                direction_correct += (
                    predicted_feedback and prediction["sentiment"] == row["expected_direction"]
                )
                false_direction_updates += (
                    predicted_feedback
                    and prediction["sentiment"] != row["expected_direction"]
                )
            else:
                non_feedback_total += 1
                false_positives += predicted_feedback
                false_positive_learning_updates += (
                    predicted_feedback and prediction["confidence"] >= 0.70
                )
        result[category] = {
            "examples": len(indices),
            "learning_threshold": 0.70,
            "feedback_detection_accuracy": detection_correct / len(indices),
            "behavior_precision": (
                true_behavior_predictions / behavior_predictions
                if behavior_predictions
                else None
            ),
            "behavior_recall": (
                true_behavior_predictions / feedback_total if feedback_total else None
            ),
            "target_accuracy": target_correct / len(indices),
            "direction_accuracy": direction_correct / feedback_total if feedback_total else None,
            "false_direction_update_rate": (
                false_direction_updates / behavior_predictions
                if behavior_predictions
                else None
            ),
            "false_positive_rate": false_positives / non_feedback_total if non_feedback_total else None,
            "false_positive_learning_rate": (
                false_positive_learning_updates / non_feedback_total
                if non_feedback_total
                else None
            ),
            "invalid_outputs": invalid,
        }
    return result


def failure_diagnostics(
    rows: list[dict[str, Any]], predictions: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for row, prediction in zip(rows, predictions):
        expected_target = row["expected_target"]
        expected_sentiment = row["expected_direction"]
        if prediction.get("invalid"):
            failures.append(
                {
                    "example_id": row["example_id"],
                    "category": row["category"],
                    "expected_target": expected_target,
                    "expected_sentiment": expected_sentiment,
                    "predicted_target": prediction.get("target"),
                    "predicted_sentiment": prediction.get("sentiment"),
                    "confidence": prediction.get("confidence"),
                    "invalid": True,
                }
            )
        elif (
            prediction["target"] != expected_target
            or prediction["sentiment"] != expected_sentiment
        ):
            failures.append(
                {
                    "example_id": row["example_id"],
                    "category": row["category"],
                    "expected_target": expected_target,
                    "expected_sentiment": expected_sentiment,
                    "predicted_target": prediction["target"],
                    "predicted_sentiment": prediction["sentiment"],
                    "confidence": prediction["confidence"],
                    "invalid": False,
                }
            )
    return failures


def write_summary(summary: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def safe_error_metadata(exc: Exception) -> dict[str, Any]:
    metadata: dict[str, Any] = {"error_type": type(exc).__name__}
    if isinstance(exc, PydanticValidationError):
        metadata["validation_errors"] = [
            {
                "type": error["type"],
                "location": [str(part) for part in error["loc"]],
            }
            for error in exc.errors(include_input=False, include_context=False)
        ]
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        metadata["status_code"] = status_code
    code = getattr(exc, "code", None)
    if isinstance(code, str):
        metadata["error_code"] = code
    return metadata


def failed_summary(
    model: str, example_count: int, error_metadata: dict[str, Any]
) -> dict[str, Any]:
    summary = {
        "benchmark_type": "real_llm",
        "status": "failed",
        "model": model,
        "example_count": example_count,
        "metrics": {
            "feedback_detection_accuracy": None,
            "behavior_precision": None,
            "behavior_recall": None,
            "target_accuracy": None,
            "direction_accuracy": None,
            "false_direction_update_rate": None,
            "false_positive_rate": None,
            "false_positive_learning_rate": None,
            "invalid_outputs": example_count,
        },
        "usage": {
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
        },
        "requests": 1,
        "store": False,
    }
    summary.update(error_metadata)
    return summary


def aggregate_metrics(
    metric_sets: list[dict[str, float | int]],
) -> tuple[dict[str, float | int], dict[str, dict[str, float]]]:
    if not metric_sets:
        raise ValueError("metric_sets must not be empty")
    count_fields = {"examples", "invalid_outputs"}
    aggregate: dict[str, float | int] = {}
    ranges: dict[str, dict[str, float]] = {}
    for name in metric_sets[0]:
        values = [float(metrics[name]) for metrics in metric_sets]
        if name == "examples":
            aggregate[name] = int(values[0])
        elif name == "invalid_outputs":
            aggregate[name] = int(sum(values))
        else:
            aggregate[name] = sum(values) / len(values)
        if name not in count_fields:
            ranges[name] = {"min": min(values), "max": max(values)}
    return aggregate, ranges


def _add_usage(total: dict[str, int | None], usage: Any) -> None:
    if usage is None:
        return
    for name in ("input_tokens", "output_tokens", "total_tokens"):
        value = getattr(usage, name, None)
        if isinstance(value, int):
            total[name] = (total[name] or 0) + value


def run_once(
    client: OpenAI,
    model: str,
    rows: list[dict[str, Any]],
    *,
    batch_size: int,
) -> dict[str, Any]:
    predictions: list[dict[str, Any]] = []
    unexpected = 0
    requests = 0
    request_errors: list[dict[str, Any]] = []
    usage_incomplete = False
    usage: dict[str, int | None] = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }
    response_model = model
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        requests += 1
        parsed: PredictionBatch | None
        response_usage: Any = None
        try:
            if len(chunk) == 1:
                single_response = client.responses.parse(
                    model=model,
                    instructions=INSTRUCTIONS,
                    input=benchmark_input(chunk),
                    text_format=Prediction,
                    store=False,
                    max_output_tokens=800,
                    reasoning={"effort": "low"},
                )
                parsed = (
                    None
                    if single_response.output_parsed is None
                    else PredictionBatch(predictions=[single_response.output_parsed])
                )
                response_model = single_response.model
                response_usage = single_response.usage
            else:
                batch_response = client.responses.parse(
                    model=model,
                    instructions=BATCH_INSTRUCTIONS,
                    input=benchmark_input(chunk),
                    text_format=PredictionBatch,
                    store=False,
                    max_output_tokens=max(800, len(chunk) * 200),
                    reasoning={"effort": "low"},
                )
                parsed = batch_response.output_parsed
                response_model = batch_response.model
                response_usage = batch_response.usage
        except PydanticValidationError as exc:
            usage_incomplete = True
            predictions.extend({"invalid": True} for _ in chunk)
            request_errors.append(
                {"chunk_start": start, **safe_error_metadata(exc)}
            )
            continue
        if parsed is None:
            usage_incomplete = True
            predictions.extend({"invalid": True} for _ in chunk)
            request_errors.append(
                {"chunk_start": start, "error_type": "MissingParsedOutput"}
            )
            continue
        chunk_predictions, chunk_unexpected = aligned_predictions(
            chunk, parsed
        )
        predictions.extend(chunk_predictions)
        unexpected += chunk_unexpected
        _add_usage(usage, response_usage)

    metrics = score(rows, predictions)
    metrics["invalid_outputs"] += unexpected
    return {
        "model": response_model,
        "metrics": metrics,
        "unexpected_outputs": unexpected,
        "metrics_by_category": metrics_by_category(rows, predictions),
        "failure_diagnostics": failure_diagnostics(rows, predictions),
        "request_errors": request_errors,
        "usage": usage,
        "usage_incomplete": usage_incomplete,
        "requests": requests,
    }


def total_usage(runs: list[dict[str, Any]]) -> dict[str, int | None]:
    result: dict[str, int | None] = {}
    for name in ("input_tokens", "output_tokens", "total_tokens"):
        values = [run["usage"][name] for run in runs]
        result[name] = None if any(value is None for value in values) else sum(values)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a bounded, opt-in real-LLM evaluator benchmark."
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Independent benchmark runs (1-5).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Examples per paid request (1-46); defaults to 10.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not 1 <= args.runs <= 5:
        raise SystemExit("--runs must be between 1 and 5")
    if not 1 <= args.batch_size <= 46:
        raise SystemExit("--batch-size must be between 1 and 46")

    load_dotenv(ROOT / ".env")
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_JUDGE_MODEL")
    if not api_key or api_key.startswith("insert-"):
        raise SystemExit("OPENAI_API_KEY is not configured in .env")
    if not model or model.startswith("insert-"):
        raise SystemExit("OPENAI_JUDGE_MODEL is not configured in .env")

    rows = load_examples()
    client = OpenAI(api_key=api_key, max_retries=0, timeout=120.0)
    runs: list[dict[str, Any]] = []
    for _ in range(args.runs):
        try:
            runs.append(run_once(client, model, rows, batch_size=args.batch_size))
        except Exception as exc:
            summary = failed_summary(model, len(rows), safe_error_metadata(exc))
            summary["runs_requested"] = args.runs
            summary["runs_completed"] = len(runs)
            summary["run_results"] = runs
            summary["usage"] = total_usage(runs) if runs else summary["usage"]
            summary["requests"] = sum(run["requests"] for run in runs) + 1
            write_summary(summary, args.output)
            raise SystemExit(1) from None

    metrics, ranges = aggregate_metrics([run["metrics"] for run in runs])
    summary = {
        "benchmark_type": "real_llm",
        "status": "completed",
        "model": runs[0]["model"],
        "example_count": len(rows),
        "metrics": metrics,
        "metric_ranges": ranges,
        "runs_requested": args.runs,
        "runs_completed": len(runs),
        "batch_size": args.batch_size,
        "run_results": runs,
        "usage": total_usage(runs),
        "requests": sum(run["requests"] for run in runs),
        "store": False,
    }
    write_summary(summary, args.output)


if __name__ == "__main__":
    main()
