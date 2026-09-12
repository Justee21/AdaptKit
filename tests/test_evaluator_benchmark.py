import json
import unittest
from pathlib import Path

from benchmarks.evaluator_benchmark import score


class EvaluatorBenchmarkTests(unittest.TestCase):
    def test_dataset_has_required_coverage(self):
        path = Path(__file__).parents[1] / "benchmarks/data/evaluator_examples.json"
        rows = json.loads(path.read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(rows), 20)
        self.assertLessEqual(len(rows), 30)
        self.assertEqual(
            {row["category"] for row in rows},
            {
                "clear_positive_feedback",
                "clear_negative_feedback",
                "preference_correction",
                "ordinary_continuation",
                "topic_change",
                "unrelated_factual_correction",
                "ambiguous_message",
            },
        )

    def test_adversarial_dataset_has_required_coverage(self):
        path = Path(__file__).parents[1] / "benchmarks/data/evaluator_adversarial_examples.json"
        rows = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(len(rows), 10)
        self.assertEqual(
            {row["category"] for row in rows},
            {
                "paraphrase",
                "indirect_feedback",
                "quoted_feedback",
                "ambiguous_continuation",
                "prompt_injection",
            },
        )

    def test_score_calculation(self):
        expected = [
            {"expected_has_feedback": True, "expected_direction": "positive"},
            {"expected_has_feedback": True, "expected_direction": "negative"},
            {"expected_has_feedback": False, "expected_direction": "none"},
            {"expected_has_feedback": False, "expected_direction": "none"},
        ]
        predictions = [
            {"has_feedback": True, "reward": 1},
            {"has_feedback": True, "reward": 1},
            {"has_feedback": True, "reward": -1},
            {"has_feedback": False, "reward": None},
        ]
        result = score(expected, predictions)
        self.assertEqual(result["feedback_detection_accuracy"], 0.75)
        self.assertEqual(result["direction_accuracy"], 0.5)
        self.assertEqual(result["false_positive_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
