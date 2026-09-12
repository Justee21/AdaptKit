import unittest

from adaptkit import ValidationError
from adaptkit.metrics import SimulationStep, cumulative, summarize
from adaptkit.simulation import SimulatedUser
from benchmarks.bandit_benchmark import averaged_series


class SimulationTests(unittest.TestCase):
    def test_simulated_user_extreme_probabilities(self):
        user = SimulatedUser({"ctx": {"always": 1, "never": 0}}, seed=1)
        self.assertEqual([user.react("ctx", "always") for _ in range(5)], [1] * 5)
        self.assertEqual([user.react("ctx", "never") for _ in range(5)], [0] * 5)
        self.assertEqual(user.best_action("ctx"), "always")

    def test_simulated_user_validation(self):
        with self.assertRaises(ValidationError):
            SimulatedUser({"ctx": {"bad": 1.1}})

    def test_metrics_match_hand_calculation(self):
        steps = [
            SimulationStep(1, 0.8, 0.8, True),
            SimulationStep(0, 0.3, 0.8, False),
            SimulationStep(1, 0.3, 0.8, False),
        ]
        self.assertEqual(cumulative([1, 0, 1]), [1, 1, 2])
        result = summarize(steps)
        self.assertAlmostEqual(result["average_reward"], 2 / 3)
        self.assertEqual(result["cumulative_reward"], 2)
        self.assertAlmostEqual(result["cumulative_regret"], 1.0)
        self.assertAlmostEqual(result["optimal_action_rate"], 1 / 3)

    def test_thompson_beats_random_on_easy_stationary_problem(self):
        random_result, _ = averaged_series("random", steps=400, seeds=30)
        thompson_result, _ = averaged_series("thompson", steps=400, seeds=30)
        self.assertGreater(thompson_result["average_reward"], random_result["average_reward"] + 0.2)
        self.assertLess(thompson_result["cumulative_regret"], random_result["cumulative_regret"])
        self.assertGreater(thompson_result["optimal_action_rate"], 0.85)


if __name__ == "__main__":
    unittest.main()
