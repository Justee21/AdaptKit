from __future__ import annotations

import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape

from adaptkit import Profile
from adaptkit.metrics import SimulationStep, cumulative, summarize
from adaptkit.simulation import SimulatedUser

ACTIONS = ("patch_first", "explanation_first")
CONTEXT = "debugging"
PREFERENCES = {CONTEXT: {"patch_first": 0.9, "explanation_first": 0.2}}
COLORS = {"random": "#d97706", "thompson": "#2563eb"}


def run_trial(learner: str, *, steps: int, seed: int) -> list[SimulationStep]:
    user = SimulatedUser(PREFERENCES, seed=seed * 2 + 1)
    profile = Profile(user_id=f"sim-{seed}", actions=ACTIONS, learner=learner, seed=seed * 2)
    best = user.best_action(CONTEXT)
    optimal_reward = user.expected_reward(CONTEXT, best)
    results: list[SimulationStep] = []
    for step in range(steps):
        decision = profile.choose(CONTEXT)
        action = decision.action
        reward = user.react(CONTEXT, action)
        if reward:
            profile.like(decision, idempotency_key=f"simulation-{step}")
        else:
            profile.dislike(decision, idempotency_key=f"simulation-{step}")
        results.append(
            SimulationStep(
                reward=reward,
                expected_reward=user.expected_reward(CONTEXT, action),
                optimal_expected_reward=optimal_reward,
                chose_optimal=action == best,
            )
        )
    return results


def averaged_series(learner: str, *, steps: int, seeds: int) -> tuple[dict[str, float], dict[str, list[float]]]:
    trials = [run_trial(learner, steps=steps, seed=seed) for seed in range(seeds)]
    reward_lines = [cumulative([step.reward for step in trial]) for trial in trials]
    regret_lines = [cumulative([step.regret for step in trial]) for trial in trials]
    optimal_lines = [cumulative([int(step.chose_optimal) for step in trial]) for trial in trials]
    series = {
        "cumulative_reward": [sum(line[i] for line in reward_lines) / seeds for i in range(steps)],
        "cumulative_regret": [sum(line[i] for line in regret_lines) / seeds for i in range(steps)],
        "optimal_action_rate": [
            sum(line[i] for line in optimal_lines) / seeds / (i + 1) for i in range(steps)
        ],
    }
    final = {
        "average_reward": series["cumulative_reward"][-1] / steps,
        "cumulative_reward": series["cumulative_reward"][-1],
        "cumulative_regret": series["cumulative_regret"][-1],
        "optimal_action_rate": series["optimal_action_rate"][-1],
    }
    return final, series


def write_svg(path: Path, title: str, ylabel: str, lines: dict[str, list[float]]) -> None:
    width, height = 840, 480
    left, right, top, bottom = 75, 25, 55, 60
    plot_w, plot_h = width - left - right, height - top - bottom
    max_x = max(len(values) for values in lines.values()) - 1
    max_y = max(max(values) for values in lines.values())
    max_y = max(max_y, 1.0)

    def point(index: int, value: float) -> tuple[float, float]:
        x = left + (index / max(max_x, 1)) * plot_w
        y = top + plot_h - (value / max_y) * plot_h
        return x, y

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="30" text-anchor="middle" font-family="sans-serif" font-size="20">{escape(title)}</text>',
    ]
    for tick in range(6):
        value = max_y * tick / 5
        y = top + plot_h - plot_h * tick / 5
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="12">{value:.1f}</text>')
    parts.extend(
        [
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#111827"/>',
            f'<line x1="{left}" y1="{top+plot_h}" x2="{width-right}" y2="{top+plot_h}" stroke="#111827"/>',
            f'<text x="{width/2}" y="{height-15}" text-anchor="middle" font-family="sans-serif" font-size="14">Interaction</text>',
            f'<text x="18" y="{height/2}" text-anchor="middle" transform="rotate(-90 18 {height/2})" font-family="sans-serif" font-size="14">{escape(ylabel)}</text>',
            f'<text x="{left}" y="{top+plot_h+22}" text-anchor="middle" font-family="sans-serif" font-size="12">1</text>',
            f'<text x="{width-right}" y="{top+plot_h+22}" text-anchor="middle" font-family="sans-serif" font-size="12">{max_x+1}</text>',
        ]
    )
    for offset, (name, values) in enumerate(lines.items()):
        coords = " ".join(f"{x:.1f},{y:.1f}" for x, y in (point(i, v) for i, v in enumerate(values)))
        color = COLORS[name]
        parts.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="3"/>')
        legend_x = width - right - 145
        legend_y = top + 18 * offset
        parts.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x+24}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{legend_x+31}" y="{legend_y+4}" font-family="sans-serif" font-size="13">{escape(name.title())}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare AdaptKit bandit learners.")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    if args.steps < 2 or args.seeds < 1:
        parser.error("steps must be at least 2 and seeds at least 1")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, dict[str, float]] = {}
    all_series: dict[str, dict[str, list[float]]] = {}
    for learner in ("random", "thompson"):
        summaries[learner], all_series[learner] = averaged_series(
            learner, steps=args.steps, seeds=args.seeds
        )
    payload = {
        "configuration": {"steps": args.steps, "seeds": args.seeds, "preferences": PREFERENCES},
        "results": summaries,
    }
    (args.output_dir / "benchmark_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    write_svg(
        args.output_dir / "cumulative_reward.svg",
        "Average cumulative reward",
        "Cumulative successes",
        {name: series["cumulative_reward"] for name, series in all_series.items()},
    )
    write_svg(
        args.output_dir / "cumulative_regret.svg",
        "Average cumulative expected regret",
        "Cumulative regret",
        {name: series["cumulative_regret"] for name, series in all_series.items()},
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
