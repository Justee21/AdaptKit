from examples.interactive_personalization import (
    ACTION_INSTRUCTIONS,
    PLAYGROUND_CONFIDENCE_THRESHOLD,
    WORKFLOW_SCENARIO,
    _offline_generate_for,
    looks_like_launch_command,
    offline_generate,
    response_follows_action,
)


def test_playground_isolates_response_order() -> None:
    assert tuple(ACTION_INSTRUCTIONS) == ("code_first", "explanation_first")
    assert PLAYGROUND_CONFIDENCE_THRESHOLD == 0.90

    code_first, _ = offline_generate("solve the problem", "code_first")
    explanation_first, _ = offline_generate("solve the problem", "explanation_first")

    assert code_first.startswith("```")
    assert explanation_first.startswith("Brief explanation.")


def test_playground_recognizes_accidentally_pasted_launch_command() -> None:
    command = ".venv/bin/python -m examples.interactive_personalization --real"
    assert looks_like_launch_command(command)
    assert not looks_like_launch_command("explain how Python imports work")


def test_playground_validates_selected_ordering() -> None:
    assert response_follows_action("```python\npass\n```\nExplanation", "code_first")
    assert not response_follows_action("Explanation\n```python\npass\n```", "code_first")
    assert response_follows_action(
        "Explanation\n```python\npass\n```", "explanation_first"
    )
    assert not response_follows_action(
        "```python\npass\n```\nExplanation", "explanation_first"
    )


def test_workflow_scenario_has_four_distinct_visible_strategies() -> None:
    assert tuple(WORKFLOW_SCENARIO.actions) == (
        "direct_solution",
        "guided_learning",
        "deep_dive",
        "examples_first",
    )

    prefixes = WORKFLOW_SCENARIO.required_prefixes
    assert prefixes is not None
    for action, prefix in prefixes.items():
        response, _ = _offline_generate_for("solve 3Sum", action, (), WORKFLOW_SCENARIO)
        assert response.startswith(prefix)
        assert response_follows_action(response, action, WORKFLOW_SCENARIO)
        other_action = next(key for key in WORKFLOW_SCENARIO.actions if key != action)
        assert not response_follows_action(response, other_action, WORKFLOW_SCENARIO)
