from examples.interactive_personalization import ACTION_INSTRUCTIONS, offline_generate


def test_playground_isolates_response_order() -> None:
    assert tuple(ACTION_INSTRUCTIONS) == ("code_first", "explanation_first")

    code_first, _ = offline_generate("solve the problem", "code_first")
    explanation_first, _ = offline_generate("solve the problem", "explanation_first")

    assert code_first.startswith("Code first:")
    assert explanation_first.startswith("Explanation first:")
