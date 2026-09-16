from types import SimpleNamespace

from examples import openai_judge, openai_prompt_router


class FakeResponses:
    def __init__(self, parsed: object) -> None:
        self.parsed = parsed
        self.calls: list[dict[str, object]] = []

    def parse(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return SimpleNamespace(output_parsed=self.parsed)


class FakeClient:
    responses: FakeResponses
    constructor_calls: list[dict[str, object]] = []
    instances: list["FakeClient"] = []
    parsed: object

    def __init__(self, **kwargs: object) -> None:
        self.constructor_calls.append(kwargs)
        self.responses = FakeResponses(self.parsed)
        self.instances.append(self)


def _configure(monkeypatch, module: object, parsed: object) -> None:
    FakeClient.constructor_calls.clear()
    FakeClient.instances.clear()
    FakeClient.parsed = parsed
    monkeypatch.setattr(module, "load_dotenv", lambda _: True)
    monkeypatch.setattr(module, "OpenAI", FakeClient)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-placeholder")
    monkeypatch.setenv("OPENAI_JUDGE_MODEL", "test-model")


def test_openai_feedback_example_uses_structured_responses_without_storage(
    monkeypatch,
) -> None:
    parsed = openai_judge.OpenAIJudgment(
        target="behavior",
        sentiment="negative",
        confidence=0.98,
        reason=None,
    )
    _configure(monkeypatch, openai_judge, parsed)
    judge, model = openai_judge.build_openai_judge()
    result = judge([{"role": "user", "content": "test interaction"}])
    client = FakeClient.constructor_calls
    assert model == "test-model"
    assert client == [
        {"api_key": "test-key-placeholder", "max_retries": 0, "timeout": 60.0}
    ]
    assert result == {
        "target": "behavior",
        "sentiment": "negative",
        "confidence": 0.98,
        "reason": None,
    }
    assert "source" not in result
    assert FakeClient.instances[0].responses.calls[0]["store"] is False
    assert (
        FakeClient.instances[0].responses.calls[0]["text_format"]
        is openai_judge.OpenAIJudgment
    )


def test_openai_prompt_example_uses_structured_responses_without_storage(
    monkeypatch,
) -> None:
    parsed = openai_prompt_router.OpenAIPromptPreference(
        action="guided_learning", confidence=0.99, reason=None
    )
    _configure(monkeypatch, openai_prompt_router, parsed)
    judge, model = openai_prompt_router.build_openai_prompt_judge()
    result = judge([{"role": "user", "content": "test interaction"}])
    assert model == "test-model"
    assert result == {
        "action": "guided_learning",
        "confidence": 0.99,
        "reason": None,
    }
    assert FakeClient.instances[0].responses.calls[0]["store"] is False
    assert (
        FakeClient.instances[0].responses.calls[0]["text_format"]
        is openai_prompt_router.OpenAIPromptPreference
    )
