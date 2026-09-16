from types import ModuleType, SimpleNamespace
from typing import cast

import pytest

from adaptkit import BetaPrior, InMemoryStore, ValidationError, cold_start_priors
from examples import letta_cold_start, mem0_cold_start
from examples.letta_cold_start import profile_from_letta
from examples.mem0_cold_start import profile_from_mem0


class FakeMem0:
    def search(self, **kwargs):
        assert kwargs == {
            "query": "durable response-style and learning preferences",
            "filters": {"user_id": "u"},
            "top_k": 5,
        }
        return {
            "results": [
                {"memory": "The user generally prefers guided explanations"}
            ]
        }


class FakeMem0ListResponse(FakeMem0):
    def search(self, **kwargs):
        super().search(**kwargs)
        return [{"memory": "The user generally prefers direct solutions"}]


class FakeLetta:
    def __init__(self, value="The user generally prefers guided explanations"):
        self.agents = SimpleNamespace(blocks=self)
        self.value = value

    def retrieve(self, **kwargs):
        assert kwargs == {"agent_id": "agent-1", "block_label": "human"}
        return SimpleNamespace(value=self.value)


def test_provider_neutral_conversion_is_modest_and_validated() -> None:
    assert cold_start_priors(
        actions=["direct_solution", "guided_learning"],
        preferred_action="guided_learning",
    ) == {"guided_learning": BetaPrior(4, 2)}
    assert cold_start_priors(actions=["a", "b"], preferred_action=None) == {}
    with pytest.raises(ValidationError):
        cold_start_priors(actions=["a", "a"], preferred_action="a")
    with pytest.raises(ValidationError):
        cold_start_priors(actions=["a"], preferred_action="missing")


@pytest.mark.parametrize(
    "factory",
    [
        lambda store: profile_from_mem0(FakeMem0(), user_id="u", store=store),
        lambda store: profile_from_letta(
            FakeLetta(), agent_id="agent-1", user_id="u", store=store
        ),
    ],
)
def test_memory_examples_translate_through_action_keys(factory) -> None:
    profile = factory(InMemoryStore())
    assert profile.state() == {}
    assert profile.policy("algorithms")["actions"]["guided_learning"][
        "posterior_mean"
    ] == pytest.approx(2 / 3)


def test_mem0_current_list_response_shape_is_supported() -> None:
    profile = profile_from_mem0(FakeMem0ListResponse(), user_id="u")
    actions = cast(dict[str, dict[str, float]], profile.policy("ctx")["actions"])
    assert actions["direct_solution"]["posterior_mean"] == pytest.approx(2 / 3)


def test_materialized_policy_wins_over_later_memory_prior() -> None:
    store = InMemoryStore()
    first = profile_from_mem0(FakeMem0(), user_id="u", store=store)
    first.policy("algorithms")
    reopened = profile_from_letta(
        FakeLetta("The user generally prefers direct solutions"),
        agent_id="agent-1",
        user_id="u",
        store=store,
    )
    assert reopened.state()["algorithms"]["guided_learning"] == {
        "alpha": 4.0,
        "beta": 2.0,
    }
    assert reopened.state()["algorithms"]["direct_solution"] == {
        "alpha": 1.0,
        "beta": 1.0,
    }


def test_temporary_cue_is_not_accepted_as_durable_memory() -> None:
    temporary = {
        "results": [{"memory": "For this answer, be concise"}]
    }

    class TemporaryMem0:
        def search(self, **kwargs):
            return temporary

    profile = profile_from_mem0(TemporaryMem0(), user_id="u")
    profile.policy("ctx")
    assert profile.state()["ctx"] == {
        "direct_solution": {"alpha": 1.0, "beta": 1.0},
        "guided_learning": {"alpha": 1.0, "beta": 1.0},
    }


@pytest.mark.parametrize(
    "memories",
    [
        [
            "The user generally prefers guided explanations",
            "The user generally prefers direct solutions",
        ],
        ["The user may prefer guided explanations or direct solutions"],
        ["For this response, the user wants a direct solution"],
    ],
)
def test_ambiguous_contradictory_or_temporary_mem0_is_neutral(
    memories: list[str],
) -> None:
    class CandidateMem0:
        def search(self, **kwargs):
            return {"results": [{"memory": memory} for memory in memories]}

    profile = profile_from_mem0(CandidateMem0(), user_id="u")
    assert profile.policy("ctx")["actions"] == {
        "direct_solution": {"alpha": 1.0, "beta": 1.0, "posterior_mean": 0.5},
        "guided_learning": {"alpha": 1.0, "beta": 1.0, "posterior_mean": 0.5},
    }


def test_contradictory_letta_memory_is_neutral() -> None:
    profile = profile_from_letta(
        FakeLetta(
            "The user generally prefers guided explanations\n"
            "The user generally prefers direct solutions"
        ),
        agent_id="agent-1",
        user_id="u",
    )
    actions = cast(dict[str, dict[str, float]], profile.policy("ctx")["actions"])
    assert all(state["posterior_mean"] == 0.5 for state in actions.values())


@pytest.mark.parametrize(
    ("module", "builder", "package"),
    [
        (mem0_cold_start, mem0_cold_start.build_mem0_client, "mem0"),
        (letta_cold_start, letta_cold_start.build_letta_client, "letta_client"),
    ],
)
def test_optional_memory_dependency_errors_are_actionable(
    monkeypatch: pytest.MonkeyPatch, module: object, builder: object, package: str
) -> None:
    del module

    def missing(name: str):
        if name == package:
            raise ModuleNotFoundError(name)
        raise AssertionError(name)

    target = mem0_cold_start if package == "mem0" else letta_cold_start
    monkeypatch.setattr(target.importlib, "import_module", missing)
    with pytest.raises(RuntimeError, match="pip install"):
        builder()  # type: ignore[operator]


@pytest.mark.parametrize(
    ("target", "builder", "package", "class_name", "credential"),
    [
        (
            mem0_cold_start,
            mem0_cold_start.build_mem0_client,
            "mem0",
            "MemoryClient",
            "MEM0_API_KEY",
        ),
        (
            letta_cold_start,
            letta_cold_start.build_letta_client,
            "letta_client",
            "Letta",
            "LETTA_API_KEY",
        ),
    ],
)
def test_optional_memory_clients_validate_credentials_and_current_constructor(
    monkeypatch: pytest.MonkeyPatch,
    target: object,
    builder: object,
    package: str,
    class_name: str,
    credential: str,
) -> None:
    calls = []

    class FakeClient:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    fake_module = ModuleType(package)
    setattr(fake_module, class_name, FakeClient)
    monkeypatch.setattr(target.importlib, "import_module", lambda name: fake_module)  # type: ignore[attr-defined]
    monkeypatch.delenv(credential, raising=False)
    with pytest.raises(RuntimeError, match=credential):
        builder()  # type: ignore[operator]
    monkeypatch.setenv(credential, "test-placeholder-secret")
    builder()  # type: ignore[operator]
    assert calls == [{"api_key": "test-placeholder-secret"}]
