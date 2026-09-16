import math
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from adaptkit import (
    ActionSetMismatchError,
    BetaPrior,
    InMemoryStore,
    Profile,
    SQLiteStore,
    ValidationError,
)


@pytest.mark.parametrize("value", [True, False, 0, -1, math.inf, -math.inf, math.nan])
@pytest.mark.parametrize("field", ["alpha", "beta"])
def test_beta_prior_rejects_invalid_parameters(value, field) -> None:
    arguments = {"alpha": 1.0, "beta": 1.0, field: value}
    with pytest.raises(ValidationError):
        BetaPrior(**arguments)


@pytest.mark.parametrize("mean", [True, False, 0, 1, -0.1, 1.1, math.inf, math.nan])
def test_mean_strength_rejects_invalid_means(mean) -> None:
    with pytest.raises(ValidationError):
        BetaPrior.from_mean_strength(mean, 2)


@pytest.mark.parametrize("strength", [True, False, 0, -1, math.inf, math.nan])
def test_mean_strength_rejects_invalid_strengths(strength) -> None:
    with pytest.raises(ValidationError):
        BetaPrior.from_mean_strength(0.75, strength)


def test_unspecified_action_uses_default_prior() -> None:
    profile = Profile(
        user_id="u",
        actions=["a", "b"],
        prior_alpha=2,
        prior_beta=3,
        action_priors={"a": BetaPrior(4, 1)},
    )
    profile.policy("ctx")
    assert profile.state()["ctx"] == {
        "a": {"alpha": 4.0, "beta": 1.0},
        "b": {"alpha": 2.0, "beta": 3.0},
    }


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_concurrent_first_use_materializes_one_complete_prior_set(
    store_kind, tmp_path
) -> None:
    store = (
        InMemoryStore()
        if store_kind == "memory"
        else SQLiteStore(tmp_path / "policy.db", busy_timeout=5)
    )
    first = Profile(
        user_id="u",
        actions=["a", "b"],
        action_priors={"a": BetaPrior(8, 1), "b": BetaPrior(2, 7)},
        store=store,
    )
    second = Profile(
        user_id="u",
        actions=["b", "a"],
        action_priors={"a": BetaPrior(3, 6), "b": BetaPrior(9, 2)},
        store=store,
    )
    barrier = threading.Barrier(3)

    def materialize(profile):
        barrier.wait(timeout=5)
        profile.policy("ctx")

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(materialize, profile) for profile in (first, second)]
        barrier.wait(timeout=5)
        for future in futures:
            future.result(timeout=10)
    state = first.state()["ctx"]
    assert state in (
        {
            "a": {"alpha": 8.0, "beta": 1.0},
            "b": {"alpha": 2.0, "beta": 7.0},
        },
        {
            "a": {"alpha": 3.0, "beta": 6.0},
            "b": {"alpha": 9.0, "beta": 2.0},
        },
    )


def test_action_order_is_identity_neutral_and_export_is_detached() -> None:
    store = InMemoryStore()
    Profile(user_id="u", actions=["b", "a"], store=store).policy("ctx")
    reopened = Profile(user_id="u", actions=["a", "b"], store=store)
    export = reopened.export_user()
    original_fingerprint = export["policies"][0]["action_set_fingerprint"]
    export["policies"][0]["actions"].append("private")
    second = reopened.export_user()
    assert second["policies"][0]["actions"] == ["a", "b"]
    assert second["policies"][0]["action_set_fingerprint"] == original_fingerprint


@pytest.mark.parametrize(
    "changed",
    [
        ["a", "b", "c"],
        ["a"],
        ["a", "renamed"],
    ],
)
def test_added_removed_and_renamed_actions_are_rejected(changed) -> None:
    store = InMemoryStore()
    Profile(user_id="u", actions=["a", "b"], store=store).policy("ctx")
    with pytest.raises(ActionSetMismatchError):
        Profile(user_id="u", actions=changed, store=store).policy("ctx")


def test_duplicate_actions_are_rejected_before_storage() -> None:
    with pytest.raises(ValidationError, match="actions must be unique"):
        Profile(user_id="u", actions=["a", "a"])
