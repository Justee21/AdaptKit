from __future__ import annotations

from collections.abc import Mapping, Sequence

from .exceptions import ValidationError
from .priors import BetaPrior


def cold_start_priors(
    *,
    actions: Sequence[str],
    preferred_action: str | None,
    preferred_prior: BetaPrior | None = None,
) -> Mapping[str, BetaPrior]:
    """Translate one app-validated durable preference into a modest action prior.

    External memory text is intentionally not accepted here. The application must
    validate and map it to an action key before calling this provider-neutral helper.
    """
    if isinstance(actions, (str, bytes)) or not actions:
        raise ValidationError("actions must be a non-empty sequence of strings")
    normalized = tuple(actions)
    if any(not isinstance(action, str) or not action.strip() for action in normalized):
        raise ValidationError("actions must contain non-empty strings")
    if len(set(normalized)) != len(normalized):
        raise ValidationError("actions must be unique")
    if preferred_action is None:
        return {}
    if not isinstance(preferred_action, str) or not preferred_action.strip():
        raise ValidationError("preferred_action must be a non-empty string or None")
    if preferred_action not in normalized:
        raise ValidationError("preferred_action must be one of actions")
    prior = BetaPrior(4, 2) if preferred_prior is None else preferred_prior
    if not isinstance(prior, BetaPrior):
        raise ValidationError("preferred_prior must be a BetaPrior or None")
    return {preferred_action: prior}
