from adaptkit import Profile

profile = Profile(
    user_id="user-123",
    actions=["patch_first", "explanation_first"],
    seed=7,
)

decision = profile.choose("debugging")
print("chosen:", decision.action)

profile.like(decision, idempotency_key="thumb-up-1")
profile.prefer(
    "debugging",
    preferred="patch_first",
    rejected="explanation_first",
    idempotency_key="pairwise-1",
)
print("state:", profile.state())
