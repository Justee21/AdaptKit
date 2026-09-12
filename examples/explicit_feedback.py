from adaptkit import Profile

profile = Profile(
    user_id="user-123",
    actions=["patch_first", "explanation_first"],
    seed=7,
)

action = profile.choose("debugging")
print("chosen:", action)

profile.like("debugging", action)
profile.prefer("debugging", preferred="patch_first", rejected="explanation_first")
print("state:", profile.state())
