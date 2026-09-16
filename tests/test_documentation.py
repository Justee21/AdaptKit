import re
from pathlib import Path

import tomllib


ROOT = Path(__file__).parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")


def test_public_alpha_positioning_and_reward_semantics_are_explicit() -> None:
    assert (
        "AdaptKit 0.1.0 is a public-alpha, provider-independent Python SDK "
        "that learns per-user agent behavior preferences using contextual bandits."
        in README
    )
    assert "Production integrations should begin in shadow mode" in README
    assert "r\\in\\{0,1\\}" in README
    assert "not a general reinforcement-learning framework" in README


def test_core_dependencies_are_empty_and_provider_integrations_are_optional() -> None:
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert metadata["project"]["dependencies"] == []
    extras = metadata["project"]["optional-dependencies"]
    assert "evaluation" in extras
    assert "memory-mem0" in extras
    assert "memory-letta" in extras


def test_repository_links_in_markdown_point_to_existing_files() -> None:
    paths = [ROOT / "README.md"]
    paths.extend((ROOT / "docs").glob("*.md"))
    markdown = "\n".join(
        path.read_text(encoding="utf-8") for path in paths
    )
    targets = re.findall(
        r"https://github\.com/Justee21/AdaptKit/blob/main/([^\s)#]+)", markdown
    )
    assert targets
    for target in targets:
        assert (ROOT / target).is_file(), target


def test_markdown_uses_github_math_delimiters() -> None:
    paths = list(ROOT.glob("*.md"))
    paths.extend((ROOT / "docs").glob("*.md"))
    for path in paths:
        markdown = path.read_text(encoding="utf-8")
        assert "\\(" not in markdown, path
        assert "\\)" not in markdown, path
        assert "\\[" not in markdown, path
        assert "\\]" not in markdown, path


def test_ci_does_not_hardcode_distribution_version_or_use_wildcards() -> None:
    workflow = (ROOT / ".github/workflows/tests.yml").read_text(encoding="utf-8")
    assert "adaptkit-0.1.0" not in workflow
    assert "dist/*" not in workflow
    assert "scripts/release_build.py --check" in workflow


def test_x_draft_fits_one_post() -> None:
    launch = (ROOT / "docs/launch.md").read_text(encoding="utf-8")
    draft = launch.split("## X draft\n\n", 1)[1].split("\n\n", 1)[0]
    assert len(draft) <= 280
