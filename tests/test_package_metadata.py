from pathlib import Path

import tomllib


ROOT = Path(__file__).parents[1]


def test_package_metadata() -> None:
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = metadata["project"]

    assert project["name"] == "adaptkit"
    assert project["version"] == "0.1.0"
    assert project["description"]
    assert project["license"] == "MIT"
    assert project["requires-python"] == ">=3.11"
    assert project["dependencies"] == []
    assert set(project["optional-dependencies"]) == {
        "dev",
        "evaluation",
        "memory-letta",
        "memory-mem0",
    }
    assert project["urls"]["Repository"] == "https://github.com/Justee21/AdaptKit"
