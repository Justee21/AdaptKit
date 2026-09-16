from pathlib import Path
import tarfile
import zipfile

import pytest

from scripts.release_build import (
    clean_generated,
    verify_current_source,
    verify_distributions,
)


def _project(root: Path) -> None:
    (root / "pyproject.toml").write_text(
        '[project]\nname = "adaptkit"\nversion = "0.1.0"\n', encoding="utf-8"
    )


def test_clean_removes_only_recognized_repository_build_outputs(tmp_path: Path) -> None:
    _project(tmp_path)
    dist = tmp_path / "dist"
    dist.mkdir()
    stale = dist / "adaptkit-0.1.0-py3-none-any 2.whl"
    stale.write_bytes(b"stale")
    (tmp_path / "build").mkdir()
    (tmp_path / "adaptkit.egg-info").mkdir()
    removed = clean_generated(tmp_path)
    assert removed == [
        "dist/adaptkit-0.1.0-py3-none-any 2.whl",
        "build/",
        "adaptkit.egg-info/",
    ]
    assert not stale.exists()


def test_clean_refuses_unexpected_dist_content_before_removing_anything(
    tmp_path: Path,
) -> None:
    _project(tmp_path)
    dist = tmp_path / "dist"
    dist.mkdir()
    wheel = dist / "adaptkit-0.1.0-py3-none-any.whl"
    wheel.write_bytes(b"wheel")
    (dist / "notes.txt").write_text("user file", encoding="utf-8")
    with pytest.raises(RuntimeError, match="unexpected"):
        clean_generated(tmp_path)
    assert wheel.exists()


def test_verify_requires_exactly_one_canonical_wheel_and_sdist(tmp_path: Path) -> None:
    _project(tmp_path)
    dist = tmp_path / "dist"
    dist.mkdir()
    wheel = dist / "adaptkit-0.1.0-py3-none-any.whl"
    sdist = dist / "adaptkit-0.1.0.tar.gz"
    wheel.write_bytes(b"wheel")
    sdist.write_bytes(b"sdist")
    assert verify_distributions(tmp_path) == (wheel, sdist)
    (dist / "adaptkit-0.1.0-py3-none-any 2.whl").write_bytes(b"stale")
    with pytest.raises(RuntimeError, match="exactly"):
        verify_distributions(tmp_path)


def test_verify_compares_archives_to_current_package_source(tmp_path: Path) -> None:
    _project(tmp_path)
    package = tmp_path / "adaptkit"
    package.mkdir()
    source = package / "__init__.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    (package / "py.typed").write_text("", encoding="utf-8")
    dist = tmp_path / "dist"
    dist.mkdir()
    wheel = dist / "adaptkit-0.1.0-py3-none-any.whl"
    sdist = dist / "adaptkit-0.1.0.tar.gz"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.write(source, "adaptkit/__init__.py")
        archive.write(package / "py.typed", "adaptkit/py.typed")
        for name in (
            "METADATA",
            "WHEEL",
            "RECORD",
            "top_level.txt",
            "licenses/LICENSE",
        ):
            archive.writestr(f"adaptkit-0.1.0.dist-info/{name}", "metadata")
    with tarfile.open(sdist, "w:gz") as archive:
        archive.add(package, arcname="adaptkit-0.1.0/adaptkit")
        archive.add(
            tmp_path / "pyproject.toml",
            arcname="adaptkit-0.1.0/pyproject.toml",
        )
    verify_current_source(tmp_path, wheel, sdist)
    source.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="current source"):
        verify_current_source(tmp_path, wheel, sdist)


def test_verify_rejects_unexpected_wheel_package_member(tmp_path: Path) -> None:
    _project(tmp_path)
    package = tmp_path / "adaptkit"
    package.mkdir()
    source = package / "__init__.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    (package / "py.typed").write_text("", encoding="utf-8")
    dist = tmp_path / "dist"
    dist.mkdir()
    wheel = dist / "adaptkit-0.1.0-py3-none-any.whl"
    sdist = dist / "adaptkit-0.1.0.tar.gz"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.write(source, "adaptkit/__init__.py")
        archive.write(package / "py.typed", "adaptkit/py.typed")
        archive.writestr("adaptkit/debug.txt", "unexpected")
    with tarfile.open(sdist, "w:gz") as archive:
        archive.add(package, arcname="adaptkit-0.1.0/adaptkit")
        archive.add(
            tmp_path / "pyproject.toml",
            arcname="adaptkit-0.1.0/pyproject.toml",
        )
    with pytest.raises(RuntimeError, match="package members"):
        verify_current_source(tmp_path, wheel, sdist)
