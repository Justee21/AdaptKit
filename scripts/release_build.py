"""Clean, build, and verify exactly one canonical AdaptKit wheel and sdist."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "adaptkit"


def project_version(root: Path) -> str:
    data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    return str(data["project"]["version"])


def expected_distributions(root: Path) -> tuple[Path, Path]:
    version = project_version(root)
    dist = root / "dist"
    return (
        dist / f"{PACKAGE}-{version}-py3-none-any.whl",
        dist / f"{PACKAGE}-{version}.tar.gz",
    )


def _validated_generated_paths(root: Path) -> list[Path]:
    dist = root / "dist"
    if not dist.exists():
        return []
    if dist.is_symlink() or not dist.is_dir():
        raise RuntimeError("dist must be a repository-local directory, not a link")
    pattern = re.compile(r"^adaptkit-[A-Za-z0-9_.+! -]+(?:\.whl|\.tar\.gz)$")
    paths = list(dist.iterdir())
    unexpected = [
        path.name
        for path in paths
        if not path.is_file() or not pattern.fullmatch(path.name)
    ]
    if unexpected:
        raise RuntimeError(f"refusing to clean unexpected dist entries: {sorted(unexpected)}")
    return paths


def clean_generated(root: Path) -> list[str]:
    """Remove only validated repository-local Python build outputs."""
    root = root.resolve()
    dist_paths = _validated_generated_paths(root)
    generated_dirs = (root / "build", root / f"{PACKAGE}.egg-info")
    for path in generated_dirs:
        if path.is_symlink():
            raise RuntimeError(f"refusing to clean linked build path: {path.name}")
        if path.exists() and not path.is_dir():
            raise RuntimeError(f"refusing to clean non-directory build path: {path.name}")
    removed = [str(path.relative_to(root)) for path in dist_paths]
    for path in dist_paths:
        path.unlink()
    dist = root / "dist"
    if dist.exists():
        dist.rmdir()
    for path in generated_dirs:
        if path.exists():
            shutil.rmtree(path)
            removed.append(str(path.relative_to(root)) + "/")
    return removed


def verify_distributions(root: Path) -> tuple[Path, Path]:
    expected = expected_distributions(root)
    dist = root / "dist"
    actual = set(dist.iterdir()) if dist.is_dir() else set()
    if actual != set(expected):
        names = sorted(path.name for path in actual)
        raise RuntimeError(f"expected exactly the canonical wheel and sdist; found {names}")
    if any(
        path.is_symlink() or not path.is_file() or path.stat().st_size == 0
        for path in expected
    ):
        raise RuntimeError("distribution artifacts must be nonempty regular files")
    return expected


def _package_source(root: Path) -> dict[str, bytes]:
    package = root / PACKAGE
    paths = [*package.rglob("*.py"), package / "py.typed"]
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in paths
        if path.is_file()
    }


def _sdist_source(root: Path) -> dict[str, bytes]:
    paths = [
        root / name
        for name in (
            ".env.example",
            "CHANGELOG.md",
            "CONTRIBUTING.md",
            "LICENSE",
            "MANIFEST.in",
            "README.md",
            "RELEASE_CHECKLIST.md",
            "SECURITY.md",
            "pyproject.toml",
        )
    ]
    for directory in (
        "adaptkit",
        "artifacts",
        "benchmarks",
        "docs",
        "examples",
        "scripts",
        "tests",
    ):
        paths.extend(
            path
            for path in (root / directory).rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and not path.name.endswith((".pyc", ".pyo"))
        )
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in paths
        if path.is_file()
    }


def _forbidden_archive_member(name: str) -> bool:
    path = Path(name)
    basename = path.name.casefold()
    parts = {part.casefold() for part in path.parts}
    return (
        "__pycache__" in parts
        or ".git" in parts
        or ".idea" in parts
        or ".vscode" in parts
        or basename == ".env"
        or (basename.startswith(".env.") and basename != ".env.example")
        or basename in {".pypirc", "credentials.json", "secrets.json"}
        or basename.endswith(
            (
                ".pyc",
                ".pyo",
                ".db",
                ".sqlite",
                ".sqlite3",
                ".jsonl",
                ".pem",
                ".key",
                ".p12",
                ".pfx",
            )
        )
        or basename in {".ds_store", "thumbs.db"}
        or basename.endswith((".swp", ".swo", "~"))
    )


def verify_current_source(root: Path, wheel: Path, sdist: Path) -> None:
    """Verify package bytes match the working source and archives exclude private files."""
    source = _package_source(root)
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        forbidden = [name for name in names if _forbidden_archive_member(name)]
        if forbidden:
            raise RuntimeError(f"wheel contains forbidden files: {sorted(forbidden)}")
        package_members = {name for name in names if name.startswith(f"{PACKAGE}/")}
        if package_members != set(source):
            raise RuntimeError("wheel package members do not exactly match current source")
        dist_info = f"{PACKAGE}-{project_version(root)}.dist-info/"
        unexpected_wheel_members = [
            name
            for name in names
            if not name.startswith(f"{PACKAGE}/") and not name.startswith(dist_info)
        ]
        if unexpected_wheel_members:
            raise RuntimeError(
                f"wheel contains unexpected members: {sorted(unexpected_wheel_members)}"
            )
        required_metadata = {
            dist_info + "METADATA",
            dist_info + "WHEEL",
            dist_info + "RECORD",
            dist_info + "top_level.txt",
            dist_info + "licenses/LICENSE",
        }
        if not required_metadata.issubset(names):
            raise RuntimeError("wheel is missing required package metadata")
        for name, expected in source.items():
            if archive.read(name) != expected:
                raise RuntimeError(f"wheel does not match current source: {name}")
    prefix = f"{PACKAGE}-{project_version(root)}/"
    with tarfile.open(sdist, "r:gz") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        forbidden = [name for name in names if _forbidden_archive_member(name)]
        if forbidden:
            raise RuntimeError(f"sdist contains forbidden files: {sorted(forbidden)}")
        expected_source = _sdist_source(root)
        archive_files = {
            member.name.removeprefix(prefix) for member in members if member.isfile()
        }
        generated_metadata = {
            "PKG-INFO",
            "setup.cfg",
            f"{PACKAGE}.egg-info/PKG-INFO",
            f"{PACKAGE}.egg-info/SOURCES.txt",
            f"{PACKAGE}.egg-info/dependency_links.txt",
            f"{PACKAGE}.egg-info/requires.txt",
            f"{PACKAGE}.egg-info/top_level.txt",
        }
        unexpected_sdist_members = (
            archive_files - set(expected_source) - generated_metadata
        )
        missing = set(expected_source) - archive_files
        if unexpected_sdist_members or missing:
            raise RuntimeError(
                "sdist members do not match intended source; "
                f"unexpected={sorted(unexpected_sdist_members)}, missing={sorted(missing)}"
            )
        for name, expected in expected_source.items():
            member = archive.extractfile(prefix + name)
            if member is None or member.read() != expected:
                raise RuntimeError(f"sdist does not match current source: {name}")


def _venv_python(environment: Path) -> Path:
    return environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def check_distributions(root: Path, wheel: Path, sdist: Path) -> None:
    """Run Twine and clean, outside-the-tree install/import checks."""
    subprocess.run(
        [sys.executable, "-m", "twine", "check", str(wheel), str(sdist)],
        cwd=root,
        check=True,
    )
    clean_environment = os.environ.copy()
    clean_environment.pop("PYTHONPATH", None)
    clean_environment["PYTHONDONTWRITEBYTECODE"] = "1"
    version = project_version(root)
    for label, distribution in (("wheel", wheel), ("sdist", sdist)):
        with tempfile.TemporaryDirectory(prefix=f"adaptkit-{label}-check-") as temporary:
            temporary_path = Path(temporary)
            environment = temporary_path / "venv"
            subprocess.run(
                [sys.executable, "-m", "venv", str(environment)],
                cwd=temporary_path,
                check=True,
                env=clean_environment,
            )
            python = _venv_python(environment)
            subprocess.run(
                [
                    str(python),
                    "-m",
                    "pip",
                    "install",
                    "--disable-pip-version-check",
                    "--no-deps",
                    str(distribution),
                ],
                cwd=temporary_path,
                check=True,
                env=clean_environment,
            )
            subprocess.run(
                [
                    str(python),
                    "-c",
                    "import adaptkit; from importlib.metadata import version; "
                    f"assert version('adaptkit') == {version!r}",
                ],
                cwd=temporary_path,
                check=True,
                env=clean_environment,
            )
            print(f"verified clean {label} install outside source tree")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-only", action="store_true", help="verify without cleaning or building"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="also run Twine and separate clean wheel/sdist install checks",
    )
    args = parser.parse_args()
    if not args.verify_only:
        removed = clean_generated(ROOT)
        for path in removed:
            print(f"removed generated artifact: {path}")
        subprocess.run([sys.executable, "-m", "build"], cwd=ROOT, check=True)
    wheel, sdist = verify_distributions(ROOT)
    verify_current_source(ROOT, wheel, sdist)
    print(f"verified wheel: {wheel.name}")
    print(f"verified sdist: {sdist.name}")
    if args.check:
        check_distributions(ROOT, wheel, sdist)


if __name__ == "__main__":
    main()
