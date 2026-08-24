from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tests._support import REPO_ROOT


def _write_pyproject(path: Path, version: str) -> None:
    path.write_text(f'[project]\nname = "flexaligner"\nversion = "{version}"\n', encoding="utf-8")
    language_pack = path.parent / "packages" / "flexaligner-g2p-en"
    language_pack.mkdir(parents=True)
    (language_pack / "pyproject.toml").write_text(
        f'[project]\nname = "flexaligner-g2p-en"\nversion = "{version}"\n',
        encoding="utf-8",
    )


def _run_guard(pyproject: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "check_release.py"),
            *arguments,
            "--pyproject",
            str(pyproject),
        ],
        check=False,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("arguments", [("--tag", "v0.1.0a1"), ("--version-only",)])
def test_release_guard_accepts_only_the_approved_alpha_boundary(
    tmp_path: Path, arguments: tuple[str, ...]
) -> None:
    pyproject = tmp_path / "pyproject.toml"
    _write_pyproject(pyproject, "0.1.0a1")

    result = _run_guard(pyproject, *arguments)

    assert result.returncode == 0, result.stderr
    assert "RELEASE_SOURCE_OK name=flexaligner version=0.1.0a1" in result.stdout


def test_release_guard_rejects_tag_version_mismatch(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    _write_pyproject(pyproject, "0.1.0a1")

    result = _run_guard(pyproject, "--tag", "v0.1.0a2")

    assert result.returncode != 0
    assert "Tag/version mismatch" in result.stderr


@pytest.mark.parametrize(
    "version",
    [
        "0.1.0.dev0",
        "0.1.0",
        "0.1.0b1",
        "0.1.0rc1",
        "0.1.0.post1",
        "0.1.0a1+local",
        "0.1.0a01",
    ],
)
def test_release_guard_rejects_noncanonical_or_nonalpha_versions(
    tmp_path: Path, version: str
) -> None:
    pyproject = tmp_path / "pyproject.toml"
    _write_pyproject(pyproject, version)

    result = _run_guard(pyproject, "--version-only")

    assert result.returncode != 0
    assert "accepts only canonical X.Y.ZaN versions" in result.stderr
