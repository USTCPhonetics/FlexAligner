from __future__ import annotations

import subprocess
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 CI path
    import tomli as tomllib


def test_incremental_language_and_audio_dependencies_are_not_in_minimal_install() -> None:
    root = Path(__file__).resolve().parents[1]
    with (root / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)["project"]
    base = tuple(project["dependencies"])
    extras = project["optional-dependencies"]
    assert all(
        not dependency.startswith(("flexaligner-g2p-en", "jieba", "pypinyin", "av"))
        for dependency in base
    )
    assert extras["en"] == ["flexaligner-g2p-en==0.3.0a1"]
    assert extras["zh"] == ["jieba==0.42.1", "pypinyin==0.55.0"]
    assert extras["audio"] == ["av==16.0.1"]


def test_top_level_import_does_not_import_incremental_dependencies() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import sys, flexaligner; "
            "assert not {'flexaligner_g2p_en', 'jieba', 'pypinyin', 'av'}.intersection(sys.modules)",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
