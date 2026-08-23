from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests._support import REPO_ROOT


def _report(versions: dict[str, str]) -> dict[str, object]:
    return {
        "install": [
            {"metadata": {"name": name, "version": version}} for name, version in versions.items()
        ]
    }


def _run_verifier(report: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "verify_inference_resolution.py"),
            str(report),
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_resolution_report_accepts_frozen_alpha_versions(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(_report({"Torch": "2.3.1", "transformers": "4.41.2"})),
        encoding="utf-8",
    )

    result = _run_verifier(report)

    assert result.returncode == 0, result.stderr
    assert "INFERENCE_RESOLUTION_OK" in result.stdout


@pytest.mark.parametrize(
    "versions",
    [
        {"torch": "2.4.0", "transformers": "4.41.2"},
        {"transformers": "4.41.2"},
    ],
)
def test_resolution_report_rejects_drift_or_missing_runtime(
    tmp_path: Path, versions: dict[str, str]
) -> None:
    report = tmp_path / "report.json"
    report.write_text(json.dumps(_report(versions)), encoding="utf-8")
    result = _run_verifier(report)
    assert result.returncode != 0
    assert "Inference resolution mismatch" in result.stderr
