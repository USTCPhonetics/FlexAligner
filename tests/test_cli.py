from __future__ import annotations

import importlib.metadata
from collections.abc import Callable
from subprocess import CompletedProcess

import pytest

from tests._support import AVAILABLE_IDS, PLACEHOLDER_IDS, parse_json_stream


def test_help_is_successful_and_lists_stage1_commands(
    run_cli: Callable[..., CompletedProcess[str]],
) -> None:
    result = run_cli("--help")
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert "usage:" in result.stdout.lower()
    for command in ("align", "capabilities", "batch", "serve", "models"):
        assert command in result.stdout


def test_version_matches_installed_distribution(
    run_cli: Callable[..., CompletedProcess[str]],
) -> None:
    result = run_cli("--version")
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert result.stdout.strip() == f"flexaligner {importlib.metadata.version('flexaligner')}"


def test_capabilities_json_is_stable_and_complete(
    run_cli: Callable[..., CompletedProcess[str]],
) -> None:
    first = run_cli("capabilities", "--json")
    second = run_cli("capabilities", "--json")
    assert first.returncode == second.returncode == 0
    assert first.stderr == second.stderr == ""
    assert first.stdout == second.stdout

    payload = parse_json_stream(first.stdout)
    assert payload["schema_version"] == 1
    serialized = first.stdout
    for capability_id in sorted(AVAILABLE_IDS | PLACEHOLDER_IDS):
        assert capability_id in serialized


def test_human_capabilities_are_deterministic(
    run_cli: Callable[..., CompletedProcess[str]],
) -> None:
    first = run_cli("capabilities")
    second = run_cli("capabilities")
    assert first.returncode == second.returncode == 0
    assert first.stderr == second.stderr == ""
    assert first.stdout == second.stdout
    assert "available" in first.stdout
    assert "placeholder" in first.stdout


@pytest.mark.parametrize(
    ("args", "capability_id"),
    [
        (("batch",), "alignment.batch"),
        (("serve",), "integration.web"),
        (("models", "fetch"), "models.auto_download"),
    ],
)
def test_placeholder_commands_emit_stable_json_error(
    run_cli: Callable[..., CompletedProcess[str]],
    args: tuple[str, ...],
    capability_id: str,
) -> None:
    result = run_cli(*args)
    assert result.returncode != 0
    assert result.stdout == ""
    payload = parse_json_stream(result.stderr)
    assert set(payload) == {"code", "message", "context"}
    assert payload["code"] == "feature_not_available"
    assert payload["context"]["capability"] == capability_id
    assert payload["context"]["status"] == "placeholder"


def test_all_placeholder_cli_commands_share_one_exit_status(
    run_cli: Callable[..., CompletedProcess[str]],
) -> None:
    statuses = {
        run_cli("batch").returncode,
        run_cli("serve").returncode,
        run_cli("models", "fetch").returncode,
    }
    assert len(statuses) == 1
    assert next(iter(statuses)) > 0


def test_align_reports_strict_input_failure_without_output(
    run_cli: Callable[..., CompletedProcess[str]],
) -> None:
    result = run_cli(
        "align",
        "--audio",
        "/definitely/missing/input.wav",
        "--text",
        "hello",
        "--lexicon",
        "/definitely/missing/lexicon.dict",
        "--chunker-model",
        "/definitely/missing/chunker",
        "--aligner-model",
        "/definitely/missing/aligner",
        "--output",
        "/definitely/missing/result.TextGrid",
    )
    assert result.returncode != 0
    assert result.stdout == ""
    payload = parse_json_stream(result.stderr)
    assert payload["code"] == "input_validation_error"
