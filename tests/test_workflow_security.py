from __future__ import annotations

import re
from pathlib import Path

from tests._support import REPO_ROOT

WORKFLOW_ROOT = REPO_ROOT / ".github" / "workflows"
FULL_SHA_ACTION = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+@[0-9a-f]{40}$")


def _workflow_sources() -> dict[str, str]:
    return {
        path.name: path.read_text(encoding="utf-8") for path in sorted(WORKFLOW_ROOT.glob("*.yml"))
    }


def _external_action_refs(source: str) -> list[str]:
    refs: list[str] = []
    for line in source.splitlines():
        stripped = line.strip()
        if not stripped.startswith("uses:"):
            continue
        ref = stripped.removeprefix("uses:").split("#", 1)[0].strip()
        if not ref.startswith("./"):
            refs.append(ref)
    return refs


def test_all_external_actions_are_pinned_to_full_commit_sha() -> None:
    sources = _workflow_sources()
    assert sources
    refs = [ref for source in sources.values() for ref in _external_action_refs(source)]
    assert refs
    assert all(FULL_SHA_ACTION.fullmatch(ref) for ref in refs), refs


def test_fast_ci_has_read_only_permissions_and_no_privileged_trigger() -> None:
    source = _workflow_sources()["ci.yml"]
    assert "pull_request_target" not in source
    assert "id-token: write" not in source
    assert "permissions:\n  contents: read" in source
    assert 'HF_HUB_OFFLINE: "1"' in source
    assert 'HF_HUB_DISABLE_TELEMETRY: "1"' in source
    assert 'TRANSFORMERS_OFFLINE: "1"' in source
    assert "--disable-socket" in source
    assert '-m "not model_e2e"' in source
    assert "python -m build --no-isolation --sdist --wheel" in source


def test_release_is_tag_only_and_oidc_is_confined_to_publish_job() -> None:
    sources = _workflow_sources()
    release = sources["release.yml"]
    assert "pull_request:" not in release
    assert "pull_request_target:" not in release
    assert "branches:" not in release
    assert 'tags:\n      - "v*"' in release
    assert "PYPI_RELEASE_AUTHORIZED == 'true'" in release
    assert "name: pypi" in release
    assert "model-e2e" in release
    assert sum(source.count("id-token: write") for source in sources.values()) == 1
    publish_block = release.split("\n  publish:\n", 1)[1]
    assert "id-token: write" in publish_block
    assert "actions/checkout@" not in publish_block
    assert "python -m build" not in publish_block


def test_build_backend_is_exact_and_used_without_isolation() -> None:
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'requires = ["hatchling==1.32.0"]' in pyproject
    build_command = "python -m build --no-isolation --sdist --wheel"
    assert build_command in _workflow_sources()["ci.yml"]
    assert build_command in _workflow_sources()["release.yml"]


def test_model_e2e_is_offline_and_fails_closed_without_manifest() -> None:
    source = _workflow_sources()["model-e2e.yml"]
    assert "pull_request:" not in source
    assert "pull_request_target:" not in source
    assert 'HF_HUB_OFFLINE: "1"' in source
    assert 'HF_HUB_DISABLE_TELEMETRY: "1"' in source
    assert 'TRANSFORMERS_OFFLINE: "1"' in source
    assert "--no-index" in source
    assert "verify_model_assets.py" in source
    assert "--check-runtime" in source
    assert "--require-approved" in source
    assert "MODEL_E2E_BLOCKED" in source
    assert "-m model_e2e" in source
    for downloader in ("curl ", "wget ", "huggingface-cli", "hf download"):
        assert downloader not in source


def test_workflow_files_are_regular_utf8_files() -> None:
    for path in sorted(WORKFLOW_ROOT.glob("*.yml")):
        assert isinstance(path, Path)
        assert path.is_file()
        path.read_text(encoding="utf-8", errors="strict")
