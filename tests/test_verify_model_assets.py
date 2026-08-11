from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "verify_model_assets.py"
CANDIDATE_MANIFEST = REPO_ROOT / "tests" / "fixtures" / "e2e" / "asset_manifest.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_manifest(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _manifest_payload(
    *,
    files: list[dict[str, str]],
    root: str | None = None,
    root_env: str | None = None,
    runtime: dict[str, str] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "fixture_id": "unit-test-fixture",
        "status": "candidate",
        "runtime": {"python": platform.python_version()} if runtime is None else runtime,
        "provenance": {"source_revision": None, "reason": "unit test"},
        "files": files,
    }
    if root is not None:
        payload["root"] = root
    if root_env is not None:
        payload["root_env"] = root_env
    return payload


def _run(
    manifest: Path,
    *,
    env: dict[str, str] | None = None,
    check_runtime: bool = False,
    require_approved: bool = False,
) -> subprocess.CompletedProcess[str]:
    clean_env = os.environ.copy()
    if env:
        clean_env.update(env)
    command = [sys.executable, str(SCRIPT), str(manifest)]
    if check_runtime:
        command.append("--check-runtime")
    if require_approved:
        command.append("--require-approved")
    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=clean_env,
    )


def test_environment_root_manifest_verifies_frozen_file(tmp_path: Path) -> None:
    root = tmp_path / "assets"
    root.mkdir()
    asset = root / "model.bin"
    asset.write_bytes(b"frozen-model")
    manifest = tmp_path / "manifest.json"
    _write_manifest(
        manifest,
        _manifest_payload(
            root_env="FLEXALIGNER_TEST_ASSET_ROOT",
            files=[{"path": "model.bin", "sha256": _sha256(asset), "role": "model"}],
        ),
    )

    result = _run(manifest, env={"FLEXALIGNER_TEST_ASSET_ROOT": str(root)})

    assert result.returncode == 0
    assert "MODEL_E2E_ASSETS_OK" in result.stdout
    assert "file_count=1" in result.stdout


def test_exact_runtime_check_accepts_current_python_and_distribution(tmp_path: Path) -> None:
    asset = tmp_path / "model.bin"
    asset.write_bytes(b"runtime-bound-model")
    manifest = tmp_path / "manifest.json"
    _write_manifest(
        manifest,
        _manifest_payload(
            root=str(tmp_path),
            runtime={
                "python": platform.python_version(),
                "pytest": importlib.metadata.version("pytest"),
            },
            files=[{"path": "model.bin", "sha256": _sha256(asset), "role": "model"}],
        ),
    )

    result = _run(manifest, check_runtime=True)

    assert result.returncode == 0
    assert "MODEL_E2E_ASSETS_OK" in result.stdout


def test_runtime_mismatch_is_an_explicit_blocker(tmp_path: Path) -> None:
    asset = tmp_path / "model.bin"
    asset.write_bytes(b"runtime-mismatch")
    manifest = tmp_path / "manifest.json"
    _write_manifest(
        manifest,
        _manifest_payload(
            root=str(tmp_path),
            runtime={"python": "0.0.0"},
            files=[{"path": "model.bin", "sha256": _sha256(asset), "role": "model"}],
        ),
    )

    result = _run(manifest, check_runtime=True)

    assert result.returncode != 0
    assert "MODEL_E2E_BLOCKED: Python version mismatch" in result.stderr


def test_release_gate_blocks_candidate_manifest_before_asset_access(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    _write_manifest(
        manifest,
        _manifest_payload(
            root=str(tmp_path),
            files=[{"path": "absent.bin", "sha256": "0" * 64, "role": "model"}],
        ),
    )

    result = _run(manifest, require_approved=True)

    assert result.returncode != 0
    assert "MODEL_E2E_BLOCKED: manifest is not approved" in result.stderr
    assert "status=candidate" in result.stderr
    assert "asset is absent" not in result.stderr


def test_release_gate_accepts_approved_manifest(tmp_path: Path) -> None:
    asset = tmp_path / "model.bin"
    asset.write_bytes(b"approved-model")
    manifest = tmp_path / "manifest.json"
    payload = _manifest_payload(
        root=str(tmp_path),
        files=[{"path": "model.bin", "sha256": _sha256(asset), "role": "model"}],
    )
    payload["status"] = "approved"
    _write_manifest(manifest, payload)

    result = _run(manifest, require_approved=True)

    assert result.returncode == 0
    assert "MODEL_E2E_ASSETS_OK" in result.stdout


def test_committed_candidate_manifest_has_frozen_runtime_and_provenance() -> None:
    payload = json.loads(CANDIDATE_MANIFEST.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 1
    assert payload["status"] == "candidate"
    assert payload["root_env"] == "FLEXALIGNER_E2E_ASSET_ROOT"
    assert payload["runtime"] == {
        "python": "3.10.8",
        "numpy": "2.2.6",
        "torch": "2.3.1",
        "transformers": "4.41.2",
    }
    assert payload["provenance"]["source_revision"] is None
    assert payload["provenance"]["oov_pronunciation_approval"] == "TBD-E2E-001"

    entries = payload["files"]
    roles = [entry["role"] for entry in entries]
    assert len(entries) == 16
    assert len(set(roles)) == len(roles)
    assert all(not Path(entry["path"]).is_absolute() for entry in entries)
    assert all(len(entry["sha256"]) == 64 for entry in entries)

    fixture_entry = next(entry for entry in entries if entry["role"] == "effective_fixture_lexicon")
    fixture_path = REPO_ROOT / "tests" / "fixtures" / "e2e" / "english_synthetic.dict"
    assert fixture_entry["sha256"] == _sha256(fixture_path)


def test_unset_environment_root_is_an_explicit_blocker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("FLEXALIGNER_ABSENT_ASSET_ROOT", raising=False)
    manifest = tmp_path / "manifest.json"
    _write_manifest(
        manifest,
        _manifest_payload(
            root_env="FLEXALIGNER_ABSENT_ASSET_ROOT",
            files=[{"path": "model.bin", "sha256": "0" * 64, "role": "model"}],
        ),
    )

    result = _run(manifest)

    assert result.returncode != 0
    assert "MODEL_E2E_BLOCKED" in result.stderr
    assert "FLEXALIGNER_ABSENT_ASSET_ROOT" in result.stderr


def test_missing_asset_is_an_explicit_blocker(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    _write_manifest(
        manifest,
        _manifest_payload(
            root=str(tmp_path),
            files=[{"path": "missing.bin", "sha256": "0" * 64, "role": "model"}],
        ),
    )

    result = _run(manifest)

    assert result.returncode != 0
    assert "MODEL_E2E_BLOCKED: asset is absent" in result.stderr


def test_manifest_rejects_path_traversal(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    _write_manifest(
        manifest,
        _manifest_payload(
            root=str(tmp_path),
            files=[{"path": "../outside.bin", "sha256": "0" * 64, "role": "model"}],
        ),
    )

    result = _run(manifest)

    assert result.returncode != 0
    assert "Unsafe relative asset path" in result.stderr
