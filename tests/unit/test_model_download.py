from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

import flexaligner.model_download as model_download
from flexaligner import ModelDownloadError, ModelValidationError


def _make_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    snapshot = tmp_path / "snapshot"
    entries: list[dict[str, object]] = []
    for role in ("chunker", "aligner"):
        for filename in (
            "config.json",
            "model.safetensors",
            "preprocessor_config.json",
            "special_tokens_map.json",
            "tokenizer_config.json",
            "vocab.json",
        ):
            relative = f"en/{role}/{filename}"
            path = snapshot / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = f"{role}:{filename}".encode()
            path.write_bytes(payload)
            entries.append(
                {
                    "path": relative,
                    "size": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
    manifest = {
        "schema_version": 1,
        "bundle_version": "0.2.0a1",
        "languages": {
            "en": {
                "chunker_path": "en/chunker",
                "aligner_path": "en/aligner",
            }
        },
        "files": entries,
    }
    manifest_path = snapshot / "model_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(
        model_download,
        "MODEL_MANIFEST_SHA256",
        hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    )
    return snapshot


def test_default_cache_precedence(tmp_path: Path) -> None:
    assert (
        model_download.default_model_cache_dir(
            {"HF_HUB_CACHE": str(tmp_path / "hub"), "HF_HOME": str(tmp_path / "home")}
        )
        == tmp_path / "hub"
    )
    assert model_download.default_model_cache_dir({"HF_HOME": str(tmp_path / "home")}) == (
        tmp_path / "home" / "hub"
    )
    assert (
        model_download.default_model_cache_dir({"XDG_CACHE_HOME": str(tmp_path / "xdg")})
        == tmp_path / "xdg" / "huggingface" / "hub"
    )


def test_validate_snapshot_returns_local_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = _make_snapshot(tmp_path, monkeypatch)
    bundle = model_download.validate_english_snapshot(snapshot)
    assert bundle.chunker_dir == snapshot / "en/chunker"
    assert bundle.aligner_dir == snapshot / "en/aligner"
    assert bundle.manifest_path == snapshot / "model_manifest.json"


def test_validate_snapshot_rejects_manifest_trust_root_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = _make_snapshot(tmp_path, monkeypatch)
    monkeypatch.setattr(model_download, "MODEL_MANIFEST_SHA256", "0" * 64)
    with pytest.raises(ModelValidationError, match="trust root"):
        model_download.validate_english_snapshot(snapshot)


def test_validate_snapshot_rejects_hash_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = _make_snapshot(tmp_path, monkeypatch)
    (snapshot / "en/chunker/config.json").write_text("changed", encoding="utf-8")
    with pytest.raises(ModelValidationError, match=r"size|SHA-256"):
        model_download.validate_english_snapshot(snapshot)


def test_validate_snapshot_rejects_extra_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = _make_snapshot(tmp_path, monkeypatch)
    (snapshot / "en/chunker/extra.bin").write_bytes(b"unexpected")
    with pytest.raises(ModelValidationError, match="file set"):
        model_download.validate_english_snapshot(snapshot)


def test_cache_lookup_is_strictly_local_and_anonymous(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = _make_snapshot(tmp_path, monkeypatch)
    captured: dict[str, Any] = {}

    def fake_snapshot_download(**kwargs: object) -> str:
        captured.update(kwargs)
        return str(snapshot)

    monkeypatch.setattr(model_download, "_snapshot_download", fake_snapshot_download)
    bundle = model_download.find_cached_english_models(cache_dir=tmp_path / "cache")
    assert bundle is not None
    assert captured["local_files_only"] is True
    assert captured["token"] is False
    assert captured["revision"] == model_download.DEFAULT_MODEL_REVISION
    assert captured["allow_patterns"] == list(model_download._ALLOW_PATTERNS)


def test_cache_miss_returns_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class Missing(Exception):
        pass

    def missing(**kwargs: object) -> str:
        del kwargs
        raise Missing

    monkeypatch.setattr(model_download, "_snapshot_download", missing)
    monkeypatch.setattr(model_download, "_is_local_cache_miss", lambda error: True)
    assert model_download.find_cached_english_models(cache_dir=tmp_path) is None


def test_download_uses_one_selected_endpoint_without_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = _make_snapshot(tmp_path, monkeypatch)
    captured: dict[str, Any] = {}

    def fake_snapshot_download(**kwargs: object) -> str:
        captured.update(kwargs)
        return str(snapshot)

    monkeypatch.setattr(model_download, "_snapshot_download", fake_snapshot_download)
    model_download.download_english_models(cache_dir=tmp_path / "cache", source="mirror")
    assert captured["endpoint"] == "https://hf-mirror.com"
    assert captured["local_files_only"] is False
    assert captured["token"] is False
    assert captured["force_download"] is False


def test_explicit_repair_forces_hub_redownload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = _make_snapshot(tmp_path, monkeypatch)
    captured: dict[str, Any] = {}

    def fake_snapshot_download(**kwargs: object) -> str:
        captured.update(kwargs)
        return str(snapshot)

    monkeypatch.setattr(model_download, "_snapshot_download", fake_snapshot_download)
    model_download.download_english_models(
        cache_dir=tmp_path / "cache",
        source="official",
        force_download=True,
    )

    assert captured["endpoint"] == "https://huggingface.co"
    assert captured["force_download"] is True


def test_forced_repair_still_revalidates_every_snapshot_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = _make_snapshot(tmp_path, monkeypatch)
    (snapshot / "en/aligner/model.safetensors").unlink()
    monkeypatch.setattr(model_download, "_snapshot_download", lambda **kwargs: str(snapshot))

    with pytest.raises(ModelValidationError, match="file set"):
        model_download.download_english_models(
            cache_dir=tmp_path / "cache",
            source="mirror",
            force_download=True,
        )


def test_download_wraps_network_error_without_sensitive_exception_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(**kwargs: object) -> str:
        del kwargs
        raise RuntimeError("https://token@example.invalid/private?secret=yes")

    monkeypatch.setattr(model_download, "_snapshot_download", fail)
    with pytest.raises(ModelDownloadError) as caught:
        model_download.download_english_models(cache_dir=tmp_path, source="official")
    serialized = json.dumps(caught.value.to_dict())
    assert "token@example" not in serialized
    assert caught.value.context["exception_type"] == "RuntimeError"
