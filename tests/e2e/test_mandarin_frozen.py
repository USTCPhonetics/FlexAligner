"""Non-skippable release E2E for the schema-2 Mandarin bundle."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path, PurePosixPath
from typing import Any

import pytest

from flexaligner import (
    AlignmentOptions,
    AlignmentRequest,
    AlignmentResult,
    Device,
    FlexAligner,
    Language,
    LocalModelBundle,
    PronunciationMode,
    TextGridOutput,
)
from flexaligner.model_download import validate_model_snapshot
from flexaligner.textgrid import parse_textgrid_long, validate_textgrid_structure

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "tests" / "fixtures" / "e2e" / "mandarin_asset_manifest.json"
EXPECTED_SYSTEM_WORDS = ("日", "月", "无", "一")
EXPECTED_SYSTEM_PHONES = ("r", "iz4", "ve4", "u2", "i1")
EXPECTED_OVERRIDE_PHONES = ("r", "i4", "ve4", "u2", "i1")
EXPECTED_G2P_WORDS = ("日月星辰",)
EXPECTED_G2P_PHONES = ("r", "iz4", "ve4", "x", "ing1", "ch", "en2")
EXPECTED_OUTPUT_HASHES = {
    "system_textgrid": "d783e25677a58c4659519e382d929bcc7be7f9cfa938b5d3b668f6254e605232",
    "system_metadata": "755e5264d226757c56f08a9b2fe7e2814627a7fe0e6c8c04fd13f6cc0eefefed",
    "override_textgrid": "1860aa131681c23f8a349992bd5cd2f0b52a7e1306c15957b8fa7de7b8df15ff",
    "override_metadata": "61216e79f729649138f0884e9c0306dbbc6b52b59d66e44638eb02eac795549f",
    "g2p_textgrid": "3f9239c16508addc082046c6ff859e9ae1d0303e3731960e6bc9dfcb5b3cfcf6",
    "g2p_metadata": "a59aaeb650168ba657b14cf38c706a6d1154a0328e8c1d080772828eb86e7d28",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _required_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        pytest.fail(f"MODEL_E2E_BLOCKED: manifest field {field!r} is not a string")
    return value


def _load_assets() -> dict[str, Path]:
    try:
        payload: Any = json.loads(MANIFEST_PATH.read_text(encoding="utf-8", errors="strict"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        pytest.fail(f"MODEL_E2E_BLOCKED: Mandarin manifest cannot be parsed: {error}")
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        pytest.fail("MODEL_E2E_BLOCKED: Mandarin manifest schema_version must be 1")
    if payload.get("status") != "approved":
        pytest.fail("MODEL_E2E_BLOCKED: Mandarin manifest is not approved")
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict) or provenance.get("model_revision") != (
        "6720609c167c8ddcd09ea4cb9223aeb569150df8"
    ):
        pytest.fail("MODEL_E2E_BLOCKED: Mandarin model revision is not frozen")
    root_env = _required_string(payload.get("root_env"), field="root_env")
    root_text = os.environ.get(root_env)
    if not root_text:
        pytest.fail(f"MODEL_E2E_BLOCKED: required asset root is unset: {root_env}")
    root = Path(root_text)
    if not root.is_absolute():
        pytest.fail("MODEL_E2E_BLOCKED: Mandarin asset root must be absolute")

    runtime = payload.get("runtime")
    if not isinstance(runtime, dict):
        pytest.fail("MODEL_E2E_BLOCKED: Mandarin runtime must be an object")
    expected_python = _required_string(runtime.get("python"), field="runtime.python")
    if platform.python_version() != expected_python:
        pytest.fail(
            "MODEL_E2E_BLOCKED: Python version mismatch: "
            f"expected={expected_python}, actual={platform.python_version()}"
        )
    for distribution, raw_expected in runtime.items():
        if distribution == "python":
            continue
        expected = _required_string(raw_expected, field=f"runtime.{distribution}")
        try:
            actual = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            pytest.fail(f"MODEL_E2E_BLOCKED: distribution is absent: {distribution}")
        if actual != expected:
            pytest.fail(
                "MODEL_E2E_BLOCKED: distribution version mismatch: "
                f"name={distribution}, expected={expected}, actual={actual}"
            )

    entries = payload.get("files")
    if not isinstance(entries, list) or not entries:
        pytest.fail("MODEL_E2E_BLOCKED: Mandarin files must be a non-empty list")
    assets: dict[str, Path] = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            pytest.fail(f"MODEL_E2E_BLOCKED: Mandarin files[{index}] is invalid")
        role = _required_string(entry.get("role"), field=f"files[{index}].role")
        relative = _required_string(entry.get("path"), field=f"files[{index}].path")
        expected_hash = _required_string(entry.get("sha256"), field=f"files[{index}].sha256")
        posix_path = PurePosixPath(relative)
        if posix_path.is_absolute() or ".." in posix_path.parts or role in assets:
            pytest.fail(f"MODEL_E2E_BLOCKED: unsafe or duplicate Mandarin asset: {entry}")
        asset = root.joinpath(*posix_path.parts)
        if not asset.is_file() or _sha256(asset) != expected_hash:
            pytest.fail(f"MODEL_E2E_BLOCKED: Mandarin asset is absent or changed: {asset}")
        assets[role] = asset
    return assets


def _run_case(
    *,
    tmp_path: Path,
    assets: dict[str, Path],
    models: LocalModelBundle,
    stem: str,
    audio_role: str,
    transcript_role: str,
    pronunciation_mode: PronunciationMode,
    lexicon_path: Path | None = None,
) -> tuple[AlignmentResult, Path, Path, dict[str, Any]]:
    output = tmp_path / f"{stem}.TextGrid"
    metadata_path = tmp_path / f"{stem}.chunker.json"
    with FlexAligner(
        models=models,
        lexicon_path=lexicon_path,
        options=AlignmentOptions(
            device=Device.CPU,
            language=Language.ZH,
            pronunciation_mode=pronunciation_mode,
            num_threads=1,
        ),
    ) as engine:
        result = engine.align(
            AlignmentRequest(
                audio_path=assets[audio_role],
                transcript=assets[transcript_role].read_text(encoding="utf-8"),
                output=TextGridOutput(path=output, chunk_metadata_path=metadata_path),
                utterance_id=f"mandarin-{stem}-e2e",
            )
        )
    parsed = parse_textgrid_long(output)
    validate_textgrid_structure(parsed, context=f"Mandarin frozen {stem} release E2E")
    metadata: dict[str, Any] = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["schema_version"] == "2"
    return result, output, metadata_path, metadata


def _lexical_phones(result: AlignmentResult) -> tuple[str, ...]:
    return tuple(interval.label for interval in result.phones if interval.word_index is not None)


@pytest.mark.model_e2e
@pytest.mark.filterwarnings("error:A test tried to use socket\\.socket\\.")
def test_frozen_mandarin_bundle_system_override_and_g2p_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name, value in {
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }.items():
        monkeypatch.setenv(name, value)
    assets = _load_assets()
    models = validate_model_snapshot(
        assets["zh_model_manifest"].parent,
        language=Language.ZH,
    )

    system, system_grid, system_metadata_path, system_metadata = _run_case(
        tmp_path=tmp_path,
        assets=assets,
        models=models,
        stem="system",
        audio_role="zh_audio",
        transcript_role="zh_transcript",
        pronunciation_mode=PronunciationMode.LEXICON_ONLY,
    )
    assert system.normalized_words == EXPECTED_SYSTEM_WORDS
    assert system.provenance.algorithm_profile == "zh-online-core-v1"
    assert _lexical_phones(system) == EXPECTED_SYSTEM_PHONES
    assert [item["pronunciation_source"] for item in system_metadata["words"]] == [
        "system",
    ] * len(EXPECTED_SYSTEM_WORDS)

    user_lexicon = tmp_path / "user.dict"
    user_lexicon.write_text("日 r i4\n", encoding="utf-8")
    override, override_grid, override_metadata_path, override_metadata = _run_case(
        tmp_path=tmp_path,
        assets=assets,
        models=models,
        stem="override",
        audio_role="zh_audio",
        transcript_role="zh_transcript",
        pronunciation_mode=PronunciationMode.LEXICON_ONLY,
        lexicon_path=user_lexicon,
    )
    assert override.normalized_words == EXPECTED_SYSTEM_WORDS
    assert _lexical_phones(override) == EXPECTED_OVERRIDE_PHONES
    assert [item["pronunciation_source"] for item in override_metadata["words"]] == [
        "user",
        "system",
        "system",
        "system",
    ]

    g2p, g2p_grid, g2p_metadata_path, g2p_metadata = _run_case(
        tmp_path=tmp_path,
        assets=assets,
        models=models,
        stem="g2p",
        audio_role="zh_g2p_audio",
        transcript_role="zh_g2p_transcript",
        pronunciation_mode=PronunciationMode.G2P,
    )
    assert g2p.normalized_words == EXPECTED_G2P_WORDS
    assert _lexical_phones(g2p) == EXPECTED_G2P_PHONES
    assert [item["pronunciation_source"] for item in g2p_metadata["words"]] == ["g2p"]
    assert len(g2p.pronunciation_notices) == 1
    assert g2p.pronunciation_notices[0].word == "日月星辰"

    actual_hashes = {
        "system_textgrid": _sha256(system_grid),
        "system_metadata": _sha256(system_metadata_path),
        "override_textgrid": _sha256(override_grid),
        "override_metadata": _sha256(override_metadata_path),
        "g2p_textgrid": _sha256(g2p_grid),
        "g2p_metadata": _sha256(g2p_metadata_path),
    }
    assert actual_hashes == EXPECTED_OUTPUT_HASHES
