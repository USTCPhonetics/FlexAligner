"""Non-skippable release E2E for the approved frozen English fixture.

Decision D-033 approves the fixture-only ``openphonetics`` pronunciation with
scope ``release-e2e-fixture-only``.  This suite independently requires that
approval in addition to the release workflow's ``--require-approved`` preflight.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import pytest

from flexaligner import (
    AlignmentOptions,
    AlignmentRequest,
    FlexAligner,
    LocalModelBundle,
    ResourceLimits,
    ScoreKind,
    TextGridOutput,
)
from flexaligner.textgrid import parse_textgrid_long, validate_textgrid_structure

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "tests" / "fixtures" / "e2e" / "asset_manifest.json"
REFERENCE_PATH = REPO_ROOT / "reference" / "align_single_cpu.py"
REFERENCE_SHA256 = "9ed4e21e615718ddfd10930359f55769fb27a0d284599cce45a3fc755e835de1"
EXPECTED_TEXTGRID_SHA256 = "d15265c207faa6c1d5b588aef645bb97e457868ec8f458112a54335c3ec2d32a"
EXPECTED_METADATA_SHA256 = "c6c5b035be5aeb3727996538c37c168e5af0c5591b08b38befeaace5a9f36140"
EXPECTED_WORDS = (
    "this",
    "synthetic",
    "example",
    "shows",
    "openphonetics",
    "word",
    "and",
    "phone",
    "alignment",
)
IGNORED_WORD_LABELS = {"", "null", "sil", "[sph]"}


@dataclass(frozen=True, slots=True)
class FrozenAssets:
    root: Path
    manifest: dict[str, Any]
    by_role: dict[str, Path]

    def require(self, role: str) -> Path:
        try:
            return self.by_role[role]
        except KeyError:
            pytest.fail(f"MODEL_E2E_BLOCKED: manifest is missing required role: {role}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        pytest.fail(f"MODEL_E2E_BLOCKED: manifest field {field!r} is not a string")
    return value


def _load_frozen_assets() -> FrozenAssets:
    if not MANIFEST_PATH.is_file():
        pytest.fail(f"MODEL_E2E_BLOCKED: manifest is absent: {MANIFEST_PATH}")
    try:
        payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8", errors="strict"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        pytest.fail(f"MODEL_E2E_BLOCKED: manifest cannot be parsed: {error}")
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        pytest.fail("MODEL_E2E_BLOCKED: manifest schema_version must be 1")

    status = payload.get("status")
    if status != "approved":
        pytest.fail(f"MODEL_E2E_BLOCKED: manifest is not approved: status={status!r}")
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        pytest.fail("MODEL_E2E_BLOCKED: manifest provenance must be an object")
    if provenance.get("oov_pronunciation_approval") != "D-033":
        pytest.fail("MODEL_E2E_BLOCKED: manifest approval decision must be D-033")
    if provenance.get("scope") != "release-e2e-fixture-only":
        pytest.fail("MODEL_E2E_BLOCKED: manifest approval scope is not release-E2E-only")
    if provenance.get("approved_on") != "2026-08-11":
        pytest.fail("MODEL_E2E_BLOCKED: manifest approval date is not frozen")

    root_env = _require_string(payload.get("root_env"), field="root_env")
    root_text = os.environ.get(root_env)
    if not root_text:
        pytest.fail(
            f"MODEL_E2E_BLOCKED: required asset-root environment variable is unset: {root_env}"
        )
    root = Path(root_text)
    if not root.is_absolute():
        pytest.fail("MODEL_E2E_BLOCKED: asset root must be absolute")

    runtime = payload.get("runtime")
    if not isinstance(runtime, dict):
        pytest.fail("MODEL_E2E_BLOCKED: manifest runtime must be an object")
    expected_python = _require_string(runtime.get("python"), field="runtime.python")
    if platform.python_version() != expected_python:
        pytest.fail(
            "MODEL_E2E_BLOCKED: Python version mismatch: "
            f"expected={expected_python}, actual={platform.python_version()}"
        )
    for distribution, expected_value in sorted(runtime.items()):
        if distribution == "python":
            continue
        expected = _require_string(expected_value, field=f"runtime.{distribution}")
        try:
            actual = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            pytest.fail(f"MODEL_E2E_BLOCKED: required distribution is absent: {distribution}")
        if actual != expected:
            pytest.fail(
                "MODEL_E2E_BLOCKED: distribution version mismatch: "
                f"name={distribution}, expected={expected}, actual={actual}"
            )

    entries = payload.get("files")
    if not isinstance(entries, list) or not entries:
        pytest.fail("MODEL_E2E_BLOCKED: manifest files must be a non-empty list")
    by_role: dict[str, Path] = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            pytest.fail(f"MODEL_E2E_BLOCKED: files[{index}] must be an object")
        role = _require_string(entry.get("role"), field=f"files[{index}].role")
        relative = _require_string(entry.get("path"), field=f"files[{index}].path")
        expected_hash = _require_string(entry.get("sha256"), field=f"files[{index}].sha256").lower()
        posix_path = PurePosixPath(relative)
        if posix_path.is_absolute() or ".." in posix_path.parts:
            pytest.fail(f"MODEL_E2E_BLOCKED: unsafe asset path: {relative!r}")
        if role in by_role:
            pytest.fail(f"MODEL_E2E_BLOCKED: duplicate asset role: {role!r}")
        asset = root.joinpath(*posix_path.parts)
        if not asset.is_file():
            pytest.fail(f"MODEL_E2E_BLOCKED: asset is absent: {asset}")
        actual_hash = _sha256(asset)
        if actual_hash != expected_hash:
            pytest.fail(
                "MODEL_E2E_BLOCKED: asset hash mismatch: "
                f"path={asset}, expected={expected_hash}, actual={actual_hash}"
            )
        by_role[role] = asset
    return FrozenAssets(root=root, manifest=payload, by_role=by_role)


def _lexical_words(labels: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(label for label in labels if label.strip().lower() not in IGNORED_WORD_LABELS)


def _run_reference(
    assets: FrozenAssets,
    *,
    output_path: Path,
    metadata_path: Path,
) -> subprocess.CompletedProcess[str]:
    if not REFERENCE_PATH.is_file() or _sha256(REFERENCE_PATH) != REFERENCE_SHA256:
        pytest.fail("MODEL_E2E_BLOCKED: frozen reference is absent or has changed")
    env = os.environ.copy()
    env.update(
        {
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONDONTWRITEBYTECODE": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )
    return subprocess.run(
        [
            sys.executable,
            str(REFERENCE_PATH),
            "--wav_path",
            str(assets.require("audio")),
            "--text_path",
            str(assets.require("transcript")),
            "--lexicon",
            str(assets.require("effective_fixture_lexicon")),
            "--chunk_model",
            str(assets.require("stage1_model_config").parent),
            "--align_model",
            str(assets.require("stage2_model_config").parent),
            "--output_path",
            str(output_path),
            "--chunker_metadata_path",
            str(metadata_path),
            "--num_threads",
            "1",
        ],
        cwd=output_path.parent,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )


@pytest.mark.model_e2e
@pytest.mark.filterwarnings("error:A test tried to use socket\\.socket\\.")
def test_frozen_english_release_e2e_fixture_matches_reference(
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
    assets = _load_frozen_assets()
    output_path = tmp_path / "new.TextGrid"
    metadata_path = tmp_path / "new.chunker.json"
    transcript = assets.require("transcript").read_text(encoding="utf-8", errors="strict")

    with FlexAligner(
        models=LocalModelBundle(
            chunker_dir=assets.require("stage1_model_config").parent,
            aligner_dir=assets.require("stage2_model_config").parent,
            manifest_path=MANIFEST_PATH,
        ),
        lexicon_path=assets.require("effective_fixture_lexicon"),
        options=AlignmentOptions(
            num_threads=1,
            limits=ResourceLimits(
                max_audio_seconds=60.0,
                max_transcript_words=100,
                max_phone_tokens=1_000,
                max_trellis_cells=1_000_000,
            ),
        ),
    ) as engine:
        result = engine.align(
            AlignmentRequest(
                audio_path=assets.require("audio"),
                transcript=transcript,
                output=TextGridOutput(
                    path=output_path,
                    chunk_metadata_path=metadata_path,
                ),
                utterance_id="english_synthetic",
            )
        )

    assert result.normalized_words == EXPECTED_WORDS
    assert _lexical_words(tuple(word.label for word in result.words)) == EXPECTED_WORDS
    assert result.calibrated_scores is None
    assert len(result.raw_scores) == len(EXPECTED_WORDS)
    assert all(not score.calibrated for score in result.raw_scores)
    assert all(
        score.kind is ScoreKind.CHUNKER_EMISSION_GEOMETRIC_MEAN for score in result.raw_scores
    )
    assert result.chunks[0].word_indices == tuple(range(len(EXPECTED_WORDS)))
    assert output_path.is_file() and metadata_path.is_file()
    assert not output_path.with_name(output_path.name + ".tmp").exists()
    assert not metadata_path.with_name(metadata_path.name + ".tmp").exists()
    assert result.output_sha256 == _sha256(output_path) == EXPECTED_TEXTGRID_SHA256
    assert _sha256(metadata_path) == EXPECTED_METADATA_SHA256

    parsed = parse_textgrid_long(output_path)
    validate_textgrid_structure(
        parsed,
        context="frozen English E2E",
        require_full_coverage=True,
    )
    assert tuple(tier.name for tier in parsed.tiers) == ("phones", "words")
    assert _lexical_words(tuple(interval.text for interval in parsed.tiers[1].intervals)) == (
        EXPECTED_WORDS
    )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8", errors="strict"))
    assert metadata["schema_version"] == "1"
    assert metadata["score_kind"] == ScoreKind.CHUNKER_EMISSION_GEOMETRIC_MEAN.value
    assert metadata["calibrated"] is False
    assert tuple(word["word"] for word in metadata["words"]) == EXPECTED_WORDS
    assert tuple(word["value"] for word in metadata["words"]) == tuple(
        score.value for score in result.raw_scores
    )

    reference_output = tmp_path / "reference.TextGrid"
    reference_metadata_path = tmp_path / "reference.chunker.json"
    reference_run = _run_reference(
        assets,
        output_path=reference_output,
        metadata_path=reference_metadata_path,
    )
    assert reference_run.returncode == 0, (
        f"reference stdout:\n{reference_run.stdout}\nreference stderr:\n{reference_run.stderr}"
    )
    reference_grid = parse_textgrid_long(reference_output)
    actual_non_null = tuple(
        tuple(
            (interval.xmin, interval.xmax, interval.text)
            for interval in tier.intervals
            if interval.text.strip().lower() != "null"
        )
        for tier in parsed.tiers
    )
    reference_non_null = tuple(
        tuple(
            (interval.xmin, interval.xmax, interval.text)
            for interval in tier.intervals
            if interval.text.strip().lower() != "null"
        )
        for tier in reference_grid.tiers
    )
    assert actual_non_null == reference_non_null

    reference_metadata = json.loads(
        reference_metadata_path.read_text(encoding="utf-8", errors="strict")
    )
    assert len(reference_metadata["words"]) == len(metadata["words"])
    for actual, reference in zip(metadata["words"], reference_metadata["words"], strict=True):
        assert actual["word_index"] == reference["word_index"]
        assert actual["word"] == reference["word"]
        assert actual["chunker_pronunciation"] == reference["chunker_pronunciation"]
        assert actual["value"] == reference["confidence"]
        assert actual["log_value"] == reference["confidence_log"]
