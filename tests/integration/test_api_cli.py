from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

import flexaligner.adapters.hf_local as hf_local
import flexaligner.cli as cli
from flexaligner import (
    AlignmentOptions,
    Device,
    EngineClosedError,
    FeatureNotAvailableError,
    FlexAligner,
    Language,
)
from tests.integration._support import FakeInferenceFactory, make_integration_fixture


def test_public_engine_lazily_runs_real_pipeline_and_closes_factory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = make_integration_fixture(tmp_path, metadata=False)
    factory = FakeInferenceFactory()
    monkeypatch.setattr(hf_local, "LocalHuggingFaceInferenceFactory", lambda: factory)
    engine = FlexAligner(models=fixture.models, lexicon_path=fixture.lexicon_path)
    assert factory.trace == []

    result = engine.align(fixture.request)
    assert result.normalized_words == ("alpha", "beta")
    assert factory.trace[:4] == ["chunk.load", "chunk.infer", "chunk.close", "align.load"]
    engine.close()
    engine.close()
    assert factory.trace.count("factory.close") == 1
    with pytest.raises(EngineClosedError):
        engine.align(fixture.request)


def test_future_option_still_fails_before_pipeline_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = make_integration_fixture(tmp_path, metadata=False)
    constructed = False

    def forbidden_factory() -> FakeInferenceFactory:
        nonlocal constructed
        constructed = True
        raise AssertionError("future option constructed inference factory")

    monkeypatch.setattr(hf_local, "LocalHuggingFaceInferenceFactory", forbidden_factory)
    engine = FlexAligner(models=fixture.models, lexicon_path=fixture.lexicon_path)
    with pytest.raises(FeatureNotAvailableError):
        engine.align(
            fixture.request,
            options=replace(AlignmentOptions(), device=Device.CUDA),
        )
    assert constructed is False
    assert not fixture.request.output.path.exists()


def test_cli_text_file_runs_once_and_prints_stable_success_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = make_integration_fixture(tmp_path, metadata=False)
    transcript_path = tmp_path / "transcript.txt"
    transcript_path.write_text("Alpha beta\n", encoding="utf-8")
    factory = FakeInferenceFactory()
    monkeypatch.setattr(hf_local, "LocalHuggingFaceInferenceFactory", lambda: factory)

    original_reader = cli.read_utf8_text
    read_paths: list[Path] = []

    def counting_reader(path: Path) -> str:
        read_paths.append(path)
        return original_reader(path)

    monkeypatch.setattr(cli, "read_utf8_text", counting_reader)
    status = cli.main(
        [
            "align",
            "--audio",
            str(fixture.request.audio_path),
            "--text-file",
            str(transcript_path),
            "--lexicon",
            str(fixture.lexicon_path),
            "--chunker-model",
            str(fixture.models.chunker_dir),
            "--aligner-model",
            str(fixture.models.aligner_dir),
            "--output",
            str(fixture.request.output.path),
            "--utterance-id",
            "cli-fixture",
        ]
    )
    streams = capsys.readouterr()
    assert status == 0
    assert streams.err == ""
    payload: dict[str, Any] = json.loads(streams.out)
    assert payload == {
        "output_path": str(fixture.request.output.path),
        "output_sha256": payload["output_sha256"],
        "schema_version": "1",
        "utterance_id": "cli-fixture",
    }
    assert len(payload["output_sha256"]) == 64
    assert read_paths == [transcript_path]


def test_cli_default_g2p_emits_structured_warning_on_stderr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = make_integration_fixture(tmp_path, metadata=False)
    fixture.lexicon_path.write_text("alpha AH\n", encoding="utf-8")
    factory = FakeInferenceFactory()
    monkeypatch.setattr(hf_local, "LocalHuggingFaceInferenceFactory", lambda: factory)

    class FakeLocalEnglishG2P:
        engine_id = "fake-local"
        engine_version = "test"

        def pronounce(self, word: str) -> tuple[str, ...]:
            assert word == "gamma"
            return ("B",)

    monkeypatch.setattr(
        "flexaligner.adapters.g2p_en_local.LocalEnglishG2P",
        FakeLocalEnglishG2P,
    )

    status = cli.main(
        [
            "align",
            "--audio",
            str(fixture.request.audio_path),
            "--text",
            "Alpha gamma",
            "--lexicon",
            str(fixture.lexicon_path),
            "--chunker-model",
            str(fixture.models.chunker_dir),
            "--aligner-model",
            str(fixture.models.aligner_dir),
            "--output",
            str(fixture.request.output.path),
        ]
    )

    streams = capsys.readouterr()
    assert status == 0
    assert json.loads(streams.out)["schema_version"] == "1"
    prefix = "WARNING "
    assert streams.err.startswith(prefix)
    warning = json.loads(streams.err.removeprefix(prefix))
    assert warning == {
        "code": "oov_g2p_fallback",
        "engine_id": "fake-local",
        "engine_version": "test",
        "pronunciation": ["B"],
        "word": "gamma",
        "word_indices": [1],
    }


def test_cli_lexicon_mode_keeps_oov_failure_strict(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = make_integration_fixture(tmp_path, metadata=False)
    fixture.lexicon_path.write_text("alpha AH\n", encoding="utf-8")

    status = cli.main(
        [
            "align",
            "--audio",
            str(fixture.request.audio_path),
            "--text",
            "Alpha gamma",
            "--lexicon",
            str(fixture.lexicon_path),
            "--chunker-model",
            str(fixture.models.chunker_dir),
            "--aligner-model",
            str(fixture.models.aligner_dir),
            "--output",
            str(fixture.request.output.path),
            "--pronunciation-mode",
            "lexicon",
        ]
    )

    streams = capsys.readouterr()
    assert status != 0
    assert streams.out == ""
    error = json.loads(streams.err)
    assert error["code"] == "input_validation_error"
    assert error["context"]["word"] == "gamma"
    assert not fixture.request.output.path.exists()


def test_cli_placeholder_guard_precedes_missing_text_file_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def forbidden_reader(path: Path) -> str:
        raise AssertionError(f"placeholder read text file: {path}")

    monkeypatch.setattr(cli, "read_utf8_text", forbidden_reader)
    status = cli.main(
        [
            "align",
            "--audio",
            str(tmp_path / "missing.wav"),
            "--text-file",
            str(tmp_path / "missing.txt"),
            "--lexicon",
            str(tmp_path / "missing.dict"),
            "--chunker-model",
            str(tmp_path / "missing-chunker"),
            "--aligner-model",
            str(tmp_path / "missing-aligner"),
            "--output",
            str(tmp_path / "result.TextGrid"),
            "--language",
            Language.ZH.value,
        ]
    )
    streams = capsys.readouterr()
    assert status != 0
    assert streams.out == ""
    payload = json.loads(streams.err)
    assert payload["code"] == "feature_not_available"
    assert payload["context"]["capability"] == "language.zh"
