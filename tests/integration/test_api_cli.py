from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

import flexaligner.adapters.hf_local as hf_local
import flexaligner.cli as cli
import flexaligner.model_download as model_download
import flexaligner.pipeline as pipeline
from flexaligner import (
    AlignmentOptions,
    AudioPolicy,
    EngineClosedError,
    FeatureNotAvailableError,
    FlexAligner,
    Language,
    LanguageMismatchError,
    LocalModelBundle,
    ModelResolution,
    PronunciationMode,
)
from tests.integration._support import FakeInferenceFactory, make_integration_fixture


def _validated_bundle_with_dictionary(
    models: LocalModelBundle,
    dictionary_path: Path,
) -> LocalModelBundle:
    manifest_path = dictionary_path.parent / "model_manifest.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    return model_download._validated_bundle(
        chunker_dir=models.chunker_dir,
        aligner_dir=models.aligner_dir,
        manifest_path=manifest_path,
        dictionary_path=dictionary_path,
    )


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


def test_public_engine_explicit_audio_policy_uses_optional_decoder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = make_integration_fixture(tmp_path, metadata=False)
    factory = FakeInferenceFactory()
    monkeypatch.setattr(hf_local, "LocalHuggingFaceInferenceFactory", lambda: factory)
    with FlexAligner(
        models=fixture.models,
        lexicon_path=fixture.lexicon_path,
        options=AlignmentOptions(audio_policy=AudioPolicy.AUTO_RESAMPLE),
    ) as engine:
        result = engine.align(fixture.request)
    assert result.output_path.is_file()
    assert factory.trace[:2] == ["chunk.load", "chunk.infer"]


def test_public_engine_runs_mandarin_online_core_profile_with_sph_enabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = make_integration_fixture(tmp_path, metadata=False)
    fixture.lexicon_path.write_text("一 i1\n无 u2\n", encoding="utf-8")
    models = _validated_bundle_with_dictionary(fixture.models, fixture.lexicon_path)
    chunk_vocabulary = {
        "<pad>": 0,
        "i1": 1,
        "u2": 2,
        "ix1": 3,
        "iy2": 4,
        "iz4": 5,
    }
    align_vocabulary = {
        "i1": 0,
        "u2": 1,
        "sil": 2,
        "ix1": 3,
        "iy2": 4,
        "iz4": 5,
        "sph": 6,
    }
    (fixture.models.chunker_dir / "vocab.json").write_text(
        json.dumps(chunk_vocabulary), encoding="utf-8"
    )
    (fixture.models.aligner_dir / "vocab.json").write_text(
        json.dumps(align_vocabulary), encoding="utf-8"
    )
    request = replace(fixture.request, transcript="一 无")
    factory = FakeInferenceFactory(
        chunk_vocabulary=chunk_vocabulary,
        align_vocabulary=align_vocabulary,
    )
    monkeypatch.setattr(hf_local, "LocalHuggingFaceInferenceFactory", lambda: factory)
    observed_enable_sph: list[bool] = []
    original_stage2 = pipeline._stage2_from_posterior

    def recording_stage2(**kwargs: Any) -> Any:
        observed_enable_sph.append(bool(kwargs["enable_sph"]))
        return original_stage2(**kwargs)

    monkeypatch.setattr(pipeline, "_stage2_from_posterior", recording_stage2)
    options = AlignmentOptions(
        language=Language.ZH,
        pronunciation_mode=PronunciationMode.LEXICON_ONLY,
    )

    with FlexAligner(
        models=models,
        options=options,
    ) as engine:
        result = engine.align(request)

    assert result.normalized_words == ("一", "无")
    assert result.provenance.language is Language.ZH
    assert result.provenance.algorithm_profile == "zh-online-core-v1"
    assert observed_enable_sph == [True]
    assert "sph" not in {interval.label for interval in result.phones}
    assert result.output_path.is_file()


def test_mandarin_user_dictionary_overrides_bundle_and_metadata_records_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = make_integration_fixture(tmp_path, metadata=True)
    system_path = tmp_path / "system.dict"
    user_path = tmp_path / "user.dict"
    system_path.write_text("一 i1\n无 u2\n", encoding="utf-8")
    user_path.write_text("一 i1\n", encoding="utf-8")
    models = _validated_bundle_with_dictionary(fixture.models, system_path)
    chunk_vocabulary = {"<pad>": 0, "i1": 1, "u2": 2, "ix1": 3, "iz4": 4}
    align_vocabulary = {
        "i1": 0,
        "u2": 1,
        "sil": 2,
        "sph": 3,
        "ix1": 4,
        "iz4": 5,
    }
    for model_dir, vocabulary in (
        (models.chunker_dir, chunk_vocabulary),
        (models.aligner_dir, align_vocabulary),
    ):
        (model_dir / "vocab.json").write_text(json.dumps(vocabulary), encoding="utf-8")
    factory = FakeInferenceFactory(
        chunk_vocabulary=chunk_vocabulary,
        align_vocabulary=align_vocabulary,
    )
    monkeypatch.setattr(hf_local, "LocalHuggingFaceInferenceFactory", lambda: factory)
    request = replace(fixture.request, transcript="一 无")

    with FlexAligner(
        models=models,
        lexicon_path=user_path,
        options=AlignmentOptions(
            language=Language.ZH,
            pronunciation_mode=PronunciationMode.LEXICON_ONLY,
        ),
    ) as engine:
        engine.align(request)

    assert request.output.chunk_metadata_path is not None
    metadata = json.loads(request.output.chunk_metadata_path.read_text(encoding="utf-8"))
    assert [item["pronunciation_source"] for item in metadata["words"]] == [
        "user",
        "system",
    ]


def test_unreferenced_system_dictionary_entry_cannot_block_stage1(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = make_integration_fixture(tmp_path, metadata=False)
    system_path = tmp_path / "system.dict"
    system_path.write_text("一 i1\n无 u2\n吹吹 ch ui5 ch ui5\n", encoding="utf-8")
    models = _validated_bundle_with_dictionary(fixture.models, system_path)
    chunk_vocabulary = {"<pad>": 0, "i1": 1, "u2": 2, "ix1": 3, "iz4": 4}
    align_vocabulary = {
        "i1": 0,
        "u2": 1,
        "sil": 2,
        "sph": 3,
        "ix1": 4,
        "iz4": 5,
    }
    for model_dir, vocabulary in (
        (models.chunker_dir, chunk_vocabulary),
        (models.aligner_dir, align_vocabulary),
    ):
        (model_dir / "vocab.json").write_text(json.dumps(vocabulary), encoding="utf-8")
    factory = FakeInferenceFactory(
        chunk_vocabulary=chunk_vocabulary,
        align_vocabulary=align_vocabulary,
    )
    monkeypatch.setattr(hf_local, "LocalHuggingFaceInferenceFactory", lambda: factory)
    request = replace(fixture.request, transcript="一 无")

    with FlexAligner(
        models=models,
        options=AlignmentOptions(
            language=Language.ZH,
            pronunciation_mode=PronunciationMode.LEXICON_ONLY,
        ),
    ) as engine:
        result = engine.align(request)

    assert result.normalized_words == ("一", "无")


def test_mandarin_rejects_english_lexicon_before_model_io(tmp_path: Path) -> None:
    fixture = make_integration_fixture(tmp_path, metadata=False)
    request = replace(fixture.request, transcript="中文")
    engine = FlexAligner(
        models=fixture.models,
        lexicon_path=fixture.lexicon_path,
        options=AlignmentOptions(
            language=Language.ZH,
            pronunciation_mode=PronunciationMode.LEXICON_ONLY,
        ),
    )
    with pytest.raises(LanguageMismatchError) as caught:
        engine.align(request)
    assert caught.value.context["component"] == "lexicon"
    assert not request.output.path.exists()


def test_mandarin_rejects_english_models_before_inference_factory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = make_integration_fixture(tmp_path, metadata=False)
    fixture.lexicon_path.write_text("中文 zh ong u en\n", encoding="utf-8")
    request = replace(fixture.request, transcript="中文")

    def forbidden_factory() -> FakeInferenceFactory:
        raise AssertionError("language mismatch constructed inference factory")

    monkeypatch.setattr(hf_local, "LocalHuggingFaceInferenceFactory", forbidden_factory)
    engine = FlexAligner(
        models=fixture.models,
        lexicon_path=fixture.lexicon_path,
        options=AlignmentOptions(
            language=Language.ZH,
            pronunciation_mode=PronunciationMode.LEXICON_ONLY,
        ),
    )
    with pytest.raises(LanguageMismatchError) as caught:
        engine.align(request)
    assert caught.value.context["component"] == "chunker model"
    assert not request.output.path.exists()


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
            options=replace(AlignmentOptions(), model_resolution=ModelResolution.AUTO_DOWNLOAD),
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


def test_cli_mandarin_language_mismatch_precedes_missing_model_io(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    status = cli.main(
        [
            "align",
            "--audio",
            str(tmp_path / "missing.wav"),
            "--text",
            "english only",
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
    assert payload["code"] == "language_mismatch"
    assert payload["context"]["component"] == "transcript"
