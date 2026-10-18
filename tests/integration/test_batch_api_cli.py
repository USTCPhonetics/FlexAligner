from __future__ import annotations

import json
import shutil
import weakref
from dataclasses import replace
from pathlib import Path

import pytest

from flexaligner import (
    AlignmentOptions,
    ConfigurationError,
    Device,
    FlexAligner,
    InputValidationError,
    Language,
    PronunciationMode,
    TextGridOutput,
    cli,
)
from flexaligner.adapters import hf_local
from flexaligner.model_download import _validated_bundle
from flexaligner.pipeline import AlignmentPipeline
from tests.integration._support import FakeInferenceFactory, make_integration_fixture


def test_batch_matches_single_and_preserves_existing(tmp_path: Path) -> None:
    fixture = make_integration_fixture(tmp_path, metadata=False)
    factory = FakeInferenceFactory()
    engine = FlexAligner(models=fixture.models, lexicon_path=fixture.lexicon_path)
    engine._pipeline = AlignmentPipeline(inference_factory=factory)
    single = engine.align(fixture.request)
    request = replace(fixture.request, output=TextGridOutput(path=tmp_path / "batch.TextGrid"))
    report = engine.align_batch([request, request])
    assert [item.status for item in report.items] == ["succeeded", "conflict"]
    assert report.exit_code == 1
    assert request.output.path.read_bytes() == single.output_path.read_bytes()
    assert report.items[0].result is not None
    assert report.items[0].result.provenance.requested_device == "auto"
    assert report.items[0].result.provenance.device is Device.CPU
    assert report.items[0].device == "cpu"
    skipped = engine.align_batch([request], skip_existing=True)
    assert skipped.counts["skipped"] == 1
    assert skipped.exit_code == 0
    temporary = request.output.path.with_name(request.output.path.name + ".tmp")
    temporary.write_bytes(b"keep")
    conflict = engine.align_batch([request], skip_existing=True)
    assert conflict.items[0].status == "conflict"
    assert temporary.read_bytes() == b"keep"
    engine.close()


def test_item_failure_continues_and_iterator_close_exits_batch(tmp_path: Path) -> None:
    fixture = make_integration_fixture(tmp_path, metadata=False)
    engine = FlexAligner(models=fixture.models, lexicon_path=fixture.lexicon_path)
    engine._pipeline = AlignmentPipeline(inference_factory=FakeInferenceFactory())
    invalid = replace(fixture.request, transcript="")
    report = engine.align_batch([invalid, fixture.request])
    assert [item.status for item in report.items] == ["failed", "succeeded"]
    iterator = engine.iter_batch([fixture.request, fixture.request], skip_existing=True)
    assert next(iterator).status == "skipped"
    assert engine._batch_active
    iterator.close()
    assert not engine._batch_active
    with engine.batch_session():
        with pytest.raises(ConfigurationError, match="already active"), engine.batch_session():
            pass
        with pytest.raises(ConfigurationError, match="cannot change"):
            engine.align(fixture.request, options=AlignmentOptions(device=Device.CPU))
    engine.close()


def test_cli_batch_partial_failure_summary_and_skip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fixture = make_integration_fixture(tmp_path, metadata=False)
    corpus = tmp_path / "corpus"
    speaker = corpus / "speaker"
    speaker.mkdir(parents=True)
    shutil.copyfile(fixture.request.audio_path, speaker / "a.wav")
    (speaker / "a.txt").write_text("Alpha beta", encoding="utf-8")
    shutil.copyfile(fixture.request.audio_path, speaker / "missing.wav")
    factory = FakeInferenceFactory()
    monkeypatch.setattr(hf_local, "LocalHuggingFaceInferenceFactory", lambda: factory)
    output = tmp_path / "out"
    args = [
        "batch",
        str(corpus),
        str(output),
        "--lexicon",
        str(fixture.lexicon_path),
        "--chunker-model",
        str(fixture.models.chunker_dir),
        "--aligner-model",
        str(fixture.models.aligner_dir),
        "--device",
        "cpu",
    ]
    assert cli.main(args) == 1
    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    assert summary["counts"]["succeeded"] == 1
    assert summary["counts"]["failed"] == 1
    assert (output / "speaker/a.TextGrid").is_file()
    assert Path(summary["summary_path"]).is_file()
    assert len(Path(summary["items_path"]).read_text().splitlines()) == 2
    assert cli.main([*args, "--skip-existing"]) == 1
    repeated = json.loads(capsys.readouterr().out)
    assert repeated["counts"]["skipped"] == 1
    assert repeated["run_id"] != summary["run_id"]


@pytest.mark.parametrize("value", [True, -1, 0.5])
def test_device_index_validation(value: object) -> None:
    with pytest.raises(ConfigurationError):
        AlignmentOptions(device_index=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("manifest_kind", ["missing", "directory"])
def test_invalid_batch_manifest_does_not_consume_requests(
    tmp_path: Path, manifest_kind: str
) -> None:
    fixture = make_integration_fixture(tmp_path, metadata=False)
    manifest = tmp_path / "manifest"
    if manifest_kind == "directory":
        manifest.mkdir()
    engine = FlexAligner(
        models=replace(fixture.models, manifest_path=manifest),
        lexicon_path=fixture.lexicon_path,
    )
    consumed = []

    def requests():
        consumed.append(True)
        yield fixture.request

    with engine, pytest.raises(ConfigurationError, match="manifest"):
        engine.align_batch(requests())
    assert consumed == []


def test_failed_outcome_releases_error_frame_before_yield(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = make_integration_fixture(tmp_path, metadata=False)
    engine = FlexAligner(models=fixture.models, lexicon_path=fixture.lexicon_path)
    engine._pipeline = AlignmentPipeline(inference_factory=FakeInferenceFactory())
    references = []

    class Heavy:
        pass

    def fail(request):
        heavy = Heavy()
        references.append(weakref.ref(heavy))
        raise InputValidationError("bad waveform")

    monkeypatch.setattr(engine, "align", fail)
    with engine:
        iterator = engine.iter_batch([fixture.request])
        try:
            assert next(iterator).status == "failed"
            assert references[0]() is None
        finally:
            iterator.close()


def test_batch_system_dictionary_punctuation_matches_single(tmp_path: Path) -> None:
    fixture = make_integration_fixture(tmp_path, metadata=False)
    dictionary = tmp_path / "system.dict"
    dictionary.write_text("一 i1\n无 u2\n\uff01 sil\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    models = _validated_bundle(
        chunker_dir=fixture.models.chunker_dir,
        aligner_dir=fixture.models.aligner_dir,
        manifest_path=manifest,
        dictionary_path=dictionary,
    )
    chunk_vocab = {"<pad>": 0, "i1": 1, "u2": 2, "ix1": 3, "iz4": 4}
    align_vocab = {"i1": 0, "u2": 1, "sil": 2, "sph": 3, "ix1": 4, "iz4": 5}
    for path, vocab in ((models.chunker_dir, chunk_vocab), (models.aligner_dir, align_vocab)):
        (path / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
    engine = FlexAligner(
        models=models,
        options=AlignmentOptions(
            device=Device.CPU,
            language=Language.ZH,
            pronunciation_mode=PronunciationMode.LEXICON_ONLY,
        ),
    )
    engine._pipeline = AlignmentPipeline(
        inference_factory=FakeInferenceFactory(
            chunk_vocabulary=chunk_vocab, align_vocabulary=align_vocab
        )
    )
    request = replace(fixture.request, transcript="一 无")
    with engine:
        single = engine.align(request)
        batch_request = replace(request, output=TextGridOutput(path=tmp_path / "batch.TextGrid"))
        report = engine.align_batch([batch_request])
    assert report.exit_code == 0
    assert report.items[0].result.normalized_words == single.normalized_words == ("一", "无")
    assert batch_request.output.path.read_bytes() == single.output_path.read_bytes()


def _batch_cli_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fixture = make_integration_fixture(tmp_path, metadata=False)
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    shutil.copyfile(fixture.request.audio_path, corpus / "sample.wav")
    (corpus / "sample.txt").write_text("Alpha gamma", encoding="utf-8")
    monkeypatch.setattr(hf_local, "LocalHuggingFaceInferenceFactory", FakeInferenceFactory)
    return fixture, [
        "batch",
        str(corpus),
        str(tmp_path / "out"),
        "--lexicon",
        str(fixture.lexicon_path),
        "--chunker-model",
        str(fixture.models.chunker_dir),
        "--aligner-model",
        str(fixture.models.aligner_dir),
        "--device",
        "cpu",
    ]


def test_batch_cli_g2p_warning_is_reported_and_persisted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fixture, args = _batch_cli_fixture(tmp_path, monkeypatch)
    fixture.lexicon_path.write_text("alpha AH\n", encoding="utf-8")

    class FakeG2P:
        engine_id = "fake-local"
        engine_version = "test"

        def pronounce(self, word: str) -> tuple[str, ...]:
            assert word == "gamma"
            return ("B",)

    monkeypatch.setattr("flexaligner.adapters.g2p_en_local.LocalEnglishG2P", FakeG2P)
    assert cli.main(args) == 0
    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    record = json.loads(Path(summary["items_path"]).read_text())
    warning = next(
        line.removeprefix("WARNING ")
        for line in captured.err.splitlines()
        if line.startswith("WARNING ")
    )
    warning_data = json.loads(warning)
    assert warning_data.pop("audio_path") == str(tmp_path / "corpus/sample.wav")
    assert warning_data == record["pronunciation_notices"][0]
    assert warning_data["word"] == "gamma"
    assert warning_data["pronunciation"] == ["B"]
    assert warning_data["engine_id"] == "fake-local"


@pytest.mark.parametrize(
    "failure", [RuntimeError("unexpected batch failure"), OSError("disk failed")]
)
def test_unknown_batch_failure_cli_matches_saved_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failure: Exception,
) -> None:
    _, args = _batch_cli_fixture(tmp_path, monkeypatch)

    def fail(self, request, **kwargs):
        raise failure

    monkeypatch.setattr(FlexAligner, "align", fail)
    assert cli.main(args) == 3
    captured = capsys.readouterr()
    assert captured.out == ""
    error = json.loads(captured.err.splitlines()[-1])
    assert error["code"] == "internal_error"
    assert error["context"]["exception_type"] == type(failure).__name__
    saved = json.loads(
        next((tmp_path / "out/_flexaligner_runs").glob("*/summary.json")).read_text()
    )
    assert saved["exit_code"] == 3
    assert saved["error"]["message"] == str(failure)


@pytest.mark.parametrize("failure", [KeyboardInterrupt(), SystemExit(17)])
def test_batch_cli_does_not_swallow_control_flow_exceptions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failure: BaseException,
) -> None:
    _, args = _batch_cli_fixture(tmp_path, monkeypatch)

    def fail(self, request, **kwargs):
        raise failure

    monkeypatch.setattr(FlexAligner, "align", fail)
    if isinstance(failure, KeyboardInterrupt):
        assert cli.main(args) == 130
        assert json.loads(capsys.readouterr().err.splitlines()[-1])["code"] == "interrupted"
    else:
        with pytest.raises(SystemExit) as caught:
            cli.main(args)
        assert caught.value.code == 17
