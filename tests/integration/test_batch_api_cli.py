from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from flexaligner import (
    AlignmentOptions,
    ConfigurationError,
    Device,
    FlexAligner,
    TextGridOutput,
    cli,
)
from flexaligner.adapters import hf_local
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
