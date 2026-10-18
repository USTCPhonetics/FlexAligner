"""Directory scheduling tests use an injected engine, without model dependencies."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from flexaligner import corpus
from flexaligner.contracts import AlignmentResult, Device, Language, RunProvenance
from flexaligner.errors import (
    AlignmentError,
    ArtifactExistsError,
    ConfigurationError,
    FlexAlignerError,
    ModelValidationError,
    OutputError,
    ResourceLimitError,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from flexaligner.contracts import AlignmentRequest


class FakeEngine:
    resolved_device: str | None = None

    def __init__(self, failures: dict[str, BaseException] | None = None) -> None:
        self.failures = failures or {}
        self.requests: list[AlignmentRequest] = []
        self.in_session = False
        self.session_count = 0
        self.observed_counts: list[int] = []
        self.setup_error: BaseException | None = None
        self.cleanup_error: BaseException | None = None

    @contextmanager
    def batch_session(self) -> Iterator[None]:
        self.session_count += 1
        if self.setup_error is not None:
            raise self.setup_error
        self.in_session = True
        self.resolved_device = "cpu"
        try:
            yield
        finally:
            self.in_session = False
        if self.cleanup_error is not None:
            raise self.cleanup_error

    def align(self, request: AlignmentRequest) -> AlignmentResult:
        assert self.in_session
        self.requests.append(request)
        if request.audio_path.stem in self.failures:
            raise self.failures[request.audio_path.stem]
        request.output.path.parent.mkdir(parents=True, exist_ok=True)
        request.output.path.write_text("fake TextGrid", encoding="utf-8")
        return AlignmentResult(
            utterance_id=request.audio_path.stem,
            audio_duration_s=1.0,
            normalized_words=(),
            words=(),
            phones=(),
            chunks=(),
            raw_scores=(),
            calibrated_scores=None,
            output_path=request.output.path,
            output_sha256="fake-hash",
            provenance=RunProvenance(
                package_version="test",
                algorithm_profile="test",
                language=Language.EN,
                device=Device.CPU,
            ),
        )


def pair(root: Path, name: str, text: str = "hello", extension: str = "txt") -> None:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix(".wav").write_bytes(b"fake wav")
    path.with_suffix("." + extension).write_text(text, encoding="utf-8")


def reports(output: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    run = next((output / "_flexaligner_runs").iterdir())
    items = [json.loads(line) for line in (run / "items.jsonl").read_text().splitlines()]
    return items, json.loads((run / "summary.json").read_text())


def test_stable_recursive_pairing_and_mirrored_paths(tmp_path: Path, capsys) -> None:
    source, output = tmp_path / "input", tmp_path / "output"
    pair(source, "z/repeat")
    pair(source, "a/repeat", "你好")
    pair(source, "first")
    (source / "orphan.txt").write_text("unused", encoding="utf-8")
    engine = FakeEngine()
    summary = corpus.run_corpus(engine, source, output)
    assert [r.audio_path.relative_to(source).as_posix() for r in engine.requests] == [
        "a/repeat.wav",
        "first.wav",
        "z/repeat.wav",
    ]
    assert [r.output.path.relative_to(output).as_posix() for r in engine.requests] == [
        "a/repeat.TextGrid",
        "first.TextGrid",
        "z/repeat.TextGrid",
    ]
    assert engine.requests[0].transcript == "你好"
    assert engine.session_count == 1 and not engine.in_session
    items, saved = reports(output)
    assert saved == summary
    assert summary["exit_code"] == 0
    assert summary["counts"] == {"succeeded": 3, "failed": 0, "skipped": 0, "conflict": 0}
    assert summary["processed"] == 3 and summary["device"] == "cpu"
    assert len(summary["notices"]) == 1
    assert len(items) == 3 and items[0]["output_sha256"] == "fake-hash"
    captured = capsys.readouterr()
    assert captured.out == "" and "[3/3] succeeded" in captured.err
    assert "orphan transcript" in captured.err


def test_lab_pairing_and_bom(tmp_path: Path) -> None:
    source, output = tmp_path / "in", tmp_path / "out"
    pair(source, "sample", "\ufeffhello", extension="lab")
    engine = FakeEngine()
    assert corpus.run_corpus(engine, source, output, text_extension=".LAB")["exit_code"] == 0
    assert engine.requests[0].transcript == "hello"


def test_missing_empty_invalid_transcript_are_item_failures(tmp_path: Path) -> None:
    source, output = tmp_path / "in", tmp_path / "out"
    pair(source, "empty", " \n")
    pair(source, "good")
    (source / "missing.wav").write_bytes(b"wav")
    pair(source, "bad")
    (source / "bad.txt").write_bytes(b"\xff")
    engine = FakeEngine()
    summary = corpus.run_corpus(engine, source, output)
    assert summary["exit_code"] == 1
    assert summary["counts"]["failed"] == 3
    assert len(engine.requests) == 1
    items, _ = reports(output)
    assert items[0]["error"]["code"] == "input_validation_error"


@pytest.mark.parametrize("error", [AlignmentError("bad item"), ResourceLimitError("oom")])
def test_per_item_failure_continues(tmp_path: Path, error: BaseException) -> None:
    source, output = tmp_path / "in", tmp_path / "out"
    pair(source, "a")
    pair(source, "b")
    engine = FakeEngine({"a": error})
    summary = corpus.run_corpus(engine, source, output)
    assert summary["exit_code"] == 1 and summary["counts"]["succeeded"] == 1
    assert (output / "b.TextGrid").exists()
    assert len(engine.requests) == 2


@pytest.mark.parametrize("skip", [False, True])
def test_existing_outputs_require_explicit_skip(tmp_path: Path, skip: bool) -> None:
    source, output = tmp_path / "in", tmp_path / "out"
    pair(source, "sample")
    output.mkdir()
    (output / "sample.TextGrid").write_text("existing", encoding="utf-8")
    engine = FakeEngine()
    summary = corpus.run_corpus(engine, source, output, skip_existing=skip)
    assert summary["counts"]["skipped" if skip else "conflict"] == 1
    assert summary["exit_code"] == (0 if skip else 1)
    assert not engine.requests
    assert (output / "sample.TextGrid").read_text() == "existing"


@pytest.mark.parametrize("formal", [False, True])
def test_temporary_conflict_is_never_skipped(tmp_path: Path, formal: bool) -> None:
    source, output = tmp_path / "in", tmp_path / "out"
    pair(source, "sample")
    output.mkdir()
    (output / "sample.TextGrid.tmp").write_text("partial", encoding="utf-8")
    if formal:
        (output / "sample.TextGrid").write_text("existing", encoding="utf-8")
    engine = FakeEngine()
    summary = corpus.run_corpus(engine, source, output, skip_existing=True)
    assert summary["counts"]["conflict"] == 1
    assert not engine.requests
    assert (output / "sample.TextGrid.tmp").read_text() == "partial"


def test_unrequested_json_is_not_an_output_conflict(tmp_path: Path) -> None:
    source, output = tmp_path / "in", tmp_path / "out"
    pair(source, "sample")
    output.mkdir()
    sidecar = output / "sample.chunker.json"
    sidecar.write_text("unrelated", encoding="utf-8")
    assert corpus.run_corpus(FakeEngine(), source, output)["exit_code"] == 0
    assert sidecar.read_text() == "unrelated"


def test_symlink_audio_text_and_directories_are_not_followed(tmp_path: Path) -> None:
    source, output, external = tmp_path / "in", tmp_path / "out", tmp_path / "external"
    pair(source, "good")
    pair(external, "external")
    (source / "linked.wav").symlink_to(external / "external.wav")
    (source / "linked.txt").symlink_to(external / "external.txt")
    (source / "directory").symlink_to(external, target_is_directory=True)
    (source / "missing.wav").write_bytes(b"wav")
    (source / "missing.txt").symlink_to(external / "external.txt")
    engine = FakeEngine()
    summary = corpus.run_corpus(engine, source, output)
    assert summary["total"] == 2
    assert summary["counts"]["failed"] == 1
    assert len(engine.requests) == 1


@pytest.mark.parametrize("nested", ["same", "output_inside", "input_inside"])
def test_input_output_containment_rejected(tmp_path: Path, nested: str) -> None:
    source = tmp_path / "in"
    pair(source, "sample")
    output = {"same": source, "output_inside": source / "out", "input_inside": tmp_path}[nested]
    with pytest.raises(ConfigurationError, match="contain each other"):
        corpus.run_corpus(FakeEngine(), source, output)
    assert not (output / "_flexaligner_runs").exists()


@pytest.mark.parametrize("kind", ["empty", "missing", "symlink"])
def test_no_audio_is_configuration_failure(tmp_path: Path, kind: str) -> None:
    source, output = tmp_path / "in", tmp_path / "out"
    if kind != "missing":
        source.mkdir()
    if kind == "symlink":
        external = tmp_path / "sample.wav"
        external.write_bytes(b"wav")
        (source / "sample.wav").symlink_to(external)
    with pytest.raises(ConfigurationError):
        corpus.run_corpus(FakeEngine(), source, output)
    assert not output.exists()


def test_casefolded_target_collision_rejected(tmp_path: Path, monkeypatch) -> None:
    source, output = tmp_path / "in", tmp_path / "out"
    pair(source, "sample")
    # Simulate a case-sensitive inventory on macOS as well, so the portability
    # guard runs instead of being skipped on a case-insensitive filesystem.
    monkeypatch.setattr(
        corpus.os, "walk", lambda *_args, **_kwargs: [(source, [], ["sample.wav", "sample.WAV"])]
    )
    with pytest.raises(ConfigurationError, match=r"ambiguous case|one output"):
        corpus.run_corpus(FakeEngine(), source, output)
    assert not output.exists()


@pytest.mark.parametrize("kind", ["case_alias", "symlink"])
def test_existing_output_path_alias_rejected(tmp_path: Path, kind: str) -> None:
    source, output = tmp_path / "in", tmp_path / "out"
    pair(source, "nested/sample")
    output.mkdir()
    if kind == "case_alias":
        (output / "NESTED").mkdir()
    else:
        external = tmp_path / "external"
        external.mkdir()
        (output / "nested").symlink_to(external, target_is_directory=True)
    with pytest.raises(ConfigurationError):
        corpus.run_corpus(FakeEngine(), source, output)
    assert not (output / "_flexaligner_runs").exists()


@pytest.mark.parametrize(
    "error", [ModelValidationError("broken model"), RuntimeError("unexpected")]
)
def test_fatal_failure_stops_and_writes_summary(tmp_path: Path, error: BaseException) -> None:
    source, output = tmp_path / "in", tmp_path / "out"
    for name in ("a", "b", "c"):
        pair(source, name)
    engine = FakeEngine({"b": error})
    with pytest.raises(type(error)):
        corpus.run_corpus(engine, source, output)
    items, summary = reports(output)
    assert summary["exit_code"] == 3 and summary["status"] == "failed"
    assert summary["counts"] == {"succeeded": 1, "failed": 1, "skipped": 0, "conflict": 0}
    assert len(items) == 2 and len(engine.requests) == 2
    assert (output / "a.TextGrid").exists() and not (output / "c.TextGrid").exists()
    assert not engine.in_session


def test_keyboard_interrupt_records_partial_summary_and_reraises(tmp_path: Path) -> None:
    source, output = tmp_path / "in", tmp_path / "out"
    pair(source, "a")
    pair(source, "b")
    engine = FakeEngine({"b": KeyboardInterrupt()})
    with pytest.raises(KeyboardInterrupt):
        corpus.run_corpus(engine, source, output)
    items, summary = reports(output)
    assert summary["exit_code"] == 130 and summary["status"] == "interrupted"
    assert len(items) == 2 and summary["counts"]["succeeded"] == 1
    assert items[-1]["error"]["code"] == "interrupted"
    assert (output / "a.TextGrid").exists()


@pytest.mark.parametrize("phase", ["setup", "cleanup"])
def test_batch_context_errors_still_write_summary(tmp_path: Path, phase: str) -> None:
    source, output = tmp_path / "in", tmp_path / "out"
    pair(source, "sample")
    engine = FakeEngine()
    setattr(engine, phase + "_error", ModelValidationError("global failure"))
    with pytest.raises(ModelValidationError):
        corpus.run_corpus(engine, source, output)
    _, summary = reports(output)
    assert summary["exit_code"] == 3
    assert summary["processed"] == (0 if phase == "setup" else 1)


def test_item_report_is_flushed_before_next_alignment(tmp_path: Path) -> None:
    source, output = tmp_path / "in", tmp_path / "out"
    pair(source, "a")
    pair(source, "b")

    class ObservingEngine(FakeEngine):
        def align(self, request):
            items_path = next(output.glob("_flexaligner_runs/*/items.jsonl"))
            self.observed_counts.append(len(items_path.read_text().splitlines()))
            return super().align(request)

    engine = ObservingEngine()
    corpus.run_corpus(engine, source, output)
    assert engine.observed_counts == [0, 1]


def test_report_creation_failure_is_output_error(tmp_path: Path) -> None:
    source, output = tmp_path / "in", tmp_path / "out"
    pair(source, "sample")
    output.mkdir()
    (output / "_flexaligner_runs").write_text("occupied", encoding="utf-8")
    engine = FakeEngine()
    with pytest.raises(OutputError, match="create batch report"):
        corpus.run_corpus(engine, source, output)
    assert not engine.requests


def test_report_append_failure_stops_without_rolling_back(tmp_path: Path, monkeypatch) -> None:
    source, output = tmp_path / "in", tmp_path / "out"
    pair(source, "a")
    pair(source, "b")

    def fail_append(*_args):
        raise OutputError("disk full")

    monkeypatch.setattr(corpus, "_write_item", fail_append)
    engine = FakeEngine()
    with pytest.raises(OutputError, match="disk full"):
        corpus.run_corpus(engine, source, output)
    _, summary = reports(output)
    assert summary["exit_code"] == 3
    assert summary["counts"]["succeeded"] == 1
    assert (output / "a.TextGrid").exists() and not (output / "b.TextGrid").exists()


def test_each_run_has_independent_uuid_report(tmp_path: Path) -> None:
    source, output = tmp_path / "in", tmp_path / "out"
    pair(source, "sample")
    first = corpus.run_corpus(FakeEngine(), source, output)
    second = corpus.run_corpus(FakeEngine(), source, output, skip_existing=True)
    assert first["run_id"] != second["run_id"]
    assert Path(first["summary_path"]).exists() and Path(second["summary_path"]).exists()


@pytest.mark.parametrize("extension", ["json", "..txt", "", None])
def test_invalid_extension_rejected(tmp_path: Path, extension) -> None:
    source, output = tmp_path / "in", tmp_path / "out"
    pair(source, "sample")
    with pytest.raises(ConfigurationError):
        corpus.run_corpus(FakeEngine(), source, output, text_extension=extension)
    assert not output.exists()


def test_directory_is_not_a_skippable_textgrid(tmp_path: Path) -> None:
    source, output = tmp_path / "in", tmp_path / "out"
    pair(source, "sample")
    (output / "sample.TextGrid").mkdir(parents=True)
    summary = corpus.run_corpus(FakeEngine(), source, output, skip_existing=True)
    assert summary["counts"]["conflict"] == 1


def test_alignment_output_race_is_item_conflict(tmp_path: Path) -> None:
    source, output = tmp_path / "in", tmp_path / "out"
    pair(source, "a")
    pair(source, "b")
    engine = FakeEngine({"a": ArtifactExistsError("racing writer")})
    summary = corpus.run_corpus(engine, source, output)
    assert summary["counts"]["conflict"] == 1
    assert summary["counts"]["succeeded"] == 1


@pytest.mark.parametrize("root", ["source", "output"])
def test_symlink_root_is_rejected(tmp_path: Path, root: str) -> None:
    source, output, alias = tmp_path / "in", tmp_path / "out", tmp_path / "alias"
    pair(source, "sample")
    if root == "source":
        alias.symlink_to(source, target_is_directory=True)
        source = alias
    else:
        output.mkdir()
        alias.symlink_to(output, target_is_directory=True)
        output = alias
    with pytest.raises(ConfigurationError):
        corpus.run_corpus(FakeEngine(), source, output)


def test_invalid_path_types_and_skip_flag_rejected(tmp_path: Path) -> None:
    source, output = tmp_path / "in", tmp_path / "out"
    pair(source, "sample")
    with pytest.raises(ConfigurationError, match=r"pathlib\.Path"):
        corpus.run_corpus(FakeEngine(), str(source), output)
    with pytest.raises(ConfigurationError, match="boolean"):
        corpus.run_corpus(FakeEngine(), source, output, skip_existing="yes")


def test_output_root_file_rejected(tmp_path: Path) -> None:
    source, output = tmp_path / "in", tmp_path / "out"
    pair(source, "sample")
    output.write_text("occupied", encoding="utf-8")
    with pytest.raises(ConfigurationError):
        corpus.run_corpus(FakeEngine(), source, output)
    assert output.read_text() == "occupied"


def test_scan_permission_error_is_configuration_error(tmp_path: Path, monkeypatch) -> None:
    source, output = tmp_path / "in", tmp_path / "out"
    pair(source, "sample")

    def failed_walk(*_args, onerror, **_kwargs):
        onerror(PermissionError("denied"))

    monkeypatch.setattr(corpus.os, "walk", failed_walk)
    with pytest.raises(ConfigurationError, match="enumerate"):
        corpus.run_corpus(FakeEngine(), source, output)
    assert not output.exists()


def test_report_serializers_convert_io_failures_to_output_error(tmp_path: Path) -> None:
    class BrokenHandle:
        def write(self, _text):
            raise OSError("disk full")

    item = corpus.BatchItemResult(
        audio_path=tmp_path / "a.wav",
        output_path=tmp_path / "a.TextGrid",
        status="failed",
        elapsed_seconds=0.0,
    )
    with pytest.raises(OutputError, match="append batch item"):
        corpus._write_item(BrokenHandle(), item)
    with pytest.raises(OutputError, match="write batch summary"):
        corpus._write_summary(tmp_path / "missing" / "summary.json", {})


def test_error_serialization_never_retains_foreign_objects() -> None:
    foreign = RuntimeError("heavy traceback")
    error = FlexAlignerError("expected", context={"nested": [{"exception": foreign}]})
    serialized = corpus._error_dict(error)
    assert serialized["context"] == {
        "nested": [{"exception": {"unsupported_type": "RuntimeError"}}]
    }
    assert "heavy traceback" not in json.dumps(serialized)


def test_failure_report_error_supersedes_interrupt_with_output_error(
    tmp_path: Path, monkeypatch
) -> None:
    source, output = tmp_path / "in", tmp_path / "out"
    pair(source, "sample")

    def fail_append(*_args):
        raise OutputError("report unavailable")

    monkeypatch.setattr(corpus, "_write_item", fail_append)
    with pytest.raises(OutputError):
        corpus.run_corpus(FakeEngine({"sample": KeyboardInterrupt()}), source, output)
    _, summary = reports(output)
    assert summary["exit_code"] == 3 and summary["error"]["code"] == "output_error"


@pytest.mark.parametrize("skip", [False, True])
@pytest.mark.parametrize("dangling", [False, True])
def test_formal_symlink_is_conflict_not_skipped_or_followed(
    tmp_path: Path, skip: bool, dangling: bool
) -> None:
    source, output, external = tmp_path / "in", tmp_path / "out", tmp_path / "external"
    pair(source, "sample")
    output.mkdir()
    if not dangling:
        external.write_text("preserve target", encoding="utf-8")
    final = output / "sample.TextGrid"
    final.symlink_to(external)
    engine = FakeEngine()
    summary = corpus.run_corpus(engine, source, output, skip_existing=skip)
    assert summary["counts"]["conflict"] == 1 and not engine.requests
    assert final.is_symlink()
    if dangling:
        assert not external.exists()
    else:
        assert external.read_text() == "preserve target"
