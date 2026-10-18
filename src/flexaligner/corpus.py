"""Deterministic directory scheduling with append-only, per-run audit reports."""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, TextIO

from .batch import is_fatal_batch_error
from .contracts import AlignmentRequest, BatchItemResult, TextGridOutput
from .errors import (
    ArtifactExistsError,
    ConfigurationError,
    FlexAlignerError,
    InputValidationError,
    OutputError,
)

if TYPE_CHECKING:
    from .api import FlexAligner


def _safe_value(value: object) -> object:
    """Copy only JSON values, never retaining exceptions or runtime resources."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_value(item) for item in value]
    return {"unsupported_type": type(value).__name__}


def _error_dict(error: BaseException) -> dict[str, object]:
    if isinstance(error, FlexAlignerError):
        return {key: _safe_value(value) for key, value in error.to_dict().items()}
    return {
        "code": "interrupted" if isinstance(error, KeyboardInterrupt) else "internal_error",
        "message": str(error),
        "context": {"exception_type": type(error).__name__},
    }


def _case_key(path: Path) -> str:
    return path.as_posix().casefold()


def _validate_roots(corpus_dir: Path, output_dir: Path) -> tuple[Path, Path]:
    if corpus_dir.is_symlink() or not corpus_dir.is_dir():
        raise ConfigurationError("corpus_dir must be an existing non-symlink directory")
    if output_dir.is_symlink():
        raise ConfigurationError("output_dir must not be a symlink")
    corpus, output = corpus_dir.resolve(), output_dir.resolve()
    if corpus == output or corpus in output.parents or output in corpus.parents:
        raise ConfigurationError("Input and output directories must not contain each other")
    # Also reject containment on case-sensitive hosts that would alias on macOS/Windows.
    corpus_key, output_key = _case_key(corpus), _case_key(output)
    if (
        corpus_key == output_key
        or corpus_key.startswith(output_key + "/")
        or output_key.startswith(corpus_key + "/")
    ):
        raise ConfigurationError("Input and output directories have ambiguous case aliases")
    if output.exists() and not output.is_dir():
        raise ConfigurationError("output_dir must be a directory")
    return corpus, output


def _scan(corpus: Path, extension: str) -> tuple[list[Path], dict[str, Path]]:
    audio: list[Path] = []
    texts: dict[str, Path] = {}
    seen: dict[str, Path] = {}

    def walk_error(error: OSError) -> None:
        raise ConfigurationError(
            "Could not enumerate corpus directory", context={"detail": str(error)}
        ) from None

    for root, directories, files in os.walk(corpus, followlinks=False, onerror=walk_error):
        base = Path(root)
        directories[:] = sorted(name for name in directories if not (base / name).is_symlink())
        for name in sorted(files):
            path = base / name
            if path.is_symlink() or not path.is_file():
                continue
            suffix = path.suffix.casefold()
            if suffix not in {".wav", "." + extension}:
                continue
            relative = path.relative_to(corpus)
            for candidate in (relative, *relative.parents[:-1]):
                key = _case_key(candidate)
                if key in seen and seen[key] != candidate:
                    raise ConfigurationError(
                        "Corpus paths have ambiguous case aliases",
                        context={"first": str(seen[key]), "second": str(candidate)},
                    )
                seen[key] = candidate
            if suffix == ".wav":
                audio.append(relative)
            else:
                texts[_case_key(relative.with_suffix(""))] = relative
    audio.sort(key=lambda path: path.as_posix())
    if not audio:
        raise ConfigurationError("Corpus contains no non-symlink WAV files")
    return audio, texts


def _check_target(output: Path, relative: Path, *, allow_leaf_symlink: bool = False) -> None:
    """Reject path aliases and symlink traversal before any output is created."""

    current = output
    for index, component in enumerate(relative.parts):
        if current.is_dir():
            for existing in current.iterdir():
                if existing.name.casefold() == component.casefold() and existing.name != component:
                    raise ConfigurationError(
                        "Output path has an ambiguous case alias", context={"path": str(existing)}
                    )
        current = current / component
        leaf = index == len(relative.parts) - 1
        if current.is_symlink() and not (leaf and allow_leaf_symlink):
            raise ConfigurationError(
                "Output path must not traverse symlinks", context={"path": str(current)}
            )


def _existing_output(output_path: Path, skip_existing: bool) -> str | None:
    temporary = output_path.with_name(output_path.name + ".tmp")
    if os.path.lexists(temporary):
        return "conflict"
    if os.path.lexists(output_path):
        return (
            "skipped"
            if skip_existing and output_path.is_file() and not output_path.is_symlink()
            else "conflict"
        )
    return None


def _item_dict(item: BatchItemResult) -> dict[str, object]:
    record: dict[str, object] = {
        "audio_path": str(item.audio_path),
        "output_path": str(item.output_path),
        "status": item.status,
        "elapsed_seconds": item.elapsed_seconds,
        "device": item.device,
        "error": item.error,
    }
    if item.result is not None:
        record["output_sha256"] = item.result.output_sha256
        record["audio_duration_s"] = item.result.audio_duration_s
        record["utterance_id"] = item.result.utterance_id
        record["pronunciation_notices"] = [
            notice.to_dict() for notice in item.result.pronunciation_notices
        ]
    return record


def _write_item(handle: TextIO, item: BatchItemResult) -> None:
    try:
        handle.write(json.dumps(_item_dict(item), ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
    except (OSError, ValueError, TypeError) as error:
        raise OutputError(
            "Could not append batch item report", context={"detail": str(error)}
        ) from None


def _write_summary(path: Path, summary: dict[str, object]) -> None:
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
    except (OSError, ValueError, TypeError) as error:
        raise OutputError("Could not write batch summary", context={"detail": str(error)}) from None


def _process_item(
    engine: FlexAligner,
    audio_path: Path,
    text_path: Path | None,
    output_path: Path,
    *,
    skip_existing: bool,
) -> BatchItemResult:
    started = time.monotonic()
    existing = _existing_output(output_path, skip_existing)
    if existing is not None:
        return BatchItemResult(
            audio_path=audio_path,
            output_path=output_path,
            status=existing,
            elapsed_seconds=time.monotonic() - started,
            error=(
                _error_dict(ArtifactExistsError("Output artifact already exists"))
                if existing == "conflict"
                else None
            ),
        )
    try:
        if text_path is None:
            raise InputValidationError("Matching transcript is missing")
        try:
            transcript = text_path.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeError) as error:
            raise InputValidationError(
                "Transcript could not be read", context={"path": str(text_path)}
            ) from error
        if not transcript.strip():
            raise InputValidationError("Transcript is empty", context={"path": str(text_path)})
        result = engine.align(
            AlignmentRequest(
                audio_path=audio_path,
                transcript=transcript,
                output=TextGridOutput(path=output_path),
            )
        )
    except FlexAlignerError as error:
        if is_fatal_batch_error(error):
            raise
        return BatchItemResult(
            audio_path=audio_path,
            output_path=output_path,
            status="conflict" if isinstance(error, ArtifactExistsError) else "failed",
            elapsed_seconds=time.monotonic() - started,
            error=_error_dict(error),
            device=engine.resolved_device,
        )
    provenance = result.provenance
    device = str(provenance.device)
    if provenance.device_index is not None:
        device += f":{provenance.device_index}"
    return BatchItemResult(
        audio_path=audio_path,
        output_path=output_path,
        status="succeeded",
        elapsed_seconds=time.monotonic() - started,
        result=result,
        device=device,
    )


def run_corpus(
    engine: FlexAligner,
    corpus_dir: Path,
    output_dir: Path,
    *,
    text_extension: str = "txt",
    skip_existing: bool = False,
) -> dict[str, object]:
    """Align a stable WAV/transcript inventory, preserving every prior artifact.

    Input configuration errors are raised before creating output directories.
    Expected item failures continue; fatal errors and interrupts write a summary
    before propagating. Reports contain JSON data, never exception objects.
    """

    started = time.monotonic()
    if not isinstance(corpus_dir, Path) or not isinstance(output_dir, Path):
        raise ConfigurationError("corpus_dir and output_dir must be pathlib.Path objects")
    if not isinstance(text_extension, str) or text_extension.lower() not in {
        "txt",
        "lab",
        ".txt",
        ".lab",
    }:
        raise ConfigurationError("text_extension must be txt or lab")
    if not isinstance(skip_existing, bool):
        raise ConfigurationError("skip_existing must be a boolean")
    extension = text_extension.lower().lstrip(".")
    corpus, output = _validate_roots(corpus_dir, output_dir)
    audio, texts = _scan(corpus, extension)
    targets: dict[str, Path] = {}
    for relative in audio:
        target = relative.with_suffix(".TextGrid")
        key = _case_key(target)
        if key in targets:
            raise ConfigurationError("Multiple WAV files map to one output target")
        targets[key] = target
        _check_target(output, target, allow_leaf_symlink=True)
    _check_target(output, Path("_flexaligner_runs"))
    audio_stems = {_case_key(relative.with_suffix("")) for relative in audio}
    notices: list[dict[str, object]] = [
        {"code": "orphan_transcript", "path": str(corpus / relative)}
        for key, relative in sorted(texts.items())
        if key not in audio_stems
    ]
    run_id = str(uuid.uuid4())
    run_dir = output / "_flexaligner_runs" / run_id
    items_path, summary_path = run_dir / "items.jsonl", run_dir / "summary.json"
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
        handle = items_path.open("x", encoding="utf-8")
    except OSError as error:
        raise OutputError("Could not create batch report", context={"detail": str(error)}) from None
    counts = {status: 0 for status in ("succeeded", "failed", "skipped", "conflict")}
    summary: dict[str, object] = {
        "schema_version": "1",
        "run_id": run_id,
        "corpus_dir": str(corpus),
        "output_dir": str(output),
        "items_path": str(items_path),
        "summary_path": str(summary_path),
        "total": len(audio),
        "counts": counts,
        "notices": notices,
    }
    active: tuple[Path, Path] | None = None
    item_started = started
    try:
        for notice in notices:
            print(f"notice: orphan transcript: {notice['path']}", file=sys.stderr)
        with engine.batch_session():
            for index, relative in enumerate(audio, 1):
                audio_path, output_path = (
                    corpus / relative,
                    output / relative.with_suffix(".TextGrid"),
                )
                active = audio_path, output_path
                item_started = time.monotonic()
                _check_target(output, relative.with_suffix(".TextGrid"), allow_leaf_symlink=True)
                text = texts.get(_case_key(relative.with_suffix("")))
                item = _process_item(
                    engine,
                    audio_path,
                    corpus / text if text is not None else None,
                    output_path,
                    skip_existing=skip_existing,
                )
                counts[item.status] += 1
                _write_item(handle, item)
                active = None
                if item.result is not None:
                    for pronunciation_notice in item.result.pronunciation_notices:
                        print(
                            "WARNING "
                            + json.dumps(
                                {
                                    **pronunciation_notice.to_dict(),
                                    "audio_path": str(item.audio_path),
                                },
                                ensure_ascii=False,
                                sort_keys=True,
                            ),
                            file=sys.stderr,
                        )
                print(f"[{index}/{len(audio)}] {item.status}: {relative}", file=sys.stderr)
        summary.update(
            status="completed", exit_code=int(bool(counts["failed"] or counts["conflict"]))
        )
    except BaseException as error:
        summary.update(
            status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            exit_code=130 if isinstance(error, KeyboardInterrupt) else 3,
            error=_error_dict(error),
        )
        if active is not None and not isinstance(error, OutputError):
            failure = BatchItemResult(
                audio_path=active[0],
                output_path=active[1],
                status="failed",
                elapsed_seconds=time.monotonic() - item_started,
                error=_error_dict(error),
                device=engine.resolved_device,
            )
            counts["failed"] += 1
            try:
                _write_item(handle, failure)
            except OutputError as report_error:
                summary.update(status="failed", exit_code=3, error=_error_dict(report_error))
                raise report_error from None
        raise
    finally:
        summary.update(
            elapsed_seconds=time.monotonic() - started,
            device=engine.resolved_device,
            processed=sum(counts.values()),
        )
        try:
            handle.close()
        except OSError as error:
            summary.update(status="failed", exit_code=3, error=_error_dict(error))
            _write_summary(summary_path, summary)
            raise OutputError("Could not close batch item report") from None
        _write_summary(summary_path, summary)
    return summary
