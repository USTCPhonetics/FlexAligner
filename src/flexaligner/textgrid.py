"""Long TextGrid records, merge semantics, and validated artifact writes."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .core.stage1 import RuntimeChunk
from .errors import (
    ArtifactExistsError,
    FlexAlignerError,
    OutputError,
    OutputValidationError,
)

EPSILON = 1.0e-6
MERGE_EPSILON = 1.0e-5
EXPECTED_TIER_ORDER = ("phones", "words")
FileIdentity = tuple[int, int]


@dataclass(frozen=True, slots=True)
class Interval:
    xmin: float
    xmax: float
    text: str


@dataclass(frozen=True, slots=True)
class IntervalTier:
    name: str
    xmin: float
    xmax: float
    intervals: tuple[Interval, ...]


@dataclass(frozen=True, slots=True)
class TextGridDocument:
    xmin: float
    xmax: float
    tiers: tuple[IntervalTier, ...]


@dataclass(frozen=True, slots=True)
class LocalAlignment:
    textgrid: TextGridDocument
    redecode_stats: object | None = None


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
        return value[1:-1].replace('""', '"')
    return value.replace('""', '"')


def _escape_textgrid(value: str) -> str:
    return value.replace('"', '""')


def serialize_textgrid_long(textgrid: TextGridDocument) -> str:
    """Serialize a long-form Praat TextGrid without writing a file."""

    lines = [
        'File type = "ooTextFile"',
        'Object class = "TextGrid"',
        "",
        f"xmin = {textgrid.xmin:.6f}",
        f"xmax = {textgrid.xmax:.6f}",
        "tiers? <exists>",
        f"size = {len(textgrid.tiers)}",
        "item []:",
    ]
    for tier_index, tier in enumerate(textgrid.tiers, start=1):
        lines.extend(
            (
                f"    item [{tier_index}]:",
                '        class = "IntervalTier"',
                f'        name = "{_escape_textgrid(tier.name)}"',
                f"        xmin = {tier.xmin:.6f}",
                f"        xmax = {tier.xmax:.6f}",
                f"        intervals: size = {len(tier.intervals)}",
            )
        )
        for interval_index, interval in enumerate(tier.intervals, start=1):
            lines.extend(
                (
                    f"        intervals [{interval_index}]:",
                    f"            xmin = {interval.xmin:.6f}",
                    f"            xmax = {interval.xmax:.6f}",
                    f'            text = "{_escape_textgrid(interval.text)}"',
                )
            )
    return "\n".join(lines) + "\n"


def parse_textgrid_long(path: Path) -> TextGridDocument:
    """Parse the long TextGrid subset emitted by :func:`serialize_textgrid_long`."""

    if not path.is_file():
        raise FileNotFoundError(f"TextGrid not found: {path}")
    lines = path.read_text(encoding="utf-8", errors="strict").splitlines()

    def first_value(pattern: str) -> str | None:
        for line in lines:
            match = re.match(pattern, line.strip())
            if match:
                return match.group(1)
        return None

    xmin_text = first_value(r"xmin\s*=\s*([0-9.eE+-]+)\s*$")
    xmax_text = first_value(r"xmax\s*=\s*([0-9.eE+-]+)\s*$")
    if xmin_text is None or xmax_text is None:
        raise ValueError(f"Failed to parse global xmin/xmax: {path}")

    tiers: list[IntervalTier] = []
    index = 0
    while index < len(lines):
        if not re.match(r"item\s*\[\d+\]\s*:", lines[index].strip()):
            index += 1
            continue
        tier_class: str | None = None
        name: str | None = None
        tier_xmin: float | None = None
        tier_xmax: float | None = None
        intervals: list[Interval] = []
        index += 1
        while index < len(lines) and not re.match(r"item\s*\[\d+\]\s*:", lines[index].strip()):
            line = lines[index].strip()
            if line.startswith("class"):
                tier_class = _strip_quotes(line.split("=", 1)[1])
            elif line.startswith("name"):
                name = _strip_quotes(line.split("=", 1)[1])
            elif line.startswith("xmin") and tier_xmin is None:
                tier_xmin = float(line.split("=", 1)[1])
            elif line.startswith("xmax") and tier_xmax is None:
                tier_xmax = float(line.split("=", 1)[1])
            elif re.match(r"intervals\s*\[\d+\]\s*:", line):
                interval_xmin: float | None = None
                interval_xmax: float | None = None
                text = ""
                index += 1
                while index < len(lines):
                    interval_line = lines[index].strip()
                    if interval_line.startswith("xmin"):
                        interval_xmin = float(interval_line.split("=", 1)[1])
                    elif interval_line.startswith("xmax"):
                        interval_xmax = float(interval_line.split("=", 1)[1])
                    elif interval_line.startswith("text"):
                        text = _strip_quotes(interval_line.split("=", 1)[1])
                    elif re.match(r"(intervals|item)\s*\[", interval_line):
                        index -= 1
                        break
                    index += 1
                if interval_xmin is None or interval_xmax is None:
                    raise ValueError(f"Bad interval near line={index} in {path}")
                intervals.append(Interval(interval_xmin, interval_xmax, text))
            index += 1
        if tier_class == "IntervalTier":
            if name is None or tier_xmin is None or tier_xmax is None:
                raise ValueError(f"Incomplete IntervalTier header: {path}")
            tiers.append(IntervalTier(name, tier_xmin, tier_xmax, tuple(intervals)))

    if not tiers:
        raise ValueError(f"No IntervalTier parsed from {path}")
    return TextGridDocument(float(xmin_text), float(xmax_text), tuple(tiers))


def validate_textgrid_structure(
    textgrid: TextGridDocument,
    *,
    context: str,
    expected_tier_order: tuple[str, ...] = EXPECTED_TIER_ORDER,
) -> None:
    """Validate bounds, exact tier order, and non-overlapping intervals.

    Continuous coverage is intentionally not required; that remains TBD-ALG-001.
    """

    if not math.isfinite(textgrid.xmin) or not math.isfinite(textgrid.xmax):
        raise ValueError(f"Non-finite TextGrid bounds for {context}")
    if textgrid.xmax <= textgrid.xmin:
        raise ValueError(f"Invalid TextGrid bounds for {context}: {textgrid.xmin}..{textgrid.xmax}")
    tier_names = tuple(tier.name for tier in textgrid.tiers)
    if tier_names != expected_tier_order:
        raise ValueError(
            f"Unexpected tier order for {context}: actual={tier_names!r}, "
            f"expected={expected_tier_order!r}"
        )
    if len(tier_names) != len(set(tier_names)):
        raise ValueError(f"Duplicate tier names for {context}: {tier_names!r}")

    for tier in textgrid.tiers:
        if abs(tier.xmin - textgrid.xmin) > EPSILON or abs(tier.xmax - textgrid.xmax) > EPSILON:
            raise ValueError(f"Tier/global bounds mismatch for {context}, tier={tier.name!r}")
        previous_end = tier.xmin
        for interval_index, interval in enumerate(tier.intervals):
            if not math.isfinite(interval.xmin) or not math.isfinite(interval.xmax):
                raise ValueError(
                    f"Non-finite interval for {context}, tier={tier.name!r}, "
                    f"interval_index={interval_index}"
                )
            if interval.xmax <= interval.xmin:
                raise ValueError(
                    f"Non-positive interval for {context}, tier={tier.name!r}, "
                    f"interval_index={interval_index}"
                )
            if interval.xmin < tier.xmin - EPSILON or interval.xmax > tier.xmax + EPSILON:
                raise ValueError(
                    f"Interval outside tier bounds for {context}, tier={tier.name!r}, "
                    f"interval_index={interval_index}"
                )
            if interval.xmin < previous_end - EPSILON:
                raise ValueError(
                    f"Overlapping/backward intervals for {context}, tier={tier.name!r}, "
                    f"interval_index={interval_index}"
                )
            previous_end = interval.xmax


def clip_shift_interval(
    local: Interval,
    *,
    chunk_start: float,
    chunk_end: float,
    local_xmax: float,
) -> Interval | None:
    """Clip a local interval to its chunk and shift it to global time."""

    chunk_duration = chunk_end - chunk_start
    if chunk_duration <= EPSILON:
        raise ValueError(f"Invalid chunk duration: {chunk_start:.6f}..{chunk_end:.6f}")
    if local.xmax - local.xmin <= EPSILON:
        return None
    valid_local_end = min(float(local_xmax), float(chunk_duration))
    local_start = max(0.0, float(local.xmin))
    local_end = min(float(local.xmax), valid_local_end)
    if local_end - local_start <= EPSILON:
        return None
    return Interval(
        xmin=chunk_start + local_start,
        xmax=chunk_start + local_end,
        text=local.text,
    )


def labels_from_intervals(
    intervals: Iterable[Interval], *, ignore_labels: set[str]
) -> tuple[str, ...]:
    ignore = {label.strip().lower() for label in ignore_labels}
    return tuple(
        label
        for interval in intervals
        if (label := interval.text.strip()) and label.lower() not in ignore
    )


def merge_adjacent_null(intervals: Iterable[Interval]) -> tuple[Interval, ...]:
    """Sort intervals and merge adjacent ``NULL`` spans only."""

    merged: list[Interval] = []
    for interval in sorted(intervals, key=lambda item: (item.xmin, item.xmax, item.text)):
        if interval.xmax - interval.xmin <= EPSILON:
            continue
        if (
            interval.text == "NULL"
            and merged
            and merged[-1].text == "NULL"
            and abs(merged[-1].xmax - interval.xmin) <= MERGE_EPSILON
        ):
            previous = merged[-1]
            merged[-1] = Interval(previous.xmin, max(previous.xmax, interval.xmax), "NULL")
        else:
            merged.append(interval)
    return tuple(merged)


def _validate_word_sequence(
    *, actual: Sequence[str], expected: Sequence[str], context: str
) -> None:
    if tuple(actual) == tuple(expected):
        return
    shared_length = min(len(actual), len(expected))
    mismatch_index = next(
        (index for index in range(shared_length) if actual[index] != expected[index]),
        shared_length,
    )
    actual_token = actual[mismatch_index] if mismatch_index < len(actual) else None
    expected_token = expected[mismatch_index] if mismatch_index < len(expected) else None
    raise ValueError(
        f"Word sequence mismatch during merge: {context}; actual_len={len(actual)}, "
        f"expected_len={len(expected)}, mismatch_pos={mismatch_index}, "
        f"actual_token={actual_token!r}, expected_token={expected_token!r}"
    )


def merge_local_alignments(
    *,
    chunks: Sequence[RuntimeChunk],
    local_alignments: Sequence[LocalAlignment],
    full_duration_s: float,
    expected_words: Sequence[str],
    word_sil_label: str,
    sph_word_label: str,
) -> TextGridDocument:
    """Merge chunk-local TextGrids while preserving reference gap behavior."""

    if len(chunks) != len(local_alignments):
        raise ValueError(
            f"Chunk/alignment count mismatch: chunks={len(chunks)}, "
            f"alignments={len(local_alignments)}"
        )
    if not chunks:
        raise ValueError("Cannot merge zero chunks")
    if not math.isfinite(full_duration_s) or full_duration_s <= 0.0:
        raise ValueError(f"Invalid full audio duration: {full_duration_s!r}")

    accumulated: dict[str, list[Interval]] = {name: [] for name in EXPECTED_TIER_ORDER}
    ignored_labels = {"", "NULL", "null", word_sil_label, sph_word_label}

    def add_gap(start_s: float, end_s: float) -> None:
        if end_s - start_s <= EPSILON:
            return
        for tier_name in EXPECTED_TIER_ORDER:
            accumulated[tier_name].append(Interval(start_s, end_s, "NULL"))

    previous_end = 0.0
    concatenated_words: list[str] = []
    for chunk, local_alignment in zip(chunks, local_alignments, strict=True):
        if chunk.start_s > previous_end + EPSILON:
            add_gap(previous_end, chunk.start_s)
        elif chunk.start_s < previous_end - EPSILON:
            raise ValueError(
                f"Chunk overlap during merge: chunk_id={chunk.chunk_id}, "
                f"start={chunk.start_s}, previous_end={previous_end}"
            )

        local_textgrid = local_alignment.textgrid
        validate_textgrid_structure(local_textgrid, context=f"local {chunk.chunk_id}")
        shifted_words: list[Interval] = []
        for tier in local_textgrid.tiers:
            for interval in tier.intervals:
                shifted = clip_shift_interval(
                    interval,
                    chunk_start=chunk.start_s,
                    chunk_end=chunk.end_s,
                    local_xmax=tier.xmax,
                )
                if shifted is None:
                    continue
                accumulated[tier.name].append(shifted)
                if tier.name == "words":
                    shifted_words.append(shifted)

        actual_chunk_words = labels_from_intervals(shifted_words, ignore_labels=ignored_labels)
        _validate_word_sequence(
            actual=actual_chunk_words,
            expected=chunk.words,
            context=f"shifted local chunk_id={chunk.chunk_id}",
        )
        concatenated_words.extend(actual_chunk_words)
        previous_end = chunk.end_s

    if full_duration_s > previous_end + EPSILON:
        add_gap(previous_end, full_duration_s)
    elif previous_end > full_duration_s + 0.1:
        raise ValueError(
            f"Last chunk end {previous_end:.6f}s exceeds full duration {full_duration_s:.6f}s"
        )

    final_tiers = tuple(
        IntervalTier(
            name=tier_name,
            xmin=0.0,
            xmax=full_duration_s,
            intervals=merge_adjacent_null(accumulated[tier_name]),
        )
        for tier_name in EXPECTED_TIER_ORDER
    )
    merged = TextGridDocument(0.0, full_duration_s, final_tiers)
    validate_textgrid_structure(merged, context="merged TextGrid")

    word_tier = merged.tiers[1]
    merged_words = labels_from_intervals(word_tier.intervals, ignore_labels=ignored_labels)
    _validate_word_sequence(
        actual=merged_words,
        expected=expected_words,
        context="final merged TextGrid",
    )
    _validate_word_sequence(
        actual=concatenated_words,
        expected=expected_words,
        context="concatenated shifted local word intervals",
    )
    return merged


def _path_present(path: Path) -> bool:
    return os.path.lexists(path)


def _temporary_path(path: Path) -> Path:
    return path.with_name(path.name + ".tmp")


def _identity(path: Path) -> FileIdentity:
    stat_result = os.lstat(path)
    return stat_result.st_dev, stat_result.st_ino


def _write_exclusive(
    path: Path,
    contents: str,
    *,
    owned: dict[Path, FileIdentity],
) -> None:
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            stat_result = os.fstat(handle.fileno())
            owned[path] = (stat_result.st_dev, stat_result.st_ino)
            handle.write(contents)
    except FileExistsError as error:
        raise ArtifactExistsError(
            f"Temporary output appeared before staging: {path}",
            context={"path": str(path), "role": "temporary"},
        ) from error


def _validate_written_textgrid(
    path: Path,
    *,
    expected_words: Sequence[str],
    word_sil_label: str,
    sph_word_label: str,
) -> TextGridDocument:
    try:
        parsed = parse_textgrid_long(path)
        validate_textgrid_structure(parsed, context=f"written TextGrid {path}")
        words = labels_from_intervals(
            parsed.tiers[1].intervals,
            ignore_labels={"", "NULL", "null", word_sil_label, sph_word_label},
        )
        _validate_word_sequence(
            actual=words,
            expected=expected_words,
            context=f"written TextGrid word tier: {path}",
        )
    except Exception as error:
        raise OutputValidationError(
            "Staged TextGrid failed read-back validation",
            context={"path": str(path)},
        ) from error
    return parsed


def _validate_written_metadata(path: Path, expected_payload: object) -> None:
    try:
        parsed = json.loads(path.read_text(encoding="utf-8", errors="strict"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise OutputValidationError(
            "Staged metadata failed read-back parsing",
            context={"path": str(path)},
        ) from error
    if parsed != expected_payload:
        raise OutputValidationError(
            "Staged metadata changed during JSON round-trip",
            context={"path": str(path)},
        )
    if not isinstance(parsed, dict) or parsed.get("calibrated") is not False:
        raise OutputValidationError(
            "Chunk metadata must explicitly declare calibrated=false",
            context={"path": str(path)},
        )


def _unlink_if_owned(path: Path, expected_identity: FileIdentity) -> tuple[bool, OSError | None]:
    """Remove one owned path without deleting a file installed by a racer."""

    try:
        actual_identity = _identity(path)
    except FileNotFoundError:
        return True, None
    except OSError as error:
        return False, error
    if actual_identity != expected_identity:
        return False, None
    try:
        path.unlink()
    except OSError as error:
        return False, error
    return True, None


def _publish_no_clobber(
    *,
    temporary: Path,
    official: Path,
    temporary_identity: FileIdentity,
    owned_officials: dict[Path, FileIdentity],
) -> None:
    """Atomically create *official* without replacing an existing directory entry."""

    try:
        if _identity(temporary) != temporary_identity:
            raise OutputValidationError(
                "Temporary artifact identity changed before publication",
                context={"path": str(temporary)},
            )
        os.link(temporary, official, follow_symlinks=False)
    except FileExistsError as error:
        raise ArtifactExistsError(
            f"Official output appeared before commit: {official}",
            context={"path": str(official), "role": "official"},
        ) from error

    owned_officials[official] = temporary_identity
    official_identity = _identity(official)
    if official_identity != temporary_identity:
        raise OutputValidationError(
            "Temporary artifact identity changed during publication",
            context={"temporary_path": str(temporary), "official_path": str(official)},
        )

    removed, unlink_error = _unlink_if_owned(temporary, temporary_identity)
    if unlink_error is not None:
        raise OutputError(
            "Published artifact but could not remove its temporary hard link",
            context={"path": str(temporary), "reason": str(unlink_error)},
        ) from unlink_error
    if not removed:
        raise OutputValidationError(
            "Temporary artifact identity changed before cleanup",
            context={"path": str(temporary)},
        )


def _validate_committed_bytes(
    path: Path,
    *,
    expected_identity: FileIdentity,
    expected_bytes: bytes,
    role: str,
) -> None:
    """Recheck identity and exact staged bytes after publication."""

    try:
        if _identity(path) != expected_identity:
            raise OutputValidationError(
                f"Committed {role} identity changed before validation",
                context={"path": str(path), "role": role},
            )
        actual_bytes = path.read_bytes()
        if actual_bytes != expected_bytes:
            raise OutputValidationError(
                f"Committed {role} bytes differ from the validated temporary artifact",
                context={"path": str(path), "role": role},
            )
        if _identity(path) != expected_identity:
            raise OutputValidationError(
                f"Committed {role} identity changed during validation",
                context={"path": str(path), "role": role},
            )
    except OutputValidationError:
        raise
    except OSError as error:
        raise OutputValidationError(
            f"Committed {role} could not be read back",
            context={"path": str(path), "role": role},
        ) from error


def write_validated_artifacts(
    *,
    textgrid: TextGridDocument,
    output_path: Path,
    expected_words: Sequence[str],
    word_sil_label: str,
    sph_word_label: str,
    metadata_path: Path | None = None,
    metadata: Mapping[str, object] | None = None,
) -> str:
    """Stage, validate and commit TextGrid plus optional uncalibrated metadata.

    The metadata file is published first through an atomic no-clobber hard link.
    The TextGrid is published last and is therefore the process-level success
    marker.  Any caught failure rolls back files still owned by this call while
    preserving directory entries installed by a racer.  Cross-file crash
    atomicity is intentionally not claimed (TBD-OUT-001).
    """

    if (metadata_path is None) != (metadata is None):
        raise OutputError("metadata_path and metadata must be provided together")

    artifacts: list[tuple[Path, Path]] = [(output_path, _temporary_path(output_path))]
    if metadata_path is not None:
        artifacts.append((metadata_path, _temporary_path(metadata_path)))
    normalized_paths = [os.path.abspath(path) for pair in artifacts for path in pair]
    if len(normalized_paths) != len(set(normalized_paths)):
        raise OutputError("Official and temporary artifact paths must be distinct")
    for official, temporary in artifacts:
        for candidate, role in ((official, "official"), (temporary, "temporary")):
            if _path_present(candidate):
                raise ArtifactExistsError(
                    f"{role.capitalize()} output already exists: {candidate}",
                    context={"path": str(candidate), "role": role},
                )

    output_temporary = artifacts[0][1]
    metadata_temporary = artifacts[1][1] if len(artifacts) == 2 else None
    owned_temporaries: dict[Path, FileIdentity] = {}
    owned_officials: dict[Path, FileIdentity] = {}
    try:
        for official, _temporary in artifacts:
            official.parent.mkdir(parents=True, exist_ok=True)

        _write_exclusive(
            output_temporary,
            serialize_textgrid_long(textgrid),
            owned=owned_temporaries,
        )
        _validate_written_textgrid(
            output_temporary,
            expected_words=expected_words,
            word_sil_label=word_sil_label,
            sph_word_label=sph_word_label,
        )
        expected_output_bytes = output_temporary.read_bytes()
        output_digest = hashlib.sha256(expected_output_bytes).hexdigest()

        expected_metadata: object | None = None
        expected_metadata_bytes: bytes | None = None
        if metadata_path is not None and metadata is not None and metadata_temporary is not None:
            try:
                serialized_metadata = (
                    json.dumps(
                        metadata,
                        ensure_ascii=False,
                        indent=2,
                        sort_keys=True,
                        allow_nan=False,
                    )
                    + "\n"
                )
                expected_metadata = json.loads(serialized_metadata)
            except (TypeError, ValueError) as error:
                raise OutputValidationError("Metadata is not strict JSON") from error
            _write_exclusive(
                metadata_temporary,
                serialized_metadata,
                owned=owned_temporaries,
            )
            _validate_written_metadata(metadata_temporary, expected_metadata)
            expected_metadata_bytes = metadata_temporary.read_bytes()

        commit_order = artifacts[1:] + artifacts[:1]
        for official, temporary in commit_order:
            _publish_no_clobber(
                temporary=temporary,
                official=official,
                temporary_identity=owned_temporaries[temporary],
                owned_officials=owned_officials,
            )

        if metadata_path is not None:
            if expected_metadata is None or expected_metadata_bytes is None:
                raise AssertionError("metadata validation state is incomplete")
            _validate_committed_bytes(
                metadata_path,
                expected_identity=owned_officials[metadata_path],
                expected_bytes=expected_metadata_bytes,
                role="metadata",
            )
            _validate_written_metadata(metadata_path, expected_metadata)
        _validate_committed_bytes(
            output_path,
            expected_identity=owned_officials[output_path],
            expected_bytes=expected_output_bytes,
            role="TextGrid",
        )
        _validate_written_textgrid(
            output_path,
            expected_words=expected_words,
            word_sil_label=word_sil_label,
            sph_word_label=sph_word_label,
        )
        return output_digest
    except Exception as error:
        cleanup_failures: list[str] = []
        owned_paths = [
            *reversed(tuple(owned_officials.items())),
            *owned_temporaries.items(),
        ]
        for path, path_identity in owned_paths:
            _removed, cleanup_error = _unlink_if_owned(path, path_identity)
            if cleanup_error is not None:
                cleanup_failures.append(f"{path}: {cleanup_error}")
        if cleanup_failures:
            raise OutputError(
                "Artifact write failed and cleanup was incomplete",
                context={"cleanup_errors": "; ".join(cleanup_failures)},
            ) from error
        if isinstance(error, FlexAlignerError):
            raise
        raise OutputError(
            "Failed to write validated alignment artifacts",
            context={"path": str(output_path)},
        ) from error


__all__ = [
    "Interval",
    "IntervalTier",
    "LocalAlignment",
    "TextGridDocument",
    "clip_shift_interval",
    "labels_from_intervals",
    "merge_adjacent_null",
    "merge_local_alignments",
    "parse_textgrid_long",
    "serialize_textgrid_long",
    "validate_textgrid_structure",
    "write_validated_artifacts",
]
