"""Independent, model-free oracle for the reference Stage 1 helpers.

The functions in this module intentionally do not import production code or the
reference script.  They provide small NumPy implementations suitable for later
differential tests.  ``load_reference_subset`` is a separate test helper: it
executes only explicitly selected definitions from the frozen script and uses a
minimal NumPy-backed torch surface, so importing torch/transformers is never
required.
"""

from __future__ import annotations

import ast
import hashlib
import itertools
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from numpy.typing import NDArray

REFERENCE_SHA256 = "9ed4e21e615718ddfd10930359f55769fb27a0d284599cce45a3fc755e835de1"
CURRENT_BEHAVIOR_REPEATED_TARGETS = (
    "current_behavior: simplified trellis permits adjacent emissions of repeated targets"
)

REFERENCE_DEFINITIONS = (
    "Point",
    "Segment",
    "SegmentWithConf",
    "WordSpan",
    "Chunk",
    "WordAnchor",
    "GreedyPronResult",
    "RuntimeChunk",
    "normalize_word",
    "strip_arpabet_stress",
    "choose_greedy_pronunciations",
    "build_trellis",
    "backtrace",
    "points_to_segments",
    "compute_segment_confidence",
    "word_segments_with_confidence",
    "first_mismatch",
    "assert_tokens_equal",
    "make_word_anchors_from_emissions",
    "merge_word_anchors_into_chunks",
    "round_chunks_to_legacy_grid",
    "write_chunker_metadata",
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Point:
    token_index: int
    time_index: int


@dataclass(frozen=True, slots=True)
class Segment:
    label: str
    start_frame: int
    end_frame: int


@dataclass(frozen=True, slots=True)
class WordSpan:
    word_index: int
    word: str
    start_frame: int
    end_frame: int
    start_s: float
    end_s: float
    conf_log: float
    pron: list[str]


@dataclass(frozen=True, slots=True)
class WordAnchor:
    word_index: int
    word: str
    emit_start_frame: int
    emit_end_frame: int
    emit_start_s: float
    emit_end_s: float
    anchor_start_s: float
    anchor_end_s: float


@dataclass(frozen=True, slots=True)
class Chunk:
    start: float
    end: float
    words: list[str]
    word_indices: list[int]


@dataclass(frozen=True, slots=True)
class RuntimeChunk:
    chunk_id: str
    start_ms: int
    end_ms: int
    start_sample: int
    end_sample: int
    words: list[str]
    word_indices: list[int]


@dataclass(frozen=True, slots=True)
class GreedyPronunciation:
    phones: list[str]
    chosen_prons: list[list[str]]
    pron_choice_idxs: list[int]


def locate_reference(path: Path | None = None) -> Path:
    """Locate the repository's immutable snapshot; absence is a hard failure."""

    if path is not None:
        return path
    repository_root = Path(__file__).resolve().parents[2]
    snapshot = repository_root / "reference" / "align_single_cpu.py"
    if not snapshot.is_file():
        raise FileNotFoundError(f"Frozen reference snapshot is absent: {snapshot}")
    return snapshot


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class _NumpyTensor:
    """Only the tensor operations used by the selected reference helpers."""

    def __init__(self, value: Any) -> None:
        self._array = np.asarray(value)

    @property
    def shape(self) -> tuple[int, ...]:
        return self._array.shape

    @property
    def dtype(self) -> np.dtype[Any]:
        return self._array.dtype

    @property
    def device(self) -> None:
        return None

    def size(self, dim: int | None = None) -> int | tuple[int, ...]:
        return self.shape if dim is None else self.shape[dim]

    def __getitem__(self, key: Any) -> Any:
        result = self._array[_unwrap_index(key)]
        return _NumpyTensor(result) if isinstance(result, np.ndarray) else result

    def __setitem__(self, key: Any, value: Any) -> None:
        self._array[_unwrap_index(key)] = _unwrap_value(value)

    def __add__(self, other: Any) -> _NumpyTensor:
        return _NumpyTensor(self._array + _unwrap_value(other))

    def __radd__(self, other: Any) -> _NumpyTensor:
        return _NumpyTensor(_unwrap_value(other) + self._array)

    def mean(self) -> Any:
        return self._array.mean()

    def item(self) -> Any:
        return self._array.item()

    def to_numpy(self) -> NDArray[Any]:
        return self._array.copy()


def _unwrap_index(key: Any) -> Any:
    if isinstance(key, _NumpyTensor):
        return key._array
    if isinstance(key, tuple):
        return tuple(_unwrap_index(item) for item in key)
    return key


def _unwrap_value(value: Any) -> Any:
    return value._array if isinstance(value, _NumpyTensor) else value


class _TorchShim:
    """NumPy implementation of the tiny torch surface in trellis helpers."""

    long = np.int64

    @staticmethod
    def tensor(
        value: Any,
        *,
        device: Any = None,
        dtype: Any = None,
    ) -> _NumpyTensor:
        del device
        return _NumpyTensor(np.asarray(value, dtype=dtype))

    @staticmethod
    def full(
        shape: tuple[int, ...],
        fill_value: float,
        *,
        device: Any = None,
        dtype: Any = None,
    ) -> _NumpyTensor:
        del device
        return _NumpyTensor(np.full(shape, fill_value, dtype=dtype))

    @staticmethod
    def cumsum(value: _NumpyTensor, *, dim: int) -> _NumpyTensor:
        return _NumpyTensor(np.cumsum(value._array, axis=dim))

    @staticmethod
    def maximum(left: _NumpyTensor, right: _NumpyTensor) -> _NumpyTensor:
        return _NumpyTensor(np.maximum(left._array, right._array))

    @staticmethod
    def max(value: _NumpyTensor) -> Any:
        return np.max(value._array)

    @staticmethod
    def isfinite(value: Any) -> Any:
        return np.isfinite(_unwrap_value(value))

    @staticmethod
    def argmax(value: _NumpyTensor) -> Any:
        return np.argmax(value._array)


def as_reference_tensor(value: Any) -> _NumpyTensor:
    """Wrap an array for functions extracted from the torch-based reference."""

    return _NumpyTensor(value)


def load_reference_subset(
    path: Path | None = None,
    names: Sequence[str] = REFERENCE_DEFINITIONS,
) -> SimpleNamespace:
    """Execute selected definitions without executing reference module imports."""

    reference_path = locate_reference(path)
    tree = ast.parse(reference_path.read_text(encoding="utf-8"), filename=str(reference_path))
    definitions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }
    missing = sorted(set(names) - definitions.keys())
    if missing:
        raise LookupError(f"Reference definitions missing: {missing!r}")

    future = ast.ImportFrom(
        module="__future__",
        names=[ast.alias(name="annotations")],
        level=0,
    )
    selected = ast.Module(
        body=[future, *(definitions[name] for name in names)],
        type_ignores=[],
    )
    ast.fix_missing_locations(selected)
    namespace: dict[str, Any] = {
        "__name__": __name__,
        "dataclass": dataclass,
        "math": math,
        "np": np,
        "re": re,
        "torch": _TorchShim(),
    }
    exec(compile(selected, str(reference_path), "exec"), namespace)
    return SimpleNamespace(**{name: namespace[name] for name in names})


def normalize_word(word: str) -> str:
    normalized = word.strip().lower()
    return re.sub(r"^[^\w']+|[^\w']+$", "", normalized)


def strip_arpabet_stress(phone: str) -> str:
    if len(phone) >= 2 and phone[-1] in {"0", "1", "2"}:
        return phone[:-1]
    return phone


def choose_greedy_pronunciations(
    words: Sequence[str],
    lexicon: Mapping[str, Sequence[Sequence[str]]],
    phone_to_id: Mapping[str, int],
    inter_word_token: str | None,
) -> GreedyPronunciation:
    phones: list[str] = []
    chosen_prons: list[list[str]] = []
    choice_indices: list[int] = []
    for word_index, word in enumerate(words):
        if word not in lexicon:
            raise KeyError(f"OOV word not found in lexicon at word_index={word_index}: {word!r}")
        pronunciations = lexicon[word]
        if not pronunciations:
            raise RuntimeError(f"Word has no pronunciations at word_index={word_index}: {word!r}")
        pronunciation = list(pronunciations[0])
        if not pronunciation:
            raise RuntimeError(f"Empty greedy pronunciation at word_index={word_index}: {word!r}")
        for phone in pronunciation:
            if phone not in phone_to_id:
                raise KeyError(
                    f"Phone not in vocab: phone={phone!r}, word={word!r}, "
                    f"word_index={word_index}, pron={pronunciation!r}"
                )
        if word_index > 0 and inter_word_token is not None:
            if inter_word_token not in phone_to_id:
                raise KeyError(f"inter_word_token not in vocab: {inter_word_token!r}")
            phones.append(inter_word_token)
        phones.extend(pronunciation)
        chosen_prons.append(pronunciation)
        choice_indices.append(0)
    if not phones:
        raise RuntimeError("Greedy pronunciation produced empty phone sequence.")
    return GreedyPronunciation(phones, chosen_prons, choice_indices)


def build_trellis(
    log_probs: NDArray[np.floating[Any]],
    targets: Sequence[int],
    blank_id: int,
) -> FloatArray:
    scores = np.asarray(log_probs, dtype=np.float64)
    if scores.ndim != 2:
        raise ValueError(f"log_probs must be two-dimensional, got {scores.shape!r}")
    time_steps, vocab_size = scores.shape
    if not targets:
        raise ValueError("Empty target token sequence.")
    if not 0 <= blank_id < vocab_size:
        raise ValueError(f"blank_id out of range: blank_id={blank_id}, V={vocab_size}")
    for target_index, token_id in enumerate(targets):
        if not 0 <= token_id < vocab_size:
            raise ValueError(
                f"target id out of range at target_index={target_index}: {token_id}, V={vocab_size}"
            )

    trellis = np.full((time_steps + 1, len(targets) + 1), -np.inf, dtype=np.float64)
    trellis[0, 0] = 0.0
    trellis[1:, 0] = np.cumsum(scores[:, blank_id])
    target_ids = np.asarray(targets, dtype=np.int64)
    for time_index in range(1, time_steps + 1):
        stay = trellis[time_index - 1, 1:] + scores[time_index - 1, blank_id]
        emit = trellis[time_index - 1, :-1] + scores[time_index - 1, target_ids]
        trellis[time_index, 1:] = np.maximum(stay, emit)
    if not np.isfinite(np.max(trellis[:, len(targets)])):
        raise RuntimeError(
            "CTC trellis failed to consume all targets: "
            f"T={time_steps}, N={len(targets)}, blank_id={blank_id}"
        )
    return trellis


def backtrace(
    trellis: FloatArray,
    log_probs: NDArray[np.floating[Any]],
    targets: Sequence[int],
    blank_id: int,
) -> list[Point]:
    scores = np.asarray(log_probs, dtype=np.float64)
    time_index = int(np.argmax(trellis[:, len(targets)]))
    target_count = len(targets)
    target_index = target_count
    path: list[Point] = []
    while time_index > 0 and target_index > 0:
        stay = trellis[time_index - 1, target_index] + scores[time_index - 1, blank_id]
        emit = (
            trellis[time_index - 1, target_index - 1]
            + scores[time_index - 1, targets[target_index - 1]]
        )
        if emit > stay:
            path.append(Point(target_index - 1, time_index - 1))
            target_index -= 1
        time_index -= 1
    path.reverse()
    if target_index != 0:
        raise RuntimeError(
            "Backtrace did not consume all targets: "
            f"remaining_targets={target_index}, N={target_count}"
        )
    return path


def brute_force_final_column(
    log_probs: NDArray[np.floating[Any]],
    targets: Sequence[int],
    blank_id: int,
) -> FloatArray:
    """Enumerate every emit/stay path for every possible finish prefix."""

    scores = np.asarray(log_probs, dtype=np.float64)
    result = np.full(scores.shape[0] + 1, -np.inf, dtype=np.float64)
    for finish in range(len(targets), scores.shape[0] + 1):
        for emission_frames in itertools.combinations(range(finish), len(targets)):
            emission_by_frame = {
                frame: target_index for target_index, frame in enumerate(emission_frames)
            }
            path_score = 0.0
            for frame in range(finish):
                target_index = emission_by_frame.get(frame)
                token_id = blank_id if target_index is None else targets[target_index]
                path_score += float(scores[frame, token_id])
            result[finish] = max(result[finish], path_score)
    return result


def points_to_segments(points: Sequence[Point], target_labels: Sequence[str]) -> list[Segment]:
    if len(points) != len(target_labels):
        raise ValueError(f"len(points)={len(points)} != len(target_labels)={len(target_labels)}")
    if not points:
        raise ValueError("Cannot convert empty points to segments.")
    segments: list[Segment] = []
    for index, point in enumerate(points):
        start = point.time_index
        end = points[index + 1].time_index if index + 1 < len(points) else start + 1
        if end <= start:
            raise RuntimeError(
                f"Non-positive token segment at token_index={point.token_index}: "
                f"start={start}, end={end}"
            )
        segments.append(Segment(target_labels[point.token_index], start, end))
    return segments


def emission_confidence_log(
    segment: Segment,
    label_to_id: Mapping[str, int],
    log_probs: NDArray[np.floating[Any]],
    emission_frame: int | None,
) -> float:
    if segment.label not in label_to_id:
        raise KeyError(f"Segment label not in vocab for confidence: {segment.label!r}")
    scores = np.asarray(log_probs, dtype=np.float64)
    frame = (
        (segment.start_frame + segment.end_frame) // 2 if emission_frame is None else emission_frame
    )
    frame = max(0, min(int(frame), scores.shape[0] - 1))
    return float(scores[frame, label_to_id[segment.label]])


def word_confidence_log(phone_confidence_logs: Sequence[float]) -> float:
    if not phone_confidence_logs:
        raise ValueError("phone_confidence_logs must not be empty")
    return float(sum(phone_confidence_logs) / len(phone_confidence_logs))


def make_word_anchors_from_emissions(
    word_spans: Sequence[WordSpan],
    token_ranges: Sequence[tuple[int, int]],
    token_emission_frames: Sequence[int],
    *,
    seconds_per_frame: float,
    anchor_pad_seconds: float,
    audio_duration_seconds: float,
) -> list[WordAnchor]:
    if len(word_spans) != len(token_ranges):
        raise ValueError(
            f"len(word_spans)={len(word_spans)} != len(token_ranges)={len(token_ranges)}"
        )
    anchors: list[WordAnchor] = []
    for word_span, (token_start, token_end) in zip(
        word_spans,
        token_ranges,
        strict=True,
    ):
        frames = token_emission_frames[token_start:token_end]
        if not frames:
            raise RuntimeError(
                f"No emission frames for word_index={word_span.word_index}, word={word_span.word!r}"
            )
        emit_start_frame = min(frames)
        emit_end_frame = max(frames)
        emit_start_seconds = emit_start_frame * seconds_per_frame
        emit_end_seconds = emit_end_frame * seconds_per_frame
        anchor_start = max(0.0, emit_start_seconds - anchor_pad_seconds)
        anchor_end = min(
            audio_duration_seconds,
            emit_end_seconds + anchor_pad_seconds,
        )
        if anchor_end <= anchor_start:
            raise RuntimeError(f"Invalid word anchor for {word_span.word!r}")
        anchors.append(
            WordAnchor(
                word_span.word_index,
                word_span.word,
                emit_start_frame,
                emit_end_frame,
                emit_start_seconds,
                emit_end_seconds,
                anchor_start,
                anchor_end,
            )
        )
    return anchors


def merge_word_anchors_into_chunks(
    word_anchors: Sequence[WordAnchor],
    *,
    anchor_merge_gap_seconds: float,
) -> list[Chunk]:
    if not word_anchors:
        raise ValueError("merge_word_anchors_into_chunks received empty word_anchors")
    anchors = sorted(
        word_anchors,
        key=lambda anchor: (
            anchor.anchor_start_s,
            anchor.anchor_end_s,
            anchor.word_index,
        ),
    )
    chunks: list[Chunk] = []
    current_start = anchors[0].anchor_start_s
    current_end = anchors[0].anchor_end_s
    current_words = [anchors[0].word]
    current_indices = [anchors[0].word_index]
    for anchor in anchors[1:]:
        gap = anchor.anchor_start_s - current_end
        if gap < anchor_merge_gap_seconds:
            current_end = max(current_end, anchor.anchor_end_s)
            current_words.append(anchor.word)
            current_indices.append(anchor.word_index)
            continue
        chunks.append(Chunk(current_start, current_end, current_words, current_indices))
        current_start = anchor.anchor_start_s
        current_end = anchor.anchor_end_s
        current_words = [anchor.word]
        current_indices = [anchor.word_index]
    chunks.append(Chunk(current_start, current_end, current_words, current_indices))
    return chunks


def round_chunks_to_legacy_grid(
    raw_chunks: Sequence[Chunk],
    *,
    utterance_id: str,
    words: Sequence[str],
    num_samples: int,
    sample_rate: int,
) -> list[RuntimeChunk]:
    if not raw_chunks:
        raise RuntimeError("Chunker returned no chunks.")
    audio_duration = num_samples / float(sample_rate)
    chunks: list[RuntimeChunk] = []
    for chunk_number, raw_chunk in enumerate(raw_chunks, start=1):
        start_ms = round(raw_chunk.start * 1000.0)
        end_ms = round(raw_chunk.end * 1000.0)
        if end_ms <= start_ms:
            raise RuntimeError("Invalid ms-rounded chunk span")
        if raw_chunk.end > audio_duration + 1e-6:
            raise RuntimeError("Raw chunk exceeds audio duration before ms rounding")
        if end_ms / 1000.0 > audio_duration:
            overflow = end_ms / 1000.0 - audio_duration
            if overflow > 0.001:
                raise RuntimeError("Chunk exceeds audio duration after ms rounding")
            end_ms = math.floor(audio_duration * 1000.0)
            if end_ms <= start_ms:
                raise RuntimeError("Invalid tail clamp after ms rounding")
        start_sample = round(start_ms / 1000.0 * sample_rate)
        end_sample = round(end_ms / 1000.0 * sample_rate)
        if start_sample < 0 or end_sample > num_samples or end_sample <= start_sample:
            raise RuntimeError("Invalid chunk sample span")
        chunks.append(
            RuntimeChunk(
                f"{utterance_id}.chunk{chunk_number:03d}",
                start_ms,
                end_ms,
                start_sample,
                end_sample,
                list(raw_chunk.words),
                list(raw_chunk.word_indices),
            )
        )

    previous_end_sample = 0
    concatenated_words: list[str] = []
    concatenated_indices: list[int] = []
    for chunk in chunks:
        if chunk.start_sample < previous_end_sample:
            raise RuntimeError("Overlapping chunks after legacy rounding")
        if sorted(chunk.word_indices) != chunk.word_indices:
            raise RuntimeError("Non-monotonic word indices in chunk")
        previous_end_sample = chunk.end_sample
        concatenated_words.extend(chunk.words)
        concatenated_indices.extend(chunk.word_indices)
    if concatenated_words != list(words):
        raise RuntimeError("Token consistency check failed: input_transcript_vs_rounded_chunks")
    expected_indices = list(range(len(words)))
    if concatenated_indices != expected_indices:
        raise RuntimeError(
            "Chunk word-index coverage mismatch: "
            f"expected={expected_indices!r}, actual={concatenated_indices!r}"
        )
    return chunks
