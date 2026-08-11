"""NumPy implementation of the reference Stage 1 alignment semantics.

This module is intentionally independent of model frameworks and the frozen
reference snapshot.  It preserves the characterized recurrence, backtrace,
confidence, anchor, merge, and millisecond-rounding behavior.  In particular,
it does not add the unresolved repeated-target CTC blank constraint.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from ..errors import ResourceLimitError

EPS = 1e-6

FloatArray = NDArray[np.floating[Any]]
LexiconMapping = Mapping[str, Sequence[Sequence[str]]]


@dataclass(frozen=True, slots=True)
class Point:
    token_index: int
    time_index: int


@dataclass(frozen=True, slots=True)
class Segment:
    label: str
    start_frame: int
    end_frame: int

    @property
    def dur_frames(self) -> int:
        return self.end_frame - self.start_frame


@dataclass(frozen=True, slots=True)
class SegmentWithConf:
    label: str
    start_frame: int
    end_frame: int
    conf_log: float

    @property
    def conf_prob(self) -> float:
        return math.exp(self.conf_log) if math.isfinite(self.conf_log) else -1.0


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

    @property
    def dur_s(self) -> float:
        return self.end_s - self.start_s

    @property
    def conf_prob(self) -> float:
        return math.exp(self.conf_log) if math.isfinite(self.conf_log) else -1.0


@dataclass(frozen=True, slots=True)
class Chunk:
    start: float
    end: float
    words: list[str]
    word_indices: list[int]

    @property
    def dur(self) -> float:
        return self.end - self.start


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

    @property
    def anchor_dur_s(self) -> float:
        return self.anchor_end_s - self.anchor_start_s


@dataclass(frozen=True, slots=True)
class GreedyPronResult:
    phones: list[str]
    chosen_prons: list[list[str]]
    pron_choice_idxs: list[int]


@dataclass(frozen=True, slots=True)
class RuntimeChunk:
    chunk_id: str
    start_ms: int
    end_ms: int
    start_sample: int
    end_sample: int
    words: list[str]
    word_indices: list[int]

    @property
    def start_s(self) -> float:
        return self.start_ms / 1000.0

    @property
    def end_s(self) -> float:
        return self.end_ms / 1000.0

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass(frozen=True, slots=True)
class TrellisResourceEstimate:
    frames: int
    targets: int
    cells: int
    bytes: int


class _LexiconContainer(Protocol):
    lex: LexiconMapping


def normalize_word(word: str) -> str:
    """Lowercase and strip non-word punctuation only at token edges."""

    normalized = word.strip().lower()
    return re.sub(r"^[^\w']+|[^\w']+$", "", normalized)


def strip_arpabet_stress(phone: str) -> str:
    """Remove only a terminal ARPAbet stress digit 0, 1, or 2."""

    if len(phone) >= 2 and phone[-1] in {"0", "1", "2"}:
        return phone[:-1]
    return phone


def build_chunk_lexicon(
    raw_lexicon: LexiconMapping | _LexiconContainer,
) -> dict[str, list[list[str]]]:
    """Create the stress-stripped Stage 1 lexicon without changing order."""

    source = raw_lexicon if isinstance(raw_lexicon, Mapping) else raw_lexicon.lex
    return {
        word: [
            [strip_arpabet_stress(phone) for phone in pronunciation]
            for pronunciation in pronunciations
        ]
        for word, pronunciations in source.items()
    }


def choose_greedy_pronunciations(
    words: Sequence[str],
    lex: LexiconMapping,
    phone_to_id: Mapping[str, int],
    inter_word_token: str | None,
) -> GreedyPronResult:
    """Select the first pronunciation of each word, matching the reference."""

    phones: list[str] = []
    chosen_prons: list[list[str]] = []
    choice_indices: list[int] = []
    for word_index, word in enumerate(words):
        if word not in lex:
            raise KeyError(f"OOV word not found in lexicon at word_index={word_index}: {word!r}")
        pronunciations = lex[word]
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
    return GreedyPronResult(
        phones=phones,
        chosen_prons=chosen_prons,
        pron_choice_idxs=choice_indices,
    )


def estimate_trellis_resources(
    frames: int,
    targets: int,
    *,
    itemsize: int = 8,
) -> TrellisResourceEstimate:
    """Estimate the dense ``(T + 1) x (N + 1)`` trellis allocation."""

    for name, value in (("frames", frames), ("targets", targets), ("itemsize", itemsize)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if frames < 0:
        raise ValueError(f"frames must be non-negative, got {frames}")
    if targets < 0:
        raise ValueError(f"targets must be non-negative, got {targets}")
    if itemsize <= 0:
        raise ValueError(f"itemsize must be positive, got {itemsize}")
    cells = (frames + 1) * (targets + 1)
    return TrellisResourceEstimate(
        frames=frames,
        targets=targets,
        cells=cells,
        bytes=cells * itemsize,
    )


def _validate_log_probs(log_probs: FloatArray) -> tuple[int, int]:
    if not isinstance(log_probs, np.ndarray):
        raise TypeError(f"log_probs must be a NumPy array, got {type(log_probs).__name__}")
    if log_probs.ndim != 2:
        raise ValueError(f"log_probs must be two-dimensional, got shape={log_probs.shape}")
    if log_probs.shape[0] <= 0 or log_probs.shape[1] <= 0:
        raise ValueError(f"log_probs dimensions must be positive, got shape={log_probs.shape}")
    if not np.issubdtype(log_probs.dtype, np.floating):
        raise TypeError(f"log_probs dtype must be floating, got {log_probs.dtype}")
    if not bool(np.isfinite(log_probs).all()):
        raise ValueError("log_probs contains NaN or infinity")
    return int(log_probs.shape[0]), int(log_probs.shape[1])


def _validate_token_ids(
    targets: Sequence[int],
    blank_id: int,
    vocab_size: int,
) -> list[int]:
    if isinstance(blank_id, bool) or not isinstance(blank_id, (int, np.integer)):
        raise TypeError(f"blank_id must be an integer, got {type(blank_id).__name__}")
    normalized_blank = int(blank_id)
    if normalized_blank < 0 or normalized_blank >= vocab_size:
        raise ValueError(f"blank_id out of range: blank_id={normalized_blank}, V={vocab_size}")
    normalized_targets: list[int] = []
    for target_index, token_id in enumerate(targets):
        if isinstance(token_id, bool) or not isinstance(token_id, (int, np.integer)):
            raise TypeError(
                "target id must be an integer at "
                f"target_index={target_index}, got {type(token_id).__name__}"
            )
        normalized_id = int(token_id)
        if normalized_id < 0 or normalized_id >= vocab_size:
            raise ValueError(
                "target id out of range at "
                f"target_index={target_index}: {normalized_id}, V={vocab_size}"
            )
        normalized_targets.append(normalized_id)
    return normalized_targets


def build_trellis(
    log_probs: FloatArray,
    targets: Sequence[int],
    blank_id: int,
    *,
    max_trellis_cells: int | None = None,
) -> FloatArray:
    """Build the dense reference recurrence without repeated-target correction."""

    frame_count, vocab_size = _validate_log_probs(log_probs)
    if len(targets) <= 0:
        raise ValueError("Empty target token sequence.")
    target_ids = _validate_token_ids(targets, blank_id, vocab_size)
    normalized_blank = int(blank_id)

    estimate = estimate_trellis_resources(
        frame_count,
        len(target_ids),
        itemsize=int(log_probs.dtype.itemsize),
    )
    if max_trellis_cells is not None:
        if isinstance(max_trellis_cells, bool) or not isinstance(max_trellis_cells, int):
            raise TypeError("max_trellis_cells must be an integer or None")
        if max_trellis_cells <= 0:
            raise ValueError("max_trellis_cells must be positive when provided")
        if estimate.cells > max_trellis_cells:
            raise ResourceLimitError(
                "CTC trellis cell limit exceeded before allocation",
                context={
                    "frames": estimate.frames,
                    "targets": estimate.targets,
                    "cells": estimate.cells,
                    "limit": max_trellis_cells,
                },
            )

    trellis = np.full(
        (frame_count + 1, len(target_ids) + 1),
        -np.inf,
        dtype=log_probs.dtype,
    )
    trellis[0, 0] = 0.0
    trellis[1:, 0] = np.cumsum(log_probs[:, normalized_blank])
    target_array = np.asarray(target_ids, dtype=np.intp)
    for time_index in range(1, frame_count + 1):
        scores = log_probs[time_index - 1]
        stay = trellis[time_index - 1, 1:] + scores[normalized_blank]
        emit = trellis[time_index - 1, :-1] + scores[target_array]
        trellis[time_index, 1:] = np.maximum(stay, emit)
    if not bool(np.isfinite(np.max(trellis[:, len(target_ids)]))):
        raise RuntimeError(
            "CTC trellis failed to consume all targets: "
            f"T={frame_count}, N={len(target_ids)}, blank_id={normalized_blank}"
        )
    return trellis


def backtrace(
    trellis: FloatArray,
    log_probs: FloatArray,
    targets: Sequence[int],
    blank_id: int,
) -> list[Point]:
    """Backtrace from the earliest maximum finish; ties prefer staying blank."""

    frame_count, vocab_size = _validate_log_probs(log_probs)
    target_ids = _validate_token_ids(targets, blank_id, vocab_size)
    if not isinstance(trellis, np.ndarray):
        raise TypeError(f"trellis must be a NumPy array, got {type(trellis).__name__}")
    expected_shape = (frame_count + 1, len(target_ids) + 1)
    if trellis.ndim != 2 or trellis.shape != expected_shape:
        raise ValueError(
            f"trellis shape mismatch: expected={expected_shape}, actual={trellis.shape}"
        )
    if not np.issubdtype(trellis.dtype, np.floating):
        raise TypeError(f"trellis dtype must be floating, got {trellis.dtype}")
    if bool(np.isnan(trellis).any()) or bool(np.isposinf(trellis).any()):
        raise ValueError("trellis contains NaN or positive infinity")

    target_count = len(target_ids)
    target_index = target_count
    time_index = int(np.argmax(trellis[:, target_index]))
    path: list[Point] = []
    normalized_blank = int(blank_id)
    while time_index > 0 and target_index > 0:
        scores = log_probs[time_index - 1]
        score_stay = trellis[time_index - 1, target_index] + scores[normalized_blank]
        score_emit = (
            trellis[time_index - 1, target_index - 1] + scores[target_ids[target_index - 1]]
        )
        if score_emit > score_stay:
            path.append(Point(token_index=target_index - 1, time_index=time_index - 1))
            target_index -= 1
        time_index -= 1
    path.reverse()
    if target_index != 0:
        raise RuntimeError(
            "Backtrace did not consume all targets: "
            f"remaining_targets={target_index}, N={target_count}, T={frame_count}"
        )
    if len(path) != target_count:
        raise RuntimeError(f"Backtrace length mismatch: len(path)={len(path)} != N={target_count}")
    return path


def points_to_segments(
    points: Sequence[Point],
    target_labels: Sequence[str],
) -> list[Segment]:
    if len(points) != len(target_labels):
        raise ValueError(f"len(points)={len(points)} != len(target_labels)={len(target_labels)}")
    if not points:
        raise ValueError("Cannot convert empty points to segments.")
    segments: list[Segment] = []
    for index, point in enumerate(points):
        if point.token_index < 0 or point.token_index >= len(target_labels):
            raise ValueError(
                f"Point token_index out of range: {point.token_index}, labels={len(target_labels)}"
            )
        start = int(point.time_index)
        end = (
            int(points[index + 1].time_index)
            if index + 1 < len(points)
            else int(point.time_index) + 1
        )
        if end <= start:
            raise RuntimeError(
                f"Non-positive token segment at token_index={point.token_index}: "
                f"start={start}, end={end}"
            )
        segments.append(
            Segment(
                label=target_labels[point.token_index],
                start_frame=start,
                end_frame=end,
            )
        )
    return segments


def compute_segment_confidence(
    seg: Segment,
    label_to_id: Mapping[str, int],
    log_probs: FloatArray,
    emission_frame: int | None,
    mode: str,
) -> float:
    _, vocab_size = _validate_log_probs(log_probs)
    if seg.label not in label_to_id:
        raise KeyError(f"Segment label not in vocab for confidence: {seg.label!r}")
    token_id = label_to_id[seg.label]
    if isinstance(token_id, bool) or not isinstance(token_id, (int, np.integer)):
        raise TypeError(f"Token ID for {seg.label!r} must be an integer")
    normalized_id = int(token_id)
    if normalized_id < 0 or normalized_id >= vocab_size:
        raise ValueError(
            f"Token ID for {seg.label!r} out of range: {normalized_id}, V={vocab_size}"
        )
    frame_count = int(log_probs.shape[0])
    if mode == "emission":
        frame = emission_frame
        if frame is None:
            frame = (seg.start_frame + seg.end_frame) // 2
        frame = max(0, min(int(frame), frame_count - 1))
        return float(log_probs[frame, normalized_id])
    if mode == "avg_frame":
        start = max(0, int(seg.start_frame))
        end = min(int(seg.end_frame), frame_count)
        if end <= start:
            raise RuntimeError(f"Invalid segment for avg_frame confidence: {seg}")
        return float(np.mean(log_probs[start:end, normalized_id]))
    raise ValueError(f"Unsupported confidence_mode={mode!r}")


def attach_phone_confidence_from_points(
    phone_token_segs: Sequence[Segment],
    points: Sequence[Point],
    target_labels: Sequence[str],
    phone_to_id: Mapping[str, int],
    log_probs: FloatArray,
    mode: str,
) -> list[SegmentWithConf]:
    if len(phone_token_segs) != len(target_labels):
        raise ValueError(
            "phone_token_segs length must equal target_labels length: "
            f"{len(phone_token_segs)} != {len(target_labels)}"
        )
    emission_frames: list[int | None] = [None] * len(target_labels)
    for point in points:
        if point.token_index < 0 or point.token_index >= len(target_labels):
            raise RuntimeError(
                "Point token_index out of range while attaching confidence: "
                f"token_index={point.token_index}, targets={len(target_labels)}"
            )
        emission_frames[point.token_index] = point.time_index
    output: list[SegmentWithConf] = []
    for index, segment in enumerate(phone_token_segs):
        if segment.label != target_labels[index]:
            raise RuntimeError(
                f"Phone token segment label mismatch at token_index={index}: "
                f"seg={segment.label!r}, target={target_labels[index]!r}"
            )
        confidence = compute_segment_confidence(
            seg=segment,
            label_to_id=phone_to_id,
            log_probs=log_probs,
            emission_frame=emission_frames[index],
            mode=mode,
        )
        output.append(
            SegmentWithConf(
                segment.label,
                segment.start_frame,
                segment.end_frame,
                confidence,
            )
        )
    return output


def phones_to_word_segments_by_offsets(
    phone_token_segs: Sequence[Segment],
    words: Sequence[str],
    prons_per_word: Sequence[Sequence[str]],
    inter_word_token: str | None,
) -> list[Segment]:
    if len(words) != len(prons_per_word):
        raise ValueError(f"len(words)={len(words)} != len(prons_per_word)={len(prons_per_word)}")
    phone_index = 0
    word_segments: list[Segment] = []
    for word_index, (word, pronunciation) in enumerate(zip(words, prons_per_word, strict=True)):
        if word_index > 0 and inter_word_token is not None:
            if phone_index >= len(phone_token_segs):
                raise RuntimeError(
                    "Ran out of phone token segments before inter_word_token "
                    f"at word_index={word_index}"
                )
            if phone_token_segs[phone_index].label != inter_word_token:
                raise RuntimeError(
                    f"Expected inter_word_token={inter_word_token!r} at "
                    f"phone_token_index={phone_index}, "
                    f"got {phone_token_segs[phone_index].label!r}"
                )
            phone_index += 1
        phone_count = len(pronunciation)
        if phone_count <= 0:
            raise RuntimeError(f"Empty pronunciation for word_index={word_index}, word={word!r}")
        if phone_index + phone_count > len(phone_token_segs):
            raise RuntimeError(
                f"Ran out of phone token segments for word_index={word_index}, "
                f"word={word!r}: need={phone_count}, "
                f"remaining={len(phone_token_segs) - phone_index}"
            )
        actual_phones = [
            segment.label for segment in phone_token_segs[phone_index : phone_index + phone_count]
        ]
        expected_phones = list(pronunciation)
        if actual_phones != expected_phones:
            raise RuntimeError(
                "Phone-token/pronunciation mismatch at "
                f"word_index={word_index}, word={word!r}: "
                f"got={actual_phones!r}, expected={expected_phones!r}"
            )
        start_frame = phone_token_segs[phone_index].start_frame
        end_frame = phone_token_segs[phone_index + phone_count - 1].end_frame
        if end_frame <= start_frame:
            raise RuntimeError(
                f"Invalid word segment frame span at word_index={word_index}, "
                f"word={word!r}: start={start_frame}, end={end_frame}"
            )
        word_segments.append(Segment(word, start_frame, end_frame))
        phone_index += phone_count
    if phone_index != len(phone_token_segs):
        remaining = [segment.label for segment in phone_token_segs[phone_index : phone_index + 20]]
        raise RuntimeError(
            "Unconsumed phone token segments after word reconstruction: "
            f"consumed={phone_index}, total={len(phone_token_segs)}, "
            f"remaining_sample={remaining!r}"
        )
    return word_segments


def word_segments_with_confidence(
    word_segs: Sequence[Segment],
    phone_segs_conf: Sequence[SegmentWithConf],
) -> list[SegmentWithConf]:
    output: list[SegmentWithConf] = []
    phone_index = 0
    for word_segment in word_segs:
        confidence_values: list[float] = []
        while (
            phone_index < len(phone_segs_conf)
            and phone_segs_conf[phone_index].end_frame <= word_segment.start_frame
        ):
            phone_index += 1
        next_phone_index = phone_index
        while (
            next_phone_index < len(phone_segs_conf)
            and phone_segs_conf[next_phone_index].start_frame < word_segment.end_frame
        ):
            confidence_values.append(phone_segs_conf[next_phone_index].conf_log)
            next_phone_index += 1
        if not confidence_values:
            raise RuntimeError(f"No overlapping phone confidence for word segment: {word_segment}")
        output.append(
            SegmentWithConf(
                word_segment.label,
                word_segment.start_frame,
                word_segment.end_frame,
                float(sum(confidence_values) / len(confidence_values)),
            )
        )
        phone_index = next_phone_index
    return output


def emission_frames_by_token_index(
    points: Sequence[Point],
    num_tokens: int,
) -> list[int]:
    if num_tokens <= 0:
        raise ValueError(f"num_tokens must be positive, got {num_tokens}")
    frames: list[int | None] = [None] * num_tokens
    for point in points:
        if point.token_index < 0 or point.token_index >= num_tokens:
            raise RuntimeError(
                f"Point token_index out of range: token_index={point.token_index}, "
                f"num_tokens={num_tokens}"
            )
        if frames[point.token_index] is not None:
            raise RuntimeError(f"Duplicate emission point for token_index={point.token_index}")
        frames[point.token_index] = int(point.time_index)
    missing = [index for index, value in enumerate(frames) if value is None]
    if missing:
        raise RuntimeError(f"Missing emission frames for token indices: {missing[:20]}")
    return [int(value) for value in frames if value is not None]


def word_phone_token_ranges(
    phone_token_segs: Sequence[Segment],
    words: Sequence[str],
    prons_per_word: Sequence[Sequence[str]],
    inter_word_token: str | None,
) -> list[tuple[int, int]]:
    if len(words) != len(prons_per_word):
        raise ValueError(f"len(words)={len(words)} != len(prons_per_word)={len(prons_per_word)}")
    token_index = 0
    ranges: list[tuple[int, int]] = []
    for word_index, (word, pronunciation) in enumerate(zip(words, prons_per_word, strict=True)):
        if word_index > 0 and inter_word_token is not None:
            if token_index >= len(phone_token_segs):
                raise RuntimeError(
                    "Ran out of phone token segments before inter_word_token "
                    f"at word_index={word_index}"
                )
            if phone_token_segs[token_index].label != inter_word_token:
                raise RuntimeError(
                    f"Expected inter_word_token={inter_word_token!r} at "
                    f"phone_token_index={token_index}, "
                    f"got {phone_token_segs[token_index].label!r}"
                )
            token_index += 1
        phone_count = len(pronunciation)
        if phone_count <= 0:
            raise RuntimeError(f"Empty pronunciation for word_index={word_index}, word={word!r}")
        if token_index + phone_count > len(phone_token_segs):
            raise RuntimeError(
                f"Ran out of phone token segments for word_index={word_index}, "
                f"word={word!r}: need={phone_count}, "
                f"remaining={len(phone_token_segs) - token_index}"
            )
        actual_phones = [
            segment.label for segment in phone_token_segs[token_index : token_index + phone_count]
        ]
        expected_phones = list(pronunciation)
        if actual_phones != expected_phones:
            raise RuntimeError(
                "Phone-token/pronunciation mismatch at "
                f"word_index={word_index}, word={word!r}: "
                f"got={actual_phones!r}, expected={expected_phones!r}"
            )
        ranges.append((token_index, token_index + phone_count))
        token_index += phone_count
    if token_index != len(phone_token_segs):
        remaining = [segment.label for segment in phone_token_segs[token_index : token_index + 20]]
        raise RuntimeError(
            "Unconsumed phone token segments after word-token range construction: "
            f"consumed={token_index}, total={len(phone_token_segs)}, "
            f"remaining_sample={remaining!r}"
        )
    return ranges


def make_word_anchors_from_emissions(
    word_spans: Sequence[WordSpan],
    token_ranges: Sequence[tuple[int, int]],
    token_emission_frames: Sequence[int],
    *,
    spf: float,
    anchor_pad_s: float,
    audio_dur_s: float,
) -> list[WordAnchor]:
    if len(word_spans) != len(token_ranges):
        raise ValueError(
            f"len(word_spans)={len(word_spans)} != len(token_ranges)={len(token_ranges)}"
        )
    if spf <= 0 or not math.isfinite(spf):
        raise ValueError(f"spf must be positive finite, got {spf}")
    if anchor_pad_s < 0 or not math.isfinite(anchor_pad_s):
        raise ValueError(f"anchor_pad_s must be non-negative finite, got {anchor_pad_s}")
    if audio_dur_s <= 0 or not math.isfinite(audio_dur_s):
        raise ValueError(f"audio_dur_s must be positive finite, got {audio_dur_s}")
    anchors: list[WordAnchor] = []
    for word_span, (token_start, token_end) in zip(word_spans, token_ranges, strict=True):
        if token_end <= token_start:
            raise RuntimeError(
                f"Empty token range for word_index={word_span.word_index}, word={word_span.word!r}"
            )
        frames = token_emission_frames[token_start:token_end]
        if not frames:
            raise RuntimeError(
                f"No emission frames for word_index={word_span.word_index}, word={word_span.word!r}"
            )
        emit_start_frame = min(frames)
        emit_end_frame = max(frames)
        emit_start_s = float(emit_start_frame) * spf
        emit_end_s = float(emit_end_frame) * spf
        anchor_start_s = max(0.0, emit_start_s - anchor_pad_s)
        anchor_end_s = min(float(audio_dur_s), emit_end_s + anchor_pad_s)
        if anchor_end_s <= anchor_start_s:
            raise RuntimeError(
                f"Invalid word anchor for word_index={word_span.word_index}, "
                f"word={word_span.word!r}: emit=({emit_start_s}, {emit_end_s}), "
                f"anchor=({anchor_start_s}, {anchor_end_s})"
            )
        anchors.append(
            WordAnchor(
                word_index=word_span.word_index,
                word=word_span.word,
                emit_start_frame=emit_start_frame,
                emit_end_frame=emit_end_frame,
                emit_start_s=emit_start_s,
                emit_end_s=emit_end_s,
                anchor_start_s=anchor_start_s,
                anchor_end_s=anchor_end_s,
            )
        )
    return anchors


def merge_word_anchors_into_chunks(
    word_anchors: Sequence[WordAnchor],
    *,
    anchor_merge_gap_s: float,
) -> list[Chunk]:
    if not word_anchors:
        raise ValueError("merge_word_anchors_into_chunks received empty word_anchors")
    if anchor_merge_gap_s < 0 or not math.isfinite(anchor_merge_gap_s):
        raise ValueError(
            f"anchor_merge_gap_s must be non-negative finite, got {anchor_merge_gap_s}"
        )
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
        if gap < anchor_merge_gap_s:
            current_end = max(current_end, anchor.anchor_end_s)
            current_words.append(anchor.word)
            current_indices.append(anchor.word_index)
            continue
        chunks.append(
            Chunk(
                start=current_start,
                end=current_end,
                words=list(current_words),
                word_indices=list(current_indices),
            )
        )
        current_start = anchor.anchor_start_s
        current_end = anchor.anchor_end_s
        current_words = [anchor.word]
        current_indices = [anchor.word_index]
    chunks.append(
        Chunk(
            start=current_start,
            end=current_end,
            words=list(current_words),
            word_indices=list(current_indices),
        )
    )
    for chunk_index, chunk in enumerate(chunks):
        if chunk.end <= chunk.start:
            raise RuntimeError(f"Invalid merged anchor chunk at index={chunk_index}: {chunk}")
        if sorted(chunk.word_indices) != chunk.word_indices:
            raise RuntimeError(
                "Merged anchor chunk word_indices are not monotonic at "
                f"index={chunk_index}: {chunk}"
            )
    return chunks


def _first_mismatch(
    expected: Sequence[str],
    actual: Sequence[str],
) -> tuple[int, str | None, str | None]:
    shared_length = min(len(expected), len(actual))
    for index in range(shared_length):
        if expected[index] != actual[index]:
            return index, expected[index], actual[index]
    if len(expected) != len(actual):
        return (
            shared_length,
            expected[shared_length] if shared_length < len(expected) else None,
            actual[shared_length] if shared_length < len(actual) else None,
        )
    return -1, None, None


def _assert_tokens_equal(
    name: str,
    expected: Sequence[str],
    actual: Sequence[str],
) -> None:
    if list(expected) == list(actual):
        return
    position, expected_token, actual_token = _first_mismatch(expected, actual)
    left = max(0, position - 5)
    right = position + 6
    raise RuntimeError(
        f"Token consistency check failed: {name}\n"
        f"expected_len={len(expected)}, actual_len={len(actual)}, "
        f"mismatch_pos={position}\n"
        f"expected_token={expected_token!r}, actual_token={actual_token!r}\n"
        f"expected_context={list(expected[left:right])!r}\n"
        f"actual_context={list(actual[left:right])!r}"
    )


def round_chunks_to_legacy_grid(
    *,
    raw_chunks: Sequence[Chunk],
    utt_id: str,
    words: Sequence[str],
    num_samples: int,
    sample_rate: int,
) -> list[RuntimeChunk]:
    """Round to the reference millisecond grid and preserve full word coverage."""

    if not raw_chunks:
        raise RuntimeError("Chunker returned no chunks.")
    if isinstance(num_samples, bool) or not isinstance(num_samples, int):
        raise TypeError("num_samples must be an integer")
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int):
        raise TypeError("sample_rate must be an integer")
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}")
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")

    audio_duration_s = num_samples / float(sample_rate)
    chunks: list[RuntimeChunk] = []
    for chunk_index, raw_chunk in enumerate(raw_chunks, start=1):
        raw_start_s = float(raw_chunk.start)
        raw_end_s = float(raw_chunk.end)
        if not math.isfinite(raw_start_s) or not math.isfinite(raw_end_s):
            raise ValueError(
                "Chunk bounds must be finite: "
                f"chunk_index={chunk_index}, start={raw_start_s}, end={raw_end_s}"
            )

        start_ms = round(raw_start_s * 1000.0)
        end_ms = round(raw_end_s * 1000.0)
        if end_ms <= start_ms:
            raise RuntimeError(
                f"Invalid ms-rounded chunk span: chunk_index={chunk_index}, "
                f"raw_start_s={raw_start_s}, raw_end_s={raw_end_s}, "
                f"start_ms={start_ms}, end_ms={end_ms}"
            )

        if raw_end_s > audio_duration_s + EPS:
            raise RuntimeError(
                "Raw chunk exceeds audio duration before ms rounding: "
                f"chunk_index={chunk_index}, raw_end_s={raw_end_s:.9f}, "
                f"audio_duration_s={audio_duration_s:.9f}"
            )

        end_s = end_ms / 1000.0
        if end_s > audio_duration_s:
            overflow_s = end_s - audio_duration_s
            if overflow_s > 0.001:
                raise RuntimeError(
                    "Chunk exceeds audio duration after ms rounding: "
                    f"chunk_index={chunk_index}, end_s={end_s:.3f}, "
                    f"audio_duration_s={audio_duration_s:.9f}, "
                    f"overflow_s={overflow_s:.9f}"
                )
            max_end_ms = math.floor(audio_duration_s * 1000.0)
            if max_end_ms <= start_ms:
                raise RuntimeError(
                    f"Invalid tail clamp after ms rounding: chunk_index={chunk_index}, "
                    f"start_ms={start_ms}, max_end_ms={max_end_ms}"
                )
            end_ms = max_end_ms

        start_sample = round((start_ms / 1000.0) * sample_rate)
        end_sample = round((end_ms / 1000.0) * sample_rate)
        if start_sample < 0 or end_sample > num_samples or end_sample <= start_sample:
            raise RuntimeError(
                f"Invalid chunk sample span: chunk_index={chunk_index}, "
                f"start_sample={start_sample}, end_sample={end_sample}, "
                f"num_samples={num_samples}"
            )

        chunks.append(
            RuntimeChunk(
                chunk_id=f"{utt_id}.chunk{chunk_index:03d}",
                start_ms=start_ms,
                end_ms=end_ms,
                start_sample=start_sample,
                end_sample=end_sample,
                words=list(raw_chunk.words),
                word_indices=list(raw_chunk.word_indices),
            )
        )

    previous_end_sample = 0
    concatenated_words: list[str] = []
    concatenated_indices: list[int] = []
    for chunk_index, chunk in enumerate(chunks):
        if chunk.start_sample < previous_end_sample:
            raise RuntimeError(
                f"Overlapping chunks after legacy rounding: chunk_index={chunk_index}, "
                f"chunk_id={chunk.chunk_id}, start_sample={chunk.start_sample}, "
                f"previous_end_sample={previous_end_sample}"
            )
        if sorted(chunk.word_indices) != chunk.word_indices:
            raise RuntimeError(
                f"Non-monotonic word indices in chunk {chunk.chunk_id}: {chunk.word_indices}"
            )
        previous_end_sample = chunk.end_sample
        concatenated_words.extend(chunk.words)
        concatenated_indices.extend(chunk.word_indices)

    _assert_tokens_equal(
        "input_transcript_vs_rounded_chunks",
        words,
        concatenated_words,
    )
    expected_indices = list(range(len(words)))
    if concatenated_indices != expected_indices:
        raise RuntimeError(
            f"Chunk word-index coverage mismatch: expected={expected_indices!r}, "
            f"actual={concatenated_indices!r}"
        )
    return chunks


__all__ = [
    "Chunk",
    "GreedyPronResult",
    "Point",
    "RuntimeChunk",
    "Segment",
    "SegmentWithConf",
    "TrellisResourceEstimate",
    "WordAnchor",
    "WordSpan",
    "attach_phone_confidence_from_points",
    "backtrace",
    "build_chunk_lexicon",
    "build_trellis",
    "choose_greedy_pronunciations",
    "compute_segment_confidence",
    "emission_frames_by_token_index",
    "estimate_trellis_resources",
    "make_word_anchors_from_emissions",
    "merge_word_anchors_into_chunks",
    "normalize_word",
    "phones_to_word_segments_by_offsets",
    "points_to_segments",
    "round_chunks_to_legacy_grid",
    "strip_arpabet_stress",
    "word_phone_token_ranges",
    "word_segments_with_confidence",
]
