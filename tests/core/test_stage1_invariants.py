"""Independent invariants for the production Stage 1 NumPy core.

The exhaustive helpers in this module do not import or execute the frozen
reference implementation.  Production comparisons are added against the actual
``flexaligner.core.stage1`` interface once that module lands.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from flexaligner.core import stage1
from flexaligner.errors import ResourceLimitError

FloatArray = NDArray[np.floating[Any]]


def _score_emission_frames(
    log_probs: FloatArray,
    targets: Sequence[int],
    blank_id: int,
    *,
    finish: int,
    emission_frames: Sequence[int],
) -> float:
    """Score one simplified CTC stay/emit path ending at ``finish``."""

    emitted = {frame: target_index for target_index, frame in enumerate(emission_frames)}
    score = 0.0
    for frame in range(finish):
        target_index = emitted.get(frame)
        token_id = blank_id if target_index is None else targets[target_index]
        score += float(log_probs[frame, token_id])
    return score


def _has_required_repeat_separators(
    targets: Sequence[int],
    emission_frames: Sequence[int],
) -> bool:
    return all(
        targets[index] != targets[index - 1]
        or emission_frames[index] >= emission_frames[index - 1] + 2
        for index in range(1, len(targets))
    )


def _brute_force_trellis(
    log_probs: FloatArray,
    targets: Sequence[int],
    blank_id: int,
) -> FloatArray:
    """Enumerate every stay/emit sequence for each trellis state.

    Entry ``[time, consumed]`` is the maximum score among all ways of emitting
    exactly the first ``consumed`` targets within the first ``time`` frames.
    """

    time_steps = log_probs.shape[0]
    target_count = len(targets)
    result = np.full((time_steps + 1, target_count + 1), -np.inf, dtype=np.float64)
    result[0, 0] = 0.0
    for time in range(1, time_steps + 1):
        result[time, 0] = float(np.sum(log_probs[:time, blank_id], dtype=np.float64))
    for time in range(1, time_steps + 1):
        for consumed in range(1, min(time, target_count) + 1):
            best = -math.inf
            for emission_frames in itertools.combinations(range(time), consumed):
                if not _has_required_repeat_separators(targets[:consumed], emission_frames):
                    continue
                candidate = _score_emission_frames(
                    log_probs,
                    targets[:consumed],
                    blank_id,
                    finish=time,
                    emission_frames=emission_frames,
                )
                best = max(best, candidate)
            result[time, consumed] = best
    return result


def _path_score(
    log_probs: FloatArray,
    targets: Sequence[int],
    blank_id: int,
    emission_frames: Sequence[int],
    *,
    finish: int,
) -> float:
    assert len(emission_frames) == len(targets)
    assert list(emission_frames) == sorted(emission_frames)
    assert len(set(emission_frames)) == len(emission_frames)
    assert all(0 <= frame < finish <= log_probs.shape[0] for frame in emission_frames)
    return _score_emission_frames(
        log_probs,
        targets,
        blank_id,
        finish=finish,
        emission_frames=emission_frames,
    )


def test_independent_brute_force_helper_has_expected_small_case() -> None:
    scores = np.asarray(
        [
            [-1.0, 2.0, -3.0],
            [0.5, -2.0, 3.0],
            [1.0, -4.0, -5.0],
        ],
        dtype=np.float64,
    )

    actual = _brute_force_trellis(scores, [1, 2], blank_id=0)

    assert actual.shape == (4, 3)
    assert actual[0, 0] == 0.0
    assert np.isneginf(actual[0, 1:]).all()
    assert actual[1, 0] == -1.0
    assert actual[1, 1] == 2.0
    assert np.isneginf(actual[1, 2])
    assert actual[2, 2] == 5.0
    assert actual[3, 2] == 6.0


@pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["float32", "float64"])
def test_fixed_seed_small_trellises_match_exhaustive_paths(dtype: Any) -> None:
    rng = np.random.default_rng(20260811)
    for frame_count in range(1, 7):
        for target_count in range(1, min(frame_count, 3) + 1):
            scores = rng.normal(size=(frame_count, 4)).astype(dtype)
            targets = rng.integers(1, 4, size=target_count).tolist()

            required_frames = target_count + sum(
                current == previous for previous, current in itertools.pairwise(targets)
            )
            if required_frames > frame_count:
                with pytest.raises(RuntimeError, match="failed to consume all targets"):
                    stage1.build_trellis(scores, targets, blank_id=0)
                continue

            actual = stage1.build_trellis(scores, targets, blank_id=0)
            expected = _brute_force_trellis(scores, targets, blank_id=0)

            assert actual.shape == (frame_count + 1, target_count + 1)
            assert actual.dtype == scores.dtype
            assert not np.isnan(actual).any()
            assert not np.isposinf(actual).any()
            np.testing.assert_allclose(
                actual.astype(np.float64),
                expected,
                rtol=2e-6,
                atol=2e-6,
            )
            for time_index in range(frame_count + 1):
                for consumed in range(target_count + 1):
                    minimum_frames = consumed + sum(
                        current == previous
                        for previous, current in itertools.pairwise(targets[:consumed])
                    )
                    if minimum_frames <= time_index:
                        assert np.isfinite(actual[time_index, consumed])
                    else:
                        assert np.isneginf(actual[time_index, consumed])

            points = stage1.backtrace(actual, scores, targets, blank_id=0)
            emission_frames = [point.time_index for point in points]
            finish = int(np.argmax(actual[:, target_count]))
            assert [point.token_index for point in points] == list(range(target_count))
            assert emission_frames == sorted(set(emission_frames))
            assert _has_required_repeat_separators(targets, emission_frames)
            assert all(0 <= frame < finish <= frame_count for frame in emission_frames)
            assert _path_score(
                scores,
                targets,
                0,
                emission_frames,
                finish=finish,
            ) == pytest.approx(float(actual[finish, target_count]), abs=2e-6)

            labels = [f"P{index}" for index in range(target_count)]
            segments = stage1.points_to_segments(points, labels)
            assert [segment.label for segment in segments] == labels
            assert [segment.start_frame for segment in segments] == emission_frames
            assert all(segment.end_frame > segment.start_frame >= 0 for segment in segments)
            assert all(
                left.end_frame == right.start_frame for left, right in itertools.pairwise(segments)
            )
            phone_confidence = stage1.attach_phone_confidence_from_points(
                segments,
                points,
                labels,
                dict(zip(labels, targets, strict=True)),
                scores,
                mode="emission",
            )
            assert [segment.label for segment in phone_confidence] == labels
            assert all(math.isfinite(segment.conf_log) for segment in phone_confidence)
            assert all(math.isfinite(segment.conf_prob) for segment in phone_confidence)


def test_backtrace_stops_at_earliest_best_completion() -> None:
    scores = np.asarray(
        [
            [-5.0, 5.0, -5.0],
            [-5.0, -5.0, 5.0],
            [-10.0, -10.0, -10.0],
            [-10.0, -10.0, -10.0],
        ],
        dtype=np.float64,
    )

    trellis = stage1.build_trellis(scores, [1, 2], blank_id=0)
    points = stage1.backtrace(trellis, scores, [1, 2], blank_id=0)

    assert int(np.argmax(trellis[:, 2])) == 2
    assert [(point.token_index, point.time_index) for point in points] == [(0, 0), (1, 1)]


def test_backtrace_tie_prefers_stay_over_late_emit() -> None:
    scores = np.asarray(
        [
            [0.0, 0.0],
            [0.0, -1.0],
            [1.0, 1.0],
        ],
        dtype=np.float64,
    )

    trellis = stage1.build_trellis(scores, [1], blank_id=0)
    points = stage1.backtrace(trellis, scores, [1], blank_id=0)

    assert trellis[3, 1] == 1.0
    assert trellis[2, 1] + scores[2, 0] == trellis[2, 0] + scores[2, 1]
    assert points == [stage1.Point(token_index=0, time_index=0)]


@pytest.mark.parametrize(
    ("frames", "targets", "itemsize", "cells", "byte_count"),
    [
        (0, 0, 4, 1, 4),
        (100, 20, 4, 2_121, 8_484),
        (100, 20, 8, 2_121, 16_968),
        (1_000, 100, 4, 101_101, 404_404),
        (1_000, 100, 8, 101_101, 808_808),
        (10_000, 1_000, 4, 10_011_001, 40_044_004),
        (10_000, 1_000, 8, 10_011_001, 80_088_008),
        (100_000, 5_000, 4, 500_105_001, 2_000_420_004),
        (100_000, 5_000, 8, 500_105_001, 4_000_840_008),
    ],
)
def test_trellis_resource_estimate_is_exact(
    frames: int,
    targets: int,
    itemsize: int,
    cells: int,
    byte_count: int,
) -> None:
    estimate = stage1.estimate_trellis_resources(frames, targets, itemsize=itemsize)

    assert estimate == stage1.TrellisResourceEstimate(
        frames=frames,
        targets=targets,
        cells=cells,
        bytes=byte_count,
    )


@pytest.mark.parametrize(
    ("frames", "targets", "itemsize", "error"),
    [
        (True, 1, 8, TypeError),
        (1.5, 1, 8, TypeError),
        (1, False, 8, TypeError),
        (1, 2.5, 8, TypeError),
        (1, 1, True, TypeError),
        (1, 1, 4.0, TypeError),
        (-1, 1, 8, ValueError),
        (1, -1, 8, ValueError),
        (1, 1, 0, ValueError),
        (1, 1, -8, ValueError),
    ],
)
def test_trellis_resource_estimate_rejects_invalid_dimensions(
    frames: Any,
    targets: Any,
    itemsize: Any,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        stage1.estimate_trellis_resources(frames, targets, itemsize=itemsize)


def test_explicit_cell_limit_fails_before_trellis_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scores = np.zeros((3, 3), dtype=np.float64)
    full_calls = 0

    def forbidden_full(*args: object, **kwargs: object) -> NDArray[Any]:
        del args, kwargs
        nonlocal full_calls
        full_calls += 1
        raise AssertionError("np.full must not run after the resource guard fails")

    monkeypatch.setattr(stage1.np, "full", forbidden_full)

    with pytest.raises(ResourceLimitError) as error:
        stage1.build_trellis(scores, [1, 2], blank_id=0, max_trellis_cells=11)

    assert full_calls == 0
    assert error.value.code == "resource_limit_exceeded"
    assert dict(error.value.context) == {
        "frames": 3,
        "targets": 2,
        "cells": 12,
        "limit": 11,
    }


def test_exact_cell_limit_allows_allocation() -> None:
    scores = np.zeros((3, 3), dtype=np.float32)

    trellis = stage1.build_trellis(
        scores,
        [1, 2],
        blank_id=0,
        max_trellis_cells=12,
    )

    assert trellis.shape == (4, 3)
    assert trellis.dtype == np.float32


@pytest.mark.parametrize("limit", [True, 1.5, "12"])
def test_trellis_rejects_non_integer_explicit_limits(limit: Any) -> None:
    with pytest.raises(TypeError, match="max_trellis_cells"):
        stage1.build_trellis(
            np.zeros((2, 2), dtype=np.float64),
            [1],
            blank_id=0,
            max_trellis_cells=limit,
        )


@pytest.mark.parametrize("limit", [0, -1])
def test_trellis_rejects_non_positive_explicit_limits(limit: int) -> None:
    with pytest.raises(ValueError, match="max_trellis_cells"):
        stage1.build_trellis(
            np.zeros((2, 2), dtype=np.float64),
            [1],
            blank_id=0,
            max_trellis_cells=limit,
        )


@pytest.mark.parametrize(
    ("bad_scores", "error"),
    [
        ([[0.0, 0.0]], TypeError),
        (np.zeros(2, dtype=np.float64), ValueError),
        (np.zeros((1, 1, 1), dtype=np.float64), ValueError),
        (np.zeros((0, 2), dtype=np.float64), ValueError),
        (np.zeros((2, 0), dtype=np.float64), ValueError),
        (np.zeros((2, 2), dtype=np.int64), TypeError),
        (np.zeros((2, 2), dtype=np.complex128), TypeError),
        (np.asarray([[0.0, np.nan], [0.0, 0.0]]), ValueError),
        (np.asarray([[0.0, np.inf], [0.0, 0.0]]), ValueError),
        (np.asarray([[0.0, -np.inf], [0.0, 0.0]]), ValueError),
    ],
)
def test_trellis_rejects_invalid_score_arrays(bad_scores: Any, error: type[Exception]) -> None:
    with pytest.raises(error):
        stage1.build_trellis(bad_scores, [1], blank_id=0)


@pytest.mark.parametrize("blank_id", [True, 0.5, "0"])
def test_trellis_rejects_non_integer_blank_id(blank_id: Any) -> None:
    with pytest.raises(TypeError, match="blank_id"):
        stage1.build_trellis(np.zeros((2, 3), dtype=np.float64), [1], blank_id)


@pytest.mark.parametrize("blank_id", [-1, 3])
def test_trellis_rejects_out_of_range_blank_id(blank_id: int) -> None:
    with pytest.raises(ValueError, match="blank_id out of range"):
        stage1.build_trellis(np.zeros((2, 3), dtype=np.float64), [1], blank_id)


@pytest.mark.parametrize("target_id", [True, 1.5, "1"])
def test_trellis_rejects_non_integer_target_id(target_id: Any) -> None:
    with pytest.raises(TypeError, match="target id must be an integer"):
        stage1.build_trellis(np.zeros((2, 3), dtype=np.float64), [target_id], blank_id=0)


@pytest.mark.parametrize("target_id", [-1, 3])
def test_trellis_rejects_out_of_range_target_id(target_id: int) -> None:
    with pytest.raises(ValueError, match="target id out of range"):
        stage1.build_trellis(np.zeros((2, 3), dtype=np.float64), [target_id], blank_id=0)


def test_trellis_requires_blank_between_repeated_numpy_integer_targets() -> None:
    scores = np.asarray(
        [[-10.0, 0.0], [0.0, -10.0], [-10.0, 0.0]],
        dtype=np.float64,
    )

    trellis = stage1.build_trellis(
        scores,
        [np.int64(1), np.int64(1)],
        blank_id=np.int64(0),
    )
    points = stage1.backtrace(trellis, scores, [1, 1], blank_id=0)

    assert trellis.shape == (4, 3)
    assert [(point.token_index, point.time_index) for point in points] == [(0, 0), (1, 2)]


def test_trellis_rejects_adjacent_repeats_without_separator_frame() -> None:
    scores = np.asarray([[-10.0, 0.0], [-10.0, 0.0]], dtype=np.float64)

    with pytest.raises(RuntimeError, match="failed to consume all targets"):
        stage1.build_trellis(scores, [1, 1], blank_id=0)


def test_trellis_rejects_empty_or_unreachable_targets() -> None:
    scores = np.zeros((1, 3), dtype=np.float64)
    with pytest.raises(ValueError, match="Empty target"):
        stage1.build_trellis(scores, [], blank_id=0)
    with pytest.raises(RuntimeError, match="failed to consume all targets"):
        stage1.build_trellis(scores, [1, 2], blank_id=0)


def test_backtrace_rejects_invalid_trellis_shape_type_dtype_and_values() -> None:
    scores = np.zeros((3, 3), dtype=np.float64)
    valid = stage1.build_trellis(scores, [1, 2], blank_id=0)

    with pytest.raises(TypeError, match="trellis must be a NumPy array"):
        stage1.backtrace(valid.tolist(), scores, [1, 2], blank_id=0)
    with pytest.raises(ValueError, match="trellis shape mismatch"):
        stage1.backtrace(valid[:, :-1], scores, [1, 2], blank_id=0)
    with pytest.raises(TypeError, match="trellis dtype must be floating"):
        stage1.backtrace(np.zeros(valid.shape, dtype=np.int64), scores, [1, 2], blank_id=0)
    for invalid_value in (np.nan, np.inf):
        invalid = valid.copy()
        invalid[0, 0] = invalid_value
        with pytest.raises(ValueError, match="trellis contains"):
            stage1.backtrace(invalid, scores, [1, 2], blank_id=0)

    unreachable = valid.copy()
    unreachable[:, -1] = -np.inf
    with pytest.raises(RuntimeError, match="did not consume all targets"):
        stage1.backtrace(unreachable, scores, [1, 2], blank_id=0)


def test_points_to_segments_rejects_length_ids_and_non_monotonic_times() -> None:
    with pytest.raises(ValueError, match=r"len\(points\)"):
        stage1.points_to_segments([stage1.Point(0, 0)], ["A", "B"])
    with pytest.raises(ValueError, match="empty points"):
        stage1.points_to_segments([], [])
    for invalid_id in (-1, 2):
        with pytest.raises(ValueError, match="token_index out of range"):
            stage1.points_to_segments([stage1.Point(invalid_id, 0)], ["A"])
    with pytest.raises(RuntimeError, match="Non-positive token segment"):
        stage1.points_to_segments(
            [stage1.Point(0, 2), stage1.Point(1, 2)],
            ["A", "B"],
        )


def _chunk(
    start: float,
    end: float,
    words: Sequence[str],
    word_indices: Sequence[int],
) -> stage1.Chunk:
    return stage1.Chunk(
        start=start,
        end=end,
        words=list(words),
        word_indices=list(word_indices),
    )


def test_anchor_and_merge_outputs_are_finite_monotonic_and_well_shaped() -> None:
    spans = [
        stage1.WordSpan(0, "alpha", 0, 2, 0.0, 0.02, -0.2, ["A", "B"]),
        stage1.WordSpan(1, "beta", 5, 6, 0.05, 0.06, -0.3, ["C"]),
    ]

    anchors = stage1.make_word_anchors_from_emissions(
        spans,
        [(0, 2), (2, 3)],
        [0, 1, 5],
        spf=0.01,
        anchor_pad_s=0.005,
        audio_dur_s=0.1,
    )
    chunks = stage1.merge_word_anchors_into_chunks(anchors, anchor_merge_gap_s=0.01)

    assert [anchor.word_index for anchor in anchors] == [0, 1]
    assert [anchor.word for anchor in anchors] == ["alpha", "beta"]
    assert all(
        math.isfinite(value)
        for anchor in anchors
        for value in (
            anchor.emit_start_s,
            anchor.emit_end_s,
            anchor.anchor_start_s,
            anchor.anchor_end_s,
            anchor.anchor_dur_s,
        )
    )
    assert all(anchor.emit_end_frame >= anchor.emit_start_frame >= 0 for anchor in anchors)
    assert all(anchor.anchor_end_s > anchor.anchor_start_s >= 0.0 for anchor in anchors)
    assert all(
        left.anchor_end_s <= right.anchor_start_s for left, right in itertools.pairwise(anchors)
    )
    assert [index for chunk in chunks for index in chunk.word_indices] == [0, 1]
    assert [word for chunk in chunks for word in chunk.words] == ["alpha", "beta"]
    assert all(math.isfinite(chunk.start) and math.isfinite(chunk.end) for chunk in chunks)
    assert all(chunk.end > chunk.start >= 0.0 for chunk in chunks)


@pytest.mark.parametrize(
    ("spf", "anchor_pad_s", "audio_dur_s"),
    [
        (0.0, 0.1, 1.0),
        (np.nan, 0.1, 1.0),
        (np.inf, 0.1, 1.0),
        (0.01, -0.1, 1.0),
        (0.01, np.nan, 1.0),
        (0.01, np.inf, 1.0),
        (0.01, 0.1, 0.0),
        (0.01, 0.1, np.nan),
        (0.01, 0.1, np.inf),
    ],
)
def test_anchor_construction_rejects_invalid_or_non_finite_scales(
    spf: float,
    anchor_pad_s: float,
    audio_dur_s: float,
) -> None:
    span = stage1.WordSpan(0, "alpha", 0, 1, 0.0, 0.01, -0.2, ["A"])
    with pytest.raises(ValueError):
        stage1.make_word_anchors_from_emissions(
            [span],
            [(0, 1)],
            [0],
            spf=spf,
            anchor_pad_s=anchor_pad_s,
            audio_dur_s=audio_dur_s,
        )


def test_rounded_chunks_are_finite_monotonic_and_complete() -> None:
    chunks = stage1.round_chunks_to_legacy_grid(
        raw_chunks=[
            _chunk(0.0, 0.2, ["alpha"], [0]),
            _chunk(0.3, 0.5, ["beta", "gamma"], [1, 2]),
        ],
        utt_id="utt",
        words=["alpha", "beta", "gamma"],
        num_samples=1_000,
        sample_rate=1_000,
    )

    assert [chunk.chunk_id for chunk in chunks] == ["utt.chunk001", "utt.chunk002"]
    assert [index for chunk in chunks for index in chunk.word_indices] == [0, 1, 2]
    assert [word for chunk in chunks for word in chunk.words] == ["alpha", "beta", "gamma"]
    assert all(
        math.isfinite(value)
        for chunk in chunks
        for value in (chunk.start_s, chunk.end_s, chunk.duration_s)
    )
    assert all(chunk.end_sample > chunk.start_sample >= 0 for chunk in chunks)
    assert all(left.end_sample <= right.start_sample for left, right in itertools.pairwise(chunks))


@pytest.mark.parametrize(
    ("raw_chunks", "words", "message"),
    [
        ([_chunk(0.0, 0.2, ["a", "b"], [0, 0])], ["a", "b"], "coverage mismatch"),
        ([_chunk(0.0, 0.2, ["a", "b"], [0])], ["a", "b"], "coverage mismatch"),
        ([_chunk(0.0, 0.2, ["a", "b"], [1, 0])], ["a", "b"], "Non-monotonic"),
        (
            [_chunk(0.0, 0.2, ["a"], [1]), _chunk(0.3, 0.5, ["b"], [0])],
            ["a", "b"],
            "coverage mismatch",
        ),
        ([_chunk(0.0, 0.2, ["a", "wrong"], [0, 1])], ["a", "b"], "Token consistency"),
    ],
    ids=["duplicate", "missing", "out-of-order-in-chunk", "out-of-order-across", "word"],
)
def test_rounded_chunks_reject_incomplete_or_inconsistent_word_coverage(
    raw_chunks: Sequence[stage1.Chunk],
    words: Sequence[str],
    message: str,
) -> None:
    with pytest.raises(RuntimeError, match=message):
        stage1.round_chunks_to_legacy_grid(
            raw_chunks=raw_chunks,
            utt_id="utt",
            words=words,
            num_samples=1_000,
            sample_rate=1_000,
        )


@pytest.mark.parametrize(
    ("start", "end"),
    [
        (np.nan, 0.2),
        (0.0, np.nan),
        (0.0, np.inf),
        (0.0, -np.inf),
    ],
)
def test_rounding_rejects_non_finite_chunk_boundaries(start: float, end: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        stage1.round_chunks_to_legacy_grid(
            raw_chunks=[_chunk(start, end, ["a"], [0])],
            utt_id="utt",
            words=["a"],
            num_samples=1_000,
            sample_rate=1_000,
        )


def test_rounding_rejects_empty_overlapping_and_invalid_audio_shapes() -> None:
    with pytest.raises(RuntimeError, match="no chunks"):
        stage1.round_chunks_to_legacy_grid(
            raw_chunks=[],
            utt_id="utt",
            words=[],
            num_samples=1_000,
            sample_rate=1_000,
        )
    with pytest.raises(RuntimeError, match="Overlapping chunks"):
        stage1.round_chunks_to_legacy_grid(
            raw_chunks=[
                _chunk(0.0, 0.3, ["a"], [0]),
                _chunk(0.2, 0.4, ["b"], [1]),
            ],
            utt_id="utt",
            words=["a", "b"],
            num_samples=1_000,
            sample_rate=1_000,
        )
    for field, value in (("num_samples", 0), ("sample_rate", 0)):
        kwargs = {"num_samples": 1_000, "sample_rate": 1_000, field: value}
        with pytest.raises(ValueError, match=field):
            stage1.round_chunks_to_legacy_grid(
                raw_chunks=[_chunk(0.0, 0.2, ["a"], [0])],
                utt_id="utt",
                words=["a"],
                **kwargs,
            )
