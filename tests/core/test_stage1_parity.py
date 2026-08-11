"""Three-way parity tests for the rebuilt, model-free Stage 1 core."""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from flexaligner.core import stage1 as production
from flexaligner.errors import ResourceLimitError
from tests.characterization import stage1_oracle as oracle
from tests.characterization.differential import assert_reference_equivalent
from tests.core._stage1_cases import (
    ANCHOR_MERGE_FAILURES,
    EARLY_FINISH_FINAL_COLUMN,
    EARLY_FINISH_LOG_PROBS,
    EARLY_FINISH_TARGETS,
    REPEATED_TARGET_LOG_PROBS,
    REPEATED_TARGETS,
    ROUNDING_FAILURES,
    TIE_STAY_LOG_PROBS,
    TIE_STAY_TARGETS,
    make_invalid_anchor_case,
    make_rounding_case,
)

REFERENCE_EXTRA_DEFINITIONS = (
    "build_chunk_lexicon",
    "attach_phone_confidence_from_points",
    "phones_to_word_segments_by_offsets",
    "emission_frames_by_token_index",
    "word_phone_token_ranges",
)


@pytest.fixture(scope="module")
def reference() -> SimpleNamespace:
    path = oracle.locate_reference()
    assert oracle.sha256_file(path) == oracle.REFERENCE_SHA256
    return oracle.load_reference_subset(
        path,
        names=(*oracle.REFERENCE_DEFINITIONS, *REFERENCE_EXTRA_DEFINITIONS),
    )


def _reference_array(value: Any) -> np.ndarray:
    return value.to_numpy()


def _point_pairs(points: Any) -> list[tuple[int, int]]:
    return [(point.token_index, point.time_index) for point in points]


def _round(module: Any, case: str) -> Any:
    chunks, words, num_samples, sample_rate, utterance_id = make_rounding_case(module, case)
    return module.round_chunks_to_legacy_grid(
        raw_chunks=chunks,
        utt_id=utterance_id,
        words=words,
        num_samples=num_samples,
        sample_rate=sample_rate,
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  !!!HELLO???  ", "hello"),
        ("'Quoted'", "'quoted'"),
        ("co-op", "co-op"),
        ("___", "___"),
        ("—", ""),
        ("ÉLAN!", "élan"),
    ],
)
def test_normalize_word_three_way_parity(
    reference: SimpleNamespace,
    raw: str,
    expected: str,
) -> None:
    assert production.normalize_word(raw) == reference.normalize_word(raw) == expected
    assert production.normalize_word(raw) == oracle.normalize_word(raw)


@pytest.mark.parametrize(
    ("phone", "expected"),
    [
        ("AH0", "AH"),
        ("EY1", "EY"),
        ("UW2", "UW"),
        ("AH3", "AH3"),
        ("B2", "B"),
        ("sil", "sil"),
        ("", ""),
    ],
)
def test_strip_stress_three_way_parity(
    reference: SimpleNamespace,
    phone: str,
    expected: str,
) -> None:
    assert (
        production.strip_arpabet_stress(phone) == reference.strip_arpabet_stress(phone) == expected
    )
    assert production.strip_arpabet_stress(phone) == oracle.strip_arpabet_stress(phone)


def test_first_pronunciation_and_inter_word_token_three_way_parity(
    reference: SimpleNamespace,
) -> None:
    words = ["go", "go"]
    lexicon = {"go": [["G", "OW1"], ["IGNORED"]]}
    vocabulary = {"G": 0, "OW1": 1, "|": 2}

    expected = reference.choose_greedy_pronunciations(words, lexicon, vocabulary, "|")
    independent = oracle.choose_greedy_pronunciations(words, lexicon, vocabulary, "|")
    actual = production.choose_greedy_pronunciations(words, lexicon, vocabulary, "|")

    assert_reference_equivalent(expected, actual)
    assert_reference_equivalent(independent, actual)
    assert actual.phones == ["G", "OW1", "|", "G", "OW1"]
    assert actual.pron_choice_idxs == [0, 0]


def test_chunk_lexicon_strips_stress_without_mutating_order(reference: SimpleNamespace) -> None:
    raw = {
        "go": [["G", "OW1"], ["G", "UW2"]],
        "odd": [["B2", "IY0"]],
    }
    container = SimpleNamespace(lex=raw)

    expected = reference.build_chunk_lexicon(container)
    from_container = production.build_chunk_lexicon(container)
    from_mapping = production.build_chunk_lexicon(raw)

    assert_reference_equivalent(expected, from_container)
    assert_reference_equivalent(expected, from_mapping)
    assert list(from_mapping) == ["go", "odd"]
    assert from_mapping["go"] == [["G", "OW"], ["G", "UW"]]
    assert raw["go"][0] == ["G", "OW1"]


def test_stage1_record_derived_properties_match_reference_meaning() -> None:
    segment = production.Segment("A", 2, 5)
    confident = production.SegmentWithConf("A", 2, 5, math.log(0.5))
    nonfinite = production.SegmentWithConf("A", 2, 5, -math.inf)
    span = production.WordSpan(0, "go", 2, 5, 0.2, 0.5, math.log(0.4), ["G"])
    chunk = production.Chunk(0.2, 0.5, ["go"], [0])
    anchor = production.WordAnchor(0, "go", 2, 5, 0.2, 0.5, 0.0, 0.8)
    runtime = production.RuntimeChunk("utt.chunk001", 200, 500, 3_200, 8_000, ["go"], [0])

    assert segment.dur_frames == 3
    assert confident.conf_prob == pytest.approx(0.5)
    assert nonfinite.conf_prob == -1.0
    assert span.dur_s == pytest.approx(0.3)
    assert span.conf_prob == pytest.approx(0.4)
    assert chunk.dur == pytest.approx(0.3)
    assert anchor.anchor_dur_s == pytest.approx(0.8)
    assert (runtime.start_s, runtime.end_s, runtime.duration_s) == pytest.approx((0.2, 0.5, 0.3))


@pytest.mark.parametrize(
    ("words", "lexicon", "vocabulary", "inter_word_token", "error", "message"),
    [
        (["missing"], {}, {}, None, KeyError, "word_index=0"),
        (["word"], {"word": []}, {}, None, RuntimeError, "no pronunciations"),
        (["word"], {"word": [[], ["W"]]}, {"W": 0}, None, RuntimeError, "Empty greedy"),
        (["word"], {"word": [["BAD"]]}, {}, None, KeyError, "word_index=0"),
        (
            ["word", "word"],
            {"word": [["W"]]},
            {"W": 0},
            "|",
            KeyError,
            "inter_word_token",
        ),
        ([], {}, {}, None, RuntimeError, "empty phone sequence"),
    ],
)
def test_greedy_pronunciation_failure_parity(
    reference: SimpleNamespace,
    words: list[str],
    lexicon: dict[str, list[list[str]]],
    vocabulary: dict[str, int],
    inter_word_token: str | None,
    error: type[Exception],
    message: str,
) -> None:
    for implementation in (
        reference.choose_greedy_pronunciations,
        production.choose_greedy_pronunciations,
    ):
        with pytest.raises(error, match=message):
            implementation(words, lexicon, vocabulary, inter_word_token)


def test_trellis_early_finish_matches_reference_and_independent_oracles(
    reference: SimpleNamespace,
) -> None:
    expected = reference.build_trellis(
        oracle.as_reference_tensor(EARLY_FINISH_LOG_PROBS),
        EARLY_FINISH_TARGETS,
        0,
    )
    independent = oracle.build_trellis(EARLY_FINISH_LOG_PROBS, EARLY_FINISH_TARGETS, 0)
    actual = production.build_trellis(EARLY_FINISH_LOG_PROBS, EARLY_FINISH_TARGETS, 0)

    assert_reference_equivalent(_reference_array(expected), actual)
    assert_reference_equivalent(independent, actual)
    assert_reference_equivalent(EARLY_FINISH_FINAL_COLUMN, actual[:, -1])
    assert_reference_equivalent(
        oracle.brute_force_final_column(EARLY_FINISH_LOG_PROBS, EARLY_FINISH_TARGETS, 0),
        actual[:, -1],
    )
    assert int(np.argmax(actual[:, -1])) == 2

    expected_points = reference.backtrace(
        expected,
        oracle.as_reference_tensor(EARLY_FINISH_LOG_PROBS),
        EARLY_FINISH_TARGETS,
        0,
    )
    actual_points = production.backtrace(
        actual,
        EARLY_FINISH_LOG_PROBS,
        EARLY_FINISH_TARGETS,
        0,
    )
    assert_reference_equivalent(expected_points, actual_points)
    assert _point_pairs(actual_points) == [(0, 0), (1, 1)]


def test_backtrace_tie_prefers_stay(reference: SimpleNamespace) -> None:
    expected_trellis = reference.build_trellis(
        oracle.as_reference_tensor(TIE_STAY_LOG_PROBS),
        TIE_STAY_TARGETS,
        0,
    )
    actual_trellis = production.build_trellis(TIE_STAY_LOG_PROBS, TIE_STAY_TARGETS, 0)
    expected_points = reference.backtrace(
        expected_trellis,
        oracle.as_reference_tensor(TIE_STAY_LOG_PROBS),
        TIE_STAY_TARGETS,
        0,
    )
    actual_points = production.backtrace(
        actual_trellis,
        TIE_STAY_LOG_PROBS,
        TIE_STAY_TARGETS,
        0,
    )

    assert_reference_equivalent(expected_points, actual_points)
    assert _point_pairs(actual_points) == [(0, 0), (1, 2)]


def test_repeated_target_keeps_named_current_behavior(reference: SimpleNamespace) -> None:
    assert oracle.CURRENT_BEHAVIOR_REPEATED_TARGETS.startswith("current_behavior:")
    expected_trellis = reference.build_trellis(
        oracle.as_reference_tensor(REPEATED_TARGET_LOG_PROBS),
        REPEATED_TARGETS,
        0,
    )
    actual_trellis = production.build_trellis(REPEATED_TARGET_LOG_PROBS, REPEATED_TARGETS, 0)
    expected_points = reference.backtrace(
        expected_trellis,
        oracle.as_reference_tensor(REPEATED_TARGET_LOG_PROBS),
        REPEATED_TARGETS,
        0,
    )
    actual_points = production.backtrace(
        actual_trellis,
        REPEATED_TARGET_LOG_PROBS,
        REPEATED_TARGETS,
        0,
    )

    assert_reference_equivalent(expected_points, actual_points)
    assert _point_pairs(actual_points) == [(0, 0), (1, 1)]


def test_trellis_resource_estimate_and_limit_fail_before_allocation() -> None:
    estimate = production.estimate_trellis_resources(4, 2)
    assert_reference_equivalent(
        {"frames": 4, "targets": 2, "cells": 15, "bytes": 120},
        {
            "frames": estimate.frames,
            "targets": estimate.targets,
            "cells": estimate.cells,
            "bytes": estimate.bytes,
        },
    )
    with pytest.raises(ResourceLimitError, match="cell limit exceeded") as caught:
        production.build_trellis(
            EARLY_FINISH_LOG_PROBS,
            EARLY_FINISH_TARGETS,
            0,
            max_trellis_cells=14,
        )
    assert caught.value.context == {
        "frames": 4,
        "targets": 2,
        "cells": 15,
        "limit": 14,
    }


@pytest.mark.parametrize(
    ("frames", "targets", "itemsize", "error", "message"),
    [
        (True, 1, 8, TypeError, "frames must be an integer"),
        (1, 1.5, 8, TypeError, "targets must be an integer"),
        (1, 1, False, TypeError, "itemsize must be an integer"),
        (-1, 1, 8, ValueError, "frames must be non-negative"),
        (1, -1, 8, ValueError, "targets must be non-negative"),
        (1, 1, 0, ValueError, "itemsize must be positive"),
    ],
)
def test_trellis_resource_estimate_failure_contracts(
    frames: Any,
    targets: Any,
    itemsize: Any,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        production.estimate_trellis_resources(frames, targets, itemsize=itemsize)


@pytest.mark.parametrize(
    ("log_probs", "targets", "blank_id", "limit", "error", "message"),
    [
        ([[0.0, 0.0]], [1], 0, None, TypeError, "NumPy array"),
        (np.zeros((0, 2)), [1], 0, None, ValueError, "dimensions must be positive"),
        (np.zeros((1, 0)), [0], 0, None, ValueError, "dimensions must be positive"),
        (np.zeros((1, 2), dtype=np.int64), [1], 0, None, TypeError, "dtype must be floating"),
        (np.asarray([[0.0, np.nan]]), [1], 0, None, ValueError, "NaN or infinity"),
        (np.zeros((1, 2)), [1], True, None, TypeError, "blank_id must be an integer"),
        (np.zeros((1, 2)), [False], 0, None, TypeError, "target id must be an integer"),
        (np.zeros((1, 2)), [1], 0, True, TypeError, "max_trellis_cells"),
        (np.zeros((1, 2)), [1], 0, 0, ValueError, "must be positive"),
    ],
)
def test_trellis_array_and_limit_validation(
    log_probs: Any,
    targets: list[Any],
    blank_id: Any,
    limit: Any,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        production.build_trellis(
            log_probs,
            targets,
            blank_id,
            max_trellis_cells=limit,
        )


@pytest.mark.parametrize(
    ("log_probs", "targets", "blank_id", "error", "message"),
    [
        (np.zeros((2, 2)), [], 0, ValueError, "Empty target"),
        (np.zeros((2, 2)), [1], -1, ValueError, "blank_id out of range"),
        (np.zeros((2, 2)), [2], 0, ValueError, "target id out of range"),
        (np.zeros((1, 2)), [1, 1], 0, RuntimeError, "failed to consume"),
        (np.zeros(2), [1], 0, ValueError, "two-dimensional"),
    ],
)
def test_trellis_failure_contracts(
    log_probs: np.ndarray,
    targets: list[int],
    blank_id: int,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        production.build_trellis(log_probs, targets, blank_id)


def test_backtrace_unreachable_target_fails(reference: SimpleNamespace) -> None:
    trellis = np.asarray([[0.0, -np.inf], [0.0, -np.inf]])
    log_probs = np.asarray([[0.0, -1.0]])
    for implementation, trellis_arg, log_probs_arg in (
        (
            reference.backtrace,
            oracle.as_reference_tensor(trellis),
            oracle.as_reference_tensor(log_probs),
        ),
        (production.backtrace, trellis, log_probs),
    ):
        with pytest.raises(RuntimeError, match="did not consume"):
            implementation(trellis_arg, log_probs_arg, [1], 0)


@pytest.mark.parametrize(
    ("trellis", "error", "message"),
    [
        ([[0.0, -np.inf], [0.0, 0.0]], TypeError, "NumPy array"),
        (np.zeros((3, 2)), ValueError, "shape mismatch"),
        (np.zeros((2, 2), dtype=np.int64), TypeError, "dtype must be floating"),
        (np.asarray([[0.0, -np.inf], [0.0, np.nan]]), ValueError, "NaN"),
        (np.asarray([[0.0, -np.inf], [0.0, np.inf]]), ValueError, "positive infinity"),
    ],
)
def test_backtrace_trellis_validation(
    trellis: Any,
    error: type[Exception],
    message: str,
) -> None:
    log_probs = np.asarray([[0.0, -1.0]])
    with pytest.raises(error, match=message):
        production.backtrace(trellis, log_probs, [1], 0)


def test_points_to_segments_last_token_is_exactly_one_frame(reference: SimpleNamespace) -> None:
    expected_points = [reference.Point(0, 2), reference.Point(1, 5), reference.Point(2, 8)]
    actual_points = [production.Point(0, 2), production.Point(1, 5), production.Point(2, 8)]
    expected = reference.points_to_segments(expected_points, ["A", "B", "C"])
    actual = production.points_to_segments(actual_points, ["A", "B", "C"])

    assert_reference_equivalent(expected, actual)
    assert [(item.label, item.start_frame, item.end_frame) for item in actual] == [
        ("A", 2, 5),
        ("B", 5, 8),
        ("C", 8, 9),
    ]


@pytest.mark.parametrize(
    ("points", "labels", "error", "message"),
    [
        ([production.Point(0, 0)], [], ValueError, r"len\(points\)"),
        ([], [], ValueError, "empty points"),
        ([production.Point(1, 0)], ["A"], ValueError, "token_index out of range"),
        (
            [production.Point(0, 2), production.Point(1, 2)],
            ["A", "B"],
            RuntimeError,
            "Non-positive",
        ),
    ],
)
def test_points_to_segments_failure_contracts(
    points: list[Any],
    labels: list[str],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        production.points_to_segments(points, labels)


@pytest.mark.parametrize(
    ("mode", "emission_frame", "expected"),
    [
        ("emission", 1, math.log(0.75)),
        ("emission", None, math.log(0.75)),
        ("emission", 99, math.log(0.60)),
        ("avg_frame", None, np.log([0.2, 0.75, 0.6]).mean()),
    ],
)
def test_segment_confidence_modes_match_reference(
    reference: SimpleNamespace,
    mode: str,
    emission_frame: int | None,
    expected: float,
) -> None:
    log_probs = np.log(np.asarray([[0.8, 0.2], [0.25, 0.75], [0.4, 0.6]]))
    reference_segment = reference.Segment("AA", 0, 3)
    production_segment = production.Segment("AA", 0, 3)
    reference_result = reference.compute_segment_confidence(
        reference_segment,
        {"AA": 1},
        oracle.as_reference_tensor(log_probs),
        emission_frame,
        mode,
    )
    actual = production.compute_segment_confidence(
        production_segment,
        {"AA": 1},
        log_probs,
        emission_frame,
        mode,
    )
    assert actual == pytest.approx(reference_result)
    assert actual == pytest.approx(expected)


def test_word_confidence_is_phone_log_mean_and_probability_geometric_mean(
    reference: SimpleNamespace,
) -> None:
    phone_logs = [math.log(0.25), math.log(0.81)]
    expected = reference.word_segments_with_confidence(
        [reference.Segment("word", 0, 2)],
        [
            reference.SegmentWithConf("W", 0, 1, phone_logs[0]),
            reference.SegmentWithConf("ER", 1, 2, phone_logs[1]),
        ],
    )
    actual = production.word_segments_with_confidence(
        [production.Segment("word", 0, 2)],
        [
            production.SegmentWithConf("W", 0, 1, phone_logs[0]),
            production.SegmentWithConf("ER", 1, 2, phone_logs[1]),
        ],
    )

    assert_reference_equivalent(expected, actual)
    assert actual[0].conf_log == pytest.approx(oracle.word_confidence_log(phone_logs))
    assert actual[0].conf_prob == pytest.approx(math.sqrt(0.25 * 0.81))


@pytest.mark.parametrize(
    ("case", "error", "message"),
    [
        ("missing_label", KeyError, "not in vocab"),
        ("invalid_avg", RuntimeError, "Invalid segment"),
        ("unsupported_mode", ValueError, "Unsupported confidence_mode"),
        ("no_word_phones", RuntimeError, "No overlapping phone confidence"),
    ],
)
def test_confidence_failure_contracts(case: str, error: type[Exception], message: str) -> None:
    log_probs = np.zeros((2, 2), dtype=np.float64)
    with pytest.raises(error, match=message):
        if case == "missing_label":
            production.compute_segment_confidence(
                production.Segment("BAD", 0, 1), {}, log_probs, 0, "emission"
            )
        elif case == "invalid_avg":
            production.compute_segment_confidence(
                production.Segment("A", 2, 2), {"A": 1}, log_probs, None, "avg_frame"
            )
        elif case == "unsupported_mode":
            production.compute_segment_confidence(
                production.Segment("A", 0, 1), {"A": 1}, log_probs, None, "unknown"
            )
        else:
            production.word_segments_with_confidence(
                [production.Segment("word", 0, 1)],
                [production.SegmentWithConf("A", 2, 3, -1.0)],
            )


@pytest.mark.parametrize(
    ("token_id", "error", "message"),
    [
        (True, TypeError, "must be an integer"),
        (2, ValueError, "out of range"),
    ],
)
def test_confidence_token_id_validation(
    token_id: Any,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        production.compute_segment_confidence(
            production.Segment("A", 0, 1),
            {"A": token_id},
            np.zeros((2, 2), dtype=np.float64),
            0,
            "emission",
        )


def test_attach_phone_confidence_uses_exact_emission_points(reference: SimpleNamespace) -> None:
    target_labels = ["A", "B"]
    log_probs = np.log(np.asarray([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]]))
    reference_segments = [reference.Segment("A", 0, 1), reference.Segment("B", 1, 3)]
    production_segments = [production.Segment("A", 0, 1), production.Segment("B", 1, 3)]
    reference_points = [reference.Point(0, 0), reference.Point(1, 2)]
    production_points = [production.Point(0, 0), production.Point(1, 2)]

    expected = reference.attach_phone_confidence_from_points(
        reference_segments,
        reference_points,
        target_labels,
        {"A": 0, "B": 1},
        oracle.as_reference_tensor(log_probs),
        "emission",
    )
    actual = production.attach_phone_confidence_from_points(
        production_segments,
        production_points,
        target_labels,
        {"A": 0, "B": 1},
        log_probs,
        "emission",
    )

    assert_reference_equivalent(expected, actual)
    assert [item.conf_log for item in actual] == pytest.approx([math.log(0.1), math.log(0.7)])


@pytest.mark.parametrize(
    ("segments", "points", "labels", "message"),
    [
        ([production.Segment("A", 0, 1)], [], [], "length must equal"),
        (
            [production.Segment("A", 0, 1)],
            [production.Point(1, 0)],
            ["A"],
            "token_index out of range",
        ),
        ([production.Segment("B", 0, 1)], [], ["A"], "label mismatch"),
    ],
)
def test_attach_phone_confidence_failure_contracts(
    segments: list[Any],
    points: list[Any],
    labels: list[str],
    message: str,
) -> None:
    with pytest.raises((ValueError, RuntimeError), match=message):
        production.attach_phone_confidence_from_points(
            segments,
            points,
            labels,
            {"A": 0, "B": 1},
            np.zeros((2, 2), dtype=np.float64),
            "emission",
        )


def test_word_reconstruction_ranges_and_emission_frames_match_reference(
    reference: SimpleNamespace,
) -> None:
    labels = ["G", "OW", "|", "G", "OW"]
    reference_segments = [
        reference.Segment(label, index, index + 1) for index, label in enumerate(labels)
    ]
    production_segments = [
        production.Segment(label, index, index + 1) for index, label in enumerate(labels)
    ]
    words = ["go", "go"]
    pronunciations = [["G", "OW"], ["G", "OW"]]

    expected_words = reference.phones_to_word_segments_by_offsets(
        reference_segments, words, pronunciations, "|"
    )
    actual_words = production.phones_to_word_segments_by_offsets(
        production_segments, words, pronunciations, "|"
    )
    expected_ranges = reference.word_phone_token_ranges(
        reference_segments, words, pronunciations, "|"
    )
    actual_ranges = production.word_phone_token_ranges(
        production_segments, words, pronunciations, "|"
    )
    reference_points = [reference.Point(index, index + 2) for index in range(len(labels))]
    production_points = [production.Point(index, index + 2) for index in range(len(labels))]

    assert_reference_equivalent(expected_words, actual_words)
    assert_reference_equivalent(expected_ranges, actual_ranges)
    assert actual_ranges == [(0, 2), (3, 5)]
    assert production.emission_frames_by_token_index(production_points, len(labels)) == (
        reference.emission_frames_by_token_index(reference_points, len(labels))
    )
    assert [item.label for item in actual_words] == ["go", "go"]


@pytest.mark.parametrize(
    ("helper_name", "segments", "words", "pronunciations", "inter_word_token", "message"),
    [
        (
            "phones_to_word_segments_by_offsets",
            [],
            ["a"],
            [],
            None,
            r"len\(words\)",
        ),
        ("word_phone_token_ranges", [], ["a"], [], None, r"len\(words\)"),
        (
            "phones_to_word_segments_by_offsets",
            [production.Segment("A", 0, 1)],
            ["a", "b"],
            [["A"], ["B"]],
            "|",
            "before inter_word_token",
        ),
        (
            "word_phone_token_ranges",
            [production.Segment("A", 0, 1)],
            ["a", "b"],
            [["A"], ["B"]],
            "|",
            "before inter_word_token",
        ),
        (
            "phones_to_word_segments_by_offsets",
            [
                production.Segment("A", 0, 1),
                production.Segment("X", 1, 2),
                production.Segment("B", 2, 3),
            ],
            ["a", "b"],
            [["A"], ["B"]],
            "|",
            "Expected inter_word_token",
        ),
        (
            "word_phone_token_ranges",
            [
                production.Segment("A", 0, 1),
                production.Segment("X", 1, 2),
                production.Segment("B", 2, 3),
            ],
            ["a", "b"],
            [["A"], ["B"]],
            "|",
            "Expected inter_word_token",
        ),
        ("phones_to_word_segments_by_offsets", [], ["a"], [[]], None, "Empty pronunciation"),
        ("word_phone_token_ranges", [], ["a"], [[]], None, "Empty pronunciation"),
        (
            "phones_to_word_segments_by_offsets",
            [production.Segment("A", 0, 1)],
            ["a"],
            [["A", "B"]],
            None,
            "Ran out of phone token segments",
        ),
        (
            "word_phone_token_ranges",
            [production.Segment("A", 0, 1)],
            ["a"],
            [["A", "B"]],
            None,
            "Ran out of phone token segments",
        ),
        (
            "phones_to_word_segments_by_offsets",
            [production.Segment("B", 0, 1)],
            ["a"],
            [["A"]],
            None,
            "pronunciation mismatch",
        ),
        (
            "word_phone_token_ranges",
            [production.Segment("B", 0, 1)],
            ["a"],
            [["A"]],
            None,
            "pronunciation mismatch",
        ),
        (
            "phones_to_word_segments_by_offsets",
            [production.Segment("A", 2, 2)],
            ["a"],
            [["A"]],
            None,
            "Invalid word segment frame span",
        ),
        (
            "phones_to_word_segments_by_offsets",
            [production.Segment("A", 0, 1), production.Segment("X", 1, 2)],
            ["a"],
            [["A"]],
            None,
            "Unconsumed phone token segments",
        ),
        (
            "word_phone_token_ranges",
            [production.Segment("A", 0, 1), production.Segment("X", 1, 2)],
            ["a"],
            [["A"]],
            None,
            "Unconsumed phone token segments",
        ),
    ],
)
def test_word_reconstruction_failure_contracts(
    helper_name: str,
    segments: list[Any],
    words: list[str],
    pronunciations: list[list[str]],
    inter_word_token: str | None,
    message: str,
) -> None:
    implementation = getattr(production, helper_name)
    with pytest.raises((ValueError, RuntimeError), match=message):
        implementation(segments, words, pronunciations, inter_word_token)


@pytest.mark.parametrize(
    ("points", "num_tokens", "error", "message"),
    [
        ([], 0, ValueError, "must be positive"),
        ([production.Point(1, 0)], 1, RuntimeError, "out of range"),
        (
            [production.Point(0, 0), production.Point(0, 1)],
            1,
            RuntimeError,
            "Duplicate emission point",
        ),
        ([production.Point(0, 0)], 2, RuntimeError, "Missing emission frames"),
    ],
)
def test_emission_frame_index_failure_contracts(
    points: list[Any],
    num_tokens: int,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        production.emission_frames_by_token_index(points, num_tokens)


def test_word_anchors_clip_padding_to_audio_bounds(reference: SimpleNamespace) -> None:
    reference_spans = [
        reference.WordSpan(0, "first", 1, 4, 0.1, 0.4, -1.0, ["F", "ER"]),
        reference.WordSpan(1, "last", 8, 10, 0.8, 1.0, -1.0, ["L", "AE"]),
    ]
    production_spans = [
        production.WordSpan(0, "first", 1, 4, 0.1, 0.4, -1.0, ["F", "ER"]),
        production.WordSpan(1, "last", 8, 10, 0.8, 1.0, -1.0, ["L", "AE"]),
    ]
    kwargs = {
        "token_ranges": [(0, 2), (2, 4)],
        "token_emission_frames": [1, 3, 8, 9],
        "spf": 0.1,
        "anchor_pad_s": 0.3,
        "audio_dur_s": 1.0,
    }
    expected = reference.make_word_anchors_from_emissions(reference_spans, **kwargs)
    actual = production.make_word_anchors_from_emissions(production_spans, **kwargs)

    assert_reference_equivalent(expected, actual)
    assert [(item.anchor_start_s, item.anchor_end_s) for item in actual] == [
        (0.0, pytest.approx(0.6)),
        (pytest.approx(0.5), 1.0),
    ]


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        (
            {
                "token_ranges": [],
                "token_emission_frames": [],
                "spf": 0.1,
                "anchor_pad_s": 0.3,
                "audio_dur_s": 1.0,
            },
            ValueError,
            "len\\(word_spans\\)",
        ),
        (
            {
                "token_ranges": [(0, 1)],
                "token_emission_frames": [0],
                "spf": 0.0,
                "anchor_pad_s": 0.3,
                "audio_dur_s": 1.0,
            },
            ValueError,
            "spf",
        ),
        (
            {
                "token_ranges": [(0, 1)],
                "token_emission_frames": [0],
                "spf": 0.1,
                "anchor_pad_s": -0.1,
                "audio_dur_s": 1.0,
            },
            ValueError,
            "anchor_pad_s",
        ),
        (
            {
                "token_ranges": [(0, 1)],
                "token_emission_frames": [0],
                "spf": 0.1,
                "anchor_pad_s": 0.3,
                "audio_dur_s": 0.0,
            },
            ValueError,
            "audio_dur_s",
        ),
        (
            {
                "token_ranges": [(0, 0)],
                "token_emission_frames": [0],
                "spf": 0.1,
                "anchor_pad_s": 0.3,
                "audio_dur_s": 1.0,
            },
            RuntimeError,
            "Empty token range",
        ),
        (
            {
                "token_ranges": [(1, 2)],
                "token_emission_frames": [0],
                "spf": 0.1,
                "anchor_pad_s": 0.3,
                "audio_dur_s": 1.0,
            },
            RuntimeError,
            "No emission frames",
        ),
        (
            {
                "token_ranges": [(0, 1)],
                "token_emission_frames": [0],
                "spf": 0.1,
                "anchor_pad_s": 0.0,
                "audio_dur_s": 1.0,
            },
            RuntimeError,
            "Invalid word anchor",
        ),
    ],
)
def test_anchor_failure_contracts(
    kwargs: dict[str, Any],
    error: type[Exception],
    message: str,
) -> None:
    span = production.WordSpan(0, "word", 0, 1, 0.0, 0.1, -1.0, ["W"])
    with pytest.raises(error, match=message):
        production.make_word_anchors_from_emissions([span], **kwargs)


@pytest.mark.parametrize(
    ("gap", "expected_count"),
    [(0.199999, 1), (0.2, 2)],
)
def test_anchor_merge_threshold_is_strict_and_preserves_repeated_word_order(
    reference: SimpleNamespace,
    gap: float,
    expected_count: int,
) -> None:
    second_start = 0.25 + gap
    reference_anchors = [
        reference.WordAnchor(0, "go", 0, 1, 0.0, 0.1, 0.0, 0.25),
        reference.WordAnchor(1, "go", 2, 3, 0.2, 0.3, second_start, second_start + 0.25),
    ]
    production_anchors = [
        production.WordAnchor(0, "go", 0, 1, 0.0, 0.1, 0.0, 0.25),
        production.WordAnchor(1, "go", 2, 3, 0.2, 0.3, second_start, second_start + 0.25),
    ]
    expected = reference.merge_word_anchors_into_chunks(reference_anchors, anchor_merge_gap_s=0.2)
    actual = production.merge_word_anchors_into_chunks(production_anchors, anchor_merge_gap_s=0.2)

    assert_reference_equivalent(expected, actual)
    assert len(actual) == expected_count
    assert [word for chunk in actual for word in chunk.words] == ["go", "go"]
    assert [index for chunk in actual for index in chunk.word_indices] == [0, 1]


@pytest.mark.parametrize(("case", "message"), ANCHOR_MERGE_FAILURES)
def test_anchor_merge_failure_contracts(case: str, message: str) -> None:
    anchors, gap = make_invalid_anchor_case(production, case)
    with pytest.raises((ValueError, RuntimeError), match=message):
        production.merge_word_anchors_into_chunks(anchors, anchor_merge_gap_s=gap)


def test_legacy_grid_rounding_tail_clamp_and_repeated_word_coverage(
    reference: SimpleNamespace,
) -> None:
    num_samples = 16_009
    duration = num_samples / 16_000.0
    reference_chunks = [
        reference.Chunk(0.0015, 0.0045, ["go"], [0]),
        reference.Chunk(0.9, duration, ["go"], [1]),
    ]
    production_chunks = [
        production.Chunk(0.0015, 0.0045, ["go"], [0]),
        production.Chunk(0.9, duration, ["go"], [1]),
    ]
    kwargs = {
        "utt_id": "utt",
        "words": ["go", "go"],
        "num_samples": num_samples,
        "sample_rate": 16_000,
    }
    expected = reference.round_chunks_to_legacy_grid(raw_chunks=reference_chunks, **kwargs)
    actual = production.round_chunks_to_legacy_grid(raw_chunks=production_chunks, **kwargs)

    assert_reference_equivalent(expected, actual)
    assert (actual[0].start_ms, actual[0].end_ms) == (2, 4)
    assert (actual[-1].end_ms, actual[-1].end_sample) == (1_000, 16_000)
    assert [word for chunk in actual for word in chunk.words] == ["go", "go"]
    assert [index for chunk in actual for index in chunk.word_indices] == [0, 1]


@pytest.mark.parametrize(("case", "message"), ROUNDING_FAILURES)
def test_legacy_grid_failure_contracts_match_reference(
    reference: SimpleNamespace,
    case: str,
    message: str,
) -> None:
    for module in (reference, production):
        with pytest.raises(RuntimeError, match=message, check=lambda error: bool(str(error))):
            _round(module, case)


@pytest.mark.parametrize(
    ("num_samples", "sample_rate", "error", "message"),
    [
        (True, 16_000, TypeError, "num_samples must be an integer"),
        (16_000, False, TypeError, "sample_rate must be an integer"),
        (0, 16_000, ValueError, "num_samples must be positive"),
        (16_000, 0, ValueError, "sample_rate must be positive"),
    ],
)
def test_legacy_grid_numeric_input_validation(
    num_samples: Any,
    sample_rate: Any,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        production.round_chunks_to_legacy_grid(
            raw_chunks=[production.Chunk(0.0, 0.1, ["a"], [0])],
            utt_id="utt",
            words=["a"],
            num_samples=num_samples,
            sample_rate=sample_rate,
        )
