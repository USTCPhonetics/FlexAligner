"""Model-free characterization of the frozen Stage 1 reference helpers."""

from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from tests.characterization import stage1_oracle as oracle


@pytest.fixture(scope="module")
def reference_path() -> Path:
    path = oracle.locate_reference()
    assert oracle.sha256_file(path) == oracle.REFERENCE_SHA256
    return path


@pytest.fixture(scope="module")
def reference(reference_path: Path) -> SimpleNamespace:
    return oracle.load_reference_subset(reference_path)


def _point_pairs(points: Any) -> list[tuple[int, int]]:
    return [(point.token_index, point.time_index) for point in points]


def _chunk_tuples(chunks: Any) -> list[tuple[float, float, list[str], list[int]]]:
    return [
        (chunk.start, chunk.end, list(chunk.words), list(chunk.word_indices)) for chunk in chunks
    ]


def _runtime_chunk_tuples(
    chunks: Any,
) -> list[tuple[str, int, int, int, int, list[str], list[int]]]:
    return [
        (
            chunk.chunk_id,
            chunk.start_ms,
            chunk.end_ms,
            chunk.start_sample,
            chunk.end_sample,
            list(chunk.words),
            list(chunk.word_indices),
        )
        for chunk in chunks
    ]


def test_reference_snapshot_is_exact(reference_path: Path) -> None:
    assert reference_path.name == "align_single_cpu.py"
    assert oracle.sha256_file(reference_path) == oracle.REFERENCE_SHA256


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
def test_normalize_word_frozen_examples(
    reference: SimpleNamespace,
    raw: str,
    expected: str,
) -> None:
    assert reference.normalize_word(raw) == expected
    assert oracle.normalize_word(raw) == expected


@pytest.mark.parametrize(
    ("phone", "expected"),
    [
        ("AH0", "AH"),
        ("EY1", "EY"),
        ("UW2", "UW"),
        ("AH3", "AH3"),
        ("0", "0"),
        ("sil", "sil"),
        ("", ""),
    ],
)
def test_strip_arpabet_stress_only_removes_terminal_zero_one_two(
    reference: SimpleNamespace,
    phone: str,
    expected: str,
) -> None:
    assert reference.strip_arpabet_stress(phone) == expected
    assert oracle.strip_arpabet_stress(phone) == expected


def test_greedy_pronunciation_uses_only_first_variant(
    reference: SimpleNamespace,
) -> None:
    words = ["hello", "world"]
    lexicon = {
        "hello": [["HH", "AH"], ["NOT", "VALIDATED"]],
        "world": [["W", "ER", "L", "D"], ["ALSO", "IGNORED"]],
    }
    phones = ["HH", "AH", "|", "W", "ER", "L", "D"]
    vocabulary = {phone: index for index, phone in enumerate(phones)}

    actual = reference.choose_greedy_pronunciations(
        words,
        lexicon,
        vocabulary,
        "|",
    )
    expected = oracle.choose_greedy_pronunciations(
        words,
        lexicon,
        vocabulary,
        "|",
    )
    assert actual.phones == expected.phones == ["HH", "AH", "|", "W", "ER", "L", "D"]
    assert actual.chosen_prons == expected.chosen_prons
    assert actual.pron_choice_idxs == expected.pron_choice_idxs == [0, 0]


def test_greedy_pronunciation_oov_error_keeps_word_index(
    reference: SimpleNamespace,
) -> None:
    with pytest.raises(KeyError, match="word_index=1"):
        reference.choose_greedy_pronunciations(
            ["known", "missing"],
            {"known": [["K"]]},
            {"K": 0},
            None,
        )
    with pytest.raises(KeyError, match="word_index=1"):
        oracle.choose_greedy_pronunciations(
            ["known", "missing"],
            {"known": [["K"]]},
            {"K": 0},
            None,
        )


def test_greedy_pronunciation_rejects_empty_first_variant_even_if_later_valid(
    reference: SimpleNamespace,
) -> None:
    lexicon = {"word": [[], ["W"]]}
    with pytest.raises(RuntimeError, match="Empty greedy pronunciation"):
        reference.choose_greedy_pronunciations(["word"], lexicon, {"W": 0}, None)
    with pytest.raises(RuntimeError, match="Empty greedy pronunciation"):
        oracle.choose_greedy_pronunciations(["word"], lexicon, {"W": 0}, None)


def test_greedy_pronunciation_reports_missing_phone_and_context(
    reference: SimpleNamespace,
) -> None:
    lexicon = {"word": [["W", "BAD"]]}
    for implementation in (
        reference.choose_greedy_pronunciations,
        oracle.choose_greedy_pronunciations,
    ):
        with pytest.raises(KeyError) as caught:
            implementation(["word"], lexicon, {"W": 0}, None)
        message = caught.value.args[0]
        assert "phone='BAD'" in message
        assert "word_index=0" in message
        assert "pron=['W', 'BAD']" in message


def test_early_finish_trellis_matches_hand_and_brute_force_oracles(
    reference: SimpleNamespace,
) -> None:
    log_probs = np.array(
        [
            [-5.0, 0.0, -9.0],
            [-5.0, -9.0, 0.0],
            [0.0, -9.0, -9.0],
            [-1.0, -9.0, -9.0],
        ],
        dtype=np.float64,
    )
    targets = [1, 2]
    expected_final_column = np.array([-np.inf, -np.inf, 0.0, 0.0, -1.0])

    reference_trellis = reference.build_trellis(
        oracle.as_reference_tensor(log_probs),
        targets,
        0,
    )
    actual = reference_trellis.to_numpy()
    independent = oracle.build_trellis(log_probs, targets, 0)
    brute_force = oracle.brute_force_final_column(log_probs, targets, 0)

    np.testing.assert_allclose(actual, independent)
    np.testing.assert_allclose(actual[:, -1], expected_final_column)
    np.testing.assert_allclose(actual[:, -1], brute_force)
    assert int(np.argmax(actual[:, -1])) == 2
    reference_points = reference.backtrace(
        reference_trellis,
        oracle.as_reference_tensor(log_probs),
        targets,
        0,
    )
    assert _point_pairs(reference_points) == [(0, 0), (1, 1)]
    assert oracle.backtrace(independent, log_probs, targets, 0) == [
        oracle.Point(0, 0),
        oracle.Point(1, 1),
    ]


def test_backtrace_tie_prefers_stay(
    reference: SimpleNamespace,
) -> None:
    log_probs = np.array(
        [
            [-0.1, -2.0, -100.0],
            [-0.1, -2.0, -100.0],
            [-100.0, -100.0, 0.0],
        ],
        dtype=np.float64,
    )
    targets = [1, 2]
    reference_trellis = reference.build_trellis(
        oracle.as_reference_tensor(log_probs),
        targets,
        0,
    )
    trellis = reference_trellis.to_numpy()
    stay = trellis[1, 1] + log_probs[1, 0]
    emit = trellis[1, 0] + log_probs[1, 1]
    assert stay == pytest.approx(emit)

    reference_points = reference.backtrace(
        reference_trellis,
        oracle.as_reference_tensor(log_probs),
        targets,
        0,
    )
    assert _point_pairs(reference_points) == [(0, 0), (1, 2)]
    independent = oracle.build_trellis(log_probs, targets, 0)
    assert oracle.backtrace(independent, log_probs, targets, 0) == [
        oracle.Point(0, 0),
        oracle.Point(1, 2),
    ]


def test_current_behavior_repeated_targets_can_emit_without_blank(
    reference: SimpleNamespace,
) -> None:
    # This is explicitly a parity lock, not approval of full CTC semantics.
    assert oracle.CURRENT_BEHAVIOR_REPEATED_TARGETS.startswith("current_behavior:")
    log_probs = np.array([[-10.0, 0.0], [-10.0, 0.0]], dtype=np.float64)
    targets = [1, 1]
    reference_trellis = reference.build_trellis(
        oracle.as_reference_tensor(log_probs),
        targets,
        0,
    )
    reference_points = reference.backtrace(
        reference_trellis,
        oracle.as_reference_tensor(log_probs),
        targets,
        0,
    )
    assert _point_pairs(reference_points) == [(0, 0), (1, 1)]
    independent = oracle.build_trellis(log_probs, targets, 0)
    assert oracle.backtrace(independent, log_probs, targets, 0) == [
        oracle.Point(0, 0),
        oracle.Point(1, 1),
    ]


def test_points_to_segments_gives_last_token_exactly_one_frame(
    reference: SimpleNamespace,
) -> None:
    reference_points = [
        reference.Point(0, 2),
        reference.Point(1, 5),
        reference.Point(2, 8),
    ]
    reference_segments = reference.points_to_segments(reference_points, ["A", "B", "C"])
    assert [
        (segment.label, segment.start_frame, segment.end_frame) for segment in reference_segments
    ] == [("A", 2, 5), ("B", 5, 8), ("C", 8, 9)]
    assert oracle.points_to_segments(
        [oracle.Point(0, 2), oracle.Point(1, 5), oracle.Point(2, 8)],
        ["A", "B", "C"],
    ) == [
        oracle.Segment("A", 2, 5),
        oracle.Segment("B", 5, 8),
        oracle.Segment("C", 8, 9),
    ]


def test_emission_confidence_is_target_log_probability_at_emission_frame(
    reference: SimpleNamespace,
) -> None:
    log_probs = np.log(
        np.array(
            [
                [0.8, 0.2],
                [0.25, 0.75],
                [0.4, 0.6],
            ],
            dtype=np.float64,
        )
    )
    reference_segment = reference.Segment("AA", 0, 3)
    actual = reference.compute_segment_confidence(
        reference_segment,
        {"AA": 1},
        oracle.as_reference_tensor(log_probs),
        1,
        "emission",
    )
    expected = oracle.emission_confidence_log(
        oracle.Segment("AA", 0, 3),
        {"AA": 1},
        log_probs,
        1,
    )
    assert actual == pytest.approx(math.log(0.75))
    assert actual == pytest.approx(expected)


def test_word_confidence_probability_is_geometric_mean_of_phone_emissions(
    reference: SimpleNamespace,
    reference_path: Path,
) -> None:
    phone_logs = [math.log(0.25), math.log(0.81)]
    reference_word = reference.Segment("word", 0, 2)
    reference_phones = [
        reference.SegmentWithConf("W", 0, 1, phone_logs[0]),
        reference.SegmentWithConf("ER", 1, 2, phone_logs[1]),
    ]
    result = reference.word_segments_with_confidence(
        [reference_word],
        reference_phones,
    )[0]
    expected_log = oracle.word_confidence_log(phone_logs)
    assert result.conf_log == pytest.approx(expected_log)
    assert math.exp(result.conf_log) == pytest.approx(math.sqrt(0.25 * 0.81))
    assert (
        "geometric mean of Chunker CTC target-emission probabilities"
        in reference_path.read_text(encoding="utf-8")
    )


def test_word_anchors_clip_padding_to_audio_bounds(
    reference: SimpleNamespace,
) -> None:
    reference_spans = [
        reference.WordSpan(0, "first", 1, 4, 0.1, 0.4, -1.0, ["F", "ER"]),
        reference.WordSpan(1, "last", 8, 10, 0.8, 1.0, -1.0, ["L", "AE"]),
    ]
    oracle_spans = [
        oracle.WordSpan(0, "first", 1, 4, 0.1, 0.4, -1.0, ["F", "ER"]),
        oracle.WordSpan(1, "last", 8, 10, 0.8, 1.0, -1.0, ["L", "AE"]),
    ]
    token_ranges = [(0, 2), (2, 4)]
    frames = [1, 3, 8, 9]
    actual = reference.make_word_anchors_from_emissions(
        reference_spans,
        token_ranges,
        frames,
        spf=0.1,
        anchor_pad_s=0.3,
        audio_dur_s=1.0,
    )
    expected = oracle.make_word_anchors_from_emissions(
        oracle_spans,
        token_ranges,
        frames,
        seconds_per_frame=0.1,
        anchor_pad_seconds=0.3,
        audio_duration_seconds=1.0,
    )
    np.testing.assert_allclose(
        [(item.anchor_start_s, item.anchor_end_s) for item in actual],
        [(0.0, 0.6), (0.5, 1.0)],
    )
    assert [asdict(item) for item in expected] == [asdict(item) for item in actual]


@pytest.mark.parametrize(
    ("gap", "expected_chunk_count"),
    [(0.199999, 1), (0.2, 2)],
)
def test_anchor_merge_threshold_is_strictly_less_than_point_two(
    reference: SimpleNamespace,
    gap: float,
    expected_chunk_count: int,
) -> None:
    second_start = 0.25 + gap
    reference_anchors = [
        reference.WordAnchor(0, "a", 0, 1, 0.0, 0.1, 0.0, 0.25),
        reference.WordAnchor(1, "b", 2, 3, 0.2, 0.3, second_start, second_start + 0.25),
    ]
    oracle_anchors = [
        oracle.WordAnchor(0, "a", 0, 1, 0.0, 0.1, 0.0, 0.25),
        oracle.WordAnchor(1, "b", 2, 3, 0.2, 0.3, second_start, second_start + 0.25),
    ]
    actual = reference.merge_word_anchors_into_chunks(
        reference_anchors,
        anchor_merge_gap_s=0.2,
    )
    expected = oracle.merge_word_anchors_into_chunks(
        oracle_anchors,
        anchor_merge_gap_seconds=0.2,
    )
    assert len(actual) == len(expected) == expected_chunk_count
    assert _chunk_tuples(actual) == _chunk_tuples(expected)


def test_legacy_grid_uses_python_millisecond_rounding(
    reference: SimpleNamespace,
) -> None:
    reference_chunks = [reference.Chunk(0.0015, 0.0045, ["one"], [0])]
    oracle_chunks = [oracle.Chunk(0.0015, 0.0045, ["one"], [0])]
    actual = reference.round_chunks_to_legacy_grid(
        raw_chunks=reference_chunks,
        utt_id="utt",
        words=["one"],
        num_samples=16_000,
        sample_rate=16_000,
    )
    expected = oracle.round_chunks_to_legacy_grid(
        oracle_chunks,
        utterance_id="utt",
        words=["one"],
        num_samples=16_000,
        sample_rate=16_000,
    )
    assert _runtime_chunk_tuples(actual) == _runtime_chunk_tuples(expected)
    assert (actual[0].start_ms, actual[0].end_ms) == (2, 4)
    assert (actual[0].start_sample, actual[0].end_sample) == (32, 64)


def test_legacy_grid_clamps_sub_millisecond_rounded_tail_overflow(
    reference: SimpleNamespace,
) -> None:
    num_samples = 16_009
    audio_duration = num_samples / 16_000.0
    reference_chunks = [reference.Chunk(0.9, audio_duration, ["tail"], [0])]
    oracle_chunks = [oracle.Chunk(0.9, audio_duration, ["tail"], [0])]
    actual = reference.round_chunks_to_legacy_grid(
        raw_chunks=reference_chunks,
        utt_id="utt",
        words=["tail"],
        num_samples=num_samples,
        sample_rate=16_000,
    )
    expected = oracle.round_chunks_to_legacy_grid(
        oracle_chunks,
        utterance_id="utt",
        words=["tail"],
        num_samples=num_samples,
        sample_rate=16_000,
    )
    assert _runtime_chunk_tuples(actual) == _runtime_chunk_tuples(expected)
    assert actual[0].end_ms == 1_000
    assert actual[0].end_sample == 16_000


def test_legacy_grid_requires_exact_word_index_coverage(
    reference: SimpleNamespace,
) -> None:
    reference_chunks = [reference.Chunk(0.0, 0.1, ["a", "b"], [0, 2])]
    oracle_chunks = [oracle.Chunk(0.0, 0.1, ["a", "b"], [0, 2])]
    with pytest.raises(RuntimeError, match="word-index coverage mismatch"):
        reference.round_chunks_to_legacy_grid(
            raw_chunks=reference_chunks,
            utt_id="utt",
            words=["a", "b"],
            num_samples=16_000,
            sample_rate=16_000,
        )
    with pytest.raises(RuntimeError, match="word-index coverage mismatch"):
        oracle.round_chunks_to_legacy_grid(
            oracle_chunks,
            utterance_id="utt",
            words=["a", "b"],
            num_samples=16_000,
            sample_rate=16_000,
        )
