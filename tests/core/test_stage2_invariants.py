"""Independent invariants for the production Stage 2 NumPy core.

This module does not import or execute the frozen reference. Small decoding
cases are compared with the independent unpruned dynamic program in
``tests.characterization.stage2_oracle``.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest

from flexaligner.core import stage2
from tests.characterization.stage2_oracle import exhaustive_viterbi, score_state_path

NO_GAPS = {
    "sil_phone": None,
    "optional_sil_between_words": False,
    "optional_sil_at_start": False,
    "optional_sil_at_end": False,
    "sph_phone": None,
    "optional_sph_between_words": False,
    "optional_sph_at_start": False,
    "optional_sph_at_end": False,
}


def _manual_graph(
    *,
    state_specs: Sequence[tuple[str, int, int | None, str | None]],
    successors: Sequence[Sequence[int]],
    start_states: Sequence[int],
    end_states: Sequence[int],
) -> stage2.PhoneGraph:
    predecessors: list[list[int]] = [[] for _ in state_specs]
    for state_id, next_states in enumerate(successors):
        for next_state in next_states:
            predecessors[next_state].append(state_id)
    states = [
        stage2.PhoneState(
            edge=stage2.EmitEdge(
                u=state_id,
                v=state_id + 1,
                phone=phone,
                phone_id=phone_id,
                word_index=word_index,
                word=word,
            ),
            preds=tuple(predecessors[state_id]),
            succs=tuple(successors[state_id]),
        )
        for state_id, (phone, phone_id, word_index, word) in enumerate(state_specs)
    ]
    return stage2.PhoneGraph(
        states=states,
        start_states=list(start_states),
        end_states=list(end_states),
    )


def _all_complete_state_paths(graph: stage2.PhoneGraph) -> set[tuple[int, ...]]:
    complete: set[tuple[int, ...]] = set()

    def visit(state_id: int, path: tuple[int, ...]) -> None:
        if state_id in path:
            raise AssertionError("pronunciation graph contains a successor cycle")
        next_path = (*path, state_id)
        if state_id in graph.end_states:
            complete.add(next_path)
        for next_state in graph.states[state_id].succs:
            visit(next_state, next_path)

    for start_state in graph.start_states:
        visit(start_state, ())
    return complete


def _assert_graph_consistent(graph: stage2.PhoneGraph, entry_bias: np.ndarray) -> None:
    state_count = len(graph.states)
    assert state_count > 0
    assert entry_bias.shape == (state_count,)
    assert np.issubdtype(entry_bias.dtype, np.floating)
    assert np.isfinite(entry_bias).all()
    assert graph.start_states
    assert graph.end_states
    assert len(set(graph.start_states)) == len(graph.start_states)
    assert len(set(graph.end_states)) == len(graph.end_states)
    assert all(0 <= state < state_count for state in (*graph.start_states, *graph.end_states))
    for state_id, state in enumerate(graph.states):
        assert state.edge.phone
        assert state.edge.phone_id >= 0
        assert len(set(state.preds)) == len(state.preds)
        assert len(set(state.succs)) == len(state.succs)
        assert all(0 <= related < state_count for related in (*state.preds, *state.succs))
        assert all(state_id in graph.states[pred].succs for pred in state.preds)
        assert all(state_id in graph.states[succ].preds for succ in state.succs)
    assert _all_complete_state_paths(graph)


def _assert_alignment_invariants(
    graph: stage2.PhoneGraph,
    alignment: stage2.ViterbiAlignment,
    frame_count: int,
) -> None:
    assert alignment.state_path.shape == (frame_count,)
    assert alignment.aligned_phone_ids.shape == (frame_count,)
    assert np.issubdtype(alignment.state_path.dtype, np.integer)
    assert np.issubdtype(alignment.aligned_phone_ids.dtype, np.integer)
    assert math.isfinite(alignment.score)
    state_path = [int(state) for state in alignment.state_path]
    assert state_path[0] in graph.start_states
    assert state_path[-1] in graph.end_states
    assert all(0 <= state < len(graph.states) for state in state_path)
    for previous, current in itertools.pairwise(state_path):
        assert current == previous or current in graph.states[previous].succs
    expected_phone_ids = [graph.states[state].edge.phone_id for state in state_path]
    assert alignment.aligned_phone_ids.tolist() == expected_phone_ids
    for segments in (alignment.phone_segments_f, alignment.word_segments_f):
        assert segments
        assert segments[0][1] == 0
        assert segments[-1][2] == frame_count
        assert all(label and end > start >= 0 for label, start, end in segments)
        assert all(left[2] == right[1] for left, right in itertools.pairwise(segments))


def test_multi_pronunciation_graph_is_a_consistent_finite_dag() -> None:
    graph, entry_bias = stage2.build_phone_graph_optional_sil_sph(
        ["read"],
        {"read": [["R", "IY", "D"], ["R", "EH", "D"]]},
        {"R": 0, "IY": 1, "EH": 2, "D": 3},
        **NO_GAPS,
    )

    _assert_graph_consistent(graph, entry_bias)
    phone_paths = {
        tuple(graph.states[state].edge.phone for state in path)
        for path in _all_complete_state_paths(graph)
    }
    assert phone_paths == {("R", "IY", "D"), ("R", "EH", "D")}


@pytest.mark.parametrize(
    ("words", "lexicon", "vocabulary", "options", "error"),
    [
        ([], {}, {"A": 0}, NO_GAPS, ValueError),
        (["word"], {"word": [["A"]]}, {}, NO_GAPS, ValueError),
        (["word"], {"word": []}, {"A": 0}, NO_GAPS, RuntimeError),
        (["word"], {"word": [[]]}, {"A": 0}, NO_GAPS, RuntimeError),
        (["word"], {"word": [["UNKNOWN"]]}, {"A": 0}, NO_GAPS, KeyError),
        (
            ["word"],
            {"word": [["A"]]},
            {"A": 0},
            {**NO_GAPS, "sil_cost": np.nan},
            ValueError,
        ),
        (
            ["word"],
            {"word": [["A"]]},
            {"A": 0},
            {**NO_GAPS, "sph_cost": np.inf},
            ValueError,
        ),
    ],
)
def test_graph_builder_rejects_invalid_or_non_finite_inputs(
    words: Sequence[str],
    lexicon: Any,
    vocabulary: Any,
    options: dict[str, Any],
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        stage2.build_phone_graph_optional_sil_sph(
            words,
            lexicon,
            vocabulary,
            **options,
        )


def _branching_graph() -> stage2.PhoneGraph:
    return _manual_graph(
        state_specs=[
            ("A", 0, 0, "one"),
            ("B", 1, 0, "one"),
            ("C", 2, 0, "one"),
            ("D", 3, 1, "two"),
        ],
        successors=[(1, 2), (3,), (3,), ()],
        start_states=[0],
        end_states=[3],
    )


@pytest.mark.parametrize("boundary_lambda", [0.0, 0.4], ids=["no-boundary", "boundary"])
def test_wide_beam_matches_unpruned_exact_dp_for_fixed_seed_random_scores(
    boundary_lambda: float,
) -> None:
    graph = _branching_graph()
    entry_bias = np.asarray([0.1, -0.2, 0.3, -0.1], dtype=np.float32)
    rng = np.random.default_rng(20260811)
    for frame_count in range(3, 8):
        logp = rng.normal(size=(frame_count, 4)).astype(np.float32)
        options = {
            "p_stay": 0.67,
            "boundary_lambda": boundary_lambda,
            "boundary_context_s": 0.02,
            "frame_hop_s": 0.01,
        }

        expected = exhaustive_viterbi(
            graph=graph,
            logp=logp,
            entry_bias=entry_bias,
            **options,
        )
        actual = stage2.align_beam_viterbi(
            logp,
            graph,
            entry_bias,
            beam_size=len(graph.states),
            **options,
        )
        rescored = score_state_path(
            graph=graph,
            state_path=actual.state_path,
            logp=logp,
            entry_bias=entry_bias,
            **options,
        )

        _assert_alignment_invariants(graph, actual, frame_count)
        assert actual.state_path.tolist() == expected.state_path.tolist()
        assert actual.score == pytest.approx(expected.score, abs=2e-5)
        assert rescored == pytest.approx(actual.score, abs=2e-5)


def test_narrow_beam_fails_closed_instead_of_returning_dead_end() -> None:
    graph = _manual_graph(
        state_specs=[
            ("DEAD", 0, None, None),
            ("A", 1, 0, "word"),
            ("B", 2, 0, "word"),
        ],
        successors=[(), (2,), ()],
        start_states=[0, 1],
        end_states=[2],
    )
    logp = np.asarray([[5.0, 0.0, 0.0], [0.0, 0.0, 5.0]], dtype=np.float32)
    entry_bias = np.zeros(3, dtype=np.float32)

    with pytest.raises(RuntimeError, match="failed to reach any end state"):
        stage2.align_beam_viterbi(
            logp,
            graph,
            entry_bias,
            p_stay=0.5,
            beam_size=1,
        )

    complete = stage2.align_beam_viterbi(
        logp,
        graph,
        entry_bias,
        p_stay=0.5,
        beam_size=3,
    )
    _assert_alignment_invariants(graph, complete, frame_count=2)
    assert complete.state_path.tolist() == [1, 2]


@pytest.mark.parametrize("start_order", [[0, 1], [1, 0]])
def test_equal_score_beam_one_stably_preserves_start_order(
    start_order: list[int],
) -> None:
    graph = _manual_graph(
        state_specs=[("A", 0, 0, "a"), ("B", 1, 1, "b")],
        successors=[(), ()],
        start_states=start_order,
        end_states=start_order,
    )

    alignment = stage2.align_beam_viterbi(
        np.zeros((2, 2), dtype=np.float32),
        graph,
        np.zeros(2, dtype=np.float32),
        p_stay=0.5,
        beam_size=1,
    )

    assert alignment.state_path.tolist() == [start_order[0], start_order[0]]


def test_repeated_word_labels_remain_distinct_by_word_index() -> None:
    graph, entry_bias = stage2.build_phone_graph_optional_sil_sph(
        ["go", "go"],
        {"go": [["G"]]},
        {"G": 0},
        **NO_GAPS,
    )

    alignment = stage2.align_beam_viterbi(
        np.zeros((2, 1), dtype=np.float32),
        graph,
        entry_bias,
        p_stay=0.5,
        beam_size=len(graph.states),
    )

    _assert_alignment_invariants(graph, alignment, frame_count=2)
    assert alignment.word_segments_f == [("go", 0, 1), ("go", 1, 2)]
    assert alignment.phone_segments_f == [("G", 0, 1), ("G", 1, 2)]
    assert [graph.states[state].edge.word_index for state in alignment.state_path] == [0, 1]


def test_silence_lock_matches_exact_dp_and_blocks_early_exit() -> None:
    graph = _manual_graph(
        state_specs=[
            ("A", 0, 0, "left"),
            ("sil", 1, None, None),
            ("B", 2, 1, "right"),
        ],
        successors=[(1,), (2,), ()],
        start_states=[0],
        end_states=[2],
    )
    logp = np.asarray(
        [
            [0.0, -10.0, -10.0],
            [-10.0, 0.0, -10.0],
            [-10.0, -5.0, 5.0],
            [-10.0, -5.0, 5.0],
            [-10.0, -5.0, 5.0],
            [-10.0, -5.0, 5.0],
            [-10.0, -5.0, 5.0],
            [-10.0, -5.0, 5.0],
            [-10.0, -5.0, 5.0],
        ],
        dtype=np.float32,
    )
    entry_bias = np.zeros(3, dtype=np.float32)
    options = {
        "p_stay": 0.5,
        "sil_phone_id": 1,
        "min_sil_dur_ms": 65.0,
        "frame_hop_s": 0.01,
    }

    expected = exhaustive_viterbi(
        graph=graph,
        logp=logp,
        entry_bias=entry_bias,
        **options,
    )
    actual = stage2.align_beam_viterbi(
        logp,
        graph,
        entry_bias,
        beam_size=32,
        **options,
    )

    _assert_alignment_invariants(graph, actual, frame_count=9)
    assert actual.state_path.tolist() == expected.state_path.tolist()
    assert actual.score == pytest.approx(expected.score)
    minimum_silence_frames = max(1, round(65.0 / 1000.0 / 0.01))
    assert np.count_nonzero(actual.aligned_phone_ids == 1) >= minimum_silence_frames


@pytest.mark.parametrize("gap_kind", ["sil", "sph"])
def test_gap_enter_cost_is_charged_once_and_matches_exact_score(gap_kind: str) -> None:
    graph = _manual_graph(
        state_specs=[
            ("A", 0, 0, "left"),
            ("B", 2, 1, "right"),
            (gap_kind, 1, None, None if gap_kind == "sil" else "[missing]"),
        ],
        successors=[(1, 2), (), (1,)],
        start_states=[0],
        end_states=[1],
    )
    logp = np.zeros((4, 3), dtype=np.float32)
    logp[1:3, 1] = 0.5
    entry_bias = np.zeros(3, dtype=np.float32)
    options = {
        "p_stay": 0.5,
        "sil_phone_id": 1 if gap_kind == "sil" else None,
        "sil_enter_cost": -0.75 if gap_kind == "sil" else 0.0,
        "sph_phone_id": 1 if gap_kind == "sph" else None,
        "sph_enter_cost": -0.75 if gap_kind == "sph" else 0.0,
    }

    expected = exhaustive_viterbi(
        graph=graph,
        logp=logp,
        entry_bias=entry_bias,
        **options,
    )
    actual = stage2.align_beam_viterbi(
        logp,
        graph,
        entry_bias,
        beam_size=16,
        **options,
    )

    assert actual.state_path.tolist() == expected.state_path.tolist() == [0, 2, 2, 1]
    assert actual.score == pytest.approx(expected.score)
    assert actual.score == pytest.approx(1.0 - 0.75 + 4.0 * math.log(0.5))


def test_boundary_contrast_moves_to_the_sharper_boundary() -> None:
    graph = _manual_graph(
        state_specs=[("A", 0, 0, "word"), ("B", 1, 0, "word")],
        successors=[(1,), ()],
        start_states=[0],
        end_states=[1],
    )
    contrast = np.asarray([2, 2, -2, 1, 1, 1, -1, -1], dtype=np.float32)
    logp = np.column_stack((contrast / 2.0, -contrast / 2.0)).astype(np.float32)
    entry_bias = np.zeros(2, dtype=np.float32)

    without = stage2.align_beam_viterbi(
        logp,
        graph,
        entry_bias,
        p_stay=0.5,
        beam_size=2,
        boundary_lambda=0.0,
        boundary_context_s=0.02,
        frame_hop_s=0.01,
    )
    with_contrast = stage2.align_beam_viterbi(
        logp,
        graph,
        entry_bias,
        p_stay=0.5,
        beam_size=2,
        boundary_lambda=3.0,
        boundary_context_s=0.02,
        frame_hop_s=0.01,
    )

    assert int(np.flatnonzero(without.state_path == 1)[0]) == 6
    assert int(np.flatnonzero(with_contrast.state_path == 1)[0]) == 2


def _fixed_spec(
    phone: str,
    phone_id: int,
    word_index: int | None,
    word: str | None,
    *,
    bias: float = 0.0,
) -> stage2.FixedStateSpec:
    return stage2.FixedStateSpec(phone, phone_id, word_index, word, bias)


@pytest.mark.parametrize(
    ("gap_phone", "duration_frames", "expected_kept"),
    [
        ("sil", 6, False),
        ("sil", 7, True),
        ("sph", 4, False),
        ("sph", 5, True),
    ],
)
def test_internal_gap_pruning_uses_ceiling_frame_thresholds(
    gap_phone: str,
    duration_frames: int,
    expected_kept: bool,
) -> None:
    gap_word = None if gap_phone == "sil" else "[missing]"
    segments = [
        (_fixed_spec("A", 0, 0, "left"), 0, 1),
        (_fixed_spec(gap_phone, 1, None, gap_word), 1, 1 + duration_frames),
        (
            _fixed_spec("B", 2, 1, "right"),
            1 + duration_frames,
            2 + duration_frames,
        ),
    ]

    kept, stats = stage2.prune_short_internal_sil_sph_segments(
        segments,
        sil_phone="sil",
        sph_phone="sph",
        min_sil_dur_ms=65.0,
        min_sph_dur_ms=50.0,
        frame_hop_s=0.01,
    )

    assert any(spec.phone == gap_phone for spec in kept) is expected_kept
    expected_drop_sil = int(gap_phone == "sil" and not expected_kept)
    expected_drop_sph = int(gap_phone == "sph" and not expected_kept)
    assert stats == stage2.RedecodeStats(
        first_pass_states=3,
        fixed_states=3 if expected_kept else 2,
        dropped_short_sil=expected_drop_sil,
        dropped_short_sph=expected_drop_sph,
    )


def test_short_boundary_gap_states_are_never_pruned() -> None:
    segments = [
        (_fixed_spec("sil", 0, None, None), 0, 1),
        (_fixed_spec("A", 1, 0, "word"), 1, 2),
        (_fixed_spec("sph", 2, None, "[missing]"), 2, 3),
    ]

    kept, stats = stage2.prune_short_internal_sil_sph_segments(
        segments,
        sil_phone="sil",
        sph_phone="sph",
        min_sil_dur_ms=1000.0,
        min_sph_dur_ms=1000.0,
        frame_hop_s=0.01,
    )

    assert [spec.phone for spec in kept] == ["sil", "A", "sph"]
    assert stats == stage2.RedecodeStats(3, 3, 0, 0)


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("min_sil_dur_ms", -1.0),
        ("min_sil_dur_ms", np.nan),
        ("min_sil_dur_ms", np.inf),
        ("min_sph_dur_ms", -1.0),
        ("min_sph_dur_ms", np.nan),
        ("min_sph_dur_ms", np.inf),
        ("frame_hop_s", 0.0),
        ("frame_hop_s", np.nan),
        ("frame_hop_s", np.inf),
    ],
)
def test_pruning_rejects_invalid_thresholds_and_frame_hop(
    option: str,
    value: float,
) -> None:
    options = {
        "sil_phone": "sil",
        "sph_phone": "sph",
        "min_sil_dur_ms": 65.0,
        "min_sph_dur_ms": 50.0,
        "frame_hop_s": 0.01,
    }
    options[option] = value
    with pytest.raises(ValueError):
        stage2.prune_short_internal_sil_sph_segments(
            [(_fixed_spec("A", 0, 0, "word"), 0, 1)],
            **options,
        )


@pytest.mark.parametrize(
    ("segments", "error"),
    [
        ([], RuntimeError),
        ([(_fixed_spec("A", 0, 0, "word"), 1, 2)], ValueError),
        ([(_fixed_spec("A", 0, 0, "word"), 0, 0)], RuntimeError),
        (
            [
                (_fixed_spec("A", 0, 0, "word"), 0, 1),
                (_fixed_spec("B", 1, 1, "next"), 2, 3),
            ],
            ValueError,
        ),
        ([(_fixed_spec("A", 0, 0, "word", bias=np.nan), 0, 1)], ValueError),
    ],
)
def test_pruning_rejects_malformed_or_noncontiguous_segments(
    segments: list[stage2.StateSegment],
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        stage2.prune_short_internal_sil_sph_segments(
            segments,
            sil_phone="sil",
            sph_phone="sph",
            min_sil_dur_ms=65.0,
            min_sph_dur_ms=50.0,
            frame_hop_s=0.01,
        )


def test_fixed_sequence_graph_is_linear_and_preserves_metadata_and_bias() -> None:
    specs = [
        _fixed_spec("A", 0, 0, "left", bias=0.1),
        _fixed_spec("sil", 1, None, None, bias=-0.5),
        _fixed_spec("B", 2, 1, "right", bias=0.2),
    ]

    graph, entry_bias = stage2.build_fixed_sequence_graph(specs)

    _assert_graph_consistent(graph, entry_bias)
    assert graph.start_states == [0]
    assert graph.end_states == [2]
    assert [state.preds for state in graph.states] == [(), (0,), (1,)]
    assert [state.succs for state in graph.states] == [(1,), (2,), ()]
    assert [state.edge.phone for state in graph.states] == ["A", "sil", "B"]
    assert [state.edge.word_index for state in graph.states] == [0, None, 1]
    assert entry_bias.dtype == np.float32
    assert entry_bias.tolist() == pytest.approx([0.1, -0.5, 0.2])


def test_fixed_redecode_drops_short_gap_and_matches_exact_dp() -> None:
    first_graph = _manual_graph(
        state_specs=[
            ("A", 0, 0, "left"),
            ("sil", 1, None, None),
            ("B", 2, 1, "right"),
        ],
        successors=[(1,), (2,), ()],
        start_states=[0],
        end_states=[2],
    )
    first_alignment = stage2.ViterbiAlignment(
        phone_segments_f=[("A", 0, 1), ("sil", 1, 2), ("B", 2, 3)],
        word_segments_f=[("left", 0, 1), ("sil", 1, 2), ("right", 2, 3)],
        state_path=np.asarray([0, 1, 2], dtype=np.int32),
        aligned_phone_ids=np.asarray([0, 1, 2], dtype=np.int32),
        score=0.0,
    )
    first_bias = np.asarray([0.0, -0.5, 0.0], dtype=np.float32)
    logp = np.asarray(
        [[0.0, -10.0, -10.0], [0.0, -10.0, -10.0], [-10.0, -10.0, 0.0]],
        dtype=np.float32,
    )
    config = stage2.Stage2DecodeConfig(
        p_stay=0.5,
        beam=16,
        boundary_lambda=0.0,
        boundary_context_s=0.02,
        frame_hop_s=0.01,
        min_sil_dur_ms=20.0,
        min_sph_dur_ms=50.0,
    )

    actual, stats = stage2.redecode_with_pruned_fixed_sequence(
        first_pass_ali=first_alignment,
        first_pass_graph=first_graph,
        first_pass_entry_bias=first_bias,
        logp=logp,
        sil_phone="sil",
        sil_phone_id=1,
        sph_phone="sph",
        sph_phone_id=None,
        config=config,
    )
    fixed_graph, fixed_bias = stage2.build_fixed_sequence_graph(
        [_fixed_spec("A", 0, 0, "left"), _fixed_spec("B", 2, 1, "right")]
    )
    expected = exhaustive_viterbi(
        graph=fixed_graph,
        logp=logp,
        entry_bias=fixed_bias,
        p_stay=config.p_stay,
        boundary_lambda=config.boundary_lambda,
        boundary_context_s=config.boundary_context_s,
        frame_hop_s=config.frame_hop_s,
        sil_phone_id=1,
    )

    _assert_alignment_invariants(fixed_graph, actual, frame_count=3)
    assert stats == stage2.RedecodeStats(3, 2, 1, 0)
    assert actual.state_path.tolist() == expected.state_path.tolist() == [0, 0, 1]
    assert actual.score == pytest.approx(expected.score)
    assert actual.aligned_phone_ids.tolist() == [0, 0, 2]
    assert actual.phone_segments_f == [("A", 0, 2), ("B", 2, 3)]
    assert actual.word_segments_f == [("left", 0, 2), ("right", 2, 3)]


def _baseline_decode_inputs() -> tuple[stage2.PhoneGraph, np.ndarray, np.ndarray]:
    graph = _manual_graph(
        state_specs=[("A", 0, 0, "word"), ("B", 1, 0, "word")],
        successors=[(1,), ()],
        start_states=[0],
        end_states=[1],
    )
    return graph, np.zeros((3, 2), dtype=np.float32), np.zeros(2, dtype=np.float32)


@pytest.mark.parametrize(
    ("bad_logp", "error"),
    [
        ([[0.0, 0.0]], TypeError),
        (np.zeros(2, dtype=np.float32), ValueError),
        (np.zeros((1, 1, 1), dtype=np.float32), ValueError),
        (np.zeros((0, 2), dtype=np.float32), ValueError),
        (np.zeros((2, 0), dtype=np.float32), ValueError),
        (np.zeros((2, 2), dtype=np.int32), TypeError),
        (np.zeros((2, 2), dtype=np.complex64), TypeError),
        (np.asarray([[0.0, np.nan], [0.0, 0.0]], dtype=np.float32), ValueError),
        (np.asarray([[0.0, np.inf], [0.0, 0.0]], dtype=np.float32), ValueError),
        (np.asarray([[0.0, -np.inf], [0.0, 0.0]], dtype=np.float32), ValueError),
    ],
)
def test_decoder_rejects_invalid_or_non_finite_posterior_arrays(
    bad_logp: Any,
    error: type[Exception],
) -> None:
    graph, _logp, entry_bias = _baseline_decode_inputs()
    with pytest.raises(error):
        stage2.align_beam_viterbi(bad_logp, graph, entry_bias)


@pytest.mark.parametrize(
    ("bad_bias", "error"),
    [
        ([0.0, 0.0], TypeError),
        (np.zeros((1, 2), dtype=np.float32), ValueError),
        (np.zeros(1, dtype=np.float32), ValueError),
        (np.zeros(2, dtype=np.int32), TypeError),
        (np.asarray([0.0, np.nan], dtype=np.float32), ValueError),
        (np.asarray([0.0, np.inf], dtype=np.float32), ValueError),
    ],
)
def test_decoder_rejects_invalid_or_non_finite_entry_bias(
    bad_bias: Any,
    error: type[Exception],
) -> None:
    graph, logp, _entry_bias = _baseline_decode_inputs()
    with pytest.raises(error):
        stage2.align_beam_viterbi(logp, graph, bad_bias)


@pytest.mark.parametrize(
    ("option", "value", "error"),
    [
        ("beam_size", True, TypeError),
        ("beam_size", 1.5, TypeError),
        ("beam_size", 0, ValueError),
        ("beam_size", -1, ValueError),
        ("p_stay", True, TypeError),
        ("p_stay", 0.0, ValueError),
        ("p_stay", 1.0, ValueError),
        ("p_stay", np.nan, ValueError),
        ("p_stay", np.inf, ValueError),
        ("frame_hop_s", 0.0, ValueError),
        ("frame_hop_s", -0.01, ValueError),
        ("frame_hop_s", np.nan, ValueError),
        ("frame_hop_s", np.inf, ValueError),
        ("boundary_context_s", 0.0, ValueError),
        ("boundary_context_s", np.nan, ValueError),
        ("boundary_lambda", np.nan, ValueError),
        ("boundary_lambda", np.inf, ValueError),
        ("min_sil_dur_ms", -1.0, ValueError),
        ("min_sil_dur_ms", np.nan, ValueError),
        ("sil_enter_cost", np.nan, ValueError),
        ("sph_enter_cost", np.inf, ValueError),
        ("sil_phone_id", -1, ValueError),
        ("sil_phone_id", 2, ValueError),
        ("sph_phone_id", True, TypeError),
    ],
)
def test_decoder_rejects_invalid_scalar_boundaries(
    option: str,
    value: Any,
    error: type[Exception],
) -> None:
    graph, logp, entry_bias = _baseline_decode_inputs()
    with pytest.raises(error):
        stage2.align_beam_viterbi(logp, graph, entry_bias, **{option: value})


def test_decoder_rejects_invalid_graph_relations_ids_cycles_and_phone_ids() -> None:
    _graph, logp, entry_bias = _baseline_decode_inputs()
    empty = stage2.PhoneGraph(states=[], start_states=[0], end_states=[0])
    with pytest.raises(ValueError, match="states must not be empty"):
        stage2.align_beam_viterbi(logp, empty, np.zeros(0, dtype=np.float32))

    duplicate_start = _manual_graph(
        state_specs=[("A", 0, 0, "word"), ("B", 1, 0, "word")],
        successors=[(1,), ()],
        start_states=[0, 0],
        end_states=[1],
    )
    with pytest.raises(ValueError, match="duplicate"):
        stage2.align_beam_viterbi(logp, duplicate_start, entry_bias)

    out_of_range = stage2.PhoneGraph(
        states=[
            stage2.PhoneState(stage2.EmitEdge(0, 1, "A", 0, 0, "word"), (), (2,)),
            stage2.PhoneState(stage2.EmitEdge(1, 2, "B", 1, 0, "word"), (), ()),
        ],
        start_states=[0],
        end_states=[1],
    )
    with pytest.raises(ValueError, match="relation out of range"):
        stage2.align_beam_viterbi(logp, out_of_range, entry_bias)

    mismatch = stage2.PhoneGraph(
        states=[
            stage2.PhoneState(stage2.EmitEdge(0, 1, "A", 0, 0, "word"), (), (1,)),
            stage2.PhoneState(stage2.EmitEdge(1, 2, "B", 1, 0, "word"), (), ()),
        ],
        start_states=[0],
        end_states=[1],
    )
    with pytest.raises(ValueError, match="successor/predecessor mismatch"):
        stage2.align_beam_viterbi(logp, mismatch, entry_bias)

    bad_phone = _manual_graph(
        state_specs=[("A", 0, 0, "word"), ("B", 2, 0, "word")],
        successors=[(1,), ()],
        start_states=[0],
        end_states=[1],
    )
    with pytest.raises(ValueError, match="phone ID out of range"):
        stage2.align_beam_viterbi(logp, bad_phone, entry_bias)

    cycle = _manual_graph(
        state_specs=[("A", 0, 0, "word"), ("B", 1, 0, "word")],
        successors=[(1,), (0,)],
        start_states=[0],
        end_states=[1],
    )
    with pytest.raises(ValueError, match="cycle"):
        stage2.align_beam_viterbi(logp, cycle, entry_bias)


def test_redecode_clears_gap_constraints_and_returns_typed_stats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _manual_graph(
        state_specs=[("sil", 0, None, None)],
        successors=[()],
        start_states=[0],
        end_states=[0],
    )
    first_pass = stage2.ViterbiAlignment(
        phone_segments_f=[("sil", 0, 2)],
        word_segments_f=[("sil", 0, 2)],
        state_path=np.asarray([0, 0], dtype=np.int32),
        aligned_phone_ids=np.asarray([0, 0], dtype=np.int32),
        score=0.0,
    )
    captured: dict[str, Any] = {}

    def capture_decoder(**kwargs: Any) -> stage2.ViterbiAlignment:
        captured.update(kwargs)
        return first_pass

    monkeypatch.setattr(stage2, "align_beam_viterbi", capture_decoder)
    _alignment, stats = stage2.redecode_with_pruned_fixed_sequence(
        first_pass_ali=first_pass,
        first_pass_graph=graph,
        first_pass_entry_bias=np.zeros(1, dtype=np.float32),
        logp=np.zeros((2, 1), dtype=np.float32),
        sil_phone="sil",
        sil_phone_id=0,
        sph_phone="sph",
        sph_phone_id=None,
        config=stage2.Stage2DecodeConfig(),
    )

    assert captured["min_sil_dur_ms"] == 0.0
    assert captured["sil_enter_cost"] == 0.0
    assert captured["sph_enter_cost"] == 0.0
    assert isinstance(stats, stage2.RedecodeStats)
