"""Model-free characterization of the vendored Stage 2 reference algorithm."""

from __future__ import annotations

import math
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

from tests.characterization.reference_loader import load_reference_module
from tests.characterization.stage2_oracle import exhaustive_viterbi, score_state_path


@pytest.fixture(scope="module")
def reference() -> ModuleType:
    return load_reference_module()


def _pronouncing_dictionary(reference: ModuleType, **entries: list[list[str]]) -> Any:
    return reference.PronouncingDictionary(lex=entries)


def _manual_graph(
    reference: ModuleType,
    *,
    state_specs: list[tuple[str, int, int | None, str | None]],
    successors: list[tuple[int, ...]],
    start_states: list[int],
    end_states: list[int],
) -> Any:
    predecessors: list[list[int]] = [[] for _ in state_specs]
    for state_id, next_states in enumerate(successors):
        for next_state in next_states:
            predecessors[next_state].append(state_id)

    states = []
    for state_id, (phone, phone_id, word_index, word) in enumerate(state_specs):
        edge = reference.EmitEdge(
            u=state_id,
            v=state_id + 1,
            phone=phone,
            phone_id=phone_id,
            word_index=word_index,
            word=word,
        )
        states.append(
            reference.PhoneState(
                edge=edge,
                preds=tuple(predecessors[state_id]),
                succs=successors[state_id],
            )
        )
    return reference.PhoneGraph(
        states=states,
        start_states=start_states,
        end_states=end_states,
    )


def _all_complete_phone_paths(graph: Any) -> set[tuple[str, ...]]:
    complete: set[tuple[str, ...]] = set()

    def visit(state_id: int, path: tuple[int, ...]) -> None:
        if state_id in path:
            raise AssertionError("The pronunciation graph unexpectedly contains a cycle")
        next_path = (*path, state_id)
        if state_id in graph.end_states:
            complete.add(tuple(graph.states[index].edge.phone for index in next_path))
        for next_state in graph.states[state_id].succs:
            visit(next_state, next_path)

    for start_state in graph.start_states:
        visit(start_state, ())
    return complete


def _transition_frame(state_path: np.ndarray, target_state: int) -> int:
    frames = np.flatnonzero(state_path == target_state)
    if frames.size == 0:
        raise AssertionError(f"state {target_state} was not visited: {state_path.tolist()}")
    return int(frames[0])


def _fixed_spec(
    reference: ModuleType,
    phone: str,
    phone_id: int,
    word_index: int | None,
    word: str | None,
    *,
    bias: float = 0.0,
) -> Any:
    return reference.FixedStateSpec(
        phone=phone,
        phone_id=phone_id,
        word_index=word_index,
        word=word,
        bias=bias,
    )


def test_epsilon_closure_is_transitive_and_includes_each_node(reference: ModuleType) -> None:
    closure = reference._eps_closure(
        6,
        [
            [1, 3],
            [2],
            [],
            [4],
            [5],
            [],
        ],
    )

    assert closure == [
        {0, 1, 2, 3, 4, 5},
        {1, 2},
        {2},
        {3, 4, 5},
        {4, 5},
        {5},
    ]


def test_all_pronunciations_form_distinct_complete_dag_paths(reference: ModuleType) -> None:
    dictionary = _pronouncing_dictionary(
        reference,
        read=[["R", "IY", "D"], ["R", "EH", "D"]],
    )
    graph, entry_bias = reference.build_phone_graph_optional_sil_sph(
        ["read"],
        dictionary,
        {"R": 0, "IY": 1, "EH": 2, "D": 3},
        sil_phone=None,
        optional_sil_between_words=False,
        optional_sil_at_start=False,
        optional_sil_at_end=False,
        sph_phone=None,
        optional_sph_between_words=False,
        optional_sph_at_start=False,
        optional_sph_at_end=False,
    )

    assert _all_complete_phone_paths(graph) == {
        ("R", "IY", "D"),
        ("R", "EH", "D"),
    }
    assert entry_bias.tolist() == [0.0] * 6
    assert len(graph.start_states) == 2
    assert len(graph.end_states) == 2


def test_internal_gap_has_exactly_six_silence_and_speech_paths(reference: ModuleType) -> None:
    dictionary = _pronouncing_dictionary(
        reference,
        left=[["L"]],
        right=[["R"]],
    )
    graph, _ = reference.build_phone_graph_optional_sil_sph(
        ["left", "right"],
        dictionary,
        {"L": 0, "R": 1, "sil": 2, "sph": 3},
        sil_phone="sil",
        optional_sil_between_words=True,
        optional_sil_at_start=False,
        optional_sil_at_end=False,
        sph_phone="sph",
        optional_sph_between_words=True,
        optional_sph_at_start=False,
        optional_sph_at_end=False,
    )
    left_state = next(
        state_id for state_id, state in enumerate(graph.states) if state.edge.word_index == 0
    )

    gap_paths: set[tuple[str, ...]] = set()

    def visit(state_id: int, gap_phones: tuple[str, ...], seen: frozenset[int]) -> None:
        if state_id in seen:
            raise AssertionError("Internal gap unexpectedly contains a cycle")
        for next_state_id in graph.states[state_id].succs:
            edge = graph.states[next_state_id].edge
            if edge.word_index == 1:
                gap_paths.add(gap_phones)
            elif edge.word_index is None:
                visit(
                    next_state_id,
                    (*gap_phones, edge.phone),
                    seen | {state_id},
                )

    visit(left_state, (), frozenset())
    assert gap_paths == {
        (),
        ("sil",),
        ("sph",),
        ("sil", "sph"),
        ("sph", "sil"),
        ("sil", "sph", "sil"),
    }


def test_sph_is_reachable_at_both_utterance_boundaries(reference: ModuleType) -> None:
    dictionary = _pronouncing_dictionary(reference, word=[["W"]])
    graph, _ = reference.build_phone_graph_optional_sil_sph(
        ["word"],
        dictionary,
        {"W": 0, "sph": 1},
        sil_phone=None,
        optional_sil_between_words=False,
        optional_sil_at_start=False,
        optional_sil_at_end=False,
        sph_phone="sph",
        optional_sph_between_words=False,
        optional_sph_at_start=True,
        optional_sph_at_end=True,
    )

    assert any(graph.states[state].edge.phone == "sph" for state in graph.start_states)
    assert any(graph.states[state].edge.phone == "sph" for state in graph.end_states)
    assert _all_complete_phone_paths(graph) == {
        ("W",),
        ("sph", "W"),
        ("W", "sph"),
        ("sph", "W", "sph"),
    }


def test_graph_builder_rejects_oov_words(reference: ModuleType) -> None:
    dictionary = _pronouncing_dictionary(reference, known=[["K"]])

    with pytest.raises(KeyError, match="Word not in lexicon"):
        reference.build_phone_graph_optional_sil_sph(
            ["missing"],
            dictionary,
            {"K": 0},
            sil_phone=None,
            optional_sil_between_words=False,
            sph_phone=None,
            optional_sph_between_words=False,
        )


@pytest.mark.parametrize(
    ("dictionary_entries", "vocabulary", "gap_options", "missing_phone"),
    [
        ({"word": [["UNKNOWN"]]}, {"W": 0}, {}, "UNKNOWN"),
        (
            {"word": [["W"]]},
            {"W": 0},
            {"sil_phone": "sil", "optional_sil_at_start": True},
            "sil",
        ),
        (
            {"word": [["W"]]},
            {"W": 0},
            {"sph_phone": "sph", "optional_sph_at_end": True},
            "sph",
        ),
    ],
)
def test_graph_builder_rejects_every_emitting_phone_missing_from_vocab(
    reference: ModuleType,
    dictionary_entries: dict[str, list[list[str]]],
    vocabulary: dict[str, int],
    gap_options: dict[str, object],
    missing_phone: str,
) -> None:
    dictionary = reference.PronouncingDictionary(lex=dictionary_entries)
    options: dict[str, object] = {
        "sil_phone": None,
        "optional_sil_between_words": False,
        "optional_sil_at_start": False,
        "optional_sil_at_end": False,
        "sph_phone": None,
        "optional_sph_between_words": False,
        "optional_sph_at_start": False,
        "optional_sph_at_end": False,
    }
    options.update(gap_options)

    with pytest.raises(KeyError, match=missing_phone):
        reference.build_phone_graph_optional_sil_sph(
            ["word"],
            dictionary,
            vocabulary,
            **options,
        )


@pytest.mark.parametrize(
    ("emission_advantage", "expected_path"),
    [
        (1.3, [0, 0]),
        (1.5, [1, 2]),
    ],
)
def test_stay_move_tradeoff_uses_log_probabilities(
    reference: ModuleType,
    emission_advantage: float,
    expected_path: list[int],
) -> None:
    graph = _manual_graph(
        reference,
        state_specs=[
            ("X", 0, None, None),
            ("X", 0, None, None),
            ("Y", 1, None, None),
        ],
        successors=[(), (2,), ()],
        start_states=[0, 1],
        end_states=[0, 2],
    )
    logp = np.asarray([[0.0, 0.0], [0.0, emission_advantage]], dtype=np.float32)

    alignment = reference.align_beam_viterbi(
        logp,
        graph,
        np.zeros(3, dtype=np.float32),
        p_stay=0.8,
        beam_size=8,
    )

    assert alignment.state_path.tolist() == expected_path
    assert math.log(0.8 / 0.2) == pytest.approx(math.log(4.0))


def test_entry_bias_is_charged_on_every_frame_not_only_entry(reference: ModuleType) -> None:
    graph = _manual_graph(
        reference,
        state_specs=[
            ("A", 0, 0, "word"),
            ("B", 1, 0, "word"),
        ],
        successors=[(1,), ()],
        start_states=[0],
        end_states=[1],
    )
    logp = np.zeros((4, 2), dtype=np.float32)
    entry_bias = np.asarray([1.0, 0.0], dtype=np.float32)

    alignment = reference.align_beam_viterbi(
        logp,
        graph,
        entry_bias,
        p_stay=0.5,
        beam_size=8,
    )

    assert alignment.state_path.tolist() == [0, 0, 0, 1]
    assert score_state_path(
        graph=graph,
        state_path=alignment.state_path,
        logp=logp,
        entry_bias=entry_bias,
        p_stay=0.5,
    ) == pytest.approx(3.0 + 4.0 * math.log(0.5))


def test_silence_entry_bias_is_per_frame(reference: ModuleType) -> None:
    graph = _manual_graph(
        reference,
        state_specs=[
            ("A", 0, 0, "left"),
            ("B", 2, 1, "right"),
            ("sil", 1, None, None),
        ],
        successors=[(1, 2), (), (1,)],
        start_states=[0],
        end_states=[1],
    )
    logp = np.zeros((4, 3), dtype=np.float32)
    logp[1:3, 1] = 0.3

    alignment = reference.align_beam_viterbi(
        logp,
        graph,
        np.asarray([0.0, 0.0, -0.4], dtype=np.float32),
        p_stay=0.5,
        beam_size=16,
        sil_phone_id=1,
    )

    assert 2 not in alignment.state_path


@pytest.mark.parametrize("gap_kind", ["sil", "sph"])
def test_gap_enter_cost_is_independent_and_charged_once(
    reference: ModuleType,
    gap_kind: str,
) -> None:
    graph = _manual_graph(
        reference,
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
    sil_phone_id = 1 if gap_kind == "sil" else None
    sil_enter_cost = -0.75 if gap_kind == "sil" else 0.0
    sph_phone_id = 1 if gap_kind == "sph" else None
    sph_enter_cost = -0.75 if gap_kind == "sph" else 0.0

    alignment = reference.align_beam_viterbi(
        logp,
        graph,
        np.zeros(3, dtype=np.float32),
        p_stay=0.5,
        beam_size=16,
        sil_phone_id=sil_phone_id,
        sil_enter_cost=sil_enter_cost,
        sph_phone_id=sph_phone_id,
        sph_enter_cost=sph_enter_cost,
    )

    assert alignment.state_path.tolist() == [0, 2, 2, 1]
    assert score_state_path(
        graph=graph,
        state_path=alignment.state_path,
        logp=logp,
        entry_bias=np.zeros(3, dtype=np.float32),
        p_stay=0.5,
        sil_phone_id=sil_phone_id,
        sil_enter_cost=sil_enter_cost,
        sph_phone_id=sph_phone_id,
        sph_enter_cost=sph_enter_cost,
    ) == pytest.approx(1.0 - 0.75 + 4.0 * math.log(0.5))


def test_boundary_contrast_moves_to_the_sharper_artificial_boundary(
    reference: ModuleType,
) -> None:
    graph = _manual_graph(
        reference,
        state_specs=[
            ("A", 0, 0, "word"),
            ("B", 1, 0, "word"),
        ],
        successors=[(1,), ()],
        start_states=[0],
        end_states=[1],
    )
    contrast = np.asarray([2, 2, -2, 1, 1, 1, -1, -1], dtype=np.float32)
    logp = np.column_stack((contrast / 2.0, -contrast / 2.0)).astype(np.float32)
    entry_bias = np.zeros(2, dtype=np.float32)

    without_contrast = reference.align_beam_viterbi(
        logp,
        graph,
        entry_bias,
        p_stay=0.5,
        beam_size=8,
        boundary_lambda=0.0,
        boundary_context_s=0.02,
        frame_hop_s=0.01,
    )
    with_contrast = reference.align_beam_viterbi(
        logp,
        graph,
        entry_bias,
        p_stay=0.5,
        beam_size=8,
        boundary_lambda=3.0,
        boundary_context_s=0.02,
        frame_hop_s=0.01,
    )

    assert _transition_frame(without_contrast.state_path, 1) == 6
    assert _transition_frame(with_contrast.state_path, 1) == 2


def test_wide_beam_matches_exact_small_graph_dp(reference: ModuleType) -> None:
    graph = _manual_graph(
        reference,
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
    logp = np.random.default_rng(20260811).normal(size=(7, 4)).astype(np.float32)
    entry_bias = np.asarray([0.1, -0.2, 0.3, -0.1], dtype=np.float32)

    expected = exhaustive_viterbi(
        graph=graph,
        logp=logp,
        entry_bias=entry_bias,
        p_stay=0.67,
        boundary_lambda=0.4,
        boundary_context_s=0.02,
        frame_hop_s=0.01,
    )
    actual = reference.align_beam_viterbi(
        logp,
        graph,
        entry_bias,
        beam_size=64,
        p_stay=0.67,
        boundary_lambda=0.4,
        boundary_context_s=0.02,
        frame_hop_s=0.01,
    )

    assert actual.state_path.tolist() == expected.state_path.tolist()
    assert score_state_path(
        graph=graph,
        state_path=actual.state_path,
        logp=logp,
        entry_bias=entry_bias,
        p_stay=0.67,
        boundary_lambda=0.4,
        boundary_context_s=0.02,
        frame_hop_s=0.01,
    ) == pytest.approx(expected.score)


def test_narrow_beam_fails_instead_of_returning_nonterminal_path(reference: ModuleType) -> None:
    graph = _manual_graph(
        reference,
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
        reference.align_beam_viterbi(
            logp,
            graph,
            entry_bias,
            p_stay=0.5,
            beam_size=1,
        )

    complete = reference.align_beam_viterbi(
        logp,
        graph,
        entry_bias,
        p_stay=0.5,
        beam_size=3,
    )
    assert complete.state_path.tolist() == [1, 2]


def test_repeated_word_labels_remain_separate_by_word_index(reference: ModuleType) -> None:
    dictionary = _pronouncing_dictionary(reference, go=[["G"]])
    graph, entry_bias = reference.build_phone_graph_optional_sil_sph(
        ["go", "go"],
        dictionary,
        {"G": 0},
        sil_phone=None,
        optional_sil_between_words=False,
        optional_sil_at_start=False,
        optional_sil_at_end=False,
        sph_phone=None,
        optional_sph_between_words=False,
        optional_sph_at_start=False,
        optional_sph_at_end=False,
    )

    alignment = reference.align_beam_viterbi(
        np.zeros((2, 1), dtype=np.float32),
        graph,
        entry_bias,
        p_stay=0.5,
        beam_size=8,
    )

    assert alignment.word_segments_f == [("go", 0, 1), ("go", 1, 2)]
    assert alignment.phone_segments_f == [("G", 0, 1), ("G", 1, 2)]
    assert [graph.states[state].edge.word_index for state in alignment.state_path] == [0, 1]


def test_equal_phone_states_inside_one_word_collapse_by_current_quirk(
    reference: ModuleType,
) -> None:
    dictionary = _pronouncing_dictionary(reference, doubled=[["AA", "AA"]])
    graph, entry_bias = reference.build_phone_graph_optional_sil_sph(
        ["doubled"],
        dictionary,
        {"AA": 0},
        sil_phone=None,
        optional_sil_between_words=False,
        optional_sil_at_start=False,
        optional_sil_at_end=False,
        sph_phone=None,
        optional_sph_between_words=False,
        optional_sph_at_start=False,
        optional_sph_at_end=False,
    )
    alignment = reference.align_beam_viterbi(
        np.zeros((2, 1), dtype=np.float32),
        graph,
        entry_bias,
        p_stay=0.5,
        beam_size=8,
    )
    segments = reference.extract_state_segments_from_path(
        graph,
        entry_bias,
        alignment.state_path,
    )

    assert np.unique(alignment.state_path).size == 2
    assert len(segments) == 1
    assert segments[0][0].phone == "AA"
    assert segments[0][1:] == (0, 2)


@pytest.mark.parametrize(
    ("duration_frames", "is_kept"),
    [(6, False), (7, True)],
)
def test_65ms_threshold_at_10ms_uses_ceil_seven_frames(
    reference: ModuleType,
    duration_frames: int,
    is_kept: bool,
) -> None:
    left = _fixed_spec(reference, "A", 0, 0, "left")
    silence = _fixed_spec(reference, "sil", 1, None, None)
    right = _fixed_spec(reference, "B", 2, 1, "right")
    segments = [
        (left, 0, 1),
        (silence, 1, 1 + duration_frames),
        (right, 1 + duration_frames, 2 + duration_frames),
    ]

    kept, stats = reference.prune_short_internal_sil_sph_segments(
        segments,
        sil_phone="sil",
        sph_phone="sph",
        min_sil_dur_ms=65.0,
        min_sph_dur_ms=50.0,
        frame_hop_s=0.01,
    )

    assert (silence in kept) is is_kept
    assert stats["dropped_short_sil"] == (0 if is_kept else 1)


@pytest.mark.parametrize(
    ("duration_frames", "is_kept"),
    [(4, False), (5, True)],
)
def test_50ms_threshold_at_10ms_is_exactly_five_frames(
    reference: ModuleType,
    duration_frames: int,
    is_kept: bool,
) -> None:
    left = _fixed_spec(reference, "A", 0, 0, "left")
    speech_gap = _fixed_spec(reference, "sph", 1, None, "[missing]")
    right = _fixed_spec(reference, "B", 2, 1, "right")
    segments = [
        (left, 0, 1),
        (speech_gap, 1, 1 + duration_frames),
        (right, 1 + duration_frames, 2 + duration_frames),
    ]

    kept, stats = reference.prune_short_internal_sil_sph_segments(
        segments,
        sil_phone="sil",
        sph_phone="sph",
        min_sil_dur_ms=65.0,
        min_sph_dur_ms=50.0,
        frame_hop_s=0.01,
    )

    assert (speech_gap in kept) is is_kept
    assert stats["dropped_short_sph"] == (0 if is_kept else 1)


def test_short_boundary_gap_states_are_never_pruned(reference: ModuleType) -> None:
    leading_silence = _fixed_spec(reference, "sil", 0, None, None)
    word = _fixed_spec(reference, "A", 1, 0, "word")
    trailing_speech = _fixed_spec(reference, "sph", 2, None, "[missing]")

    kept, stats = reference.prune_short_internal_sil_sph_segments(
        [
            (leading_silence, 0, 1),
            (word, 1, 2),
            (trailing_speech, 2, 3),
        ],
        sil_phone="sil",
        sph_phone="sph",
        min_sil_dur_ms=1000.0,
        min_sph_dur_ms=1000.0,
        frame_hop_s=0.01,
    )

    assert kept == [leading_silence, word, trailing_speech]
    assert stats["dropped_short_sil"] == 0
    assert stats["dropped_short_sph"] == 0


def test_fixed_sequence_graph_is_linear_and_preserves_specs(reference: ModuleType) -> None:
    specs = [
        _fixed_spec(reference, "A", 0, 0, "left", bias=0.1),
        _fixed_spec(reference, "sil", 1, None, None, bias=-0.5),
        _fixed_spec(reference, "B", 2, 1, "right", bias=0.2),
    ]

    graph, entry_bias = reference.build_fixed_sequence_graph(specs)

    assert graph.start_states == [0]
    assert graph.end_states == [2]
    assert [state.preds for state in graph.states] == [(), (0,), (1,)]
    assert [state.succs for state in graph.states] == [(1,), (2,), ()]
    assert [state.edge.phone for state in graph.states] == ["A", "sil", "B"]
    assert [state.edge.word_index for state in graph.states] == [0, None, 1]
    assert entry_bias.dtype == np.float32
    assert entry_bias.tolist() == pytest.approx([0.1, -0.5, 0.2])


def test_second_decode_drops_short_internal_gap_and_reestimates_boundaries(
    reference: ModuleType,
) -> None:
    first_graph = _manual_graph(
        reference,
        state_specs=[
            ("A", 0, 0, "left"),
            ("sil", 1, None, None),
            ("B", 2, 1, "right"),
        ],
        successors=[(1,), (2,), ()],
        start_states=[0],
        end_states=[2],
    )
    first_alignment = reference.AlignmentResult(
        phone_segments_f=[("A", 0, 1), ("sil", 1, 2), ("B", 2, 3)],
        word_segments_f=[("left", 0, 1), ("sil", 1, 2), ("right", 2, 3)],
        state_path=np.asarray([0, 1, 2], dtype=np.int32),
        aligned_phone_ids=np.asarray([0, 1, 2], dtype=np.int32),
    )
    logp = np.asarray(
        [
            [0.0, -10.0, -10.0],
            [0.0, -10.0, -10.0],
            [-10.0, -10.0, 0.0],
        ],
        dtype=np.float32,
    )
    args = SimpleNamespace(
        min_sil_dur_ms=20.0,
        min_sph_dur_ms=50.0,
        frame_hop_s=0.01,
        p_stay=0.5,
        beam=16,
        word_sil_label="sil",
        boundary_lambda=0.0,
        boundary_context_s=0.02,
    )

    second_alignment, stats = reference.redecode_with_pruned_fixed_sequence(
        first_pass_ali=first_alignment,
        first_pass_graph=first_graph,
        first_pass_entry_bias=np.asarray([0.0, -0.5, 0.0], dtype=np.float32),
        logp=logp,
        sil_phone="sil",
        sil_phone_id=1,
        sph_phone="sph",
        sph_phone_id=None,
        args=args,
    )

    assert stats == {
        "first_pass_states": 3,
        "fixed_states": 2,
        "dropped_short_sil": 1,
        "dropped_short_sph": 0,
    }
    assert second_alignment.aligned_phone_ids.tolist() == [0, 0, 2]
    assert second_alignment.phone_segments_f == [("A", 0, 2), ("B", 2, 3)]
    assert second_alignment.word_segments_f == [("left", 0, 2), ("right", 2, 3)]
