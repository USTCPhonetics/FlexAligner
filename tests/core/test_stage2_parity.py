"""Reference, exact-oracle, and production parity for the Stage 2 core."""

from __future__ import annotations

import math
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from flexaligner.core import stage2 as production
from tests.characterization.differential import assert_reference_equivalent
from tests.characterization.reference_loader import load_reference_module
from tests.characterization.stage2_oracle import exhaustive_viterbi, score_state_path
from tests.core._stage2_cases import (
    CURRENT_BEHAVIOR_EQUAL_PHONE,
    NO_GAPS,
    SIX_INTERNAL_GAP_PATHS,
    WIDE_BEAM_ENTRY_BIAS,
    WIDE_BEAM_LOGP,
    all_complete_phone_paths,
    fixed_spec,
    internal_gap_paths,
    manual_graph,
    pronouncing_dictionary,
    redecode_args,
    transition_frame,
)


@pytest.fixture(scope="module")
def reference() -> ModuleType:
    return load_reference_module()


def _graph_pair(
    reference: ModuleType,
    *,
    state_specs: list[tuple[str, int, int | None, str | None]],
    successors: list[tuple[int, ...]],
    start_states: list[int],
    end_states: list[int],
) -> tuple[Any, Any]:
    kwargs = {
        "state_specs": state_specs,
        "successors": successors,
        "start_states": start_states,
        "end_states": end_states,
    }
    return manual_graph(reference, **kwargs), manual_graph(production, **kwargs)


def _build_graph_pair(
    reference: ModuleType,
    words: list[str],
    entries: dict[str, list[list[str]]],
    vocabulary: dict[str, int],
    **options: Any,
) -> tuple[tuple[Any, np.ndarray], tuple[Any, np.ndarray]]:
    reference_result = reference.build_phone_graph_optional_sil_sph(
        words,
        pronouncing_dictionary(reference, **entries),
        vocabulary,
        **options,
    )
    production_result = production.build_phone_graph_optional_sil_sph(
        words,
        pronouncing_dictionary(production, **entries),
        vocabulary,
        **options,
    )
    return reference_result, production_result


def _alignment_pair(
    reference: ModuleType,
    reference_graph: Any,
    production_graph: Any,
    logp: np.ndarray,
    entry_bias: np.ndarray,
    **options: Any,
) -> tuple[Any, Any]:
    expected = reference.align_beam_viterbi(
        logp,
        reference_graph,
        entry_bias,
        **options,
    )
    actual = production.align_beam_viterbi(
        logp,
        production_graph,
        entry_bias,
        **options,
    )
    assert_reference_equivalent(expected.phone_segments_f, actual.phone_segments_f)
    assert_reference_equivalent(expected.word_segments_f, actual.word_segments_f)
    assert_reference_equivalent(expected.state_path, actual.state_path)
    assert_reference_equivalent(expected.aligned_phone_ids, actual.aligned_phone_ids)
    score_options = {
        name: options[name]
        for name in (
            "p_stay",
            "boundary_lambda",
            "boundary_context_s",
            "frame_hop_s",
            "sil_phone_id",
            "sil_enter_cost",
            "sph_phone_id",
            "sph_enter_cost",
        )
        if name in options
    }
    assert actual.score == pytest.approx(
        score_state_path(
            graph=production_graph,
            state_path=actual.state_path,
            logp=logp,
            entry_bias=entry_bias,
            **score_options,
        )
    )
    return expected, actual


def _stats_mapping(stats: Any) -> dict[str, int]:
    if isinstance(stats, dict):
        return stats
    return {
        "first_pass_states": stats.first_pass_states,
        "fixed_states": stats.fixed_states,
        "dropped_short_sil": stats.dropped_short_sil,
        "dropped_short_sph": stats.dropped_short_sph,
    }


def _assert_prune_equivalent(expected: tuple[Any, Any], actual: tuple[Any, Any]) -> None:
    assert_reference_equivalent(expected[0], actual[0])
    assert_reference_equivalent(_stats_mapping(expected[1]), _stats_mapping(actual[1]))


def test_epsilon_closure_is_transitive_and_matches_reference(reference: ModuleType) -> None:
    adjacency = [[1, 3], [2], [], [4], [5], []]
    expected = reference._eps_closure(6, adjacency)
    actual = production.epsilon_closure(6, adjacency)

    assert (
        actual
        == expected
        == [
            {0, 1, 2, 3, 4, 5},
            {1, 2},
            {2},
            {3, 4, 5},
            {4, 5},
            {5},
        ]
    )


@pytest.mark.parametrize(
    ("num_nodes", "adjacency", "error", "message"),
    [
        (0, [], ValueError, "num_nodes must be positive"),
        (2, "bad", TypeError, "epsilon_adjacency"),
        (2, [[]], ValueError, "length must equal"),
        (2, ["bad", []], TypeError, r"epsilon_adjacency\[0\]"),
        (2, [[2], []], ValueError, "out of range"),
        (2, [[True], []], TypeError, "must be an integer"),
    ],
)
def test_epsilon_closure_validation(
    num_nodes: Any,
    adjacency: Any,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        production.epsilon_closure(num_nodes, adjacency)


def test_stage2_decode_config_matches_accepted_reference_profile() -> None:
    config = production.Stage2DecodeConfig()
    assert_reference_equivalent(
        {
            "p_stay": 0.92,
            "beam": 400,
            "boundary_lambda": 200.0,
            "boundary_context_s": 0.03,
            "frame_hop_s": 0.01,
            "min_sil_dur_ms": 65.0,
            "min_sph_dur_ms": 50.0,
            "sil_phone": "sil",
            "sph_phone": "sph",
            "sph_word_label": "[missing]",
            "word_sil_label": "sil",
        },
        {
            "p_stay": config.p_stay,
            "beam": config.beam,
            "boundary_lambda": config.boundary_lambda,
            "boundary_context_s": config.boundary_context_s,
            "frame_hop_s": config.frame_hop_s,
            "min_sil_dur_ms": config.min_sil_dur_ms,
            "min_sph_dur_ms": config.min_sph_dur_ms,
            "sil_phone": config.sil_phone,
            "sph_phone": config.sph_phone,
            "sph_word_label": config.sph_word_label,
            "word_sil_label": config.word_sil_label,
        },
    )


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"p_stay": 0.0}, ValueError, "p_stay"),
        ({"p_stay": math.nan}, ValueError, "finite"),
        ({"beam": 0}, ValueError, "beam"),
        ({"beam": True}, TypeError, "beam"),
        ({"boundary_lambda": math.inf}, ValueError, "finite"),
        ({"boundary_context_s": 0.0}, ValueError, "positive"),
        ({"frame_hop_s": 0.0}, ValueError, "positive"),
        ({"min_sil_dur_ms": -1.0}, ValueError, "non-negative"),
        ({"min_sph_dur_ms": -1.0}, ValueError, "non-negative"),
        ({"sil_phone": ""}, ValueError, "non-empty"),
        ({"sph_word_label": ""}, ValueError, "non-empty"),
    ],
)
def test_stage2_decode_config_validation(
    kwargs: dict[str, Any],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        production.Stage2DecodeConfig(**kwargs)


def test_multi_pronunciation_graph_has_distinct_complete_paths(reference: ModuleType) -> None:
    (expected_graph, expected_bias), (actual_graph, actual_bias) = _build_graph_pair(
        reference,
        ["read"],
        {"read": [["R", "IY", "D"], ["R", "EH", "D"]]},
        {"R": 0, "IY": 1, "EH": 2, "D": 3},
        **NO_GAPS,
    )

    assert_reference_equivalent(expected_graph, actual_graph)
    assert_reference_equivalent(expected_bias, actual_bias)
    assert all_complete_phone_paths(actual_graph) == {
        ("R", "IY", "D"),
        ("R", "EH", "D"),
    }
    assert len(actual_graph.start_states) == len(actual_graph.end_states) == 2


def test_internal_gap_has_exact_six_silence_and_speech_paths(reference: ModuleType) -> None:
    options = {
        "sil_phone": "sil",
        "optional_sil_between_words": True,
        "optional_sil_at_start": False,
        "optional_sil_at_end": False,
        "sph_phone": "sph",
        "optional_sph_between_words": True,
        "optional_sph_at_start": False,
        "optional_sph_at_end": False,
    }
    (expected_graph, expected_bias), (actual_graph, actual_bias) = _build_graph_pair(
        reference,
        ["left", "right"],
        {"left": [["L"]], "right": [["R"]]},
        {"L": 0, "R": 1, "sil": 2, "sph": 3},
        **options,
    )

    assert_reference_equivalent(expected_graph, actual_graph)
    assert_reference_equivalent(expected_bias, actual_bias)
    assert internal_gap_paths(actual_graph) == SIX_INTERNAL_GAP_PATHS


def test_sph_is_reachable_at_start_and_end(reference: ModuleType) -> None:
    options = {
        "sil_phone": None,
        "optional_sil_between_words": False,
        "optional_sil_at_start": False,
        "optional_sil_at_end": False,
        "sph_phone": "sph",
        "optional_sph_between_words": False,
        "optional_sph_at_start": True,
        "optional_sph_at_end": True,
    }
    (expected_graph, expected_bias), (actual_graph, actual_bias) = _build_graph_pair(
        reference,
        ["word"],
        {"word": [["W"]]},
        {"W": 0, "sph": 1},
        **options,
    )

    assert_reference_equivalent(expected_graph, actual_graph)
    assert_reference_equivalent(expected_bias, actual_bias)
    assert all_complete_phone_paths(actual_graph) == {
        ("W",),
        ("sph", "W"),
        ("W", "sph"),
        ("sph", "W", "sph"),
    }


@pytest.mark.parametrize(
    ("words", "entries", "vocabulary", "options", "message"),
    [
        (["missing"], {"known": [["K"]]}, {"K": 0}, NO_GAPS, "Word not in lexicon"),
        (
            ["word"],
            {"word": [["UNKNOWN"]]},
            {"W": 0},
            NO_GAPS,
            "UNKNOWN",
        ),
        (
            ["word"],
            {"word": [["W"]]},
            {"W": 0},
            {**NO_GAPS, "sil_phone": "sil", "optional_sil_at_start": True},
            "sil",
        ),
        (
            ["word"],
            {"word": [["W"]]},
            {"W": 0},
            {**NO_GAPS, "sph_phone": "sph", "optional_sph_at_end": True},
            "sph",
        ),
    ],
)
def test_graph_builder_failure_parity(
    reference: ModuleType,
    words: list[str],
    entries: dict[str, list[list[str]]],
    vocabulary: dict[str, int],
    options: dict[str, Any],
    message: str,
) -> None:
    for module in (reference, production):
        with pytest.raises((KeyError, RuntimeError), match=message):
            module.build_phone_graph_optional_sil_sph(
                words,
                pronouncing_dictionary(module, **entries),
                vocabulary,
                **options,
            )


@pytest.mark.parametrize(
    ("words", "lexicon", "vocabulary", "options", "error", "message"),
    [
        ("word", {"word": [["W"]]}, {"W": 0}, NO_GAPS, TypeError, "words must be"),
        ([], {}, {"W": 0}, NO_GAPS, ValueError, "must not be empty"),
        ([""], {"": [["W"]]}, {"W": 0}, NO_GAPS, ValueError, "non-empty"),
        (["word"], {"word": [["W"]]}, {}, NO_GAPS, ValueError, "must not be empty"),
        (["word"], {"word": [["W"]]}, {"W": -1}, NO_GAPS, ValueError, "non-negative"),
        (["word"], {"word": []}, {"W": 0}, NO_GAPS, RuntimeError, "no pronunciations"),
        (["word"], {"word": [[]]}, {"W": 0}, NO_GAPS, RuntimeError, "Empty pronunciation"),
        (
            ["word"],
            {"word": [["W"]]},
            {"W": 0},
            {**NO_GAPS, "optional_sil_between_words": 1},
            TypeError,
            "must be bool",
        ),
        (
            ["word"],
            {"word": [["W"]]},
            {"W": 0},
            {**NO_GAPS, "sil_cost": math.nan},
            ValueError,
            "finite",
        ),
    ],
)
def test_graph_builder_strict_validation(
    words: Any,
    lexicon: Any,
    vocabulary: Any,
    options: dict[str, Any],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        production.build_phone_graph_optional_sil_sph(
            words,
            lexicon,
            vocabulary,
            **options,
        )


@pytest.mark.parametrize(
    ("emission_advantage", "expected_path"),
    [(1.3, [0, 0]), (1.5, [1, 2])],
)
def test_stay_move_tradeoff_uses_log_probabilities(
    reference: ModuleType,
    emission_advantage: float,
    expected_path: list[int],
) -> None:
    expected_graph, actual_graph = _graph_pair(
        reference,
        state_specs=[("X", 0, None, None), ("X", 0, None, None), ("Y", 1, None, None)],
        successors=[(), (2,), ()],
        start_states=[0, 1],
        end_states=[0, 2],
    )
    logp = np.asarray([[0.0, 0.0], [0.0, emission_advantage]], dtype=np.float32)
    _, actual = _alignment_pair(
        reference,
        expected_graph,
        actual_graph,
        logp,
        np.zeros(3, dtype=np.float32),
        p_stay=0.8,
        beam_size=8,
    )

    assert actual.state_path.tolist() == expected_path
    assert math.log(0.8 / 0.2) == pytest.approx(math.log(4.0))


def test_entry_bias_is_charged_per_frame(reference: ModuleType) -> None:
    expected_graph, actual_graph = _graph_pair(
        reference,
        state_specs=[("A", 0, 0, "word"), ("B", 1, 0, "word")],
        successors=[(1,), ()],
        start_states=[0],
        end_states=[1],
    )
    logp = np.zeros((4, 2), dtype=np.float32)
    entry_bias = np.asarray([1.0, 0.0], dtype=np.float32)
    _, actual = _alignment_pair(
        reference,
        expected_graph,
        actual_graph,
        logp,
        entry_bias,
        p_stay=0.5,
        beam_size=8,
    )

    assert actual.state_path.tolist() == [0, 0, 0, 1]
    assert score_state_path(
        graph=actual_graph,
        state_path=actual.state_path,
        logp=logp,
        entry_bias=entry_bias,
        p_stay=0.5,
    ) == pytest.approx(3.0 + 4.0 * math.log(0.5))


def test_silence_lock_uses_current_round_six_frames_at_65ms(reference: ModuleType) -> None:
    expected_graph, actual_graph = _graph_pair(
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
    _, actual = _alignment_pair(
        reference,
        expected_graph,
        actual_graph,
        logp,
        np.zeros(3, dtype=np.float32),
        p_stay=0.5,
        beam_size=32,
        sil_phone_id=1,
        min_sil_dur_ms=65.0,
        frame_hop_s=0.01,
    )

    sil_frames = np.flatnonzero(actual.aligned_phone_ids == 1)
    # Decoder locking preserves reference round(6.5) == 6 behavior. The later
    # pruning stage deliberately uses ceil and is locked separately at 7 frames.
    assert sil_frames.size == 6
    independent = exhaustive_viterbi(
        graph=actual_graph,
        logp=logp,
        entry_bias=np.zeros(3, dtype=np.float32),
        p_stay=0.5,
        sil_phone_id=1,
        min_sil_dur_ms=65.0,
        frame_hop_s=0.01,
    )
    assert_reference_equivalent(independent.state_path, actual.state_path)
    assert independent.score == pytest.approx(actual.score)


@pytest.mark.parametrize("gap_kind", ["sil", "sph"])
def test_gap_enter_cost_is_charged_once(
    reference: ModuleType,
    gap_kind: str,
) -> None:
    expected_graph, actual_graph = _graph_pair(
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
    kwargs = {
        "p_stay": 0.5,
        "beam_size": 16,
        "sil_phone_id": 1 if gap_kind == "sil" else None,
        "sil_enter_cost": -0.75 if gap_kind == "sil" else 0.0,
        "sph_phone_id": 1 if gap_kind == "sph" else None,
        "sph_enter_cost": -0.75 if gap_kind == "sph" else 0.0,
    }
    _, actual = _alignment_pair(
        reference,
        expected_graph,
        actual_graph,
        logp,
        np.zeros(3, dtype=np.float32),
        **kwargs,
    )

    assert actual.state_path.tolist() == [0, 2, 2, 1]
    assert score_state_path(
        graph=actual_graph,
        state_path=actual.state_path,
        logp=logp,
        entry_bias=np.zeros(3, dtype=np.float32),
        p_stay=0.5,
        sil_phone_id=kwargs["sil_phone_id"],
        sil_enter_cost=kwargs["sil_enter_cost"],
        sph_phone_id=kwargs["sph_phone_id"],
        sph_enter_cost=kwargs["sph_enter_cost"],
    ) == pytest.approx(1.0 - 0.75 + 4.0 * math.log(0.5))


def test_boundary_contrast_moves_to_sharper_boundary(reference: ModuleType) -> None:
    expected_graph, actual_graph = _graph_pair(
        reference,
        state_specs=[("A", 0, 0, "word"), ("B", 1, 0, "word")],
        successors=[(1,), ()],
        start_states=[0],
        end_states=[1],
    )
    contrast = np.asarray([2, 2, -2, 1, 1, 1, -1, -1], dtype=np.float32)
    logp = np.column_stack((contrast / 2.0, -contrast / 2.0)).astype(np.float32)
    bias = np.zeros(2, dtype=np.float32)

    _, without = _alignment_pair(
        reference,
        expected_graph,
        actual_graph,
        logp,
        bias,
        p_stay=0.5,
        beam_size=8,
        boundary_lambda=0.0,
        boundary_context_s=0.02,
        frame_hop_s=0.01,
    )
    _, with_contrast = _alignment_pair(
        reference,
        expected_graph,
        actual_graph,
        logp,
        bias,
        p_stay=0.5,
        beam_size=8,
        boundary_lambda=3.0,
        boundary_context_s=0.02,
        frame_hop_s=0.01,
    )

    assert transition_frame(without.state_path, 1) == 6
    assert transition_frame(with_contrast.state_path, 1) == 2


def test_wide_beam_matches_exact_dp_and_path_score_includes_final_move(
    reference: ModuleType,
) -> None:
    expected_graph, actual_graph = _graph_pair(
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
    options = {
        "p_stay": 0.67,
        "boundary_lambda": 0.4,
        "boundary_context_s": 0.02,
        "frame_hop_s": 0.01,
    }
    independent = exhaustive_viterbi(
        graph=actual_graph,
        logp=WIDE_BEAM_LOGP,
        entry_bias=WIDE_BEAM_ENTRY_BIAS,
        **options,
    )
    _, actual = _alignment_pair(
        reference,
        expected_graph,
        actual_graph,
        WIDE_BEAM_LOGP,
        WIDE_BEAM_ENTRY_BIAS,
        beam_size=64,
        **options,
    )
    scored = score_state_path(
        graph=actual_graph,
        state_path=actual.state_path,
        logp=WIDE_BEAM_LOGP,
        entry_bias=WIDE_BEAM_ENTRY_BIAS,
        **options,
    )

    assert_reference_equivalent(independent.state_path, actual.state_path)
    assert scored == pytest.approx(independent.score)
    without_final_move = scored - math.log(1.0 - options["p_stay"])
    assert scored == pytest.approx(without_final_move + math.log(1.0 - options["p_stay"]))


def test_narrow_beam_fails_and_wide_beam_reaches_terminal(reference: ModuleType) -> None:
    expected_graph, actual_graph = _graph_pair(
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
    bias = np.zeros(3, dtype=np.float32)

    for module, graph in ((reference, expected_graph), (production, actual_graph)):
        with pytest.raises(RuntimeError, match="failed to reach any end state"):
            module.align_beam_viterbi(logp, graph, bias, p_stay=0.5, beam_size=1)
    _, actual = _alignment_pair(
        reference,
        expected_graph,
        actual_graph,
        logp,
        bias,
        p_stay=0.5,
        beam_size=3,
    )
    assert actual.state_path.tolist() == [1, 2]


def _base_production_decoder_case() -> tuple[Any, np.ndarray, np.ndarray]:
    graph = manual_graph(
        production,
        state_specs=[("A", 0, 0, "word"), ("B", 1, 0, "word")],
        successors=[(1,), ()],
        start_states=[0],
        end_states=[1],
    )
    return graph, np.zeros((2, 2), dtype=np.float32), np.zeros(2, dtype=np.float32)


@pytest.mark.parametrize(
    ("field", "value", "error", "message"),
    [
        ("logp", [[0.0, 0.0]], TypeError, "NumPy array"),
        ("logp", np.zeros(2), ValueError, "shape"),
        ("logp", np.zeros((0, 2)), ValueError, "dimensions must be positive"),
        ("logp", np.zeros((2, 2), dtype=np.int64), TypeError, "dtype must be floating"),
        ("logp", np.asarray([[0.0, np.nan], [0.0, 0.0]]), ValueError, "NaN or infinity"),
        ("entry_bias", np.zeros(1), ValueError, "shape"),
        ("entry_bias", np.zeros(2, dtype=np.int64), TypeError, "dtype must be floating"),
        ("entry_bias", np.asarray([0.0, np.inf]), ValueError, "NaN or infinity"),
        ("p_stay", 1.0, ValueError, "strictly between"),
        ("beam_size", 0, ValueError, "must be positive"),
        ("word_sil_label", "", ValueError, "non-empty"),
        ("boundary_lambda", math.inf, ValueError, "finite"),
        ("boundary_context_s", 0.0, ValueError, "positive"),
        ("frame_hop_s", 0.0, ValueError, "positive"),
        ("min_sil_dur_ms", -1.0, ValueError, "non-negative"),
        ("sil_enter_cost", math.nan, ValueError, "finite"),
        ("sph_enter_cost", math.inf, ValueError, "finite"),
        ("sil_phone_id", 2, ValueError, "out of range"),
        ("sph_phone_id", -1, ValueError, "non-negative"),
    ],
)
def test_decoder_strict_shape_finite_and_configuration_failures(
    field: str,
    value: Any,
    error: type[Exception],
    message: str,
) -> None:
    graph, logp, entry_bias = _base_production_decoder_case()
    arguments: dict[str, Any] = {
        "logp": logp,
        "graph": graph,
        "entry_bias": entry_bias,
        "p_stay": 0.5,
        "beam_size": 8,
        "word_sil_label": "sil",
        "boundary_lambda": 0.0,
        "boundary_context_s": 0.02,
        "frame_hop_s": 0.01,
        "sil_phone_id": None,
        "min_sil_dur_ms": 0.0,
        "sil_enter_cost": 0.0,
        "sph_phone_id": None,
        "sph_enter_cost": 0.0,
    }
    arguments[field] = value
    with pytest.raises(error, match=message):
        production.align_beam_viterbi(**arguments)


@pytest.mark.parametrize(
    ("graph", "message"),
    [
        (
            production.PhoneGraph(states=[], start_states=[], end_states=[]),
            "states must not be empty",
        ),
        (
            production.PhoneGraph(
                states=[
                    production.PhoneState(
                        edge=production.EmitEdge(0, 1, "A", 0, 0, "word"),
                        preds=(),
                        succs=(1,),
                    ),
                    production.PhoneState(
                        edge=production.EmitEdge(1, 2, "B", 1, 0, "word"),
                        preds=(),
                        succs=(),
                    ),
                ],
                start_states=[0],
                end_states=[1],
            ),
            "successor/predecessor mismatch",
        ),
        (
            production.PhoneGraph(
                states=[
                    production.PhoneState(
                        edge=production.EmitEdge(0, 1, "A", 2, 0, "word"),
                        preds=(),
                        succs=(),
                    )
                ],
                start_states=[0],
                end_states=[0],
            ),
            "phone ID out of range",
        ),
    ],
)
def test_decoder_rejects_invalid_graphs(graph: Any, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        production.align_beam_viterbi(
            np.zeros((2, 2), dtype=np.float32),
            graph,
            np.zeros(len(graph.states), dtype=np.float32),
            p_stay=0.5,
            beam_size=8,
        )


@pytest.mark.parametrize(
    ("path", "message"),
    [
        (np.zeros((1, 1), dtype=np.int32), "one-dimensional"),
        (np.asarray([0.0, 1.0]), "dtype must be integer"),
        (np.asarray([2], dtype=np.int32), "out of range"),
        (np.asarray([1], dtype=np.int32), "begin at a graph start"),
        (np.asarray([0], dtype=np.int32), "finish at a graph end"),
    ],
)
def test_extract_state_segments_rejects_invalid_paths(path: np.ndarray, message: str) -> None:
    graph, _logp, entry_bias = _base_production_decoder_case()
    with pytest.raises((TypeError, ValueError), match=message):
        production.extract_state_segments_from_path(graph, entry_bias, path)


def test_phone_word_segments_preserve_repeated_word_indices(reference: ModuleType) -> None:
    (expected_graph, expected_bias), (actual_graph, actual_bias) = _build_graph_pair(
        reference,
        ["go", "go"],
        {"go": [["G"]]},
        {"G": 0},
        **NO_GAPS,
    )
    _, actual = _alignment_pair(
        reference,
        expected_graph,
        actual_graph,
        np.zeros((2, 1), dtype=np.float32),
        actual_bias,
        p_stay=0.5,
        beam_size=8,
    )

    assert_reference_equivalent(expected_bias, actual_bias)
    assert actual.word_segments_f == [("go", 0, 1), ("go", 1, 2)]
    assert actual.phone_segments_f == [("G", 0, 1), ("G", 1, 2)]
    assert [actual_graph.states[state].edge.word_index for state in actual.state_path] == [0, 1]


def test_equal_phone_same_word_collapse_is_named_current_behavior(reference: ModuleType) -> None:
    assert CURRENT_BEHAVIOR_EQUAL_PHONE.startswith("current_behavior:")
    (expected_graph, expected_bias), (actual_graph, actual_bias) = _build_graph_pair(
        reference,
        ["doubled"],
        {"doubled": [["AA", "AA"]]},
        {"AA": 0},
        **NO_GAPS,
    )
    _, actual = _alignment_pair(
        reference,
        expected_graph,
        actual_graph,
        np.zeros((2, 1), dtype=np.float32),
        actual_bias,
        p_stay=0.5,
        beam_size=8,
    )
    expected_segments = reference.extract_state_segments_from_path(
        expected_graph, expected_bias, actual.state_path
    )
    actual_segments = production.extract_state_segments_from_path(
        actual_graph, actual_bias, actual.state_path
    )

    assert_reference_equivalent(expected_segments, actual_segments)
    assert np.unique(actual.state_path).size == 2
    assert len(actual_segments) == 1
    assert actual_segments[0][0].phone == "AA"
    assert actual_segments[0][1:] == (0, 2)


@pytest.mark.parametrize(
    ("gap_phone", "threshold_ms", "duration_frames", "is_kept"),
    [
        ("sil", 65.0, 6, False),
        ("sil", 65.0, 7, True),
        ("sph", 50.0, 4, False),
        ("sph", 50.0, 5, True),
    ],
)
def test_internal_gap_prune_thresholds_match_reference(
    reference: ModuleType,
    gap_phone: str,
    threshold_ms: float,
    duration_frames: int,
    is_kept: bool,
) -> None:
    def segments(module: Any) -> list[tuple[Any, int, int]]:
        left = fixed_spec(module, "A", 0, 0, "left")
        gap = fixed_spec(
            module,
            gap_phone,
            1,
            None,
            None if gap_phone == "sil" else "[missing]",
        )
        right = fixed_spec(module, "B", 2, 1, "right")
        return [
            (left, 0, 1),
            (gap, 1, 1 + duration_frames),
            (right, 1 + duration_frames, 2 + duration_frames),
        ]

    options = {
        "sil_phone": "sil",
        "sph_phone": "sph",
        "min_sil_dur_ms": threshold_ms if gap_phone == "sil" else 65.0,
        "min_sph_dur_ms": threshold_ms if gap_phone == "sph" else 50.0,
        "frame_hop_s": 0.01,
    }
    expected = reference.prune_short_internal_sil_sph_segments(segments(reference), **options)
    actual = production.prune_short_internal_sil_sph_segments(segments(production), **options)

    _assert_prune_equivalent(expected, actual)
    assert any(spec.phone == gap_phone for spec in actual[0]) is is_kept


def test_short_boundary_gap_states_are_never_pruned(reference: ModuleType) -> None:
    def segments(module: Any) -> list[tuple[Any, int, int]]:
        return [
            (fixed_spec(module, "sil", 0, None, None), 0, 1),
            (fixed_spec(module, "A", 1, 0, "word"), 1, 2),
            (fixed_spec(module, "sph", 2, None, "[missing]"), 2, 3),
        ]

    options = {
        "sil_phone": "sil",
        "sph_phone": "sph",
        "min_sil_dur_ms": 1000.0,
        "min_sph_dur_ms": 1000.0,
        "frame_hop_s": 0.01,
    }
    expected = reference.prune_short_internal_sil_sph_segments(segments(reference), **options)
    actual = production.prune_short_internal_sil_sph_segments(segments(production), **options)

    _assert_prune_equivalent(expected, actual)
    assert [spec.phone for spec in actual[0]] == ["sil", "A", "sph"]
    assert actual[1].dropped_short_sil == actual[1].dropped_short_sph == 0


def test_fixed_sequence_graph_is_linear_and_preserves_specs(reference: ModuleType) -> None:
    expected_specs = [
        fixed_spec(reference, "A", 0, 0, "left", bias=0.1),
        fixed_spec(reference, "sil", 1, None, None, bias=-0.5),
        fixed_spec(reference, "B", 2, 1, "right", bias=0.2),
    ]
    actual_specs = [
        fixed_spec(production, "A", 0, 0, "left", bias=0.1),
        fixed_spec(production, "sil", 1, None, None, bias=-0.5),
        fixed_spec(production, "B", 2, 1, "right", bias=0.2),
    ]
    expected = reference.build_fixed_sequence_graph(expected_specs)
    actual = production.build_fixed_sequence_graph(actual_specs)

    assert_reference_equivalent(expected, actual)
    graph, entry_bias = actual
    assert graph.start_states == [0]
    assert graph.end_states == [2]
    assert [state.preds for state in graph.states] == [(), (0,), (1,)]
    assert [state.succs for state in graph.states] == [(1,), (2,), ()]
    assert entry_bias.tolist() == pytest.approx([0.1, -0.5, 0.2])


def test_second_decode_drops_short_gap_and_reestimates_boundaries(reference: ModuleType) -> None:
    expected_graph, actual_graph = _graph_pair(
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
    expected_first = reference.AlignmentResult(
        phone_segments_f=[("A", 0, 1), ("sil", 1, 2), ("B", 2, 3)],
        word_segments_f=[("left", 0, 1), ("sil", 1, 2), ("right", 2, 3)],
        state_path=np.asarray([0, 1, 2], dtype=np.int32),
        aligned_phone_ids=np.asarray([0, 1, 2], dtype=np.int32),
    )
    actual_first = production.ViterbiAlignment(
        phone_segments_f=[("A", 0, 1), ("sil", 1, 2), ("B", 2, 3)],
        word_segments_f=[("left", 0, 1), ("sil", 1, 2), ("right", 2, 3)],
        state_path=np.asarray([0, 1, 2], dtype=np.int32),
        aligned_phone_ids=np.asarray([0, 1, 2], dtype=np.int32),
        score=0.0,
    )
    logp = np.asarray(
        [[0.0, -10.0, -10.0], [0.0, -10.0, -10.0], [-10.0, -10.0, 0.0]],
        dtype=np.float32,
    )
    shared_options = {
        "first_pass_entry_bias": np.asarray([0.0, -0.5, 0.0], dtype=np.float32),
        "logp": logp,
        "sil_phone": "sil",
        "sil_phone_id": 1,
        "sph_phone": "sph",
        "sph_phone_id": None,
    }
    expected = reference.redecode_with_pruned_fixed_sequence(
        first_pass_ali=expected_first,
        first_pass_graph=expected_graph,
        args=redecode_args(),
        **shared_options,
    )
    actual = production.redecode_with_pruned_fixed_sequence(
        first_pass_ali=actual_first,
        first_pass_graph=actual_graph,
        config=production.Stage2DecodeConfig(
            p_stay=0.5,
            beam=16,
            boundary_lambda=0.0,
            boundary_context_s=0.02,
            frame_hop_s=0.01,
            min_sil_dur_ms=20.0,
            min_sph_dur_ms=50.0,
        ),
        **shared_options,
    )

    assert_reference_equivalent(expected[0].phone_segments_f, actual[0].phone_segments_f)
    assert_reference_equivalent(expected[0].word_segments_f, actual[0].word_segments_f)
    assert_reference_equivalent(expected[0].state_path, actual[0].state_path)
    assert_reference_equivalent(expected[0].aligned_phone_ids, actual[0].aligned_phone_ids)
    assert_reference_equivalent(_stats_mapping(expected[1]), _stats_mapping(actual[1]))
    second_alignment, stats = actual
    assert _stats_mapping(stats) == {
        "first_pass_states": 3,
        "fixed_states": 2,
        "dropped_short_sil": 1,
        "dropped_short_sph": 0,
    }
    assert second_alignment.aligned_phone_ids.tolist() == [0, 0, 2]
    assert second_alignment.phone_segments_f == [("A", 0, 2), ("B", 2, 3)]
    assert second_alignment.word_segments_f == [("left", 0, 2), ("right", 2, 3)]
