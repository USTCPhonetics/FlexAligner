"""Deterministic graph, decoder, and pruning cases for Stage 2 parity tests."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

CURRENT_BEHAVIOR_EQUAL_PHONE = (
    "accepted_behavior: adjacent equal-phone states keep pronunciation-position identity"
)

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


def pronouncing_dictionary(module: Any, **entries: list[list[str]]) -> Any:
    constructor = getattr(module, "PronouncingDictionary", None)
    return entries if constructor is None else constructor(lex=entries)


def manual_graph(
    module: Any,
    *,
    state_specs: list[tuple[str, int, int | None, str | None]],
    successors: list[tuple[int, ...]],
    start_states: list[int],
    end_states: list[int],
) -> Any:
    """Build an isomorphic graph using either reference or production records."""

    predecessors: list[list[int]] = [[] for _ in state_specs]
    for state_id, next_states in enumerate(successors):
        for next_state in next_states:
            predecessors[next_state].append(state_id)

    states = []
    for state_id, (phone, phone_id, word_index, word) in enumerate(state_specs):
        edge = module.EmitEdge(
            u=state_id,
            v=state_id + 1,
            phone=phone,
            phone_id=phone_id,
            word_index=word_index,
            word=word,
        )
        states.append(
            module.PhoneState(
                edge=edge,
                preds=tuple(predecessors[state_id]),
                succs=successors[state_id],
            )
        )
    return module.PhoneGraph(
        states=states,
        start_states=start_states,
        end_states=end_states,
    )


def all_complete_phone_paths(graph: Any) -> set[tuple[str, ...]]:
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


def internal_gap_paths(graph: Any, *, left_word_index: int = 0) -> set[tuple[str, ...]]:
    left_state = next(
        state_id
        for state_id, state in enumerate(graph.states)
        if state.edge.word_index == left_word_index
    )
    gap_paths: set[tuple[str, ...]] = set()

    def visit(state_id: int, gap_phones: tuple[str, ...], seen: frozenset[int]) -> None:
        if state_id in seen:
            raise AssertionError("Internal gap unexpectedly contains a cycle")
        for next_state_id in graph.states[state_id].succs:
            edge = graph.states[next_state_id].edge
            if edge.word_index == left_word_index + 1:
                gap_paths.add(gap_phones)
            elif edge.word_index is None:
                visit(next_state_id, (*gap_phones, edge.phone), seen | {state_id})

    visit(left_state, (), frozenset())
    return gap_paths


def transition_frame(state_path: np.ndarray, target_state: int) -> int:
    frames = np.flatnonzero(state_path == target_state)
    if frames.size == 0:
        raise AssertionError(f"state {target_state} was not visited: {state_path.tolist()}")
    return int(frames[0])


def fixed_spec(
    module: Any,
    phone: str,
    phone_id: int,
    word_index: int | None,
    word: str | None,
    *,
    bias: float = 0.0,
) -> Any:
    return module.FixedStateSpec(
        phone=phone,
        phone_id=phone_id,
        word_index=word_index,
        word=word,
        bias=bias,
    )


def redecode_args(**overrides: Any) -> SimpleNamespace:
    values: dict[str, Any] = {
        "min_sil_dur_ms": 20.0,
        "min_sph_dur_ms": 50.0,
        "frame_hop_s": 0.01,
        "p_stay": 0.5,
        "beam": 16,
        "word_sil_label": "sil",
        "boundary_lambda": 0.0,
        "boundary_context_s": 0.02,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


WIDE_BEAM_LOGP = np.random.default_rng(20260811).normal(size=(7, 4)).astype(np.float32)
WIDE_BEAM_ENTRY_BIAS = np.asarray([0.1, -0.2, 0.3, -0.1], dtype=np.float32)

SIX_INTERNAL_GAP_PATHS = {
    (),
    ("sil",),
    ("sph",),
    ("sil", "sph"),
    ("sph", "sil"),
    ("sil", "sph", "sil"),
}
