"""NumPy implementation of the reference Stage 2 alignment semantics.

The module is deliberately independent of model frameworks and of the frozen
reference snapshot.  It preserves the characterized pronunciation graph,
beam-Viterbi scoring, segmentation, short-gap pruning, and fixed-sequence
second decode behavior.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from numbers import Real
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from ..errors import ResourceLimitError

FloatArray = NDArray[np.floating[Any]]
IntArray = NDArray[np.integer[Any]]
FrameSegment = tuple[str, int, int]
PhoneProvenanceSegment = tuple[str, int, int, int | None, int | None, int | None]
StateSegment = tuple["FixedStateSpec", int, int]

NEGATIVE_INFINITY = -1.0e30


@dataclass(frozen=True, slots=True)
class EmitEdge:
    u: int
    v: int
    phone: str
    phone_id: int
    word_index: int | None
    word: str | None
    pronunciation_index: int | None = None
    phone_index: int | None = None


@dataclass(frozen=True, slots=True)
class PhoneState:
    edge: EmitEdge
    preds: tuple[int, ...]
    succs: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class PhoneGraph:
    states: list[PhoneState]
    start_states: list[int]
    end_states: list[int]


@dataclass(frozen=True, slots=True)
class ViterbiAlignment:
    phone_segments_f: list[FrameSegment]
    word_segments_f: list[FrameSegment]
    state_path: NDArray[np.int32]
    aligned_phone_ids: NDArray[np.int32]
    score: float
    phone_provenance_f: list[PhoneProvenanceSegment] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class FixedStateSpec:
    phone: str
    phone_id: int
    word_index: int | None
    word: str | None
    bias: float = 0.0
    pronunciation_index: int | None = None
    phone_index: int | None = None


@dataclass(frozen=True, slots=True)
class RedecodeStats:
    first_pass_states: int
    fixed_states: int
    dropped_short_sil: int
    dropped_short_sph: int


@dataclass(slots=True)
class BeamWorkBudget:
    """Request-scoped counter for actual beam candidate visits."""

    limit: int | None = 200_000_000
    used: int = 0

    def __post_init__(self) -> None:
        if self.limit is not None:
            _validate_positive_integer("limit", self.limit)
        if isinstance(self.used, bool) or not isinstance(self.used, int):
            raise TypeError(f"used must be an integer, got {type(self.used).__name__}")
        if self.used < 0:
            raise ValueError(f"used must be non-negative, got {self.used}")
        if self.limit is not None and self.used > self.limit:
            raise ValueError(
                f"used must not exceed limit, got used={self.used}, limit={self.limit}"
            )

    def consume(self, units: int) -> None:
        """Charge candidate visits, failing before the configured limit is crossed."""

        _validate_positive_integer("units", units)
        if self.limit is not None and self.used + units > self.limit:
            raise ResourceLimitError(
                "Stage 2 beam work limit exceeded",
                context={"used": self.used, "requested": units, "limit": self.limit},
            )
        self.used += units


@dataclass(frozen=True, slots=True)
class Stage2DecodeConfig:
    """Accepted English CPU Stage 2 profile.

    The beam is inherited parity behavior, not a measured resource guarantee.
    """

    p_stay: float = 0.92
    beam: int = 400
    boundary_lambda: float = 200.0
    boundary_context_s: float = 0.03
    frame_hop_s: float = 0.01
    min_sil_dur_ms: float = 65.0
    min_sph_dur_ms: float = 50.0
    sil_phone: str = "sil"
    sph_phone: str = "sph"
    sph_word_label: str = "[missing]"
    word_sil_label: str = "sil"

    def __post_init__(self) -> None:
        _validate_probability("p_stay", self.p_stay)
        _validate_positive_integer("beam", self.beam)
        _validate_finite_real("boundary_lambda", self.boundary_lambda)
        _validate_positive_real("boundary_context_s", self.boundary_context_s)
        _validate_positive_real("frame_hop_s", self.frame_hop_s)
        _validate_nonnegative_real("min_sil_dur_ms", self.min_sil_dur_ms)
        _validate_nonnegative_real("min_sph_dur_ms", self.min_sph_dur_ms)
        for name, value in (
            ("sil_phone", self.sil_phone),
            ("sph_phone", self.sph_phone),
            ("sph_word_label", self.sph_word_label),
            ("word_sil_label", self.word_sil_label),
        ):
            _validate_nonempty_string(name, value)


class PronunciationSource(Protocol):
    def get_prons(self, word: str) -> Sequence[Sequence[str]]: ...


LexiconMapping = Mapping[str, Sequence[Sequence[str]]]


def _is_string_like(value: object) -> bool:
    return isinstance(value, (str, bytes))


def _validate_nonempty_string(name: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string, got {value!r}")
    return value


def _validate_finite_real(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return result


def _validate_positive_real(name: str, value: object) -> float:
    result = _validate_finite_real(name, value)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return result


def _validate_nonnegative_real(name: str, value: object) -> float:
    result = _validate_finite_real(name, value)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative, got {value!r}")
    return result


def _validate_probability(name: str, value: object) -> float:
    result = _validate_finite_real(name, value)
    if not 0.0 < result < 1.0:
        raise ValueError(f"{name} must be strictly between 0 and 1, got {value!r}")
    return result


def _validate_positive_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _validate_optional_phone(name: str, value: object) -> str | None:
    if value is None:
        return None
    return _validate_nonempty_string(name, value)


def _validate_flag(name: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be bool, got {type(value).__name__}")
    return value


def _validate_optional_flag(name: str, value: object) -> bool | None:
    if value is None:
        return None
    return _validate_flag(name, value)


def _validate_nonnegative_id(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value


def epsilon_closure(
    num_nodes: int,
    epsilon_adjacency: Sequence[Sequence[int]],
) -> list[set[int]]:
    """Return epsilon-reachable nodes for every node, including itself."""

    _validate_positive_integer("num_nodes", num_nodes)
    if _is_string_like(epsilon_adjacency) or not isinstance(epsilon_adjacency, Sequence):
        raise TypeError("epsilon_adjacency must be a sequence of neighbor sequences")
    if len(epsilon_adjacency) != num_nodes:
        raise ValueError(
            "epsilon_adjacency length must equal num_nodes, "
            f"got length={len(epsilon_adjacency)}, num_nodes={num_nodes}"
        )
    adjacency: list[tuple[int, ...]] = []
    for node, neighbors in enumerate(epsilon_adjacency):
        if isinstance(neighbors, (str, bytes)) or not isinstance(neighbors, Sequence):
            raise TypeError(f"epsilon_adjacency[{node}] must be a sequence")
        validated: list[int] = []
        for position, neighbor in enumerate(neighbors):
            neighbor_id = _validate_nonnegative_id(
                f"epsilon_adjacency[{node}][{position}]", neighbor
            )
            if neighbor_id >= num_nodes:
                raise ValueError(
                    f"epsilon adjacency node out of range: source={node}, "
                    f"target={neighbor_id}, num_nodes={num_nodes}"
                )
            validated.append(neighbor_id)
        adjacency.append(tuple(validated))

    closure: list[set[int]] = []
    for node in range(num_nodes):
        seen = {node}
        stack = [node]
        while stack:
            current = stack.pop()
            for neighbor in adjacency[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        closure.append(seen)
    return closure


def _pronunciations_for_word(
    lexicon: LexiconMapping | PronunciationSource,
    word: str,
    word_index: int,
) -> list[list[str]]:
    if isinstance(lexicon, Mapping):
        if word not in lexicon:
            raise KeyError(f"Word not in lexicon: {word!r}")
        raw_pronunciations = lexicon[word]
    else:
        getter = getattr(lexicon, "get_prons", None)
        if not callable(getter):
            raise TypeError("lexicon must be a Mapping or provide callable get_prons(word)")
        raw_pronunciations = getter(word)
    if isinstance(raw_pronunciations, (str, bytes)) or not isinstance(raw_pronunciations, Sequence):
        raise TypeError(
            f"pronunciations must be a sequence at word_index={word_index}, word={word!r}"
        )
    if not raw_pronunciations:
        raise RuntimeError(f"Word has no pronunciations: {word!r}")
    pronunciations: list[list[str]] = []
    for pronunciation_index, raw_pronunciation in enumerate(raw_pronunciations):
        if isinstance(raw_pronunciation, (str, bytes)) or not isinstance(
            raw_pronunciation, Sequence
        ):
            raise TypeError(
                "pronunciation must be a phone sequence: "
                f"word_index={word_index}, pronunciation_index={pronunciation_index}"
            )
        if not raw_pronunciation:
            raise RuntimeError(
                "Empty pronunciation: "
                f"word_index={word_index}, word={word!r}, "
                f"pronunciation_index={pronunciation_index}"
            )
        pronunciation: list[str] = []
        for phone_index, phone in enumerate(raw_pronunciation):
            pronunciation.append(
                _validate_nonempty_string(
                    "phone at "
                    f"word_index={word_index}, pronunciation_index={pronunciation_index}, "
                    f"phone_index={phone_index}",
                    phone,
                )
            )
        pronunciations.append(pronunciation)
    return pronunciations


def _validate_phone_vocabulary(phone_to_id: Mapping[str, int]) -> None:
    if not isinstance(phone_to_id, Mapping):
        raise TypeError(f"phone_to_id must be a Mapping, got {type(phone_to_id).__name__}")
    if not phone_to_id:
        raise ValueError("phone_to_id must not be empty")
    for phone, phone_id in phone_to_id.items():
        _validate_nonempty_string("phone_to_id key", phone)
        _validate_nonnegative_id(f"phone_to_id[{phone!r}]", phone_id)


def build_phone_graph_optional_sil_sph(
    words: Sequence[str],
    lexicon: LexiconMapping | PronunciationSource,
    phone_to_id: Mapping[str, int],
    sil_phone: str | None = "SIL",
    optional_sil_between_words: bool = True,
    optional_sil_at_start: bool | None = None,
    optional_sil_at_end: bool | None = None,
    sil_cost: float = 0.0,
    sph_phone: str | None = "sph",
    optional_sph_between_words: bool = False,
    optional_sph_at_start: bool | None = None,
    optional_sph_at_end: bool | None = None,
    sph_cost: float = -2.5,
    sph_word_label: str = "[missing]",
    max_graph_states: int | None = None,
) -> tuple[PhoneGraph, NDArray[np.float32]]:
    """Build the characterized multi-pronunciation DAG with six gap paths."""

    if isinstance(words, (str, bytes)) or not isinstance(words, Sequence):
        raise TypeError(f"words must be a sequence of strings, got {type(words).__name__}")
    if not words:
        raise ValueError("words must not be empty")
    normalized_words = [
        _validate_nonempty_string(f"words[{word_index}]", word)
        for word_index, word in enumerate(words)
    ]
    _validate_phone_vocabulary(phone_to_id)
    sil_phone = _validate_optional_phone("sil_phone", sil_phone)
    sph_phone = _validate_optional_phone("sph_phone", sph_phone)
    optional_sil_between_words = _validate_flag(
        "optional_sil_between_words", optional_sil_between_words
    )
    optional_sph_between_words = _validate_flag(
        "optional_sph_between_words", optional_sph_between_words
    )
    optional_sil_at_start = _validate_optional_flag("optional_sil_at_start", optional_sil_at_start)
    optional_sil_at_end = _validate_optional_flag("optional_sil_at_end", optional_sil_at_end)
    optional_sph_at_start = _validate_optional_flag("optional_sph_at_start", optional_sph_at_start)
    optional_sph_at_end = _validate_optional_flag("optional_sph_at_end", optional_sph_at_end)
    sil_cost = _validate_finite_real("sil_cost", sil_cost)
    sph_cost = _validate_finite_real("sph_cost", sph_cost)
    sph_word_label = _validate_nonempty_string("sph_word_label", sph_word_label)
    if max_graph_states is not None:
        if isinstance(max_graph_states, bool) or not isinstance(max_graph_states, int):
            raise TypeError("max_graph_states must be an integer or None")
        if max_graph_states <= 0:
            raise ValueError("max_graph_states must be positive when provided")

    if optional_sil_at_start is None:
        optional_sil_at_start = optional_sil_between_words
    if optional_sil_at_end is None:
        optional_sil_at_end = optional_sil_between_words
    if optional_sph_at_start is None:
        optional_sph_at_start = optional_sph_between_words
    if optional_sph_at_end is None:
        optional_sph_at_end = optional_sph_between_words

    next_node = 0

    def new_node() -> int:
        nonlocal next_node
        node = next_node
        next_node += 1
        return node

    start_anchor = new_node()
    emit_edges: list[EmitEdge] = []
    epsilon_edges: list[tuple[int, int]] = []
    entry_bias: list[float] = []

    def add_emit(
        source: int,
        target: int,
        phone: str,
        word_index: int | None,
        word: str | None,
        bias: float = 0.0,
        pronunciation_index: int | None = None,
        phone_index: int | None = None,
    ) -> None:
        if phone not in phone_to_id:
            raise KeyError(f"Phone {phone!r} not in model vocab.")
        next_state_count = len(emit_edges) + 1
        if max_graph_states is not None and next_state_count > max_graph_states:
            raise ResourceLimitError(
                "Stage 2 graph-state limit exceeded before graph materialization",
                context={"states": next_state_count, "limit": max_graph_states},
            )
        emit_edges.append(
            EmitEdge(
                u=source,
                v=target,
                phone=phone,
                phone_id=phone_to_id[phone],
                word_index=word_index,
                word=word,
                pronunciation_index=pronunciation_index,
                phone_index=phone_index,
            )
        )
        entry_bias.append(bias)

    def add_epsilon(source: int, target: int) -> None:
        epsilon_edges.append((source, target))

    def add_silence(source: int, target: int) -> None:
        if sil_phone is not None:
            add_emit(source, target, sil_phone, None, None, sil_cost)

    def add_speech_gap(source: int, target: int) -> None:
        if sph_phone is not None:
            add_emit(source, target, sph_phone, None, sph_word_label, sph_cost)

    def add_optional_gap(source: int, target: int, allow_sil: bool, allow_sph: bool) -> None:
        add_epsilon(source, target)
        if allow_sil and sil_phone is not None:
            add_silence(source, target)
        if allow_sph and sph_phone is not None:
            add_speech_gap(source, target)
        if allow_sil and allow_sph and sil_phone is not None and sph_phone is not None:
            middle_one = new_node()
            add_silence(source, middle_one)
            add_speech_gap(middle_one, target)
            middle_two = new_node()
            add_speech_gap(source, middle_two)
            add_silence(middle_two, target)
            middle_three = new_node()
            middle_four = new_node()
            add_silence(source, middle_three)
            add_speech_gap(middle_three, middle_four)
            add_silence(middle_four, target)

    first_word_node = new_node()
    add_optional_gap(
        start_anchor,
        first_word_node,
        allow_sil=optional_sil_at_start,
        allow_sph=optional_sph_at_start,
    )
    current_node = first_word_node
    for word_index, word in enumerate(normalized_words):
        end_of_word = new_node()
        pronunciations = _pronunciations_for_word(lexicon, word, word_index)
        for pronunciation_index, pronunciation in enumerate(pronunciations):
            pronunciation_node = current_node
            for phone_index, phone in enumerate(pronunciation):
                next_phone_node = (
                    end_of_word if phone_index == len(pronunciation) - 1 else new_node()
                )
                add_emit(
                    pronunciation_node,
                    next_phone_node,
                    phone,
                    word_index,
                    word,
                    pronunciation_index=pronunciation_index,
                    phone_index=phone_index,
                )
                pronunciation_node = next_phone_node
        current_node = end_of_word
        if word_index != len(normalized_words) - 1:
            next_word_node = new_node()
            add_optional_gap(
                current_node,
                next_word_node,
                allow_sil=optional_sil_between_words,
                allow_sph=optional_sph_between_words,
            )
            current_node = next_word_node

    end_anchor = new_node()
    add_optional_gap(
        current_node,
        end_anchor,
        allow_sil=optional_sil_at_end,
        allow_sph=optional_sph_at_end,
    )

    epsilon_adjacency: list[list[int]] = [[] for _ in range(next_node)]
    reverse_epsilon_adjacency: list[list[int]] = [[] for _ in range(next_node)]
    for source, target in epsilon_edges:
        epsilon_adjacency[source].append(target)
        reverse_epsilon_adjacency[target].append(source)
    forward_closure = epsilon_closure(next_node, epsilon_adjacency)
    backward_closure = epsilon_closure(next_node, reverse_epsilon_adjacency)

    outgoing_emit: dict[int, list[int]] = {}
    incoming_emit: dict[int, list[int]] = {}
    for edge_index, edge in enumerate(emit_edges):
        outgoing_emit.setdefault(edge.u, []).append(edge_index)
        incoming_emit.setdefault(edge.v, []).append(edge_index)

    states: list[PhoneState] = []
    for edge in emit_edges:
        predecessors: list[int] = []
        for node in backward_closure[edge.u]:
            predecessors.extend(incoming_emit.get(node, ()))
        successors: list[int] = []
        for node in forward_closure[edge.v]:
            successors.extend(outgoing_emit.get(node, ()))
        states.append(
            PhoneState(
                edge=edge,
                preds=tuple(sorted(set(predecessors))),
                succs=tuple(sorted(set(successors))),
            )
        )

    start_states: list[int] = []
    for node in forward_closure[start_anchor]:
        start_states.extend(outgoing_emit.get(node, ()))
    start_states = sorted(set(start_states))
    if not start_states:
        raise RuntimeError("No start states. Check transcript/lexicon/SIL/SPH settings.")

    end_states = [
        state_index
        for state_index, state in enumerate(states)
        if end_anchor in forward_closure[state.edge.v]
    ]
    if not end_states:
        end_states = [state_index for state_index, state in enumerate(states) if not state.succs]
    if (
        optional_sph_at_start
        and sph_phone is not None
        and not any(states[state].edge.phone == sph_phone for state in start_states)
    ):
        raise RuntimeError("optional_sph_at_start=True, but SPH is not reachable from START.")
    if (
        optional_sph_at_end
        and sph_phone is not None
        and not any(states[state].edge.phone == sph_phone for state in end_states)
    ):
        raise RuntimeError("optional_sph_at_end=True, but SPH cannot terminate at END.")

    graph = PhoneGraph(states=states, start_states=start_states, end_states=end_states)
    _validate_graph(graph)
    return graph, np.asarray(entry_bias, dtype=np.float32)


def _validate_graph(graph: PhoneGraph) -> int:
    if not isinstance(graph, PhoneGraph):
        raise TypeError(f"graph must be PhoneGraph, got {type(graph).__name__}")
    state_count = len(graph.states)
    if state_count <= 0:
        raise ValueError("graph.states must not be empty")

    def validate_state_ids(name: str, state_ids: object) -> list[int]:
        if not isinstance(state_ids, list):
            raise TypeError(f"graph.{name} must be a list")
        if not state_ids:
            raise ValueError(f"graph.{name} must not be empty")
        validated: list[int] = []
        for position, state_id in enumerate(state_ids):
            value = _validate_nonnegative_id(f"graph.{name}[{position}]", state_id)
            if value >= state_count:
                raise ValueError(
                    f"graph.{name} state ID out of range: id={value}, states={state_count}"
                )
            validated.append(value)
        if len(set(validated)) != len(validated):
            raise ValueError(f"graph.{name} contains duplicate state IDs")
        return validated

    validate_state_ids("start_states", graph.start_states)
    validate_state_ids("end_states", graph.end_states)
    for state_id, state in enumerate(graph.states):
        if not isinstance(state, PhoneState):
            raise TypeError(f"graph.states[{state_id}] must be PhoneState")
        edge = state.edge
        if not isinstance(edge, EmitEdge):
            raise TypeError(f"graph.states[{state_id}].edge must be EmitEdge")
        _validate_nonnegative_id(f"graph.states[{state_id}].edge.u", edge.u)
        _validate_nonnegative_id(f"graph.states[{state_id}].edge.v", edge.v)
        _validate_nonempty_string(f"graph.states[{state_id}].edge.phone", edge.phone)
        _validate_nonnegative_id(f"graph.states[{state_id}].edge.phone_id", edge.phone_id)
        if edge.word_index is not None:
            _validate_nonnegative_id(f"graph.states[{state_id}].edge.word_index", edge.word_index)
        if edge.pronunciation_index is not None:
            _validate_nonnegative_id(
                f"graph.states[{state_id}].edge.pronunciation_index",
                edge.pronunciation_index,
            )
        if edge.phone_index is not None:
            _validate_nonnegative_id(f"graph.states[{state_id}].edge.phone_index", edge.phone_index)
        if edge.word_index is None and (
            edge.pronunciation_index is not None or edge.phone_index is not None
        ):
            raise ValueError(
                f"non-lexical graph state {state_id} must not have pronunciation provenance"
            )
        if edge.word is not None:
            _validate_nonempty_string(f"graph.states[{state_id}].edge.word", edge.word)
        for relation_name, related in (("preds", state.preds), ("succs", state.succs)):
            if not isinstance(related, tuple):
                raise TypeError(f"graph.states[{state_id}].{relation_name} must be a tuple")
            if len(set(related)) != len(related):
                raise ValueError(f"graph.states[{state_id}].{relation_name} contains duplicate IDs")
            for position, related_id in enumerate(related):
                value = _validate_nonnegative_id(
                    f"graph.states[{state_id}].{relation_name}[{position}]", related_id
                )
                if value >= state_count:
                    raise ValueError(
                        f"graph relation out of range: state={state_id}, "
                        f"relation={relation_name}, target={value}, states={state_count}"
                    )
    for state_id, state in enumerate(graph.states):
        for predecessor in state.preds:
            if state_id not in graph.states[predecessor].succs:
                raise ValueError(
                    f"graph predecessor/successor mismatch: {predecessor} -> {state_id}"
                )
        for successor in state.succs:
            if state_id not in graph.states[successor].preds:
                raise ValueError(f"graph successor/predecessor mismatch: {state_id} -> {successor}")
    remaining_predecessors = [len(state.preds) for state in graph.states]
    frontier = [
        state_id
        for state_id, predecessor_count in enumerate(remaining_predecessors)
        if predecessor_count == 0
    ]
    visited = 0
    while frontier:
        state_id = frontier.pop()
        visited += 1
        for successor in graph.states[state_id].succs:
            remaining_predecessors[successor] -= 1
            if remaining_predecessors[successor] == 0:
                frontier.append(successor)
    if visited != state_count:
        raise ValueError(
            f"graph contains a successor cycle: visited={visited}, states={state_count}"
        )
    return state_count


def _validate_float_matrix(name: str, value: object) -> FloatArray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array, got {type(value).__name__}")
    if value.ndim != 2:
        raise ValueError(f"{name} must have shape [frames, vocabulary], got {value.shape}")
    if value.shape[0] <= 0 or value.shape[1] <= 0:
        raise ValueError(f"{name} dimensions must be positive, got shape={value.shape}")
    if not np.issubdtype(value.dtype, np.floating):
        raise TypeError(f"{name} dtype must be floating, got {value.dtype}")
    if not bool(np.isfinite(value).all()):
        raise ValueError(f"{name} contains NaN or infinity")
    return value


def _validate_entry_bias(value: object, state_count: int) -> FloatArray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"entry_bias must be a NumPy array, got {type(value).__name__}")
    if value.ndim != 1 or value.shape != (state_count,):
        raise ValueError(f"entry_bias must have shape ({state_count},), got shape={value.shape}")
    if not np.issubdtype(value.dtype, np.floating):
        raise TypeError(f"entry_bias dtype must be floating, got {value.dtype}")
    if not bool(np.isfinite(value).all()):
        raise ValueError("entry_bias contains NaN or infinity")
    return value


def _validate_optional_vocab_id(
    name: str,
    value: object,
    vocabulary_size: int,
) -> int | None:
    if value is None:
        return None
    phone_id = _validate_nonnegative_id(name, value)
    if phone_id >= vocabulary_size:
        raise ValueError(f"{name} out of range: id={phone_id}, vocabulary_size={vocabulary_size}")
    return phone_id


def _validate_decoder_inputs(
    *,
    logp: object,
    graph: PhoneGraph,
    entry_bias: object,
    p_stay: object,
    beam_size: object,
    word_sil_label: object,
    boundary_lambda: object,
    boundary_context_s: object,
    frame_hop_s: object,
    sil_phone_id: object,
    min_sil_dur_ms: object,
    sil_enter_cost: object,
    sph_phone_id: object,
    sph_enter_cost: object,
) -> tuple[FloatArray, FloatArray, int, int, int | None, int | None]:
    logp_array = _validate_float_matrix("logp", logp)
    state_count = _validate_graph(graph)
    entry_bias_array = _validate_entry_bias(entry_bias, state_count)
    _validate_probability("p_stay", p_stay)
    beam = _validate_positive_integer("beam_size", beam_size)
    _validate_nonempty_string("word_sil_label", word_sil_label)
    _validate_finite_real("boundary_lambda", boundary_lambda)
    boundary_context = _validate_positive_real("boundary_context_s", boundary_context_s)
    frame_hop = _validate_positive_real("frame_hop_s", frame_hop_s)
    context_ratio = boundary_context / frame_hop
    if not math.isfinite(context_ratio):
        raise ValueError(
            "boundary_context_s/frame_hop_s must be finite, "
            f"got boundary_context_s={boundary_context}, frame_hop_s={frame_hop}"
        )
    minimum_silence = _validate_nonnegative_real("min_sil_dur_ms", min_sil_dur_ms)
    silence_ratio = minimum_silence / 1000.0 / frame_hop
    if not math.isfinite(silence_ratio):
        raise ValueError(
            "min_sil_dur_ms/frame_hop_s must be finite, "
            f"got min_sil_dur_ms={minimum_silence}, frame_hop_s={frame_hop}"
        )
    _validate_finite_real("sil_enter_cost", sil_enter_cost)
    _validate_finite_real("sph_enter_cost", sph_enter_cost)
    vocabulary_size = int(logp_array.shape[1])
    silence_id = _validate_optional_vocab_id("sil_phone_id", sil_phone_id, vocabulary_size)
    speech_gap_id = _validate_optional_vocab_id("sph_phone_id", sph_phone_id, vocabulary_size)
    for state_id, state in enumerate(graph.states):
        if state.edge.phone_id >= vocabulary_size:
            raise ValueError(
                "graph phone ID out of range for logp: "
                f"state={state_id}, phone_id={state.edge.phone_id}, "
                f"vocabulary_size={vocabulary_size}"
            )
    return (
        logp_array,
        entry_bias_array,
        state_count,
        beam,
        silence_id,
        speech_gap_id,
    )


def _prune_beam(
    scores: dict[tuple[int, int], float],
    backpointers: dict[tuple[int, int], tuple[int, int]],
    beam_size: int,
) -> tuple[dict[tuple[int, int], float], dict[tuple[int, int], tuple[int, int]]]:
    if len(scores) <= beam_size:
        return scores, backpointers
    top = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:beam_size]
    return (
        {key: score for key, score in top},
        {key: backpointers[key] for key, _score in top},
    )


def _frame_segments_from_path(
    graph: PhoneGraph,
    path: NDArray[np.int32],
    *,
    by_word: bool,
    word_sil_label: str,
) -> list[FrameSegment]:
    first_edge = graph.states[int(path[0])].edge
    current_label = first_edge.word if by_word and first_edge.word is not None else first_edge.phone
    if by_word and first_edge.word is None:
        current_label = word_sil_label
    current_word_index = first_edge.word_index
    current_pronunciation_index = first_edge.pronunciation_index
    current_phone_index = first_edge.phone_index
    start = 0
    segments: list[FrameSegment] = []
    for frame in range(1, int(path.size)):
        edge = graph.states[int(path[frame])].edge
        label = (edge.word if edge.word is not None else word_sil_label) if by_word else edge.phone
        phone_identity_changed = not by_word and (
            edge.word_index != current_word_index
            or edge.pronunciation_index != current_pronunciation_index
            or edge.phone_index != current_phone_index
        )
        if (
            label != current_label
            or (by_word and edge.word_index != current_word_index)
            or phone_identity_changed
        ):
            segments.append((current_label, start, frame))
            current_label = label
            current_word_index = edge.word_index
            current_pronunciation_index = edge.pronunciation_index
            current_phone_index = edge.phone_index
            start = frame
    segments.append((current_label, start, int(path.size)))
    return segments


def _phone_provenance_segments_from_path(
    graph: PhoneGraph,
    path: NDArray[np.int32],
) -> list[PhoneProvenanceSegment]:
    segments = _frame_segments_from_path(
        graph,
        path,
        by_word=False,
        word_sil_label="sil",
    )
    result: list[PhoneProvenanceSegment] = []
    for label, start, end in segments:
        edge = graph.states[int(path[start])].edge
        result.append(
            (
                label,
                start,
                end,
                edge.word_index,
                edge.pronunciation_index,
                edge.phone_index,
            )
        )
    return result


def align_beam_viterbi(
    logp: FloatArray,
    graph: PhoneGraph,
    entry_bias: FloatArray,
    p_stay: float = 0.92,
    beam_size: int = 300,
    word_sil_label: str = "sil",
    boundary_lambda: float = 0.0,
    boundary_context_s: float = 0.015,
    frame_hop_s: float = 0.01,
    sil_phone_id: int | None = None,
    min_sil_dur_ms: float = 0.0,
    sil_enter_cost: float = 0.0,
    sph_phone_id: int | None = None,
    sph_enter_cost: float = 0.0,
    beam_work_budget: BeamWorkBudget | None = None,
) -> ViterbiAlignment:
    """Decode the best complete emitting-state path with a stable beam."""

    (
        logp,
        entry_bias,
        state_count,
        beam_size,
        sil_phone_id,
        sph_phone_id,
    ) = _validate_decoder_inputs(
        logp=logp,
        graph=graph,
        entry_bias=entry_bias,
        p_stay=p_stay,
        beam_size=beam_size,
        word_sil_label=word_sil_label,
        boundary_lambda=boundary_lambda,
        boundary_context_s=boundary_context_s,
        frame_hop_s=frame_hop_s,
        sil_phone_id=sil_phone_id,
        min_sil_dur_ms=min_sil_dur_ms,
        sil_enter_cost=sil_enter_cost,
        sph_phone_id=sph_phone_id,
        sph_enter_cost=sph_enter_cost,
    )
    if beam_work_budget is not None and not isinstance(beam_work_budget, BeamWorkBudget):
        raise TypeError(
            "beam_work_budget must be a BeamWorkBudget or None, "
            f"got {type(beam_work_budget).__name__}"
        )
    frame_count, vocabulary_size = (int(logp.shape[0]), int(logp.shape[1]))
    log_stay = math.log(float(p_stay))
    log_move = math.log(1.0 - float(p_stay))
    context_frames = max(1, round(boundary_context_s / frame_hop_s))

    if boundary_lambda != 0.0:
        prefix = np.zeros((frame_count + 1, vocabulary_size), dtype=np.float32)
        prefix[1:] = np.cumsum(logp, axis=0)

        def mean(phone_id: int, start: int, end: int) -> float:
            if end <= start:
                return 0.0
            return float((prefix[end, phone_id] - prefix[start, phone_id]) / (end - start))

        def boundary_score(frame: int, left_phone_id: int, right_phone_id: int) -> float:
            left_start = max(0, frame - context_frames)
            right_end = min(frame_count, frame + context_frames)
            left = mean(left_phone_id, left_start, frame) - mean(right_phone_id, left_start, frame)
            right = mean(right_phone_id, frame, right_end) - mean(left_phone_id, frame, right_end)
            return left + right

    else:

        def boundary_score(frame: int, left_phone_id: int, right_phone_id: int) -> float:
            del frame, left_phone_id, right_phone_id
            return 0.0

    minimum_silence_frames = 0
    if min_sil_dur_ms > 0.0 and sil_phone_id is not None:
        minimum_silence_frames = max(
            1,
            round(min_sil_dur_ms / 1000.0 / frame_hop_s),
        )

    def is_silence(phone_id: int) -> bool:
        return sil_phone_id is not None and phone_id == sil_phone_id

    def is_speech_gap(phone_id: int) -> bool:
        return sph_phone_id is not None and phone_id == sph_phone_id

    all_backpointers: list[dict[tuple[int, int], tuple[int, int]]] = []
    current_scores: dict[tuple[int, int], float] = {}
    initial_backpointers: dict[tuple[int, int], tuple[int, int]] = {}
    if beam_work_budget is not None:
        beam_work_budget.consume(len(graph.start_states))
    for state_id in graph.start_states:
        phone_id = graph.states[state_id].edge.phone_id
        lock = (
            minimum_silence_frames - 1 if is_silence(phone_id) and minimum_silence_frames > 0 else 0
        )
        key = (int(state_id), int(lock))
        current_scores[key] = float(logp[0, phone_id]) + float(entry_bias[state_id])
        initial_backpointers[key] = key
    current_scores, initial_backpointers = _prune_beam(
        current_scores,
        initial_backpointers,
        beam_size,
    )
    all_backpointers.append(initial_backpointers)

    for frame in range(1, frame_count):
        next_scores: dict[tuple[int, int], float] = {}
        next_backpointers: dict[tuple[int, int], tuple[int, int]] = {}
        for (state_id, previous_lock), score in current_scores.items():
            state = graph.states[state_id]
            if beam_work_budget is not None:
                beam_work_budget.consume(1 + len(state.succs))
            previous_phone_id = state.edge.phone_id
            previous_is_silence = is_silence(previous_phone_id)
            previous_is_speech_gap = is_speech_gap(previous_phone_id)

            stay_score = (
                score
                + log_stay
                + float(logp[frame, previous_phone_id])
                + float(entry_bias[state_id])
            )
            stay_lock = previous_lock - 1 if previous_is_silence and previous_lock > 0 else 0
            stay_key = (state_id, int(stay_lock if previous_is_silence else 0))
            if stay_score > next_scores.get(stay_key, NEGATIVE_INFINITY):
                next_scores[stay_key] = stay_score
                next_backpointers[stay_key] = (state_id, previous_lock)

            move_base = score + log_move
            for next_state_id in state.succs:
                next_state = graph.states[next_state_id]
                next_phone_id = next_state.edge.phone_id
                next_is_silence = is_silence(next_phone_id)
                next_is_speech_gap = is_speech_gap(next_phone_id)
                if previous_is_silence and previous_lock > 0 and not next_is_silence:
                    continue
                if next_is_silence:
                    if previous_is_silence:
                        next_lock = previous_lock - 1 if previous_lock > 0 else 0
                    else:
                        next_lock = minimum_silence_frames - 1 if minimum_silence_frames > 0 else 0
                else:
                    next_lock = 0
                next_key = (int(next_state_id), int(next_lock))
                enter_cost = 0.0
                if not previous_is_silence and next_is_silence:
                    enter_cost += float(sil_enter_cost)
                if not previous_is_speech_gap and next_is_speech_gap:
                    enter_cost += float(sph_enter_cost)
                move_score = (
                    move_base
                    + float(logp[frame, next_phone_id])
                    + float(entry_bias[next_state_id])
                    + enter_cost
                    + float(boundary_lambda)
                    * boundary_score(frame, previous_phone_id, next_phone_id)
                )
                if move_score > next_scores.get(next_key, NEGATIVE_INFINITY):
                    next_scores[next_key] = move_score
                    next_backpointers[next_key] = (state_id, previous_lock)
        current_scores, next_backpointers = _prune_beam(
            next_scores,
            next_backpointers,
            beam_size,
        )
        all_backpointers.append(next_backpointers)

    end_states = set(graph.end_states)
    best_key: tuple[int, int] | None = None
    best_score = NEGATIVE_INFINITY
    if beam_work_budget is not None:
        beam_work_budget.consume(len(current_scores))
    for key, score in current_scores.items():
        terminal_score = score + log_move
        if key[0] in end_states and terminal_score > best_score:
            best_key = key
            best_score = terminal_score
    if best_key is None:
        raise RuntimeError(
            "Viterbi failed to reach any end state. "
            f"T={frame_count}, num_states={state_count}, beam_size={beam_size}, "
            f"num_end_states={len(graph.end_states)}, active_states={len(current_scores)}"
        )

    state_path = np.empty((frame_count,), dtype=np.int32)
    current_key = best_key
    for frame in range(frame_count - 1, -1, -1):
        state_path[frame] = current_key[0]
        current_key = all_backpointers[frame].get(current_key, current_key)
    aligned_phone_ids = np.asarray(
        [graph.states[int(state_id)].edge.phone_id for state_id in state_path],
        dtype=np.int32,
    )
    phone_segments = _frame_segments_from_path(
        graph,
        state_path,
        by_word=False,
        word_sil_label=word_sil_label,
    )
    word_segments = _frame_segments_from_path(
        graph,
        state_path,
        by_word=True,
        word_sil_label=word_sil_label,
    )
    return ViterbiAlignment(
        phone_segments_f=phone_segments,
        word_segments_f=word_segments,
        state_path=state_path,
        aligned_phone_ids=aligned_phone_ids,
        score=float(best_score),
        phone_provenance_f=_phone_provenance_segments_from_path(graph, state_path),
    )


def _validate_state_path(
    graph: PhoneGraph,
    path: object,
    *,
    require_complete: bool,
) -> IntArray:
    state_count = _validate_graph(graph)
    if not isinstance(path, np.ndarray):
        raise TypeError(f"path must be a NumPy array, got {type(path).__name__}")
    if path.ndim != 1 or path.size <= 0:
        raise ValueError(f"path must be a non-empty one-dimensional array, got {path.shape}")
    if not np.issubdtype(path.dtype, np.integer):
        raise TypeError(f"path dtype must be integer, got {path.dtype}")
    state_ids = [int(state_id) for state_id in path]
    for frame, state_id in enumerate(state_ids):
        if state_id < 0 or state_id >= state_count:
            raise ValueError(
                f"path state ID out of range: frame={frame}, state_id={state_id}, "
                f"states={state_count}"
            )
    if state_ids[0] not in graph.start_states:
        raise ValueError(f"path must begin at a graph start state, got state_id={state_ids[0]}")
    for frame, (previous_state, next_state) in enumerate(
        pairwise(state_ids),
        start=1,
    ):
        if next_state != previous_state and next_state not in graph.states[previous_state].succs:
            raise ValueError(
                "path contains an illegal graph transition: "
                f"frame={frame}, previous_state={previous_state}, next_state={next_state}"
            )
    if require_complete and state_ids[-1] not in graph.end_states:
        raise ValueError(f"path must finish at a graph end state, got state_id={state_ids[-1]}")
    return path


def extract_state_segments_from_path(
    graph: PhoneGraph,
    entry_bias: FloatArray,
    path: IntArray,
) -> list[StateSegment]:
    """Collapse a frame path by phone metadata, retaining the current quirk."""

    state_count = _validate_graph(graph)
    entry_bias = _validate_entry_bias(entry_bias, state_count)
    path = _validate_state_path(graph, path, require_complete=True)

    first_state_id = int(path[0])
    current_edge = graph.states[first_state_id].edge
    current_bias = float(entry_bias[first_state_id])
    start = 0
    segments: list[StateSegment] = []
    for frame in range(1, int(path.size)):
        state_id = int(path[frame])
        edge = graph.states[state_id].edge
        bias = float(entry_bias[state_id])
        if (
            edge.phone != current_edge.phone
            or edge.phone_id != current_edge.phone_id
            or edge.word_index != current_edge.word_index
            or edge.word != current_edge.word
            or edge.pronunciation_index != current_edge.pronunciation_index
            or edge.phone_index != current_edge.phone_index
        ):
            segments.append(
                (
                    FixedStateSpec(
                        phone=current_edge.phone,
                        phone_id=current_edge.phone_id,
                        word_index=current_edge.word_index,
                        word=current_edge.word,
                        bias=current_bias,
                        pronunciation_index=current_edge.pronunciation_index,
                        phone_index=current_edge.phone_index,
                    ),
                    start,
                    frame,
                )
            )
            current_edge = edge
            current_bias = bias
            start = frame
    segments.append(
        (
            FixedStateSpec(
                phone=current_edge.phone,
                phone_id=current_edge.phone_id,
                word_index=current_edge.word_index,
                word=current_edge.word,
                bias=current_bias,
                pronunciation_index=current_edge.pronunciation_index,
                phone_index=current_edge.phone_index,
            ),
            start,
            int(path.size),
        )
    )
    return segments


def _validate_fixed_state_spec(spec: object, *, context: str) -> FixedStateSpec:
    if not isinstance(spec, FixedStateSpec):
        raise TypeError(f"{context} must be FixedStateSpec, got {type(spec).__name__}")
    _validate_nonempty_string(f"{context}.phone", spec.phone)
    _validate_nonnegative_id(f"{context}.phone_id", spec.phone_id)
    if spec.word_index is not None:
        _validate_nonnegative_id(f"{context}.word_index", spec.word_index)
    if spec.pronunciation_index is not None:
        _validate_nonnegative_id(f"{context}.pronunciation_index", spec.pronunciation_index)
    if spec.phone_index is not None:
        _validate_nonnegative_id(f"{context}.phone_index", spec.phone_index)
    if spec.word_index is None and (
        spec.pronunciation_index is not None or spec.phone_index is not None
    ):
        raise ValueError(f"{context} non-lexical state must not have pronunciation provenance")
    if spec.word is not None:
        _validate_nonempty_string(f"{context}.word", spec.word)
    _validate_finite_real(f"{context}.bias", spec.bias)
    return spec


def prune_short_internal_sil_sph_segments(
    state_segments: Sequence[StateSegment],
    *,
    sil_phone: str | None,
    sph_phone: str | None,
    min_sil_dur_ms: float,
    min_sph_dur_ms: float,
    frame_hop_s: float,
) -> tuple[list[FixedStateSpec], RedecodeStats]:
    """Drop only sub-threshold internal silence and speech-gap states."""

    if _is_string_like(state_segments) or not isinstance(state_segments, Sequence):
        raise TypeError("state_segments must be a sequence")
    if not state_segments:
        raise RuntimeError("Cannot prune an empty first-pass state sequence.")
    sil_phone = _validate_optional_phone("sil_phone", sil_phone)
    sph_phone = _validate_optional_phone("sph_phone", sph_phone)
    minimum_silence = _validate_nonnegative_real("min_sil_dur_ms", min_sil_dur_ms)
    minimum_speech_gap = _validate_nonnegative_real("min_sph_dur_ms", min_sph_dur_ms)
    frame_hop = _validate_positive_real("frame_hop_s", frame_hop_s)
    silence_ratio = minimum_silence / 1000.0 / frame_hop
    speech_gap_ratio = minimum_speech_gap / 1000.0 / frame_hop
    if not math.isfinite(silence_ratio) or not math.isfinite(speech_gap_ratio):
        raise ValueError(
            "duration threshold/frame_hop_s must be finite, "
            f"min_sil_dur_ms={minimum_silence}, "
            f"min_sph_dur_ms={minimum_speech_gap}, frame_hop_s={frame_hop}"
        )
    silence_threshold_frames = math.ceil(silence_ratio)
    speech_gap_threshold_frames = math.ceil(speech_gap_ratio)

    validated_segments: list[StateSegment] = []
    previous_end: int | None = None
    for segment_index, raw_segment in enumerate(state_segments):
        if not isinstance(raw_segment, tuple) or len(raw_segment) != 3:
            raise TypeError(f"state_segments[{segment_index}] must be a (spec, start, end) tuple")
        spec, raw_start, raw_end = raw_segment
        spec = _validate_fixed_state_spec(spec, context=f"state_segments[{segment_index}][0]")
        start = _validate_nonnegative_id(f"state_segments[{segment_index}].start", raw_start)
        end = _validate_nonnegative_id(f"state_segments[{segment_index}].end", raw_end)
        if end <= start:
            raise RuntimeError(
                "Non-positive first-pass state duration at "
                f"segment {segment_index}: phone={spec.phone!r}, start={start}, end={end}"
            )
        if previous_end is None and start != 0:
            raise ValueError(f"state_segments must start at frame 0, got first_start={start}")
        if previous_end is not None and start != previous_end:
            raise ValueError(
                "state_segments must be ordered and contiguous: "
                f"segment={segment_index}, previous_end={previous_end}, start={start}"
            )
        validated_segments.append((spec, start, end))
        previous_end = end

    kept: list[FixedStateSpec] = []
    dropped_silence = 0
    dropped_speech_gap = 0
    for segment_index, (spec, start, end) in enumerate(validated_segments):
        duration_frames = end - start
        is_boundary = segment_index == 0 or segment_index == len(validated_segments) - 1
        if is_boundary:
            kept.append(spec)
            continue
        is_silence = sil_phone is not None and spec.phone == sil_phone
        is_speech_gap = sph_phone is not None and spec.phone == sph_phone
        if is_silence and duration_frames < silence_threshold_frames:
            dropped_silence += 1
            continue
        if is_speech_gap and duration_frames < speech_gap_threshold_frames:
            dropped_speech_gap += 1
            continue
        kept.append(spec)
    if not kept:
        raise RuntimeError(
            "All first-pass states were removed during short SIL/SPH pruning. "
            "Check min_sil_dur_ms/min_sph_dur_ms."
        )
    return kept, RedecodeStats(
        first_pass_states=len(validated_segments),
        fixed_states=len(kept),
        dropped_short_sil=dropped_silence,
        dropped_short_sph=dropped_speech_gap,
    )


def build_fixed_sequence_graph(
    specs: Sequence[FixedStateSpec],
) -> tuple[PhoneGraph, NDArray[np.float32]]:
    """Build the linear graph used to re-estimate fixed token boundaries."""

    if _is_string_like(specs) or not isinstance(specs, Sequence):
        raise TypeError("specs must be a sequence of FixedStateSpec records")
    if not specs:
        raise RuntimeError("Cannot build a fixed-sequence graph from an empty sequence.")
    validated_specs = [
        _validate_fixed_state_spec(spec, context=f"specs[{index}]")
        for index, spec in enumerate(specs)
    ]
    states: list[PhoneState] = []
    biases: list[float] = []
    for state_id, spec in enumerate(validated_specs):
        states.append(
            PhoneState(
                edge=EmitEdge(
                    u=state_id,
                    v=state_id + 1,
                    phone=spec.phone,
                    phone_id=spec.phone_id,
                    word_index=spec.word_index,
                    word=spec.word,
                    pronunciation_index=spec.pronunciation_index,
                    phone_index=spec.phone_index,
                ),
                preds=(state_id - 1,) if state_id > 0 else (),
                succs=(state_id + 1,) if state_id + 1 < len(validated_specs) else (),
            )
        )
        biases.append(float(spec.bias))
    graph = PhoneGraph(
        states=states,
        start_states=[0],
        end_states=[len(states) - 1],
    )
    _validate_graph(graph)
    return graph, np.asarray(biases, dtype=np.float32)


def _validate_frame_segments(
    name: str,
    segments: object,
    frame_count: int,
) -> None:
    if not isinstance(segments, list) or not segments:
        raise ValueError(f"{name} must be a non-empty list")
    previous_end = 0
    for segment_index, segment in enumerate(segments):
        if not isinstance(segment, tuple) or len(segment) != 3:
            raise TypeError(f"{name}[{segment_index}] must be a (label, start, end) tuple")
        label, raw_start, raw_end = segment
        _validate_nonempty_string(f"{name}[{segment_index}].label", label)
        start = _validate_nonnegative_id(f"{name}[{segment_index}].start", raw_start)
        end = _validate_nonnegative_id(f"{name}[{segment_index}].end", raw_end)
        if start != previous_end or end <= start:
            raise ValueError(
                f"{name} must be positive, ordered, and contiguous: "
                f"segment={segment_index}, previous_end={previous_end}, "
                f"start={start}, end={end}"
            )
        previous_end = end
    if previous_end != frame_count:
        raise ValueError(
            f"{name} must cover all frames: covered_end={previous_end}, frames={frame_count}"
        )


def _validate_alignment(
    alignment: object,
    graph: PhoneGraph,
) -> ViterbiAlignment:
    if not isinstance(alignment, ViterbiAlignment):
        raise TypeError(f"first_pass_ali must be ViterbiAlignment, got {type(alignment).__name__}")
    path = _validate_state_path(graph, alignment.state_path, require_complete=True)
    if not isinstance(alignment.aligned_phone_ids, np.ndarray):
        raise TypeError("first_pass_ali.aligned_phone_ids must be a NumPy array")
    if alignment.aligned_phone_ids.shape != path.shape:
        raise ValueError(
            "first_pass_ali.aligned_phone_ids shape must match state_path, "
            f"got ids={alignment.aligned_phone_ids.shape}, path={path.shape}"
        )
    if not np.issubdtype(alignment.aligned_phone_ids.dtype, np.integer):
        raise TypeError(
            "first_pass_ali.aligned_phone_ids dtype must be integer, "
            f"got {alignment.aligned_phone_ids.dtype}"
        )
    expected_phone_ids = np.asarray(
        [graph.states[int(state_id)].edge.phone_id for state_id in path],
        dtype=np.int64,
    )
    if not bool(np.array_equal(alignment.aligned_phone_ids, expected_phone_ids)):
        raise ValueError("first_pass_ali.aligned_phone_ids does not match state_path")
    _validate_frame_segments(
        "first_pass_ali.phone_segments_f",
        alignment.phone_segments_f,
        int(path.size),
    )
    _validate_frame_segments(
        "first_pass_ali.word_segments_f",
        alignment.word_segments_f,
        int(path.size),
    )
    _validate_finite_real("first_pass_ali.score", alignment.score)
    return alignment


def redecode_with_pruned_fixed_sequence(
    *,
    first_pass_ali: ViterbiAlignment,
    first_pass_graph: PhoneGraph,
    first_pass_entry_bias: FloatArray,
    logp: FloatArray,
    sil_phone: str | None,
    sil_phone_id: int | None,
    sph_phone: str | None,
    sph_phone_id: int | None,
    config: Stage2DecodeConfig,
    beam_work_budget: BeamWorkBudget | None = None,
) -> tuple[ViterbiAlignment, RedecodeStats]:
    """Prune short internal gaps and decode the remaining fixed sequence."""

    if not isinstance(config, Stage2DecodeConfig):
        raise TypeError(f"config must be Stage2DecodeConfig, got {type(config).__name__}")
    first_pass_ali = _validate_alignment(first_pass_ali, first_pass_graph)
    first_pass_segments = extract_state_segments_from_path(
        graph=first_pass_graph,
        entry_bias=first_pass_entry_bias,
        path=first_pass_ali.state_path,
    )
    fixed_specs, stats = prune_short_internal_sil_sph_segments(
        first_pass_segments,
        sil_phone=sil_phone,
        sph_phone=sph_phone,
        min_sil_dur_ms=config.min_sil_dur_ms,
        min_sph_dur_ms=config.min_sph_dur_ms,
        frame_hop_s=config.frame_hop_s,
    )
    fixed_graph, fixed_entry_bias = build_fixed_sequence_graph(fixed_specs)
    second_pass_alignment = align_beam_viterbi(
        logp=logp,
        graph=fixed_graph,
        entry_bias=fixed_entry_bias,
        p_stay=config.p_stay,
        beam_size=config.beam,
        word_sil_label=config.word_sil_label,
        boundary_lambda=config.boundary_lambda,
        boundary_context_s=config.boundary_context_s,
        frame_hop_s=config.frame_hop_s,
        sil_phone_id=sil_phone_id,
        min_sil_dur_ms=0.0,
        sil_enter_cost=0.0,
        sph_phone_id=sph_phone_id,
        sph_enter_cost=0.0,
        beam_work_budget=beam_work_budget,
    )
    return second_pass_alignment, stats
