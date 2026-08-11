"""Independent exact-DP oracle for small Stage 2 phone graphs.

The reference decoder prunes to a beam after each frame.  This module keeps
every reachable ``(state, silence_lock)`` key, so a sufficiently wide reference
beam must agree with it.  It intentionally mirrors the *documented scoring
contract* rather than importing any implementation helpers from the reference.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

NEGATIVE_INFINITY = -1.0e30


@dataclass(frozen=True, slots=True)
class OracleAlignment:
    """Best complete path and its score, including the final move cost."""

    score: float
    state_path: np.ndarray


def _boundary_score_function(
    logp: np.ndarray,
    *,
    boundary_lambda: float,
    boundary_context_s: float,
    frame_hop_s: float,
) -> Callable[[int, int, int], float]:
    if boundary_lambda == 0.0:
        return lambda _time, _left_phone, _right_phone: 0.0

    frame_count, vocab_size = logp.shape
    context_frames = max(1, round(boundary_context_s / frame_hop_s))
    prefix = np.zeros((frame_count + 1, vocab_size), dtype=np.float32)
    prefix[1:] = np.cumsum(logp, axis=0)

    def mean(phone_id: int, start: int, end: int) -> float:
        if end <= start:
            return 0.0
        return float((prefix[end, phone_id] - prefix[start, phone_id]) / (end - start))

    def boundary_score(time: int, left_phone: int, right_phone: int) -> float:
        left_start = max(0, time - context_frames)
        right_end = min(frame_count, time + context_frames)
        left = mean(left_phone, left_start, time) - mean(right_phone, left_start, time)
        right = mean(right_phone, time, right_end) - mean(left_phone, time, right_end)
        return left + right

    return boundary_score


def _validate_inputs(graph: Any, logp: np.ndarray, entry_bias: np.ndarray, p_stay: float) -> None:
    if logp.ndim != 2 or logp.shape[0] == 0:
        raise ValueError(f"logp must have non-empty shape [T,V], got {logp.shape}")
    if entry_bias.shape != (len(graph.states),):
        raise ValueError(
            f"entry_bias must have shape ({len(graph.states)},), got {entry_bias.shape}"
        )
    if not 0.0 < p_stay < 1.0:
        raise ValueError(f"p_stay must be strictly between zero and one, got {p_stay}")


def exhaustive_viterbi(
    *,
    graph: Any,
    logp: np.ndarray,
    entry_bias: np.ndarray,
    p_stay: float = 0.92,
    boundary_lambda: float = 0.0,
    boundary_context_s: float = 0.015,
    frame_hop_s: float = 0.01,
    sil_phone_id: int | None = None,
    min_sil_dur_ms: float = 0.0,
    sil_enter_cost: float = 0.0,
    sph_phone_id: int | None = None,
    sph_enter_cost: float = 0.0,
) -> OracleAlignment:
    """Return the exact best complete path without beam pruning."""

    _validate_inputs(graph, logp, entry_bias, p_stay)
    frame_count = int(logp.shape[0])
    log_stay = math.log(p_stay)
    log_move = math.log(1.0 - p_stay)
    boundary_score = _boundary_score_function(
        logp,
        boundary_lambda=boundary_lambda,
        boundary_context_s=boundary_context_s,
        frame_hop_s=frame_hop_s,
    )
    min_sil_frames = 0
    if min_sil_dur_ms > 0.0 and sil_phone_id is not None:
        min_sil_frames = max(
            1,
            round(min_sil_dur_ms / 1000.0 / frame_hop_s),
        )

    def is_silence(phone_id: int) -> bool:
        return sil_phone_id is not None and phone_id == sil_phone_id

    def is_speech_gap(phone_id: int) -> bool:
        return sph_phone_id is not None and phone_id == sph_phone_id

    scores: dict[tuple[int, int], float] = {}
    initial_backpointers: dict[tuple[int, int], tuple[int, int]] = {}
    for state_id in graph.start_states:
        phone_id = int(graph.states[state_id].edge.phone_id)
        lock = min_sil_frames - 1 if is_silence(phone_id) and min_sil_frames else 0
        key = (int(state_id), int(lock))
        scores[key] = float(logp[0, phone_id]) + float(entry_bias[state_id])
        initial_backpointers[key] = key

    backpointers = [initial_backpointers]
    for time in range(1, frame_count):
        next_scores: dict[tuple[int, int], float] = {}
        next_backpointers: dict[tuple[int, int], tuple[int, int]] = {}
        for (state_id, previous_lock), score in scores.items():
            state = graph.states[state_id]
            previous_phone = int(state.edge.phone_id)
            previous_is_silence = is_silence(previous_phone)
            previous_is_speech_gap = is_speech_gap(previous_phone)

            stay_lock = previous_lock - 1 if previous_is_silence and previous_lock else 0
            stay_key = (state_id, int(stay_lock if previous_is_silence else 0))
            stay_score = (
                score + log_stay + float(logp[time, previous_phone]) + float(entry_bias[state_id])
            )
            if stay_score > next_scores.get(stay_key, NEGATIVE_INFINITY):
                next_scores[stay_key] = stay_score
                next_backpointers[stay_key] = (state_id, previous_lock)

            for next_state_id in state.succs:
                next_state = graph.states[next_state_id]
                next_phone = int(next_state.edge.phone_id)
                next_is_silence = is_silence(next_phone)
                next_is_speech_gap = is_speech_gap(next_phone)
                if previous_is_silence and previous_lock > 0 and not next_is_silence:
                    continue

                if next_is_silence:
                    if previous_is_silence:
                        next_lock = previous_lock - 1 if previous_lock else 0
                    else:
                        next_lock = min_sil_frames - 1 if min_sil_frames else 0
                else:
                    next_lock = 0
                next_key = (int(next_state_id), int(next_lock))

                enter_cost = 0.0
                if not previous_is_silence and next_is_silence:
                    enter_cost += sil_enter_cost
                if not previous_is_speech_gap and next_is_speech_gap:
                    enter_cost += sph_enter_cost
                move_score = (
                    score
                    + log_move
                    + float(logp[time, next_phone])
                    + float(entry_bias[next_state_id])
                    + enter_cost
                    + boundary_lambda * boundary_score(time, previous_phone, next_phone)
                )
                if move_score > next_scores.get(next_key, NEGATIVE_INFINITY):
                    next_scores[next_key] = move_score
                    next_backpointers[next_key] = (state_id, previous_lock)

        scores = next_scores
        backpointers.append(next_backpointers)

    end_states = set(graph.end_states)
    best_key: tuple[int, int] | None = None
    best_score = NEGATIVE_INFINITY
    for key, score in scores.items():
        terminal_score = score + log_move
        if key[0] in end_states and terminal_score > best_score:
            best_key = key
            best_score = terminal_score
    if best_key is None:
        raise RuntimeError("Exact Viterbi failed to reach a complete end state")

    state_path = np.empty(frame_count, dtype=np.int32)
    current_key = best_key
    for time in range(frame_count - 1, -1, -1):
        state_path[time] = current_key[0]
        current_key = backpointers[time].get(current_key, current_key)
    return OracleAlignment(score=best_score, state_path=state_path)


def score_state_path(
    *,
    graph: Any,
    state_path: np.ndarray,
    logp: np.ndarray,
    entry_bias: np.ndarray,
    p_stay: float = 0.92,
    boundary_lambda: float = 0.0,
    boundary_context_s: float = 0.015,
    frame_hop_s: float = 0.01,
    sil_phone_id: int | None = None,
    sil_enter_cost: float = 0.0,
    sph_phone_id: int | None = None,
    sph_enter_cost: float = 0.0,
) -> float:
    """Score one legal path, making one-time and per-frame terms explicit."""

    _validate_inputs(graph, logp, entry_bias, p_stay)
    if state_path.shape != (logp.shape[0],):
        raise ValueError(f"state_path must have shape ({logp.shape[0]},), got {state_path.shape}")
    if int(state_path[0]) not in graph.start_states:
        raise ValueError("state_path does not begin at a graph start state")
    if int(state_path[-1]) not in graph.end_states:
        raise ValueError("state_path does not finish at a graph end state")

    log_stay = math.log(p_stay)
    log_move = math.log(1.0 - p_stay)
    boundary_score = _boundary_score_function(
        logp,
        boundary_lambda=boundary_lambda,
        boundary_context_s=boundary_context_s,
        frame_hop_s=frame_hop_s,
    )
    first_state = int(state_path[0])
    first_phone = int(graph.states[first_state].edge.phone_id)
    score = float(logp[0, first_phone]) + float(entry_bias[first_state])

    for time in range(1, int(state_path.size)):
        previous_state = int(state_path[time - 1])
        next_state = int(state_path[time])
        previous_phone = int(graph.states[previous_state].edge.phone_id)
        next_phone = int(graph.states[next_state].edge.phone_id)
        score += float(logp[time, next_phone]) + float(entry_bias[next_state])
        if next_state == previous_state:
            score += log_stay
            continue
        if next_state not in graph.states[previous_state].succs:
            raise ValueError(
                f"Illegal graph move at frame {time}: {previous_state} -> {next_state}"
            )
        score += log_move
        if sil_phone_id is not None and previous_phone != sil_phone_id == next_phone:
            score += sil_enter_cost
        if sph_phone_id is not None and previous_phone != sph_phone_id == next_phone:
            score += sph_enter_cost
        score += boundary_lambda * boundary_score(time, previous_phone, next_phone)

    return score + log_move
