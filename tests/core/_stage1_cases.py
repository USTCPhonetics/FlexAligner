"""Shared deterministic cases for Stage 1 reference/oracle/production parity."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np

EARLY_FINISH_LOG_PROBS = np.asarray(
    [
        [-5.0, 0.0, -9.0],
        [-5.0, -9.0, 0.0],
        [0.0, -9.0, -9.0],
        [-1.0, -9.0, -9.0],
    ],
    dtype=np.float64,
)
EARLY_FINISH_TARGETS = [1, 2]
EARLY_FINISH_FINAL_COLUMN = np.asarray([-np.inf, -np.inf, 0.0, 0.0, -1.0])

TIE_STAY_LOG_PROBS = np.asarray(
    [
        [-0.1, -2.0, -100.0],
        [-0.1, -2.0, -100.0],
        [-100.0, -100.0, 0.0],
    ],
    dtype=np.float64,
)
TIE_STAY_TARGETS = [1, 2]

REPEATED_TARGET_LOG_PROBS = np.asarray(
    [[-10.0, 0.0], [-10.0, 0.0]],
    dtype=np.float64,
)
REPEATED_TARGETS = [1, 1]


def make_rounding_case(
    module: Any,
    case: str,
) -> tuple[list[Any], list[str], int, int, str]:
    """Build one legacy-grid failure case using a module's own record class."""

    chunk: Callable[..., Any] = module.Chunk
    sample_rate = 16_000
    utterance_id = "utt"
    if case == "empty":
        return [], ["a"], sample_rate, sample_rate, utterance_id
    if case == "rounded_zero_duration":
        return [chunk(0.0011, 0.0014, ["a"], [0])], ["a"], sample_rate, sample_rate, utterance_id
    if case == "raw_tail_overflow":
        return [chunk(0.9, 1.000_002, ["a"], [0])], ["a"], sample_rate, sample_rate, utterance_id
    if case == "tail_clamp_zero_duration":
        samples = 30
        duration = samples / sample_rate
        return [chunk(0.0014, duration, ["a"], [0])], ["a"], samples, sample_rate, utterance_id
    if case == "negative_sample":
        return [chunk(-0.001, 0.1, ["a"], [0])], ["a"], sample_rate, sample_rate, utterance_id
    if case == "rounded_overlap":
        return (
            [
                chunk(0.0, 0.1, ["a"], [0]),
                chunk(0.0994, 0.2, ["b"], [1]),
            ],
            ["a", "b"],
            sample_rate,
            sample_rate,
            utterance_id,
        )
    if case == "nonmonotonic_indices":
        return (
            [chunk(0.0, 0.1, ["a", "b"], [1, 0])],
            ["a", "b"],
            sample_rate,
            sample_rate,
            utterance_id,
        )
    if case == "word_mismatch":
        return (
            [chunk(0.0, 0.1, ["a", "x"], [0, 1])],
            ["a", "b"],
            sample_rate,
            sample_rate,
            utterance_id,
        )
    if case == "word_mismatch_short":
        return (
            [chunk(0.0, 0.1, ["a"], [0])],
            ["a", "b"],
            sample_rate,
            sample_rate,
            utterance_id,
        )
    if case == "word_mismatch_extra":
        return (
            [chunk(0.0, 0.1, ["a", "b"], [0, 1])],
            ["a"],
            sample_rate,
            sample_rate,
            utterance_id,
        )
    if case == "index_coverage":
        return (
            [chunk(0.0, 0.1, ["a", "b"], [0, 2])],
            ["a", "b"],
            sample_rate,
            sample_rate,
            utterance_id,
        )
    raise AssertionError(f"Unknown Stage 1 rounding case: {case}")


ROUNDING_FAILURES = [
    ("empty", "no chunks"),
    ("rounded_zero_duration", "ms-rounded chunk span"),
    ("raw_tail_overflow", "Raw chunk exceeds audio duration"),
    ("tail_clamp_zero_duration", "tail clamp"),
    ("negative_sample", "chunk sample span"),
    ("rounded_overlap", "Overlapping chunks"),
    ("nonmonotonic_indices", "Non-monotonic word indices"),
    ("word_mismatch", "Token consistency check failed"),
    ("word_mismatch_short", "Token consistency check failed"),
    ("word_mismatch_extra", "Token consistency check failed"),
    ("index_coverage", "word-index coverage mismatch"),
]


def make_invalid_anchor_case(module: Any, case: str) -> tuple[list[Any], float]:
    """Build merge failure cases with the module's own WordAnchor class."""

    anchor: Callable[..., Any] = module.WordAnchor
    if case == "empty":
        return [], 0.2
    if case == "negative_gap":
        return [anchor(0, "a", 0, 1, 0.0, 0.1, 0.0, 0.2)], -0.1
    if case == "nonfinite_gap":
        return [anchor(0, "a", 0, 1, 0.0, 0.1, 0.0, 0.2)], math.nan
    if case == "invalid_span":
        return [anchor(0, "a", 0, 1, 0.0, 0.1, 0.2, 0.2)], 0.2
    if case == "nonmonotonic_indices":
        return (
            [
                anchor(1, "b", 0, 1, 0.0, 0.1, 0.0, 0.2),
                anchor(0, "a", 1, 2, 0.1, 0.2, 0.1, 0.3),
            ],
            0.2,
        )
    raise AssertionError(f"Unknown Stage 1 anchor case: {case}")


ANCHOR_MERGE_FAILURES = [
    ("empty", "empty word_anchors"),
    ("negative_gap", "anchor_merge_gap_s"),
    ("nonfinite_gap", "anchor_merge_gap_s"),
    ("invalid_span", "Invalid merged anchor chunk"),
    ("nonmonotonic_indices", "word_indices are not monotonic"),
]
