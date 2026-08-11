from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from tests.characterization.differential import (
    assert_reference_equivalent,
    first_mismatch,
)


@dataclass(frozen=True)
class _Interval:
    xmin: float
    xmax: float
    text: str


@dataclass(frozen=True)
class _Result:
    intervals: list[_Interval]
    scores: np.ndarray


def test_equivalent_nested_records_and_arrays_pass() -> None:
    expected = _Result([_Interval(0.0, 0.5, "one")], np.array([0.1, np.nan]))
    actual = _Result([_Interval(0.0, 0.5 + 1e-10, "one")], np.array([0.1, np.nan]))

    assert first_mismatch(expected, actual) is None
    assert_reference_equivalent(expected, actual)


def test_nested_record_difference_reports_exact_field_path() -> None:
    expected = _Result(
        [_Interval(0.0, 0.5, "one"), _Interval(0.5, 1.0, "two")],
        np.array([0.1, 0.2]),
    )
    actual = _Result(
        [_Interval(0.0, 0.5, "one"), _Interval(0.5, 1.2, "two")],
        np.array([0.1, 0.2]),
    )

    mismatch = first_mismatch(expected, actual)

    assert mismatch is not None
    assert mismatch.path == "$.intervals[1].xmax"
    assert mismatch.reason == "floating-point mismatch"


def test_array_difference_reports_first_index_and_values() -> None:
    expected = {"scores": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)}
    actual = {"scores": np.array([[1.0, 2.0], [3.5, 4.0]], dtype=np.float32)}

    mismatch = first_mismatch(expected, actual)

    assert mismatch is not None
    assert mismatch.path == "$['scores'][1][0]"
    assert mismatch.expected == 3.0
    assert mismatch.actual == 3.5


def test_assertion_message_carries_field_level_evidence() -> None:
    with pytest.raises(AssertionError, match=r"\$\['word_index'\].*expected=2, actual=3"):
        assert_reference_equivalent({"word_index": 2}, {"word_index": 3})
