"""Field-level comparison helpers for reference-versus-rebuild tests.

This module intentionally has no golden-update function. Behavioral changes are
classified and approved outside the comparator before expected values change.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass

import numpy as np


@dataclass(frozen=True)
class DifferentialMismatch:
    path: str
    expected: object
    actual: object
    reason: str

    def describe(self) -> str:
        return (
            f"differential mismatch at {self.path}: {self.reason}; "
            f"expected={self.expected!r}, actual={self.actual!r}"
        )


def first_mismatch(
    expected: object,
    actual: object,
    *,
    path: str = "$",
    rel_tol: float = 1e-7,
    abs_tol: float = 1e-9,
) -> DifferentialMismatch | None:
    """Return the first stable, field-addressed difference, if one exists."""
    if is_dataclass(expected) or is_dataclass(actual):
        if not (is_dataclass(expected) and is_dataclass(actual)):
            return DifferentialMismatch(path, expected, actual, "dataclass type mismatch")
        expected_fields = tuple(field.name for field in fields(expected))
        actual_fields = tuple(field.name for field in fields(actual))
        if expected_fields != actual_fields:
            return DifferentialMismatch(
                path,
                expected_fields,
                actual_fields,
                "dataclass field mismatch",
            )
        for field_name in expected_fields:
            mismatch = first_mismatch(
                getattr(expected, field_name),
                getattr(actual, field_name),
                path=f"{path}.{field_name}",
                rel_tol=rel_tol,
                abs_tol=abs_tol,
            )
            if mismatch is not None:
                return mismatch
        return None

    if isinstance(expected, np.ndarray) or isinstance(actual, np.ndarray):
        if not (isinstance(expected, np.ndarray) and isinstance(actual, np.ndarray)):
            return DifferentialMismatch(path, expected, actual, "array type mismatch")
        if expected.shape != actual.shape:
            return DifferentialMismatch(path, expected.shape, actual.shape, "array shape mismatch")
        if expected.dtype != actual.dtype:
            return DifferentialMismatch(path, expected.dtype, actual.dtype, "array dtype mismatch")
        if expected.size == 0:
            return None
        if np.issubdtype(expected.dtype, np.inexact):
            equal = np.isclose(expected, actual, rtol=rel_tol, atol=abs_tol, equal_nan=True)
        else:
            equal = expected == actual
        if bool(np.all(equal)):
            return None
        first_index = tuple(int(part) for part in np.argwhere(~equal)[0])
        index_path = "".join(f"[{part}]" for part in first_index)
        return DifferentialMismatch(
            f"{path}{index_path}",
            expected[first_index].item(),
            actual[first_index].item(),
            "array value mismatch",
        )

    if isinstance(expected, Mapping) or isinstance(actual, Mapping):
        if not (isinstance(expected, Mapping) and isinstance(actual, Mapping)):
            return DifferentialMismatch(path, expected, actual, "mapping type mismatch")
        expected_keys = set(expected)
        actual_keys = set(actual)
        if expected_keys != actual_keys:
            return DifferentialMismatch(
                path,
                sorted(expected_keys, key=repr),
                sorted(actual_keys, key=repr),
                "mapping key mismatch",
            )
        for key in sorted(expected_keys, key=repr):
            mismatch = first_mismatch(
                expected[key],
                actual[key],
                path=f"{path}[{key!r}]",
                rel_tol=rel_tol,
                abs_tol=abs_tol,
            )
            if mismatch is not None:
                return mismatch
        return None

    sequence_types = (list, tuple)
    if isinstance(expected, sequence_types) or isinstance(actual, sequence_types):
        if not (isinstance(expected, sequence_types) and isinstance(actual, sequence_types)):
            return DifferentialMismatch(path, expected, actual, "sequence type mismatch")
        expected_sequence: Sequence[object] = expected
        actual_sequence: Sequence[object] = actual
        if len(expected_sequence) != len(actual_sequence):
            return DifferentialMismatch(
                path,
                len(expected_sequence),
                len(actual_sequence),
                "sequence length mismatch",
            )
        for index, (expected_item, actual_item) in enumerate(
            zip(expected_sequence, actual_sequence, strict=True)
        ):
            mismatch = first_mismatch(
                expected_item,
                actual_item,
                path=f"{path}[{index}]",
                rel_tol=rel_tol,
                abs_tol=abs_tol,
            )
            if mismatch is not None:
                return mismatch
        return None

    if isinstance(expected, (float, np.floating)) or isinstance(actual, (float, np.floating)):
        if not isinstance(expected, (int, float, np.integer, np.floating)) or not isinstance(
            actual, (int, float, np.integer, np.floating)
        ):
            return DifferentialMismatch(path, expected, actual, "numeric type mismatch")
        if math.isclose(float(expected), float(actual), rel_tol=rel_tol, abs_tol=abs_tol):
            return None
        return DifferentialMismatch(path, expected, actual, "floating-point mismatch")

    if expected != actual:
        return DifferentialMismatch(path, expected, actual, "value mismatch")
    return None


def assert_reference_equivalent(
    expected: object,
    actual: object,
    *,
    rel_tol: float = 1e-7,
    abs_tol: float = 1e-9,
) -> None:
    mismatch = first_mismatch(expected, actual, rel_tol=rel_tol, abs_tol=abs_tol)
    if mismatch is not None:
        raise AssertionError(mismatch.describe())
