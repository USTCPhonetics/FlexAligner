"""Model-free characterization of the reference TextGrid/output behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.characterization.reference_loader import load_reference_module


def _two_tier_textgrid(reference):
    return reference.TextGrid(
        xmin=0.0,
        xmax=1.0,
        tiers=[
            reference.IntervalTier(
                name="phones",
                xmin=0.0,
                xmax=1.0,
                intervals=[
                    reference.Interval(xmin=0.0, xmax=0.5, text='HH "quoted"'),
                    reference.Interval(xmin=0.5, xmax=1.0, text="AH"),
                ],
            ),
            reference.IntervalTier(
                name="words",
                xmin=0.0,
                xmax=1.0,
                intervals=[reference.Interval(xmin=0.0, xmax=1.0, text='say "hello"')],
            ),
        ],
    )


def _valid_atomic_textgrid(reference):
    return reference.TextGrid(
        xmin=0.0,
        xmax=1.0,
        tiers=[
            reference.IntervalTier(
                name="phones",
                xmin=0.0,
                xmax=1.0,
                intervals=[reference.Interval(xmin=0.0, xmax=1.0, text="HH")],
            ),
            reference.IntervalTier(
                name="words",
                xmin=0.0,
                xmax=1.0,
                intervals=[reference.Interval(xmin=0.0, xmax=1.0, text="hello")],
            ),
        ],
    )


def test_long_textgrid_round_trip_preserves_tier_order_and_quotes(tmp_path: Path) -> None:
    reference = load_reference_module()
    original = _two_tier_textgrid(reference)
    output = tmp_path / "nested" / "quoted.TextGrid"

    reference.write_textgrid_long(original, output)
    serialized = output.read_text(encoding="utf-8")
    parsed = reference.parse_textgrid_long(output)

    assert 'text = "HH ""quoted"""' in serialized
    assert 'text = "say ""hello"""' in serialized
    assert [tier.name for tier in parsed.tiers] == ["phones", "words"]
    assert [interval.text for interval in parsed.tiers[0].intervals] == [
        'HH "quoted"',
        "AH",
    ]
    assert [interval.text for interval in parsed.tiers[1].intervals] == ['say "hello"']
    assert parsed.xmin == 0.0
    assert parsed.xmax == 1.0


def test_merge_adjacent_merges_only_explicitly_allowed_null_labels() -> None:
    reference = load_reference_module()
    intervals = [
        reference.Interval(xmin=0.0, xmax=0.1, text="NULL"),
        reference.Interval(xmin=0.1, xmax=0.2, text="NULL"),
        reference.Interval(xmin=0.2, xmax=0.3, text="word"),
        reference.Interval(xmin=0.3, xmax=0.4, text="word"),
        reference.Interval(xmin=0.4, xmax=0.5, text="sil"),
        reference.Interval(xmin=0.5, xmax=0.6, text="sil"),
    ]

    merged = reference.merge_adjacent(intervals, merge_texts={"NULL"})

    assert [(item.xmin, item.xmax, item.text) for item in merged] == [
        (0.0, 0.2, "NULL"),
        (0.2, 0.3, "word"),
        (0.3, 0.4, "word"),
        (0.4, 0.5, "sil"),
        (0.5, 0.6, "sil"),
    ]


def test_clip_shift_interval_uses_local_coordinates_and_chunk_bounds() -> None:
    reference = load_reference_module()

    clipped_left = reference.clip_shift_interval(
        reference.Interval(xmin=-0.1, xmax=0.2, text="left"),
        chunk_start=10.0,
        chunk_end=11.0,
        local_xmax=1.0,
    )
    clipped_right = reference.clip_shift_interval(
        reference.Interval(xmin=0.8, xmax=1.2, text="right"),
        chunk_start=10.0,
        chunk_end=11.0,
        local_xmax=1.0,
    )
    outside = reference.clip_shift_interval(
        reference.Interval(xmin=1.1, xmax=1.2, text="outside"),
        chunk_start=10.0,
        chunk_end=11.0,
        local_xmax=1.0,
    )

    assert clipped_left == reference.Interval(xmin=10.0, xmax=10.2, text="left")
    assert clipped_right == reference.Interval(xmin=10.8, xmax=11.0, text="right")
    assert outside is None
    with pytest.raises(RuntimeError, match="Invalid chunk duration"):
        reference.clip_shift_interval(
            reference.Interval(xmin=0.0, xmax=0.1, text="bad"),
            chunk_start=2.0,
            chunk_end=2.0,
            local_xmax=0.1,
        )


def test_known_gap_limitation_validator_accepts_leading_internal_and_tail_gaps() -> None:
    """Lock the current limitation; this is not a continuous-coverage claim."""

    reference = load_reference_module()
    gapped = [
        reference.Interval(xmin=0.1, xmax=0.2, text="first"),
        reference.Interval(xmin=0.4, xmax=0.5, text="second"),
    ]
    textgrid = reference.TextGrid(
        xmin=0.0,
        xmax=1.0,
        tiers=[
            reference.IntervalTier(name="phones", xmin=0.0, xmax=1.0, intervals=list(gapped)),
            reference.IntervalTier(name="words", xmin=0.0, xmax=1.0, intervals=list(gapped)),
        ],
    )

    reference.validate_textgrid_structure(textgrid, context="known-gap-limitation")


def test_write_validated_textgrid_replaces_temp_after_success(tmp_path: Path) -> None:
    reference = load_reference_module()
    output = tmp_path / "result.TextGrid"
    temporary = tmp_path / "result.TextGrid.tmp"

    reference.write_validated_textgrid(
        textgrid=_valid_atomic_textgrid(reference),
        output_path=output,
        expected_words=["hello"],
        config=reference.AlignConfig(),
    )

    assert output.is_file()
    assert not temporary.exists()
    parsed = reference.parse_textgrid_long(output)
    assert [tier.name for tier in parsed.tiers] == ["phones", "words"]


def test_write_validated_textgrid_cleans_temp_after_validation_error(tmp_path: Path) -> None:
    reference = load_reference_module()
    output = tmp_path / "result.TextGrid"
    temporary = tmp_path / "result.TextGrid.tmp"

    with pytest.raises(RuntimeError, match="Word sequence mismatch"):
        reference.write_validated_textgrid(
            textgrid=_valid_atomic_textgrid(reference),
            output_path=output,
            expected_words=["different"],
            config=reference.AlignConfig(),
        )

    assert not output.exists()
    assert not temporary.exists()
