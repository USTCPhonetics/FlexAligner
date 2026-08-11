from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from flexaligner.core.stage1 import RuntimeChunk
from flexaligner.textgrid import (
    Interval,
    IntervalTier,
    LocalAlignment,
    TextGridDocument,
    clip_shift_interval,
    labels_from_intervals,
    merge_adjacent_null,
    merge_local_alignments,
    parse_textgrid_long,
    serialize_textgrid_long,
    validate_textgrid_structure,
)


def _document(
    *,
    duration: float = 1.0,
    phone_intervals: tuple[Interval, ...] | None = None,
    word_intervals: tuple[Interval, ...] | None = None,
) -> TextGridDocument:
    phones = phone_intervals or (Interval(0.0, duration, "HH"),)
    words = word_intervals or (Interval(0.0, duration, "hello"),)
    return TextGridDocument(
        0.0,
        duration,
        (
            IntervalTier("phones", 0.0, duration, phones),
            IntervalTier("words", 0.0, duration, words),
        ),
    )


def _chunk(
    chunk_id: str,
    start_ms: int,
    end_ms: int,
    words: list[str],
    word_indices: list[int],
) -> RuntimeChunk:
    return RuntimeChunk(
        chunk_id=chunk_id,
        start_ms=start_ms,
        end_ms=end_ms,
        start_sample=start_ms * 16,
        end_sample=end_ms * 16,
        words=words,
        word_indices=word_indices,
    )


def test_records_are_frozen_and_use_immutable_collections() -> None:
    document = _document()

    with pytest.raises(FrozenInstanceError):
        document.xmax = 2.0  # type: ignore[misc]
    assert isinstance(document.tiers, tuple)
    assert isinstance(document.tiers[0].intervals, tuple)


def test_long_textgrid_round_trip_preserves_quotes_and_tier_order(tmp_path: Path) -> None:
    document = _document(
        phone_intervals=(
            Interval(0.0, 0.5, 'HH "quoted"'),
            Interval(0.5, 1.0, "AH"),
        ),
        word_intervals=(Interval(0.0, 1.0, 'say "hello"'),),
    )
    path = tmp_path / "quoted.TextGrid"
    serialized = serialize_textgrid_long(document)
    path.write_text(serialized, encoding="utf-8")

    parsed = parse_textgrid_long(path)

    assert 'text = "HH ""quoted"""' in serialized
    assert parsed == document
    assert tuple(tier.name for tier in parsed.tiers) == ("phones", "words")


def test_parser_rejects_missing_bounds_and_missing_tiers(tmp_path: Path) -> None:
    missing_bounds = tmp_path / "bounds.TextGrid"
    missing_bounds.write_text('File type = "ooTextFile"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="global xmin/xmax"):
        parse_textgrid_long(missing_bounds)

    missing_tiers = tmp_path / "tiers.TextGrid"
    missing_tiers.write_text("xmin = 0\nxmax = 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="No IntervalTier"):
        parse_textgrid_long(missing_tiers)


def test_validator_requires_exact_tier_order_and_rejects_overlap() -> None:
    valid = _document()
    reversed_document = TextGridDocument(valid.xmin, valid.xmax, tuple(reversed(valid.tiers)))
    with pytest.raises(ValueError, match="Unexpected tier order"):
        validate_textgrid_structure(reversed_document, context="reversed")

    overlapping = _document(phone_intervals=(Interval(0.0, 0.7, "HH"), Interval(0.6, 1.0, "AH")))
    with pytest.raises(ValueError, match="Overlapping/backward"):
        validate_textgrid_structure(overlapping, context="overlap")


def test_validator_preserves_known_leading_internal_and_tail_gap_limitation() -> None:
    gapped = _document(
        phone_intervals=(Interval(0.1, 0.2, "HH"), Interval(0.4, 0.5, "AH")),
        word_intervals=(Interval(0.1, 0.2, "hello"), Interval(0.4, 0.5, "again")),
    )

    validate_textgrid_structure(gapped, context="TBD-ALG-001")


def test_clip_shift_uses_local_coordinates_and_chunk_bounds() -> None:
    assert clip_shift_interval(
        Interval(-0.1, 0.2, "left"),
        chunk_start=10.0,
        chunk_end=11.0,
        local_xmax=1.0,
    ) == Interval(10.0, 10.2, "left")
    assert clip_shift_interval(
        Interval(0.8, 1.2, "right"),
        chunk_start=10.0,
        chunk_end=11.0,
        local_xmax=1.0,
    ) == Interval(10.8, 11.0, "right")
    assert (
        clip_shift_interval(
            Interval(1.1, 1.2, "outside"),
            chunk_start=10.0,
            chunk_end=11.0,
            local_xmax=1.0,
        )
        is None
    )
    with pytest.raises(ValueError, match="Invalid chunk duration"):
        clip_shift_interval(
            Interval(0.0, 0.1, "bad"),
            chunk_start=2.0,
            chunk_end=2.0,
            local_xmax=0.1,
        )


def test_only_adjacent_null_intervals_are_merged() -> None:
    intervals = (
        Interval(0.0, 0.1, "NULL"),
        Interval(0.1, 0.2, "NULL"),
        Interval(0.2, 0.3, "word"),
        Interval(0.3, 0.4, "word"),
        Interval(0.4, 0.5, "sil"),
        Interval(0.5, 0.6, "sil"),
    )

    assert merge_adjacent_null(intervals) == (
        Interval(0.0, 0.2, "NULL"),
        Interval(0.2, 0.3, "word"),
        Interval(0.3, 0.4, "word"),
        Interval(0.4, 0.5, "sil"),
        Interval(0.5, 0.6, "sil"),
    )


def test_labels_are_trimmed_case_insensitively_and_ordered() -> None:
    intervals = (
        Interval(0.0, 0.1, " NULL "),
        Interval(0.1, 0.2, " Hello "),
        Interval(0.2, 0.3, "sil"),
        Interval(0.3, 0.4, "again"),
    )
    assert labels_from_intervals(intervals, ignore_labels={"null", "SIL"}) == (
        "Hello",
        "again",
    )


def test_merge_local_alignments_adds_outer_gaps_and_preserves_word_instances() -> None:
    chunks = (
        _chunk("utt.chunk001", 200, 500, ["same"], [0]),
        _chunk("utt.chunk002", 700, 1000, ["same"], [1]),
    )
    local = LocalAlignment(
        _document(
            duration=0.3,
            phone_intervals=(Interval(0.0, 0.3, "S"),),
            word_intervals=(Interval(0.0, 0.3, "same"),),
        )
    )

    merged = merge_local_alignments(
        chunks=chunks,
        local_alignments=(local, local),
        full_duration_s=1.2,
        expected_words=("same", "same"),
        word_sil_label="sil",
        sph_word_label="[missing]",
    )

    assert merged.tiers[1].intervals == (
        Interval(0.0, 0.2, "NULL"),
        Interval(0.2, 0.5, "same"),
        Interval(0.5, 0.7, "NULL"),
        Interval(0.7, 1.0, "same"),
        Interval(1.0, 1.2, "NULL"),
    )
    assert labels_from_intervals(
        merged.tiers[1].intervals,
        ignore_labels={"NULL", "sil", "[missing]"},
    ) == ("same", "same")


def test_merge_clips_local_intervals_to_chunk_duration() -> None:
    chunk = _chunk("utt.chunk001", 100, 300, ["hello"], [0])
    local = LocalAlignment(
        _document(
            duration=0.4,
            phone_intervals=(Interval(0.0, 0.4, "HH"),),
            word_intervals=(Interval(0.0, 0.4, "hello"),),
        )
    )

    merged = merge_local_alignments(
        chunks=(chunk,),
        local_alignments=(local,),
        full_duration_s=0.5,
        expected_words=("hello",),
        word_sil_label="sil",
        sph_word_label="[missing]",
    )

    assert Interval(0.1, 0.3, "hello") in merged.tiers[1].intervals


def test_merge_rejects_word_mismatch_count_mismatch_and_chunk_overlap() -> None:
    first = _chunk("utt.chunk001", 0, 400, ["hello"], [0])
    second = _chunk("utt.chunk002", 300, 600, ["again"], [1])
    hello = LocalAlignment(_document(duration=0.4))
    again = LocalAlignment(_document(duration=0.3, word_intervals=(Interval(0.0, 0.3, "again"),)))

    with pytest.raises(ValueError, match="count mismatch"):
        merge_local_alignments(
            chunks=(first,),
            local_alignments=(),
            full_duration_s=1.0,
            expected_words=("hello",),
            word_sil_label="sil",
            sph_word_label="[missing]",
        )
    with pytest.raises(ValueError, match="Word sequence mismatch"):
        merge_local_alignments(
            chunks=(first,),
            local_alignments=(hello,),
            full_duration_s=1.0,
            expected_words=("different",),
            word_sil_label="sil",
            sph_word_label="[missing]",
        )
    with pytest.raises(ValueError, match="Chunk overlap"):
        merge_local_alignments(
            chunks=(first, second),
            local_alignments=(hello, again),
            full_duration_s=1.0,
            expected_words=("hello", "again"),
            word_sil_label="sil",
            sph_word_label="[missing]",
        )
