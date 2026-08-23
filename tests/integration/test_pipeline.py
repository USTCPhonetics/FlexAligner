from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from flexaligner import AlignmentOptions, ResourceLimits, ScoreKind
from flexaligner.adapters.lexicon_file import PronouncingLexicon
from flexaligner.adapters.wav_pcm16 import DecodedAudio
from flexaligner.errors import (
    AlignmentError,
    FlexAlignerError,
    InternalError,
    ModelCompatibilityError,
    ResourceLimitError,
    UnreachableAlignmentError,
)
from flexaligner.pipeline import (
    AlignmentPipeline,
    _stage1_from_posterior,
    _validate_posterior_vocabulary,
)
from flexaligner.ports import CtcPosterior
from flexaligner.textgrid import labels_from_intervals, parse_textgrid_long
from tests.integration._support import FakeInferenceFactory, make_integration_fixture


@pytest.mark.parametrize(
    ("words", "entries", "phone"),
    [
        pytest.param(
            ["within"],
            {"within": (("AA1", "AA2"),)},
            "AA",
            id="same-word-after-stress-stripping",
        ),
        pytest.param(
            ["first", "second"],
            {"first": (("S",),), "second": (("S",),)},
            "S",
            id="cross-word",
        ),
    ],
)
def test_stage1_pipeline_requires_blank_for_repeats_from_lexicon_context(
    words: list[str],
    entries: dict[str, tuple[tuple[str, ...], ...]],
    phone: str,
) -> None:
    log_probs = np.asarray(
        [[-10.0, 0.0], [0.0, -10.0], [-10.0, 0.0]],
        dtype=np.float32,
    )
    posterior = CtcPosterior(log_probs=log_probs, seconds_per_frame=0.01)
    audio = DecodedAudio(
        samples=np.zeros(16_000, dtype=np.float32),
        sample_rate=16_000,
        duration_s=1.0,
    )

    chunks, word_spans = _stage1_from_posterior(
        posterior=posterior,
        audio=audio,
        words=words,
        lexicon=PronouncingLexicon(entries=entries),
        vocabulary={"<pad>": 0, phone: 1},
        blank_id=0,
        utterance_id="repeat",
        options=AlignmentOptions(),
    )

    assert [span.word for span in word_spans] == words
    assert [span.pron for span in word_spans] == [[phone] * len(entries[word][0]) for word in words]
    assert word_spans[0].start_frame == 0
    assert word_spans[-1].end_frame == 3
    assert [word for chunk in chunks for word in chunk.words] == words


def test_real_numpy_cores_produce_validated_outputs_and_public_result(tmp_path: Path) -> None:
    fixture = make_integration_fixture(tmp_path)
    factory = FakeInferenceFactory()
    pipeline = AlignmentPipeline(inference_factory=factory)

    result = pipeline.align(
        request=fixture.request,
        models=fixture.models,
        lexicon_path=fixture.lexicon_path,
        options=AlignmentOptions(),
    )

    assert result.normalized_words == ("alpha", "beta")
    assert result.utterance_id == "fixture"
    assert result.audio_duration_s == pytest.approx(1.0)
    assert result.calibrated_scores is None
    assert len(result.raw_scores) == 2
    assert all(
        score.kind is ScoreKind.CHUNKER_EMISSION_GEOMETRIC_MEAN for score in result.raw_scores
    )
    assert all(score.calibrated is False for score in result.raw_scores)
    assert all(0.0 <= score.value <= 1.0 for score in result.raw_scores)
    assert result.output_path == fixture.request.output.path
    assert result.output_sha256 == hashlib.sha256(result.output_path.read_bytes()).hexdigest()
    assert [chunk.word_indices for chunk in result.chunks] == [(0, 1)]
    lexical_words = [interval for interval in result.words if interval.word_index is not None]
    assert lexical_words[0].start_s == pytest.approx(0.0)
    assert lexical_words[-1].end_s == pytest.approx(result.chunks[0].end_s, abs=0.011)

    parsed = parse_textgrid_long(result.output_path)
    assert [tier.name for tier in parsed.tiers] == ["phones", "words"]
    assert labels_from_intervals(
        parsed.tiers[1].intervals,
        ignore_labels={"NULL", "null", "sil", "[missing]", ""},
    ) == ("alpha", "beta")
    metadata_path = fixture.request.output.chunk_metadata_path
    assert metadata_path is not None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["schema_version"] == "1"
    assert metadata["score_kind"] == "chunker_emission_geometric_mean"
    assert metadata["calibrated"] is False
    assert [item["word"] for item in metadata["words"]] == ["alpha", "beta"]
    assert [item["value"] for item in metadata["words"]] == pytest.approx(
        [score.value for score in result.raw_scores]
    )


@pytest.mark.parametrize(
    ("fail_enter", "fail_infer"),
    [
        ("chunk", None),
        (None, "chunk"),
        ("align", None),
        (None, "align"),
    ],
)
def test_model_stage_failures_leave_no_official_or_temporary_artifact(
    tmp_path: Path,
    fail_enter: str | None,
    fail_infer: str | None,
) -> None:
    fixture = make_integration_fixture(tmp_path)
    pipeline = AlignmentPipeline(
        inference_factory=FakeInferenceFactory(
            fail_enter=fail_enter,
            fail_infer=fail_infer,
        )
    )

    with pytest.raises(FlexAlignerError):
        pipeline.align(
            request=fixture.request,
            models=fixture.models,
            lexicon_path=fixture.lexicon_path,
            options=AlignmentOptions(),
        )

    output_paths = [fixture.request.output.path, fixture.request.output.chunk_metadata_path]
    for path in output_paths:
        assert path is not None
        assert not path.exists()
        assert not path.with_name(path.name + ".tmp").exists()


def test_unreachable_stage2_is_typed_and_leaves_no_artifact(tmp_path: Path) -> None:
    fixture = make_integration_fixture(tmp_path)
    pipeline = AlignmentPipeline(inference_factory=FakeInferenceFactory(unreachable=True))

    with pytest.raises(UnreachableAlignmentError):
        pipeline.align(
            request=fixture.request,
            models=fixture.models,
            lexicon_path=fixture.lexicon_path,
            options=AlignmentOptions(),
        )
    assert not fixture.request.output.path.exists()


def test_stage2_beam_work_limit_is_typed_and_leaves_no_artifact(tmp_path: Path) -> None:
    fixture = make_integration_fixture(tmp_path)
    pipeline = AlignmentPipeline(inference_factory=FakeInferenceFactory())

    with pytest.raises(ResourceLimitError) as caught:
        pipeline.align(
            request=fixture.request,
            models=fixture.models,
            lexicon_path=fixture.lexicon_path,
            options=AlignmentOptions(limits=ResourceLimits(max_beam_work_units=1)),
        )

    assert caught.value.code == "resource_limit_exceeded"
    assert caught.value.context["limit"] == 1
    official_paths = (fixture.request.output.path, fixture.request.output.chunk_metadata_path)
    for path in official_paths:
        assert path is not None
        assert not path.exists()
        assert not path.with_name(path.name + ".tmp").exists()


def test_chunk_tokenizer_mapping_must_exactly_match_external_vocab(
    tmp_path: Path,
) -> None:
    fixture = make_integration_fixture(tmp_path)
    pipeline = AlignmentPipeline(
        inference_factory=FakeInferenceFactory(
            chunk_vocabulary={"<pad>": 0, "AH": 2, "B": 1},
        )
    )

    with pytest.raises(ModelCompatibilityError, match=r"does not match vocab\.json") as caught:
        pipeline.align(
            request=fixture.request,
            models=fixture.models,
            lexicon_path=fixture.lexicon_path,
            options=AlignmentOptions(),
        )
    assert caught.value.context["token"] == "AH"
    assert not fixture.request.output.path.exists()
    metadata_path = fixture.request.output.chunk_metadata_path
    assert metadata_path is not None and not metadata_path.exists()


def test_unknown_session_exception_is_chained_as_stable_internal_error(
    tmp_path: Path,
) -> None:
    fixture = make_integration_fixture(tmp_path)
    sentinel = OSError("injected raw port failure")
    factory = FakeInferenceFactory(raw_infer_failure=sentinel)
    pipeline = AlignmentPipeline(inference_factory=factory)

    with pytest.raises(InternalError) as caught:
        pipeline.align(
            request=fixture.request,
            models=fixture.models,
            lexicon_path=fixture.lexicon_path,
            options=AlignmentOptions(),
        )
    assert caught.value.__cause__ is sentinel
    assert caught.value.code == "internal_error"
    assert factory.active is None
    assert not fixture.request.output.path.exists()


def test_existing_output_fails_before_reading_missing_inputs(tmp_path: Path) -> None:
    fixture = make_integration_fixture(tmp_path)
    fixture.request.output.path.write_text("owned", encoding="utf-8")
    fixture.lexicon_path.unlink()
    pipeline = AlignmentPipeline(inference_factory=FakeInferenceFactory())

    with pytest.raises(FlexAlignerError) as caught:
        pipeline.align(
            request=fixture.request,
            models=fixture.models,
            lexicon_path=fixture.lexicon_path,
            options=AlignmentOptions(),
        )
    assert caught.value.code == "output_exists"
    assert fixture.request.output.path.read_text(encoding="utf-8") == "owned"


@pytest.mark.parametrize("reserved_word", ["NULL", "sil"])
def test_reserved_tier_word_is_rejected_before_model_loading(
    tmp_path: Path,
    reserved_word: str,
) -> None:
    fixture = make_integration_fixture(tmp_path)
    request = replace(fixture.request, transcript=f"alpha {reserved_word}")
    factory = FakeInferenceFactory()
    pipeline = AlignmentPipeline(inference_factory=factory)

    with pytest.raises(FlexAlignerError, match="reserved") as caught:
        pipeline.align(
            request=request,
            models=fixture.models,
            lexicon_path=fixture.lexicon_path,
            options=AlignmentOptions(),
        )
    assert caught.value.code == "input_validation_error"
    assert factory.trace == []
    assert not request.output.path.exists()


def test_core_failure_is_chained_as_alignment_error(tmp_path: Path) -> None:
    fixture = make_integration_fixture(tmp_path, metadata=False)
    factory = FakeInferenceFactory()
    pipeline = AlignmentPipeline(inference_factory=factory)
    fixture.lexicon_path.write_text("alpha AH\nbeta UNKNOWN\n", encoding="utf-8")

    with pytest.raises((AlignmentError, FlexAlignerError)) as caught:
        pipeline.align(
            request=fixture.request,
            models=fixture.models,
            lexicon_path=fixture.lexicon_path,
            options=AlignmentOptions(),
        )
    assert caught.value.code in {"alignment_failed", "input_validation_error"}
    assert not fixture.request.output.path.exists()


@pytest.mark.parametrize(
    "log_probs",
    [
        cast(Any, [[-0.69, -0.69]]),
        np.asarray([[0, 0]], dtype=np.int64),
        np.asarray([[float("nan"), -1.0]], dtype=np.float32),
        np.asarray([[0.1, -1.0]], dtype=np.float32),
        np.asarray([[-2.0, -2.0]], dtype=np.float32),
    ],
)
def test_posterior_contract_rejects_non_log_probability_arrays(log_probs: Any) -> None:
    posterior = CtcPosterior(log_probs=log_probs, seconds_per_frame=0.01)
    with pytest.raises(ModelCompatibilityError):
        _validate_posterior_vocabulary(posterior, expected_size=2, role="test")
