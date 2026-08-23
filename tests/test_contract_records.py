from __future__ import annotations

import inspect
from dataclasses import fields, is_dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest

IMMUTABLE_KW_ONLY_RECORDS = (
    "AlignmentRequest",
    "AlignmentOptions",
    "LocalModelBundle",
    "TextGridOutput",
    "AlignmentResult",
    "PhoneInterval",
    "PronunciationNotice",
)


@pytest.mark.parametrize("record_name", IMMUTABLE_KW_ONLY_RECORDS)
def test_public_records_are_frozen_keyword_only(public_api: ModuleType, record_name: str) -> None:
    record_type = cast(Any, getattr(public_api, record_name))
    assert is_dataclass(record_type), f"{record_name} must be a dataclass record"
    record_class = cast(Any, record_type)
    assert record_class.__dataclass_params__.frozen is True

    signature = inspect.signature(record_class)
    record_fields = fields(record_class)
    assert record_fields, f"{record_name} must have an explicit schema"
    for field in record_fields:
        parameter = signature.parameters[field.name]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY, (
            f"{record_name}.{field.name} must be keyword-only"
        )

    with pytest.raises(TypeError):
        record_class(*([None] * len(record_fields)))


def test_required_record_field_names(public_api: ModuleType) -> None:
    expected = {
        "LocalModelBundle": {"chunker_dir", "aligner_dir", "manifest_path"},
        "TextGridOutput": {"path", "chunk_metadata_path"},
        "AlignmentRequest": {
            "audio_path",
            "transcript",
            "output",
            "utterance_id",
        },
        "AlignmentOptions": {
            "language",
            "device",
            "algorithm_profile",
            "num_threads",
            "audio_policy",
            "pronunciation_mode",
            "model_resolution",
            "confidence_calibration",
            "limits",
        },
        "PhoneInterval": {
            "label",
            "start_s",
            "end_s",
            "word_index",
            "pronunciation_index",
            "phone_index",
        },
        "PronunciationNotice": {
            "code",
            "word",
            "word_indices",
            "pronunciation",
            "engine_id",
            "engine_version",
        },
    }
    for record_name, field_names in expected.items():
        record_type = cast(Any, getattr(public_api, record_name))
        actual = {field.name for field in fields(record_type)}
        assert actual == field_names


@pytest.mark.parametrize("value", [True, 1.5, 0, -1])
def test_integer_resource_limits_are_strict(
    public_api: ModuleType,
    value: object,
) -> None:
    with pytest.raises(public_api.ConfigurationError):
        public_api.ResourceLimits(max_trellis_cells=value)


def test_beam_work_limit_has_approved_default_and_rejects_none(
    public_api: ModuleType,
) -> None:
    assert public_api.ResourceLimits().max_beam_work_units == 200_000_000
    with pytest.raises(public_api.ConfigurationError):
        public_api.ResourceLimits(max_beam_work_units=None)


def test_approved_default_limits_cannot_be_disabled_with_none(public_api: ModuleType) -> None:
    for field_name in (
        "max_audio_seconds",
        "max_phone_tokens",
        "max_trellis_cells",
        "max_stage2_graph_states",
        "max_beam_work_units",
    ):
        with pytest.raises(public_api.ConfigurationError):
            public_api.ResourceLimits(**{field_name: None})
    with pytest.raises(public_api.ConfigurationError):
        public_api.AlignmentOptions(limits=None)


def test_alpha_resource_limits_have_approved_initial_defaults(public_api: ModuleType) -> None:
    limits = public_api.ResourceLimits()
    assert limits.max_audio_seconds == 900.0
    assert limits.max_phone_tokens == 10_000
    assert limits.max_trellis_cells == 200_000_000
    assert limits.max_stage2_graph_states == 10_000
    assert limits.max_beam_work_units == 200_000_000
    assert public_api.AlignmentOptions().limits == limits


@pytest.mark.parametrize("value", [True, 1.5, 0, -1])
def test_beam_work_limit_is_a_strict_positive_integer(
    public_api: ModuleType,
    value: object,
) -> None:
    with pytest.raises(public_api.ConfigurationError):
        public_api.ResourceLimits(max_beam_work_units=value)


@pytest.mark.parametrize("value", [True, 0.0, -1.0, float("nan"), float("inf")])
def test_audio_resource_limit_is_positive_finite_real(
    public_api: ModuleType,
    value: object,
) -> None:
    with pytest.raises(public_api.ConfigurationError):
        public_api.ResourceLimits(max_audio_seconds=value)


def test_path_records_reject_stringly_typed_paths(public_api: ModuleType) -> None:
    with pytest.raises(public_api.ConfigurationError):
        public_api.LocalModelBundle(chunker_dir="chunk", aligner_dir=Path("align"))
    with pytest.raises(public_api.ConfigurationError):
        public_api.TextGridOutput(path="result.TextGrid")
    with pytest.raises(public_api.ConfigurationError):
        public_api.AlignmentRequest(
            audio_path="input.wav",
            transcript="hello",
            output=public_api.TextGridOutput(path=Path("result.TextGrid")),
        )


@pytest.mark.parametrize("value", [True, 1.5, 0, -1])
def test_num_threads_is_a_positive_non_boolean_integer(
    public_api: ModuleType,
    value: object,
) -> None:
    with pytest.raises(public_api.ConfigurationError):
        public_api.AlignmentOptions(num_threads=value)
