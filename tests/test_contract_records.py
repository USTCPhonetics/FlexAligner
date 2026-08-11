from __future__ import annotations

import inspect
from dataclasses import fields, is_dataclass
from types import ModuleType
from typing import Any, cast

import pytest

IMMUTABLE_KW_ONLY_RECORDS = (
    "AlignmentRequest",
    "AlignmentOptions",
    "LocalModelBundle",
    "TextGridOutput",
    "AlignmentResult",
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
    }
    for record_name, field_names in expected.items():
        record_type = cast(Any, getattr(public_api, record_name))
        actual = {field.name for field in fields(record_type)}
        assert actual == field_names
