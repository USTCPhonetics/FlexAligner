from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from enum import Enum
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from tests._support import ExplodingIterable


def _member(current: object, *candidate_names: str) -> Enum:
    enum_type = type(current)
    assert issubclass(enum_type, Enum), f"expected enum value, got {current!r}"
    by_name = {member.name.lower(): member for member in enum_type}
    by_value = {str(member.value).lower(): member for member in enum_type}
    for candidate in candidate_names:
        lowered = candidate.lower()
        if lowered in by_name:
            return by_name[lowered]
        if lowered in by_value:
            return by_value[lowered]
    raise AssertionError(
        f"{enum_type.__name__} is missing future placeholder member; "
        f"tried={candidate_names!r}, actual={list(enum_type)!r}"
    )


@pytest.fixture
def lazy_engine_and_request(public_api: ModuleType, tmp_path: Path) -> tuple[Any, Any]:
    models = public_api.LocalModelBundle(
        chunker_dir=tmp_path / "must-not-read-chunker",
        aligner_dir=tmp_path / "must-not-read-aligner",
    )
    output = public_api.TextGridOutput(path=tmp_path / "must-not-exist.TextGrid")
    request = public_api.AlignmentRequest(
        audio_path=tmp_path / "must-not-read.wav",
        transcript="hello",
        output=output,
        utterance_id="placeholder-probe",
    )
    return public_api.FlexAligner(models=models), request


def _assert_feature_error(
    public_api: ModuleType,
    expected_capability: str,
    action: Callable[[], object],
) -> None:
    with pytest.raises(public_api.FeatureNotAvailableError) as caught:
        action()
    payload = caught.value.to_dict()
    assert payload["code"] == "feature_not_available"
    assert payload["context"]["capability"] == expected_capability
    assert payload["context"]["status"] == "placeholder"


def test_default_alignment_requires_explicit_lexicon_before_input_io(
    public_api: ModuleType, lazy_engine_and_request: tuple[Any, Any]
) -> None:
    engine, request = lazy_engine_and_request
    with pytest.raises(public_api.ConfigurationError, match="lexicon_path"):
        engine.align(request)
    assert not request.output.path.exists()


@pytest.mark.parametrize(
    ("field_name", "future_names", "capability_id"),
    [
        ("device", ("gpu", "cuda"), "device.gpu"),
        (
            "model_resolution",
            ("auto_download", "remote", "download"),
            "models.auto_resolution.python",
        ),
        (
            "confidence_calibration",
            ("calibrated", "enabled"),
            "confidence.calibration",
        ),
    ],
)
def test_future_alignment_options_fail_before_core_or_input_io(
    public_api: ModuleType,
    lazy_engine_and_request: tuple[Any, Any],
    field_name: str,
    future_names: tuple[str, ...],
    capability_id: str,
) -> None:
    engine, request = lazy_engine_and_request
    defaults = public_api.AlignmentOptions()
    future_value = _member(getattr(defaults, field_name), *future_names)
    options = replace(defaults, **{field_name: future_value})

    _assert_feature_error(
        public_api,
        capability_id,
        lambda: engine.align(request, options=options),
    )
    assert not request.output.path.exists()


def test_align_batch_fails_without_consuming_iterable(
    public_api: ModuleType, lazy_engine_and_request: tuple[Any, Any]
) -> None:
    engine, _request = lazy_engine_and_request
    _assert_feature_error(
        public_api,
        "alignment.batch",
        lambda: engine.align_batch(ExplodingIterable()),
    )


def test_engine_capabilities_match_root_report(
    public_api: ModuleType, lazy_engine_and_request: tuple[Any, Any]
) -> None:
    engine, _request = lazy_engine_and_request
    assert engine.capabilities().to_dict() == public_api.get_capabilities().to_dict()


def test_close_and_context_manager_are_safe_before_models_exist(
    public_api: ModuleType, tmp_path: Path
) -> None:
    models = public_api.LocalModelBundle(
        chunker_dir=tmp_path / "missing-chunker",
        aligner_dir=tmp_path / "missing-aligner",
    )
    engine = public_api.FlexAligner(models=models)
    assert engine.__enter__() is engine
    engine.__exit__(None, None, None)
    engine.close()
