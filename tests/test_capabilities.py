from __future__ import annotations

import json
from types import ModuleType
from typing import Any, cast

import pytest

from tests._support import AVAILABLE_IDS, PLACEHOLDER_IDS


def _status_value(capability: object) -> str:
    status = cast(Any, capability).status
    return str(getattr(status, "value", status))


def test_capability_matrix_is_complete_and_versioned(public_api: ModuleType) -> None:
    report = public_api.get_capabilities()
    assert isinstance(report, public_api.CapabilityReport)

    payload = report.to_dict()
    json.dumps(payload, sort_keys=True)
    assert isinstance(payload, dict)
    assert payload.get("schema_version") == 1

    expected_ids = AVAILABLE_IDS | PLACEHOLDER_IDS
    for capability_id in sorted(expected_ids):
        capability = report.get(capability_id)
        assert isinstance(capability, public_api.Capability)

    assert {
        capability_id
        for capability_id in expected_ids
        if _status_value(report.get(capability_id)) == "available"
    } == AVAILABLE_IDS
    assert {
        capability_id
        for capability_id in expected_ids
        if _status_value(report.get(capability_id)) == "placeholder"
    } == PLACEHOLDER_IDS


def test_available_capability_require_returns_the_capability(
    public_api: ModuleType,
) -> None:
    report = public_api.get_capabilities()
    capability = report.require("api.python")
    assert capability is report.get("api.python")


@pytest.mark.parametrize("capability_id", sorted(PLACEHOLDER_IDS))
def test_placeholder_require_raises_stable_typed_error(
    public_api: ModuleType, capability_id: str
) -> None:
    report = public_api.get_capabilities()
    with pytest.raises(public_api.FeatureNotAvailableError) as caught:
        report.require(capability_id)

    error = caught.value
    assert isinstance(error, public_api.FlexAlignerError)
    payload = error.to_dict()
    assert set(payload) == {"code", "message", "context"}
    assert payload["code"] == "feature_not_available"
    assert isinstance(payload["message"], str) and payload["message"]
    assert payload["context"]["capability"] == capability_id
    assert payload["context"]["status"] == "placeholder"
    assert isinstance(payload["context"]["reason"], str)
    assert payload["context"]["reason"]
    json.dumps(payload, sort_keys=True)


def test_capability_report_is_deterministic(public_api: ModuleType) -> None:
    first = public_api.get_capabilities().to_dict()
    second = public_api.get_capabilities().to_dict()
    assert first == second
    assert json.dumps(first, sort_keys=True, separators=(",", ":")) == json.dumps(
        second, sort_keys=True, separators=(",", ":")
    )
