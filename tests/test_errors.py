from __future__ import annotations

import json
from types import ModuleType

import pytest


def test_feature_error_has_machine_readable_json_shape(public_api: ModuleType) -> None:
    with pytest.raises(public_api.FeatureNotAvailableError) as caught:
        public_api.get_capabilities().require("device.gpu")

    payload = caught.value.to_dict()

    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    assert json.loads(encoded) == payload
    assert payload == {
        "code": "feature_not_available",
        "message": payload["message"],
        "context": {
            "capability": "device.gpu",
            "status": "placeholder",
            "reason": payload["context"]["reason"],
        },
    }
    assert str(caught.value) == payload["message"]
