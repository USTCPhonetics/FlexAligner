from __future__ import annotations

import subprocess
import sys

import pytest

from flexaligner.contracts import Device
from flexaligner.devices import parse_device_spec
from flexaligner.errors import ConfigurationError


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("auto", (Device.AUTO, 0)),
        ("cpu", (Device.CPU, 0)),
        ("cuda", (Device.CUDA, 0)),
        ("cuda:2", (Device.CUDA, 2)),
    ],
)
def test_parse_device_spec(text: str, expected: tuple[Device, int]) -> None:
    assert parse_device_spec(text) == expected


@pytest.mark.parametrize(
    "text", ["", "CUDA", "cuda:-1", "cuda:x", "cpu:0", "auto:1", "mps", "cuda:1:2"]
)
def test_invalid_device_specs_are_configuration_errors(text: str) -> None:
    with pytest.raises(ConfigurationError):
        parse_device_spec(text)


def test_parse_device_does_not_import_torch() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from flexaligner.devices import parse_device_spec; parse_device_spec('cuda:3'); assert 'torch' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
