"""Dependency-free parsing for public device specifications."""

from __future__ import annotations

import re

from flexaligner.contracts import Device
from flexaligner.errors import ConfigurationError


def parse_device_spec(value: str) -> tuple[Device, int]:
    """Parse syntax only; CUDA availability is checked at runtime configuration."""

    if value in ("auto", "cpu", "cuda"):
        return Device(value), 0
    match = re.fullmatch(r"cuda:([0-9]+)", value) if isinstance(value, str) else None
    if match is not None:
        return Device.CUDA, int(match.group(1))
    raise ConfigurationError(
        "Device must be 'auto', 'cpu', 'cuda', or 'cuda:N' with N >= 0; MPS is unsupported.",
        context={"device": str(value)},
    )
