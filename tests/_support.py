from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

AVAILABLE_IDS = {
    "api.python",
    "cli",
    "capabilities.discovery",
    "alignment.single_file.en.cpu",
    "models.auto_download",
    "language.zh",
    "text.zh_segmentation",
    "pronunciation.g2p.zh.local",
    "audio.multi_format",
    "audio.auto_resample",
    "audio.transcode",
    "pronunciation.g2p.default",
    "pronunciation.g2p.en.local",
}

PLACEHOLDER_IDS = {
    "device.gpu",
    "alignment.batch",
    "integration.web",
    "models.auto_resolution.python",
    "confidence.calibration",
}


def parse_json_stream(text: str) -> dict[str, Any]:
    stripped = text.strip()
    assert stripped, "expected one JSON document, got an empty stream"
    payload = json.loads(stripped)
    assert isinstance(payload, dict), f"expected a JSON object, got {type(payload)!r}"
    return payload


class ExplodingIterable:
    """Detects batch implementations that consume input before capability checks."""

    def __iter__(self) -> Iterator[object]:
        raise AssertionError("placeholder align_batch consumed its iterable")
