#!/usr/bin/env python3
"""Verify that the public inference extra resolves to the frozen alpha versions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

EXPECTED = {"torch": "2.3.1", "transformers": "4.41.2"}


def canonicalize(name: str) -> str:
    return name.lower().replace("_", "-").replace(".", "-")


def resolved_versions(report: dict[str, Any]) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for entry in report.get("install", []):
        metadata = entry.get("metadata", {})
        name = metadata.get("name")
        version = metadata.get("version")
        if isinstance(name, str) and isinstance(version, str):
            resolved[canonicalize(name)] = version
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    args = parser.parse_args()

    with args.report.open(encoding="utf-8") as handle:
        report = json.load(handle)
    resolved = resolved_versions(report)
    actual = {name: resolved.get(name) for name in EXPECTED}
    if actual != EXPECTED:
        raise RuntimeError(f"Inference resolution mismatch: expected={EXPECTED}, actual={actual}")
    print(f"INFERENCE_RESOLUTION_OK versions={actual}")


if __name__ == "__main__":
    main()
