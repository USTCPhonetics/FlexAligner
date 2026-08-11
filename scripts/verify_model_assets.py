#!/usr/bin/env python3
"""Verify a frozen, local-only model E2E asset manifest without downloading."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"Manifest field {field!r} must be a non-empty string")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()

    if not args.manifest.is_file():
        raise RuntimeError(f"MODEL_E2E_BLOCKED: frozen asset manifest is absent: {args.manifest}")
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise RuntimeError("MODEL_E2E_BLOCKED: manifest schema_version must be 1")

    root = Path(require_string(payload.get("root"), field="root"))
    if not root.is_absolute():
        raise RuntimeError("MODEL_E2E_BLOCKED: manifest root must be an absolute path")
    entries = payload.get("files")
    if not isinstance(entries, list) or not entries:
        raise RuntimeError("MODEL_E2E_BLOCKED: manifest files must be a non-empty list")

    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise RuntimeError(f"Manifest files[{index}] must be an object")
        relative = require_string(entry.get("path"), field=f"files[{index}].path")
        expected = require_string(entry.get("sha256"), field=f"files[{index}].sha256")
        posix_path = PurePosixPath(relative)
        if posix_path.is_absolute() or ".." in posix_path.parts:
            raise RuntimeError(f"Unsafe relative asset path: {relative!r}")
        if re_full_sha256(expected) is False:
            raise RuntimeError(f"Invalid SHA-256 for asset {relative!r}: {expected!r}")

        asset = root.joinpath(*posix_path.parts)
        if not asset.is_file():
            raise RuntimeError(f"MODEL_E2E_BLOCKED: asset is absent: {asset}")
        actual = file_sha256(asset)
        if actual != expected.lower():
            raise RuntimeError(
                f"MODEL_E2E_BLOCKED: hash mismatch for {asset}: "
                f"expected={expected.lower()}, actual={actual}"
            )

    print(f"MODEL_E2E_ASSETS_OK manifest={args.manifest} root={root} file_count={len(entries)}")


def re_full_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdefABCDEF" for char in value)


if __name__ == "__main__":
    main()
