#!/usr/bin/env python3
"""Verify a frozen, local-only model E2E asset manifest without downloading."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
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


def require_mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"Manifest field {field!r} must be an object")
    return value


def validate_manifest_metadata(payload: dict[str, Any]) -> dict[str, str]:
    require_string(payload.get("fixture_id"), field="fixture_id")
    status = require_string(payload.get("status"), field="status")
    if status not in {"candidate", "approved"}:
        raise RuntimeError("Manifest status must be 'candidate' or 'approved'")

    runtime_payload = require_mapping(payload.get("runtime"), field="runtime")
    if "python" not in runtime_payload:
        raise RuntimeError("Manifest runtime must record an exact Python version")
    runtime = {
        require_string(name, field="runtime package name"): require_string(
            version, field=f"runtime.{name}"
        )
        for name, version in runtime_payload.items()
    }
    require_mapping(payload.get("provenance"), field="provenance")
    return runtime


def verify_runtime(runtime: dict[str, str]) -> None:
    expected_python = runtime["python"]
    actual_python = platform.python_version()
    if actual_python != expected_python:
        raise RuntimeError(
            "MODEL_E2E_BLOCKED: Python version mismatch: "
            f"expected={expected_python}, actual={actual_python}"
        )

    for distribution, expected in sorted(runtime.items()):
        if distribution == "python":
            continue
        try:
            actual = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as error:
            raise RuntimeError(
                f"MODEL_E2E_BLOCKED: required distribution is absent: {distribution}"
            ) from error
        if actual != expected:
            raise RuntimeError(
                "MODEL_E2E_BLOCKED: distribution version mismatch: "
                f"name={distribution}, expected={expected}, actual={actual}"
            )


def resolve_root(payload: dict[str, Any]) -> Path:
    """Resolve exactly one explicit root source without guessing a location."""
    root_value = payload.get("root")
    root_env_value = payload.get("root_env")
    if (root_value is None) == (root_env_value is None):
        raise RuntimeError("Manifest must define exactly one of 'root' or 'root_env'")

    if root_env_value is not None:
        root_env = require_string(root_env_value, field="root_env")
        if not root_env.replace("_", "A").isalnum() or not root_env[0].isalpha():
            raise RuntimeError(f"Manifest root_env is not a safe environment name: {root_env!r}")
        root_text = os.environ.get(root_env)
        if not root_text:
            raise RuntimeError(
                f"MODEL_E2E_BLOCKED: required asset-root environment variable is unset: {root_env}"
            )
        root = Path(root_text)
    else:
        root = Path(require_string(root_value, field="root"))

    if not root.is_absolute():
        raise RuntimeError("MODEL_E2E_BLOCKED: manifest root must resolve to an absolute path")
    return root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--check-runtime",
        action="store_true",
        help="also require exact Python and installed-distribution versions",
    )
    args = parser.parse_args()

    if not args.manifest.is_file():
        raise RuntimeError(f"MODEL_E2E_BLOCKED: frozen asset manifest is absent: {args.manifest}")
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise RuntimeError("MODEL_E2E_BLOCKED: manifest schema_version must be 1")
    runtime = validate_manifest_metadata(payload)
    if args.check_runtime:
        verify_runtime(runtime)

    root = resolve_root(payload)
    entries = payload.get("files")
    if not isinstance(entries, list) or not entries:
        raise RuntimeError("MODEL_E2E_BLOCKED: manifest files must be a non-empty list")

    roles: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise RuntimeError(f"Manifest files[{index}] must be an object")
        relative = require_string(entry.get("path"), field=f"files[{index}].path")
        expected = require_string(entry.get("sha256"), field=f"files[{index}].sha256")
        role = require_string(entry.get("role"), field=f"files[{index}].role")
        if role in roles:
            raise RuntimeError(f"Duplicate asset role in manifest: {role!r}")
        roles.add(role)
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
