#!/usr/bin/env python3
"""Validate the immutable source/version boundary for a production release."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised by the configured Python 3.10 type/runtime gate
    import tomli as tomllib

STABLE_VERSION = re.compile(r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--pyproject", type=Path, default=Path("pyproject.toml"))
    args = parser.parse_args()

    with args.pyproject.open("rb") as handle:
        project = tomllib.load(handle)["project"]
    version = str(project["version"])
    expected_tag = f"v{version}"

    if args.tag != expected_tag:
        raise RuntimeError(f"Tag/version mismatch: tag={args.tag!r}, expected={expected_tag!r}")
    if STABLE_VERSION.fullmatch(version) is None:
        raise RuntimeError(
            "Production PyPI workflow accepts only stable X.Y.Z versions; "
            f"got {version!r}. TBD-PKG-003 remains unresolved."
        )
    print(f"RELEASE_SOURCE_OK name={project['name']} version={version} tag={args.tag}")


if __name__ == "__main__":
    main()
