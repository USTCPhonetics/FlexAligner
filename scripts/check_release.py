#!/usr/bin/env python3
"""Validate the immutable source/version boundary for a public alpha release."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised by the configured Python 3.10 type/runtime gate
    import tomli as tomllib

ALPHA_VERSION = re.compile(
    r"(?:0|[1-9][0-9]*)\."
    r"(?:0|[1-9][0-9]*)\."
    r"(?:0|[1-9][0-9]*)a(?:0|[1-9][0-9]*)"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    boundary = parser.add_mutually_exclusive_group(required=True)
    boundary.add_argument("--tag")
    boundary.add_argument("--version-only", action="store_true")
    parser.add_argument("--pyproject", type=Path, default=Path("pyproject.toml"))
    args = parser.parse_args()

    with args.pyproject.open("rb") as handle:
        project = tomllib.load(handle)["project"]
    version = str(project["version"])
    if ALPHA_VERSION.fullmatch(version) is None:
        raise RuntimeError(
            f"The public-alpha workflow accepts only canonical X.Y.ZaN versions; got {version!r}."
        )
    expected_tag = f"v{version}"
    language_pack_pyproject = args.pyproject.parent / "packages/flexaligner-g2p-en/pyproject.toml"
    with language_pack_pyproject.open("rb") as handle:
        language_pack = tomllib.load(handle)["project"]
    if (
        language_pack.get("name") != "flexaligner-g2p-en"
        or str(language_pack.get("version")) != version
    ):
        raise RuntimeError(
            "English G2P language-pack name/version must match the main release: "
            f"main={version!r}, language_pack={language_pack!r}"
        )
    if args.tag is not None and args.tag != expected_tag:
        raise RuntimeError(f"Tag/version mismatch: tag={args.tag!r}, expected={expected_tag!r}")
    mode = "version-only" if args.version_only else f"tag={args.tag}"
    print(f"RELEASE_SOURCE_OK name={project['name']} version={version} {mode}")


if __name__ == "__main__":
    main()
