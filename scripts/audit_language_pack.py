#!/usr/bin/env python3
"""Fail-closed inventory audit for the optional English G2P distribution."""

from __future__ import annotations

import argparse
import hashlib
import tarfile
import zipfile
from email.parser import BytesParser
from pathlib import Path, PurePosixPath

CHECKPOINT_SUFFIX = "flexaligner_g2p_en/checkpoint20.npz"
CHECKPOINT_SHA256 = "b8af35e4596d8dd5836dfd3fe9b2ba4f97b9c311efe8879544cbcfcbd566d8c6"
EXPECTED_NAME = "flexaligner-g2p-en"
EXPECTED_VERSION = "0.3.0a1"


def _safe(names: list[str], archive: Path) -> None:
    for raw_name in names:
        path = PurePosixPath(raw_name.replace("\\", "/"))
        if path.is_absolute() or ".." in path.parts:
            raise RuntimeError(f"Unsafe archive member in {archive}: {raw_name!r}")
        if {part.lower() for part in path.parts} & {".git", "__pycache__", "tests"}:
            raise RuntimeError(f"Denied archive member in {archive}: {raw_name!r}")


def _suffix(names: list[str], suffix: str) -> str:
    matches = [name for name in names if name.replace("\\", "/").endswith(suffix)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {suffix!r}, got {matches}")
    return matches[0]


def _check_digest(payload: bytes) -> None:
    if hashlib.sha256(payload).hexdigest() != CHECKPOINT_SHA256:
        raise RuntimeError("English G2P checkpoint digest does not match the frozen trust root")


def audit_wheel(path: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        _safe(names, path)
        _check_digest(archive.read(_suffix(names, CHECKPOINT_SUFFIX)))
        _suffix(names, "flexaligner_g2p_en/__init__.py")
        _suffix(names, "flexaligner_g2p_en/NOTICE.md")
        if any(name.replace("\\", "/").startswith("flexaligner/") for name in names):
            raise RuntimeError("English G2P wheel must not contain the main flexaligner package")
        metadata_name = _suffix(names, ".dist-info/METADATA")
        metadata = BytesParser().parsebytes(archive.read(metadata_name))
        if metadata.get("Name") != EXPECTED_NAME or metadata.get("Version") != EXPECTED_VERSION:
            raise RuntimeError("English G2P wheel metadata name/version mismatch")
        if metadata.get("License-Expression") != "Apache-2.0":
            raise RuntimeError("English G2P wheel must declare Apache-2.0")
        expected_numpy = {
            "numpy<2.3,>=1.26; python_version < '3.14'",
            "numpy<3,>=2.3.5; python_version >= '3.14'",
        }
        if set(metadata.get_all("Requires-Dist", [])) != expected_numpy:
            raise RuntimeError("English G2P wheel has unexpected runtime dependencies")
        if not any(".dist-info/licenses/LICENSE" in name for name in names):
            raise RuntimeError("English G2P wheel is missing its Apache-2.0 license")


def audit_sdist(path: Path) -> None:
    with tarfile.open(path, mode="r:gz") as archive:
        names = archive.getnames()
        _safe(names, path)
        checkpoint = archive.extractfile(_suffix(names, CHECKPOINT_SUFFIX))
        if checkpoint is None:
            raise RuntimeError("Unable to read English G2P checkpoint from sdist")
        _check_digest(checkpoint.read())
        for suffix in ("LICENSE", "README.md", "pyproject.toml", "flexaligner_g2p_en/NOTICE.md"):
            _suffix(names, suffix)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dist_dir", type=Path)
    args = parser.parse_args()
    wheels = sorted(args.dist_dir.glob("flexaligner_g2p_en-*.whl"))
    sdists = sorted(args.dist_dir.glob("flexaligner_g2p_en-*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise RuntimeError(f"Expected one language-pack wheel and sdist: {wheels}, {sdists}")
    audit_wheel(wheels[0])
    audit_sdist(sdists[0])
    print(f"LANGUAGE_PACK_DIST_OK wheel={wheels[0]} sdist={sdists[0]}")


if __name__ == "__main__":
    main()
