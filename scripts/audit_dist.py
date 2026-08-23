#!/usr/bin/env python3
"""Fail-closed inventory and metadata audit for FlexAligner distributions."""

from __future__ import annotations

import argparse
import hashlib
import tarfile
import zipfile
from email.parser import BytesParser
from pathlib import Path, PurePosixPath

import tomli as tomllib

DENIED_SUFFIXES = {
    ".bin",
    ".ckpt",
    ".flac",
    ".incomplete",
    ".mp3",
    ".onnx",
    ".npz",
    ".pt",
    ".pth",
    ".safetensors",
    ".textgrid",
    ".wav",
}
G2P_CHECKPOINT_SUFFIX = "flexaligner/_vendor/g2p_en/checkpoint20.npz"
G2P_CHECKPOINT_SHA256 = "b8af35e4596d8dd5836dfd3fe9b2ba4f97b9c311efe8879544cbcfcbd566d8c6"
G2P_LICENSE_SUFFIX = "flexaligner/_vendor/g2p_en/LICENSE.g2p-en.txt"
G2P_NOTICE_SUFFIX = "flexaligner/_vendor/g2p_en/NOTICE.md"
DENIED_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "artifacts",
    "htmlcov",
    "models",
    "reference",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_member_names(names: list[str], *, archive: Path, wheel: bool) -> None:
    for raw_name in names:
        normalized = raw_name.replace("\\", "/")
        path = PurePosixPath(normalized)
        if path.is_absolute() or ".." in path.parts:
            raise RuntimeError(f"Unsafe archive member in {archive}: {raw_name!r}")
        lower_parts = {part.lower() for part in path.parts}
        if lower_parts & DENIED_PARTS:
            raise RuntimeError(f"Denied directory in {archive}: {raw_name!r}")
        if path.suffix.lower() in DENIED_SUFFIXES and not normalized.endswith(
            G2P_CHECKPOINT_SUFFIX
        ):
            raise RuntimeError(f"Denied file type in {archive}: {raw_name!r}")
        if wheel and "tests" in lower_parts:
            raise RuntimeError(f"Tests must not be included in the wheel: {raw_name!r}")


def contains_suffix(names: list[str], suffix: str) -> bool:
    return any(name.replace("\\", "/").endswith(suffix) for name in names)


def require_g2p_assets(names: list[str], *, archive: Path) -> None:
    required = (G2P_CHECKPOINT_SUFFIX, G2P_LICENSE_SUFFIX, G2P_NOTICE_SUFFIX)
    missing = [suffix for suffix in required if not contains_suffix(names, suffix)]
    if missing:
        raise RuntimeError(f"Distribution is missing local English G2P assets: {missing}")


def audit_wheel(path: Path, *, expected_name: str, expected_version: str) -> None:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        validate_member_names(names, archive=path, wheel=True)
        require_g2p_assets(names, archive=path)
        checkpoint_name = next(
            name for name in names if name.replace("\\", "/").endswith(G2P_CHECKPOINT_SUFFIX)
        )
        checkpoint_digest = hashlib.sha256(archive.read(checkpoint_name)).hexdigest()
        if checkpoint_digest != G2P_CHECKPOINT_SHA256:
            raise RuntimeError("Wheel contains an unaudited local English G2P checkpoint")
        metadata_names = [name for name in names if name.endswith(".dist-info/METADATA")]
        if len(metadata_names) != 1:
            raise RuntimeError(f"Expected exactly one wheel METADATA file: {metadata_names}")
        metadata = BytesParser().parsebytes(archive.read(metadata_names[0]))

    actual_name = metadata.get("Name")
    actual_version = metadata.get("Version")
    if actual_name != expected_name or actual_version != expected_version:
        raise RuntimeError(
            "Wheel metadata mismatch: "
            f"expected={expected_name}=={expected_version}, "
            f"actual={actual_name}=={actual_version}"
        )
    if metadata.get("License-Expression") != "MIT":
        raise RuntimeError("Wheel must contain the audited SPDX License-Expression: MIT")
    if not contains_suffix(names, "flexaligner/__init__.py"):
        raise RuntimeError("Wheel does not contain flexaligner/__init__.py")
    if not contains_suffix(names, "flexaligner/py.typed"):
        raise RuntimeError(
            "Wheel declares Typing :: Typed but does not contain flexaligner/py.typed"
        )
    if not any(".dist-info/licenses/LICENSE" in name for name in names):
        raise RuntimeError("Wheel does not contain the declared LICENSE file")


def audit_sdist(path: Path) -> None:
    with tarfile.open(path, mode="r:gz") as archive:
        names = archive.getnames()
        require_g2p_assets(names, archive=path)
        checkpoint_member = next(
            member
            for member in archive.getmembers()
            if member.name.replace("\\", "/").endswith(G2P_CHECKPOINT_SUFFIX)
        )
        checkpoint_file = archive.extractfile(checkpoint_member)
        if checkpoint_file is None:
            raise RuntimeError("Unable to read the local English G2P checkpoint from sdist")
        checkpoint_digest = hashlib.sha256(checkpoint_file.read()).hexdigest()
        if checkpoint_digest != G2P_CHECKPOINT_SHA256:
            raise RuntimeError("sdist contains an unaudited local English G2P checkpoint")
    validate_member_names(names, archive=path, wheel=False)
    required = (
        "LICENSE",
        "README.md",
        "pyproject.toml",
        "src/flexaligner/__init__.py",
    )
    missing = [suffix for suffix in required if not contains_suffix(names, suffix)]
    if missing:
        raise RuntimeError(f"sdist is missing required members: {missing}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dist_dir", nargs="?", type=Path, default=Path("dist"))
    parser.add_argument("--pyproject", type=Path, default=Path("pyproject.toml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wheels = sorted(args.dist_dir.glob("*.whl"))
    sdists = sorted(args.dist_dir.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise RuntimeError(f"Expected one wheel and one sdist; wheels={wheels}, sdists={sdists}")

    with args.pyproject.open("rb") as handle:
        project = tomllib.load(handle)["project"]
    expected_name = str(project["name"])
    expected_version = str(project["version"])

    audit_wheel(wheels[0], expected_name=expected_name, expected_version=expected_version)
    audit_sdist(sdists[0])
    for artifact in (*wheels, *sdists):
        print(f"DIST_OK sha256={sha256(artifact)} path={artifact}")


if __name__ == "__main__":
    main()
