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
        if path.suffix.lower() in DENIED_SUFFIXES:
            raise RuntimeError(f"Denied file type in {archive}: {raw_name!r}")
        if wheel and "tests" in lower_parts:
            raise RuntimeError(f"Tests must not be included in the wheel: {raw_name!r}")


def contains_suffix(names: list[str], suffix: str) -> bool:
    return any(name.replace("\\", "/").endswith(suffix) for name in names)


def audit_wheel(path: Path, *, expected_name: str, expected_version: str) -> None:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        validate_member_names(names, archive=path, wheel=True)
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
    if "flexaligner-g2p-en==0.3.0a1; extra == 'en'" not in metadata.get_all("Requires-Dist", []):
        raise RuntimeError("Main wheel must expose the exact English language pack via [en]")
    if not contains_suffix(names, "flexaligner/__init__.py"):
        raise RuntimeError("Wheel does not contain flexaligner/__init__.py")
    if not contains_suffix(names, "flexaligner/py.typed"):
        raise RuntimeError(
            "Wheel declares Typing :: Typed but does not contain flexaligner/py.typed"
        )
    if not any(".dist-info/licenses/LICENSE" in name for name in names):
        raise RuntimeError("Wheel does not contain the declared LICENSE file")
    forbidden = ("checkpoint20.npz", "flexaligner_g2p_en", "LICENSE.g2p-en", "g2p_en/NOTICE")
    leaked = [name for name in names if any(marker in name for marker in forbidden)]
    if leaked:
        raise RuntimeError(f"Base wheel leaks optional English G2P assets: {leaked}")


def audit_sdist(path: Path) -> None:
    with tarfile.open(path, mode="r:gz") as archive:
        names = archive.getnames()
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
    wheels = sorted(args.dist_dir.glob("flexaligner-[0-9]*.whl"))
    sdists = sorted(args.dist_dir.glob("flexaligner-[0-9]*.tar.gz"))
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
