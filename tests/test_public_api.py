from __future__ import annotations

import importlib.metadata
import re
from types import ModuleType

PUBLIC_SYMBOLS = {
    "AlignmentOptions",
    "AlignmentRequest",
    "AlignmentResult",
    "Capability",
    "CapabilityId",
    "CapabilityReport",
    "CapabilityStatus",
    "FeatureNotAvailableError",
    "FlexAligner",
    "FlexAlignerError",
    "LocalModelBundle",
    "PhoneInterval",
    "PronunciationGenerationError",
    "PronunciationNotice",
    "TextGridOutput",
    "get_capabilities",
}


def test_intended_stage1_symbols_are_public(public_api: ModuleType) -> None:
    missing = sorted(name for name in PUBLIC_SYMBOLS if not hasattr(public_api, name))
    assert not missing, f"missing public API symbols: {missing}"

    exported = set(getattr(public_api, "__all__", ()))
    assert exported >= PUBLIC_SYMBOLS


def test_version_is_canonical_and_matches_distribution(public_api: ModuleType) -> None:
    version = public_api.__version__
    assert isinstance(version, str)
    assert version
    assert re.fullmatch(r"[0-9]+(?:\.[0-9]+)*(?:[A-Za-z0-9.+!-]*)?", version)
    assert importlib.metadata.version("flexaligner") == version


def test_capability_discovery_is_available_from_root(public_api: ModuleType) -> None:
    report = public_api.get_capabilities()
    assert isinstance(report, public_api.CapabilityReport)
