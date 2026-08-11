"""Public package surface for the clean FlexAligner rebuild."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("flexaligner")
except PackageNotFoundError:
    __version__ = "0+unknown"

from .api import FlexAligner
from .capabilities import (
    Capability,
    CapabilityId,
    CapabilityReport,
    CapabilityStatus,
    get_capabilities,
)
from .contracts import (
    AlignmentOptions,
    AlignmentRequest,
    AlignmentResult,
    AudioPolicy,
    CalibrationMode,
    ChunkResult,
    Device,
    Language,
    LocalModelBundle,
    ModelResolution,
    PhoneInterval,
    PronunciationMode,
    ResourceLimits,
    RunProvenance,
    Score,
    ScoreKind,
    TextGridOutput,
    WordInterval,
)
from .errors import (
    AlignmentError,
    ArtifactExistsError,
    AudioFormatError,
    ConfigurationError,
    EngineClosedError,
    ErrorCode,
    FeatureNotAvailableError,
    FlexAlignerError,
    InputValidationError,
    InternalError,
    ModelCompatibilityError,
    ModelValidationError,
    OutputError,
    OutputValidationError,
    ResourceLimitError,
    UnreachableAlignmentError,
)

__all__ = [
    "AlignmentError",
    "AlignmentOptions",
    "AlignmentRequest",
    "AlignmentResult",
    "ArtifactExistsError",
    "AudioFormatError",
    "AudioPolicy",
    "CalibrationMode",
    "Capability",
    "CapabilityId",
    "CapabilityReport",
    "CapabilityStatus",
    "ChunkResult",
    "ConfigurationError",
    "Device",
    "EngineClosedError",
    "ErrorCode",
    "FeatureNotAvailableError",
    "FlexAligner",
    "FlexAlignerError",
    "InputValidationError",
    "InternalError",
    "Language",
    "LocalModelBundle",
    "ModelCompatibilityError",
    "ModelResolution",
    "ModelValidationError",
    "OutputError",
    "OutputValidationError",
    "PhoneInterval",
    "PronunciationMode",
    "ResourceLimitError",
    "ResourceLimits",
    "RunProvenance",
    "Score",
    "ScoreKind",
    "TextGridOutput",
    "UnreachableAlignmentError",
    "WordInterval",
    "__version__",
    "get_capabilities",
]
