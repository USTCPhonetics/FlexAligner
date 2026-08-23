"""Typed, machine-readable errors for FlexAligner."""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from types import MappingProxyType

JsonScalar = str | int | float | bool | None


class ErrorCode(str, Enum):
    """Stable string codes exposed by the Python and command-line APIs."""

    CONFIGURATION_ERROR = "configuration_error"
    FEATURE_NOT_AVAILABLE = "feature_not_available"
    INPUT_VALIDATION_ERROR = "input_validation_error"
    AUDIO_FORMAT_UNSUPPORTED = "audio_format_unsupported"
    MODEL_VALIDATION_ERROR = "model_validation_error"
    MODEL_INCOMPATIBLE = "model_incompatible"
    MODEL_CACHE_MISS = "model_cache_miss"
    MODEL_DOWNLOAD_ERROR = "model_download_error"
    PRONUNCIATION_GENERATION_ERROR = "pronunciation_generation_error"
    RESOURCE_LIMIT_EXCEEDED = "resource_limit_exceeded"
    ALIGNMENT_FAILED = "alignment_failed"
    ALIGNMENT_END_UNREACHABLE = "alignment_end_unreachable"
    OUTPUT_ERROR = "output_error"
    OUTPUT_EXISTS = "output_exists"
    OUTPUT_VALIDATION_FAILED = "output_validation_failed"
    ENGINE_CLOSED = "engine_closed"
    INTERNAL_ERROR = "internal_error"


class FlexAlignerError(Exception):
    """Base class for expected FlexAligner failures."""

    default_code = ErrorCode.INTERNAL_ERROR

    def __init__(
        self,
        message: str,
        *,
        code: ErrorCode | str | None = None,
        context: Mapping[str, JsonScalar] | None = None,
    ) -> None:
        super().__init__(message)
        selected_code = self.default_code if code is None else code
        self.code = (
            selected_code.value if isinstance(selected_code, ErrorCode) else str(selected_code)
        )
        self.message = message
        self.context: Mapping[str, JsonScalar] = MappingProxyType(dict(context or {}))

    def to_dict(self) -> dict[str, object]:
        """Return the stable serialization used by automation and the CLI."""

        return {
            "code": self.code,
            "message": self.message,
            "context": dict(self.context),
        }


class ConfigurationError(FlexAlignerError):
    default_code = ErrorCode.CONFIGURATION_ERROR


class FeatureNotAvailableError(FlexAlignerError):
    """Raised when a declared placeholder capability is requested."""

    default_code = ErrorCode.FEATURE_NOT_AVAILABLE

    def __init__(
        self,
        capability: str,
        *,
        status: str = "placeholder",
        reason: str | None = None,
    ) -> None:
        context: dict[str, JsonScalar] = {
            "capability": capability,
            "status": status,
        }
        if reason is not None:
            context["reason"] = reason
        detail = f": {reason}" if reason else ""
        super().__init__(
            f"Capability {capability!r} is {status}{detail}",
            context=context,
        )


class InputValidationError(FlexAlignerError):
    default_code = ErrorCode.INPUT_VALIDATION_ERROR


class AudioFormatError(InputValidationError):
    default_code = ErrorCode.AUDIO_FORMAT_UNSUPPORTED


class ModelValidationError(FlexAlignerError):
    default_code = ErrorCode.MODEL_VALIDATION_ERROR


class ModelCompatibilityError(ModelValidationError):
    default_code = ErrorCode.MODEL_INCOMPATIBLE


class ModelCacheMissError(FlexAlignerError):
    default_code = ErrorCode.MODEL_CACHE_MISS


class ModelDownloadError(FlexAlignerError):
    default_code = ErrorCode.MODEL_DOWNLOAD_ERROR


class PronunciationGenerationError(FlexAlignerError):
    default_code = ErrorCode.PRONUNCIATION_GENERATION_ERROR


class ResourceLimitError(FlexAlignerError):
    default_code = ErrorCode.RESOURCE_LIMIT_EXCEEDED


class AlignmentError(FlexAlignerError):
    default_code = ErrorCode.ALIGNMENT_FAILED


class UnreachableAlignmentError(AlignmentError):
    default_code = ErrorCode.ALIGNMENT_END_UNREACHABLE


class OutputError(FlexAlignerError):
    default_code = ErrorCode.OUTPUT_ERROR


class ArtifactExistsError(OutputError):
    default_code = ErrorCode.OUTPUT_EXISTS


class OutputValidationError(OutputError):
    default_code = ErrorCode.OUTPUT_VALIDATION_FAILED


class EngineClosedError(ConfigurationError):
    default_code = ErrorCode.ENGINE_CLOSED


class InternalError(FlexAlignerError):
    default_code = ErrorCode.INTERNAL_ERROR
