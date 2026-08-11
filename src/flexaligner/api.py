"""Lazy public engine for the clean FlexAligner package."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from types import TracebackType

from .capabilities import CapabilityId, CapabilityReport, get_capabilities
from .contracts import (
    AlignmentOptions,
    AlignmentRequest,
    AlignmentResult,
    AudioPolicy,
    CalibrationMode,
    Device,
    Language,
    LocalModelBundle,
    ModelResolution,
    PronunciationMode,
)
from .errors import ConfigurationError, EngineClosedError


class FlexAligner:
    """Import-safe, lazy facade for one English CPU alignment engine.

    Stage 1 exposes contracts and truthful placeholder failures only.  It does
    not inspect paths, load models, import inference dependencies, or consume
    batch iterables.
    """

    def __init__(
        self,
        *,
        models: LocalModelBundle,
        lexicon_path: Path | None = None,
        options: AlignmentOptions | None = None,
    ) -> None:
        if not isinstance(models, LocalModelBundle):
            raise ConfigurationError("models must be a LocalModelBundle")
        if lexicon_path is not None and not isinstance(lexicon_path, Path):
            raise ConfigurationError("lexicon_path must be a pathlib.Path or None")
        if options is not None and not isinstance(options, AlignmentOptions):
            raise ConfigurationError("options must be AlignmentOptions or None")

        self._models = models
        self._lexicon_path = lexicon_path
        self._options = AlignmentOptions() if options is None else options
        self._closed = False

    def capabilities(self) -> CapabilityReport:
        """Return declared capabilities without probing the environment."""

        return get_capabilities()

    def align(
        self,
        request: AlignmentRequest,
        *,
        options: AlignmentOptions | None = None,
    ) -> AlignmentResult:
        """Align one request once the production core becomes available."""

        self._ensure_open()
        if not isinstance(request, AlignmentRequest):
            raise ConfigurationError("request must be an AlignmentRequest")
        if options is not None and not isinstance(options, AlignmentOptions):
            raise ConfigurationError("options must be AlignmentOptions or None")
        self._require_supported_options(self._options if options is None else options)
        self.capabilities().require(CapabilityId.SINGLE_FILE_EN_CPU)
        raise AssertionError("unreachable: placeholder capability unexpectedly available")

    def align_batch(self, requests: Iterable[AlignmentRequest]) -> tuple[AlignmentResult, ...]:
        """Declare the batch boundary without consuming ``requests``."""

        self._ensure_open()
        self.capabilities().require(CapabilityId.BATCH)
        raise AssertionError("unreachable: placeholder capability unexpectedly available")

    def close(self) -> None:
        """Close the lazy facade; safe and idempotent before models exist."""

        self._closed = True

    def __enter__(self) -> FlexAligner:
        self._ensure_open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc, traceback
        self.close()

    def _ensure_open(self) -> None:
        if self._closed:
            raise EngineClosedError("FlexAligner engine is closed")

    def _require_supported_options(self, options: AlignmentOptions) -> None:
        report = self.capabilities()

        if options.language is Language.ZH:
            report.require(CapabilityId.MANDARIN)
        if options.device is not Device.CPU:
            report.require(CapabilityId.GPU)
        if options.audio_policy is AudioPolicy.AUTO_DECODE:
            report.require(CapabilityId.MULTI_FORMAT_AUDIO)
        if options.audio_policy is AudioPolicy.AUTO_RESAMPLE:
            report.require(CapabilityId.AUTO_RESAMPLE)
        if options.pronunciation_mode is PronunciationMode.G2P:
            report.require(CapabilityId.DEFAULT_G2P)
        if options.model_resolution is ModelResolution.AUTO_DOWNLOAD:
            report.require(CapabilityId.AUTO_MODEL_DOWNLOAD)
        if options.confidence_calibration is CalibrationMode.CALIBRATED:
            report.require(CapabilityId.CONFIDENCE_CALIBRATION)
