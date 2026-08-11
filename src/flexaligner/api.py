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
from .ports import AlignmentPipelinePort


def require_supported_options(options: AlignmentOptions) -> None:
    """Fail every future option before input, model, output, or network I/O."""

    if not isinstance(options, AlignmentOptions):
        raise ConfigurationError("options must be AlignmentOptions")
    report = get_capabilities()
    if options.language is Language.ZH:
        report.require(CapabilityId.MANDARIN)
    if options.device is not Device.CPU:
        report.require(CapabilityId.GPU)
    if options.audio_policy is AudioPolicy.MULTI_FORMAT:
        report.require(CapabilityId.MULTI_FORMAT_AUDIO)
    if options.audio_policy is AudioPolicy.AUTO_RESAMPLE:
        report.require(CapabilityId.AUTO_RESAMPLE)
    if options.pronunciation_mode is PronunciationMode.G2P:
        report.require(CapabilityId.DEFAULT_G2P)
    if options.model_resolution is ModelResolution.AUTO_DOWNLOAD:
        report.require(CapabilityId.AUTO_MODEL_DOWNLOAD)
    if options.confidence_calibration is CalibrationMode.CALIBRATED:
        report.require(CapabilityId.CONFIDENCE_CALIBRATION)
    if options.algorithm_profile != "en-reference-v1":
        raise ConfigurationError(
            "Only algorithm_profile='en-reference-v1' is implemented",
            context={"algorithm_profile": options.algorithm_profile},
        )


class FlexAligner:
    """Import-safe, lazy facade for one English CPU alignment engine.

    Construction and capability discovery do not inspect paths, load models,
    import inference dependencies, or consume batch iterables.
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
        self._pipeline: AlignmentPipelinePort | None = None
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
        """Align one strict local English/CPU request."""

        self._ensure_open()
        if not isinstance(request, AlignmentRequest):
            raise ConfigurationError("request must be an AlignmentRequest")
        if options is not None and not isinstance(options, AlignmentOptions):
            raise ConfigurationError("options must be AlignmentOptions or None")
        selected_options = self._options if options is None else options
        require_supported_options(selected_options)
        self.capabilities().require(CapabilityId.SINGLE_FILE_EN_CPU)
        if self._lexicon_path is None:
            raise ConfigurationError("lexicon_path is required for English alignment")
        if self._pipeline is None:
            from .pipeline import AlignmentPipeline

            self._pipeline = AlignmentPipeline()
        return self._pipeline.align(
            request=request,
            models=self._models,
            lexicon_path=self._lexicon_path,
            options=selected_options,
        )

    def align_batch(self, requests: Iterable[AlignmentRequest]) -> tuple[AlignmentResult, ...]:
        """Declare the batch boundary without consuming ``requests``."""

        self._ensure_open()
        self.capabilities().require(CapabilityId.BATCH)
        raise AssertionError("unreachable: placeholder capability unexpectedly available")

    def close(self) -> None:
        """Close the lazy facade and any constructed pipeline, idempotently."""

        if self._closed:
            return
        if self._pipeline is not None:
            self._pipeline.close()
            self._pipeline = None
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


__all__ = ["FlexAligner", "require_supported_options"]
