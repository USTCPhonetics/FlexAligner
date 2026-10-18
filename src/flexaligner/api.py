"""Lazy public engine for the clean FlexAligner package."""

from __future__ import annotations

import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import TracebackType

from .batch import is_fatal_batch_error
from .capabilities import CapabilityId, CapabilityReport, get_capabilities
from .contracts import (
    AlignmentOptions,
    AlignmentRequest,
    AlignmentResult,
    AudioPolicy,
    BatchItemResult,
    BatchReport,
    CalibrationMode,
    Device,
    Language,
    LocalModelBundle,
    ModelResolution,
    PronunciationMode,
)
from .errors import ArtifactExistsError, ConfigurationError, EngineClosedError, FlexAlignerError
from .model_download import has_validated_system_dictionary
from .ports import AlignmentPipelinePort


def require_supported_options(options: AlignmentOptions) -> None:
    """Fail every future option before input, model, output, or network I/O."""

    if not isinstance(options, AlignmentOptions):
        raise ConfigurationError("options must be AlignmentOptions")
    report = get_capabilities()
    if options.language is Language.ZH:
        report.require(CapabilityId.MANDARIN)
        report.require(CapabilityId.CHINESE_SEGMENTATION)
    if options.device is Device.MPS:
        raise ConfigurationError("MPS inference is not supported; choose auto, cpu or cuda:N")
    if options.device is Device.CUDA:
        report.require(CapabilityId.GPU)
    if options.audio_policy is AudioPolicy.MULTI_FORMAT:
        report.require(CapabilityId.MULTI_FORMAT_AUDIO)
    if options.audio_policy is AudioPolicy.AUTO_RESAMPLE:
        report.require(CapabilityId.AUTO_RESAMPLE)
    if options.pronunciation_mode is PronunciationMode.G2P:
        report.require(
            CapabilityId.LOCAL_ENGLISH_G2P
            if options.language is Language.EN
            else CapabilityId.LOCAL_MANDARIN_G2P
        )
    if options.model_resolution is ModelResolution.AUTO_DOWNLOAD:
        report.require(CapabilityId.PYTHON_AUTO_MODEL_RESOLUTION)
    if options.confidence_calibration is CalibrationMode.CALIBRATED:
        report.require(CapabilityId.CONFIDENCE_CALIBRATION)
    expected_profile = "en-reference-v1" if options.language is Language.EN else "zh-online-core-v1"
    if options.algorithm_profile not in {"auto", expected_profile}:
        raise ConfigurationError(
            "algorithm_profile does not match the selected language",
            context={
                "algorithm_profile": options.algorithm_profile,
                "expected": expected_profile,
                "language": options.language.value,
            },
        )


class FlexAligner:
    """Import-safe, lazy facade for one English or Mandarin CPU alignment engine.

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
        self._batch_active = False
        self._resolved_device: str | None = None

    @property
    def resolved_device(self) -> str | None:
        """The actual device used by the latest completed alignment."""

        resolved = getattr(self._pipeline, "resolved_device", None)
        return resolved if isinstance(resolved, str) else self._resolved_device

    def capabilities(self) -> CapabilityReport:
        """Return declared capabilities without probing the environment."""

        return get_capabilities()

    def align(
        self,
        request: AlignmentRequest,
        *,
        options: AlignmentOptions | None = None,
    ) -> AlignmentResult:
        """Align one local English/CPU or Mandarin/CPU request."""

        self._ensure_open()
        if not isinstance(request, AlignmentRequest):
            raise ConfigurationError("request must be an AlignmentRequest")
        if options is not None and not isinstance(options, AlignmentOptions):
            raise ConfigurationError("options must be AlignmentOptions or None")
        selected_options = self._options if options is None else options
        if self._batch_active and selected_options != self._options:
            raise ConfigurationError("Alignment options cannot change within a batch")
        if selected_options.algorithm_profile == "auto":
            selected_options = replace(
                selected_options,
                algorithm_profile=(
                    "en-reference-v1"
                    if selected_options.language is Language.EN
                    else "zh-online-core-v1"
                ),
            )
        require_supported_options(selected_options)
        self.capabilities().require(
            CapabilityId.SINGLE_FILE_EN_CPU
            if selected_options.language is Language.EN
            else CapabilityId.MANDARIN
        )
        if self._lexicon_path is None and not (
            selected_options.language is Language.ZH
            and has_validated_system_dictionary(self._models)
        ):
            raise ConfigurationError(
                "lexicon_path is required unless a validated Mandarin model bundle supplies "
                "dictionary_path"
            )
        if self._pipeline is None:
            from .pipeline import AlignmentPipeline

            self._pipeline = AlignmentPipeline()
        result = self._pipeline.align(
            request=request,
            models=self._models,
            lexicon_path=self._lexicon_path,
            options=selected_options,
        )
        self._resolved_device = (
            f"cuda:{result.provenance.device_index}"
            if result.provenance.device is Device.CUDA
            else result.provenance.device.value
        )
        return result

    @contextmanager
    def batch_session(self) -> Iterator[None]:
        """Own one sequential batch and release cached models on every exit."""

        self._ensure_open()
        self.capabilities().require(CapabilityId.BATCH)
        if self._batch_active:
            raise ConfigurationError("A batch is already active on this engine")
        require_supported_options(self._options)
        self._validate_batch_setup()
        if self._pipeline is None:
            from .pipeline import AlignmentPipeline

            self._pipeline = AlignmentPipeline()
        context = getattr(self._pipeline, "batch_session", None)
        if not callable(context):
            raise ConfigurationError("This pipeline does not support model reuse")
        self._batch_active = True
        try:
            with context(self._options):
                yield
        finally:
            self._batch_active = False

    def _validate_batch_setup(self) -> None:
        from .adapters.lexicon_file import (
            load_dense_token_vocab,
            load_lexicon,
            validate_local_model_dir,
        )
        from .language_validation import validate_lexicon_language, validate_model_language

        try:
            for kind, path in (
                ("chunker", self._models.chunker_dir),
                ("aligner", self._models.aligner_dir),
            ):
                validate_local_model_dir(path, kind)
                vocabulary = load_dense_token_vocab(path / "vocab.json")
                validate_model_language(((kind, vocabulary),), self._options.language)
            paths = [self._lexicon_path]
            if has_validated_system_dictionary(self._models):
                paths.append(self._models.dictionary_path)
            if not any(path is not None for path in paths):
                raise ConfigurationError("A user or validated system lexicon is required")
            for lexicon_file in paths:
                if lexicon_file is not None:
                    validate_lexicon_language(load_lexicon(lexicon_file), self._options.language)
        except FlexAlignerError as error:
            raise ConfigurationError(
                "Batch model or dictionary configuration is invalid: " + error.message,
                context={"cause_code": error.code},
            ) from error

    def iter_batch(
        self,
        requests: Iterable[AlignmentRequest],
        *,
        skip_existing: bool = False,
    ) -> Iterator[BatchItemResult]:
        """Yield outcomes with bounded memory; close the iterator on early exit."""

        if type(skip_existing) is not bool:
            raise ConfigurationError("skip_existing must be a bool")
        with self.batch_session():
            for request in requests:
                if not isinstance(request, AlignmentRequest):
                    raise ConfigurationError("Batch requests must be AlignmentRequest instances")
                started = time.monotonic()
                target = request.output.path
                outputs = [target]
                if request.output.chunk_metadata_path is not None:
                    outputs.append(request.output.chunk_metadata_path)
                temporary_conflict = any(
                    path.with_name(path.name + ".tmp").exists()
                    or path.with_name(path.name + ".tmp").is_symlink()
                    for path in outputs
                )
                if (
                    skip_existing
                    and target.is_file()
                    and not target.is_symlink()
                    and not temporary_conflict
                    and all(path.is_file() and not path.is_symlink() for path in outputs)
                ):
                    yield BatchItemResult(
                        audio_path=request.audio_path,
                        output_path=target,
                        status="skipped",
                        elapsed_seconds=time.monotonic() - started,
                    )
                    continue
                try:
                    result = self.align(request)
                except FlexAlignerError as error:
                    if is_fatal_batch_error(error):
                        raise
                    yield BatchItemResult(
                        audio_path=request.audio_path,
                        output_path=target,
                        status="conflict" if isinstance(error, ArtifactExistsError) else "failed",
                        elapsed_seconds=time.monotonic() - started,
                        error=error.to_dict(),
                        device=self.resolved_device,
                    )
                else:
                    yield BatchItemResult(
                        audio_path=request.audio_path,
                        output_path=target,
                        status="succeeded",
                        elapsed_seconds=time.monotonic() - started,
                        result=result,
                        device=self.resolved_device,
                    )

    def align_batch(
        self,
        requests: Iterable[AlignmentRequest],
        *,
        skip_existing: bool = False,
    ) -> BatchReport:
        """Collect batch outcomes; iter_batch is preferable for large corpora."""

        return BatchReport(items=tuple(self.iter_batch(requests, skip_existing=skip_existing)))

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
