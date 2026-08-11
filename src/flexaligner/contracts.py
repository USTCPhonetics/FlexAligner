"""Immutable public records for the FlexAligner package."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .errors import ConfigurationError


class StringEnum(str, Enum):
    """A Python 3.10-compatible string enum."""

    def __str__(self) -> str:
        return str(self.value)


class Language(StringEnum):
    EN = "en"
    ZH = "zh"


class Device(StringEnum):
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


class AudioPolicy(StringEnum):
    STRICT_PCM16_WAV = "strict"
    MULTI_FORMAT = "multi-format"
    AUTO_DECODE = "multi-format"
    AUTO_RESAMPLE = "auto-resample"


class PronunciationMode(StringEnum):
    LEXICON_ONLY = "lexicon"
    G2P = "g2p"


class ModelResolution(StringEnum):
    LOCAL_ONLY = "local"
    AUTO_DOWNLOAD = "auto-download"


class CalibrationMode(StringEnum):
    NONE = "none"
    CALIBRATED = "calibrated"


class ScoreKind(StringEnum):
    CHUNKER_EMISSION_GEOMETRIC_MEAN = "chunker_emission_geometric_mean"


@dataclass(frozen=True, slots=True, kw_only=True)
class ResourceLimits:
    """Optional caller limits; approved package defaults remain TBD."""

    max_audio_seconds: float | None = None
    max_transcript_words: int | None = None
    max_phone_tokens: int | None = None
    max_trellis_cells: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "max_audio_seconds",
            "max_transcript_words",
            "max_phone_tokens",
            "max_trellis_cells",
        ):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ConfigurationError(
                    f"{name} must be positive when provided",
                    context={"field": name, "value": value},
                )


@dataclass(frozen=True, slots=True, kw_only=True)
class LocalModelBundle:
    chunker_dir: Path
    aligner_dir: Path
    manifest_path: Path | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class TextGridOutput:
    path: Path
    chunk_metadata_path: Path | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class AlignmentRequest:
    audio_path: Path
    transcript: str
    output: TextGridOutput
    utterance_id: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class AlignmentOptions:
    language: Language = Language.EN
    device: Device = Device.CPU
    algorithm_profile: str = "en-reference-v1"
    num_threads: int = 1
    audio_policy: AudioPolicy = AudioPolicy.STRICT_PCM16_WAV
    pronunciation_mode: PronunciationMode = PronunciationMode.LEXICON_ONLY
    model_resolution: ModelResolution = ModelResolution.LOCAL_ONLY
    confidence_calibration: CalibrationMode = CalibrationMode.NONE
    limits: ResourceLimits | None = None

    def __post_init__(self) -> None:
        enum_fields = (
            ("language", self.language, Language),
            ("device", self.device, Device),
            ("audio_policy", self.audio_policy, AudioPolicy),
            ("pronunciation_mode", self.pronunciation_mode, PronunciationMode),
            ("model_resolution", self.model_resolution, ModelResolution),
            (
                "confidence_calibration",
                self.confidence_calibration,
                CalibrationMode,
            ),
        )
        for name, value, enum_type in enum_fields:
            if not isinstance(value, enum_type):
                raise ConfigurationError(
                    f"{name} must be a {enum_type.__name__}",
                    context={"field": name, "value": str(value)},
                )
        if not self.algorithm_profile.strip():
            raise ConfigurationError("algorithm_profile must not be empty")
        if self.num_threads <= 0:
            raise ConfigurationError(
                "num_threads must be positive",
                context={"field": "num_threads", "value": self.num_threads},
            )
        if self.limits is not None and not isinstance(self.limits, ResourceLimits):
            raise ConfigurationError(
                "limits must be ResourceLimits or None",
                context={"field": "limits"},
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class WordInterval:
    label: str
    start_s: float
    end_s: float
    word_index: int | None


@dataclass(frozen=True, slots=True, kw_only=True)
class PhoneInterval:
    label: str
    start_s: float
    end_s: float
    word_index: int | None
    phone_index: int | None


@dataclass(frozen=True, slots=True, kw_only=True)
class Score:
    value: float
    kind: ScoreKind
    calibrated: bool


@dataclass(frozen=True, slots=True, kw_only=True)
class ChunkResult:
    chunk_id: str
    start_s: float
    end_s: float
    word_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class RunProvenance:
    package_version: str
    algorithm_profile: str
    language: Language
    device: Device
    model_fingerprints: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True, kw_only=True)
class AlignmentResult:
    utterance_id: str
    audio_duration_s: float
    normalized_words: tuple[str, ...]
    words: tuple[WordInterval, ...]
    phones: tuple[PhoneInterval, ...]
    chunks: tuple[ChunkResult, ...]
    raw_scores: tuple[Score, ...]
    calibrated_scores: tuple[Score, ...] | None
    output_path: Path
    output_sha256: str
    provenance: RunProvenance
    schema_version: str = "1"
