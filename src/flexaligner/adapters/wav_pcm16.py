"""Strict, dependency-light decoder for the implemented WAV input contract."""

from __future__ import annotations

import math
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from flexaligner.contracts import ResourceLimits
from flexaligner.errors import AudioFormatError, InputValidationError, ResourceLimitError

TARGET_SAMPLE_RATE = 16_000
PCM16_SAMPLE_WIDTH_BYTES = 2

Float32Array = NDArray[np.float32]


@dataclass(frozen=True, slots=True, kw_only=True)
class DecodedAudio:
    """A validated mono waveform represented as contiguous float32 samples."""

    samples: Float32Array
    sample_rate: int
    duration_s: float


def _require_file(path: Path) -> None:
    if not isinstance(path, Path):
        raise InputValidationError(
            "Audio path must be a pathlib.Path",
            context={"path_type": type(path).__name__},
        )
    if not path.is_file():
        raise InputValidationError(
            f"Audio path is not a file: {path}",
            context={"path": str(path)},
        )


def _validate_limits(limits: ResourceLimits | None) -> None:
    if limits is not None and not isinstance(limits, ResourceLimits):
        raise InputValidationError(
            "limits must be ResourceLimits or None",
            context={"limits_type": type(limits).__name__},
        )
    if (
        limits is not None
        and limits.max_audio_seconds is not None
        and not math.isfinite(limits.max_audio_seconds)
    ):
        raise InputValidationError(
            "max_audio_seconds must be finite when provided",
            context={"max_audio_seconds": limits.max_audio_seconds},
        )


def _format_error(path: Path, message: str, **context: str | int) -> AudioFormatError:
    return AudioFormatError(message, context={"path": str(path), **context})


def load_strict_pcm16_wav(
    path: Path,
    limits: ResourceLimits | None = None,
) -> DecodedAudio:
    """Decode one strict 16 kHz, mono, uncompressed PCM16 WAV file.

    The duration limit is checked from the validated header before sample
    allocation. No decoding fallback, resampling, or channel conversion occurs.
    """

    _require_file(path)
    _validate_limits(limits)

    try:
        with wave.open(str(path), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            compression = wav_file.getcomptype()
            num_frames = wav_file.getnframes()

            if channels != 1:
                raise _format_error(
                    path,
                    f"Expected mono WAV, got channels={channels}: {path}",
                    channels=channels,
                )
            if sample_width != PCM16_SAMPLE_WIDTH_BYTES:
                raise _format_error(
                    path,
                    f"Expected PCM16 WAV, got sample_width={sample_width}: {path}",
                    sample_width=sample_width,
                )
            if sample_rate != TARGET_SAMPLE_RATE:
                raise _format_error(
                    path,
                    f"Expected {TARGET_SAMPLE_RATE} Hz WAV, got sample_rate={sample_rate}: {path}",
                    sample_rate=sample_rate,
                    expected_sample_rate=TARGET_SAMPLE_RATE,
                )
            if compression != "NONE":
                raise _format_error(
                    path,
                    f"Expected uncompressed PCM WAV, got compression={compression!r}: {path}",
                    compression=compression,
                )
            if num_frames <= 0:
                raise _format_error(path, f"Empty WAV: {path}", num_frames=num_frames)

            duration_s = num_frames / sample_rate
            if not math.isfinite(duration_s) or duration_s <= 0.0:
                raise _format_error(
                    path,
                    f"Invalid WAV duration={duration_s!r}: {path}",
                    num_frames=num_frames,
                    sample_rate=sample_rate,
                )
            if (
                limits is not None
                and limits.max_audio_seconds is not None
                and duration_s > limits.max_audio_seconds
            ):
                raise ResourceLimitError(
                    "Audio duration limit exceeded",
                    context={
                        "path": str(path),
                        "duration_s": duration_s,
                        "max_audio_seconds": limits.max_audio_seconds,
                    },
                )

            raw = wav_file.readframes(num_frames)
    except (AudioFormatError, ResourceLimitError):
        raise
    except (wave.Error, EOFError) as exc:
        raise AudioFormatError(
            f"Invalid PCM WAV file: {path}: {exc}",
            context={"path": str(path), "reason": str(exc)},
        ) from exc
    except OSError as exc:
        raise InputValidationError(
            f"Unable to read audio file: {path}: {exc}",
            context={"path": str(path), "reason": str(exc)},
        ) from exc

    expected_bytes = num_frames * PCM16_SAMPLE_WIDTH_BYTES
    if len(raw) != expected_bytes:
        raise AudioFormatError(
            "WAV frame count does not match decoded sample data",
            context={
                "path": str(path),
                "header_frames": num_frames,
                "expected_bytes": expected_bytes,
                "decoded_bytes": len(raw),
            },
        )

    try:
        samples_i16 = np.frombuffer(raw, dtype="<i2")
    except ValueError as exc:
        raise AudioFormatError(
            f"Invalid PCM16 sample payload: {path}: {exc}",
            context={"path": str(path), "reason": str(exc)},
        ) from exc

    if samples_i16.size != num_frames:
        raise AudioFormatError(
            "WAV frame count does not match decoded sample count",
            context={
                "path": str(path),
                "header_frames": num_frames,
                "decoded_samples": int(samples_i16.size),
            },
        )

    samples = np.ascontiguousarray(samples_i16.astype(np.float32) / 32768.0)
    if samples.ndim != 1 or samples.size == 0:
        raise AudioFormatError(
            f"Invalid decoded waveform shape={samples.shape}: {path}",
            context={"path": str(path), "ndim": samples.ndim, "size": int(samples.size)},
        )
    if not bool(np.isfinite(samples).all()):
        raise AudioFormatError(
            f"Decoded waveform contains NaN/Inf: {path}",
            context={"path": str(path)},
        )

    return DecodedAudio(
        samples=samples,
        sample_rate=sample_rate,
        duration_s=duration_s,
    )
