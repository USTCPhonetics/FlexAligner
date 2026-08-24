"""Optional PyAV decoder and explicit canonical-WAV converter."""

from __future__ import annotations

import math
import os
import wave
from contextlib import suppress
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..contracts import ResourceLimits
from ..errors import (
    ArtifactExistsError,
    AudioFormatError,
    InputValidationError,
    OptionalDependencyError,
    OutputError,
    ResourceLimitError,
)
from .wav_pcm16 import DecodedAudio, load_strict_pcm16_wav

TARGET_SAMPLE_RATE = 16_000


def _require_av() -> Any:
    try:
        import av
    except ImportError as error:
        raise OptionalDependencyError(
            "Audio decoding and conversion require the optional audio extra",
            context={
                "dependency": "av==16.0.1",
                "extra": "audio",
                "suggested_command": "python -m pip install 'flexaligner[audio]'",
            },
        ) from error
    return av


def load_audio_with_av(
    path: Path,
    limits: ResourceLimits | None = None,
    *,
    require_wav_container: bool = False,
) -> DecodedAudio:
    """Decode the first audio stream to contiguous 16 kHz mono float32 samples."""

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
    if require_wav_container and path.suffix.lower() != ".wav":
        raise AudioFormatError(
            "auto-resample accepts WAV input only; use multi-format for other containers",
            context={"path": str(path), "suffix": path.suffix.lower()},
        )
    if limits is not None and not isinstance(limits, ResourceLimits):
        raise InputValidationError(
            "limits must be ResourceLimits or None",
            context={"limits_type": type(limits).__name__},
        )

    av = _require_av()
    max_samples = (
        None if limits is None else math.floor(limits.max_audio_seconds * TARGET_SAMPLE_RATE)
    )
    chunks: list[NDArray[np.int16]] = []
    decoded_samples = 0
    try:
        with av.open(str(path), mode="r") as container:
            streams = tuple(container.streams.audio)
            if not streams:
                raise AudioFormatError(
                    "Input container has no audio stream",
                    context={"path": str(path)},
                )
            stream = streams[0]
            resampler = av.audio.resampler.AudioResampler(
                format="s16",
                layout="mono",
                rate=TARGET_SAMPLE_RATE,
            )
            for frame in container.decode(stream):
                converted = resampler.resample(frame)
                for output_frame in () if converted is None else converted:
                    chunk_samples = _frame_to_mono_i16(output_frame, path)
                    decoded_samples += int(chunk_samples.size)
                    _check_sample_limit(path, decoded_samples, max_samples, limits)
                    chunks.append(chunk_samples)
            flushed = resampler.resample(None)
            for output_frame in () if flushed is None else flushed:
                chunk_samples = _frame_to_mono_i16(output_frame, path)
                decoded_samples += int(chunk_samples.size)
                _check_sample_limit(path, decoded_samples, max_samples, limits)
                chunks.append(chunk_samples)
    except (AudioFormatError, ResourceLimitError):
        raise
    except Exception as error:
        raise AudioFormatError(
            "Unable to decode audio with the optional audio backend",
            context={
                "path": str(path),
                "exception_type": type(error).__name__,
            },
        ) from error

    if not chunks or decoded_samples <= 0:
        raise AudioFormatError(
            "Decoded audio stream is empty",
            context={"path": str(path)},
        )
    samples_i16 = np.ascontiguousarray(np.concatenate(chunks), dtype=np.int16)
    float_samples = np.ascontiguousarray(samples_i16.astype(np.float32) / 32768.0)
    if float_samples.ndim != 1 or not bool(np.isfinite(float_samples).all()):
        raise AudioFormatError(
            "Decoded audio has an invalid waveform",
            context={"path": str(path), "shape": str(float_samples.shape)},
        )
    return DecodedAudio(
        samples=float_samples,
        sample_rate=TARGET_SAMPLE_RATE,
        duration_s=float(float_samples.size) / TARGET_SAMPLE_RATE,
    )


def _frame_to_mono_i16(frame: Any, path: Path) -> NDArray[np.int16]:
    array = np.asarray(frame.to_ndarray())
    flattened = np.ascontiguousarray(array.reshape(-1))
    if flattened.dtype != np.int16:
        raise AudioFormatError(
            "PyAV resampler did not produce PCM16 samples",
            context={"path": str(path), "dtype": str(flattened.dtype)},
        )
    return flattened


def _check_sample_limit(
    path: Path,
    decoded_samples: int,
    max_samples: int | None,
    limits: ResourceLimits | None,
) -> None:
    if max_samples is not None and decoded_samples > max_samples:
        assert limits is not None
        raise ResourceLimitError(
            "Audio duration limit exceeded during decoding",
            context={
                "path": str(path),
                "duration_s": decoded_samples / TARGET_SAMPLE_RATE,
                "max_audio_seconds": limits.max_audio_seconds,
            },
        )


def convert_to_pcm16_wav(
    input_path: Path,
    output_path: Path,
    *,
    sample_rate: int = TARGET_SAMPLE_RATE,
) -> DecodedAudio:
    """Create a validated mono PCM16 WAV without overwriting any existing path."""

    if sample_rate != TARGET_SAMPLE_RATE:
        raise InputValidationError(
            "The current model contract only permits 16000 Hz conversion output",
            context={"sample_rate": sample_rate, "expected_sample_rate": TARGET_SAMPLE_RATE},
        )
    if not isinstance(output_path, Path):
        raise InputValidationError("Audio output path must be a pathlib.Path")
    if output_path.suffix.lower() != ".wav":
        raise InputValidationError(
            "Audio conversion output must use a .wav suffix",
            context={"path": str(output_path)},
        )
    temporary = output_path.with_name(output_path.name + ".tmp")
    for candidate, role in ((output_path, "output"), (temporary, "temporary")):
        if os.path.lexists(candidate):
            raise ArtifactExistsError(
                f"Audio conversion {role} already exists: {candidate}",
                context={"path": str(candidate), "role": role},
            )
    audio = load_audio_with_av(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    quantized = np.clip(np.rint(audio.samples * 32768.0), -32768, 32767).astype("<i2")
    try:
        with wave.open(str(temporary), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(quantized.tobytes())
        validated = load_strict_pcm16_wav(temporary)
        os.link(temporary, output_path)
    except (ArtifactExistsError, AudioFormatError, InputValidationError):
        raise
    except FileExistsError as error:
        raise ArtifactExistsError(
            f"Audio conversion output already exists: {output_path}",
            context={"path": str(output_path), "role": "output"},
        ) from error
    except OSError as error:
        raise OutputError(
            "Unable to publish converted WAV",
            context={"path": str(output_path), "exception_type": type(error).__name__},
        ) from error
    finally:
        with suppress(OSError):
            temporary.unlink(missing_ok=True)
    return validated


__all__ = ["convert_to_pcm16_wav", "load_audio_with_av"]
