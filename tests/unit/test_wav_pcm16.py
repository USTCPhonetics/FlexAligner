from __future__ import annotations

import struct
import wave
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import flexaligner.adapters.wav_pcm16 as wav_adapter
from flexaligner.adapters.wav_pcm16 import DecodedAudio, load_strict_pcm16_wav
from flexaligner.contracts import ResourceLimits
from flexaligner.errors import (
    AudioFormatError,
    ConfigurationError,
    InputValidationError,
    ResourceLimitError,
)


def _write_wav(
    path: Path,
    *,
    channels: int = 1,
    sample_width: int = 2,
    sample_rate: int = 16_000,
    frames: bytes = b"",
) -> None:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(sample_width)
        output.setframerate(sample_rate)
        output.writeframes(frames)


def _write_truncated_wav(path: Path) -> None:
    fmt_chunk = struct.pack("<HHIIHH", 1, 1, 16_000, 32_000, 2, 16)
    declared_data_bytes = 4
    payload = struct.pack("<h", 7)
    riff_size = 4 + (8 + len(fmt_chunk)) + (8 + declared_data_bytes)
    path.write_bytes(
        b"RIFF"
        + struct.pack("<I", riff_size)
        + b"WAVEfmt "
        + struct.pack("<I", len(fmt_chunk))
        + fmt_chunk
        + b"data"
        + struct.pack("<I", declared_data_bytes)
        + payload
    )


def test_decodes_strict_pcm16_to_contiguous_float32(tmp_path: Path) -> None:
    path = tmp_path / "valid.wav"
    _write_wav(path, frames=struct.pack("<hhh", -32768, 0, 32767))

    decoded = load_strict_pcm16_wav(path)

    assert isinstance(decoded, DecodedAudio)
    assert decoded.sample_rate == 16_000
    assert decoded.duration_s == pytest.approx(3 / 16_000)
    assert decoded.samples.dtype == np.float32
    assert decoded.samples.flags.c_contiguous
    assert decoded.samples.tolist() == pytest.approx([-1.0, 0.0, 32767 / 32768])


def test_duration_limit_accepts_exact_boundary_and_rejects_excess(tmp_path: Path) -> None:
    path = tmp_path / "one-second.wav"
    _write_wav(path, frames=np.zeros(16_000, dtype="<i2").tobytes())

    accepted = load_strict_pcm16_wav(path, ResourceLimits(max_audio_seconds=1.0))
    assert accepted.duration_s == 1.0

    with pytest.raises(ResourceLimitError, match="duration limit exceeded") as caught:
        load_strict_pcm16_wav(path, ResourceLimits(max_audio_seconds=0.999))
    assert caught.value.context == {
        "path": str(path),
        "duration_s": 1.0,
        "max_audio_seconds": 0.999,
    }


@pytest.mark.parametrize(
    ("name", "channels", "sample_width", "sample_rate", "frames", "message"),
    [
        ("stereo.wav", 2, 2, 16_000, struct.pack("<hhhh", 0, 0, 1, 1), "mono"),
        ("width.wav", 1, 1, 16_000, b"\x00\x01", "PCM16"),
        ("rate.wav", 1, 2, 8_000, struct.pack("<hh", 0, 1), "16000 Hz"),
        ("empty.wav", 1, 2, 16_000, b"", "Empty WAV"),
    ],
)
def test_rejects_wav_contract_violations(
    tmp_path: Path,
    name: str,
    channels: int,
    sample_width: int,
    sample_rate: int,
    frames: bytes,
    message: str,
) -> None:
    path = tmp_path / name
    _write_wav(
        path,
        channels=channels,
        sample_width=sample_width,
        sample_rate=sample_rate,
        frames=frames,
    )

    with pytest.raises(AudioFormatError, match=message) as caught:
        load_strict_pcm16_wav(path)
    assert caught.value.code == "audio_format_unsupported"
    assert caught.value.context["path"] == str(path)


def test_rejects_non_none_compression_without_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "compressed.wav"
    path.write_bytes(b"placeholder")

    class FakeCompressedWave:
        def __enter__(self) -> FakeCompressedWave:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def getnchannels(self) -> int:
            return 1

        def getsampwidth(self) -> int:
            return 2

        def getframerate(self) -> int:
            return 16_000

        def getcomptype(self) -> str:
            return "ULAW"

        def getnframes(self) -> int:
            return 1

    monkeypatch.setattr(wav_adapter.wave, "open", lambda *_args, **_kwargs: FakeCompressedWave())

    with pytest.raises(AudioFormatError, match="uncompressed") as caught:
        load_strict_pcm16_wav(path)
    assert caught.value.context["compression"] == "ULAW"


def test_rejects_truncated_payload_against_header(tmp_path: Path) -> None:
    path = tmp_path / "truncated.wav"
    _write_truncated_wav(path)

    with pytest.raises(AudioFormatError, match="frame count") as caught:
        load_strict_pcm16_wav(path)
    assert caught.value.context["header_frames"] == 2
    assert caught.value.context["decoded_bytes"] == 2


def test_invalid_container_preserves_wave_error_as_cause(tmp_path: Path) -> None:
    path = tmp_path / "invalid.wav"
    path.write_bytes(b"not a RIFF/WAVE file")

    with pytest.raises(AudioFormatError, match="Invalid PCM WAV file") as caught:
        load_strict_pcm16_wav(path)

    assert isinstance(caught.value.__cause__, wave.Error)


def test_os_read_failure_is_typed_and_preserves_cause(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "exists.wav"
    path.write_bytes(b"placeholder")
    failure = OSError("simulated read failure")

    def fail_open(*_args: object, **_kwargs: object) -> Any:
        raise failure

    monkeypatch.setattr(wav_adapter.wave, "open", fail_open)

    with pytest.raises(InputValidationError, match="Unable to read audio") as caught:
        load_strict_pcm16_wav(path)
    assert caught.value.__cause__ is failure


def test_nonfinite_decoded_waveform_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "valid.wav"
    _write_wav(path, frames=struct.pack("<h", 0))

    class FalseFinite:
        def all(self) -> bool:
            return False

    monkeypatch.setattr(wav_adapter.np, "isfinite", lambda _samples: FalseFinite())

    with pytest.raises(AudioFormatError, match="NaN/Inf"):
        load_strict_pcm16_wav(path)


def test_missing_directory_non_path_and_bad_limits_are_typed(tmp_path: Path) -> None:
    with pytest.raises(InputValidationError, match="not a file"):
        load_strict_pcm16_wav(tmp_path / "missing.wav")
    with pytest.raises(InputValidationError, match="not a file"):
        load_strict_pcm16_wav(tmp_path)
    with pytest.raises(InputValidationError, match=r"pathlib\.Path"):
        load_strict_pcm16_wav("audio.wav")  # type: ignore[arg-type]

    path = tmp_path / "valid.wav"
    _write_wav(path, frames=struct.pack("<h", 0))
    with pytest.raises(InputValidationError, match="ResourceLimits"):
        load_strict_pcm16_wav(path, object())  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="positive and finite"):
        ResourceLimits(max_audio_seconds=float("nan"))
