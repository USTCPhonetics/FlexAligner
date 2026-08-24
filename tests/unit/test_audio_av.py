from __future__ import annotations

import json
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

import flexaligner.cli as cli
from flexaligner.adapters.audio_av import convert_to_pcm16_wav, load_audio_with_av
from flexaligner.adapters.wav_pcm16 import load_strict_pcm16_wav
from flexaligner.errors import ArtifactExistsError, AudioFormatError, OptionalDependencyError


def _write_stereo_8k(path: Path) -> None:
    mono = (np.sin(np.linspace(0.0, 20.0, 800)) * 12_000).astype("<i2")
    stereo = np.column_stack((mono, -mono)).reshape(-1).astype("<i2")
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(2)
        wav_file.setsampwidth(2)
        wav_file.setframerate(8_000)
        wav_file.writeframes(stereo.tobytes())


def _write_mono_8k_flac(path: Path) -> None:
    import av

    samples = (np.sin(np.linspace(0.0, 20.0, 800)) * 12_000).astype(np.int16).reshape(1, -1)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("flac", rate=8_000)
        stream.layout = "mono"
        frame = av.AudioFrame.from_ndarray(samples, format="s16", layout="mono")
        frame.sample_rate = 8_000
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)


def test_optional_audio_decoder_resamples_and_folds_to_mono(tmp_path: Path) -> None:
    source = tmp_path / "stereo-8k.wav"
    _write_stereo_8k(source)
    decoded = load_audio_with_av(source)
    assert decoded.sample_rate == 16_000
    assert decoded.samples.ndim == 1
    assert decoded.duration_s == pytest.approx(0.1, abs=0.002)


def test_multi_format_decoder_accepts_real_flac(tmp_path: Path) -> None:
    source = tmp_path / "audio.flac"
    _write_mono_8k_flac(source)
    decoded = load_audio_with_av(source)
    assert decoded.sample_rate == 16_000
    assert decoded.duration_s == pytest.approx(0.1, abs=0.002)


def test_explicit_conversion_creates_strict_pcm16_wav_without_overwrite(
    tmp_path: Path,
) -> None:
    source = tmp_path / "stereo-8k.wav"
    output = tmp_path / "canonical.wav"
    _write_stereo_8k(source)
    converted = convert_to_pcm16_wav(source, output)
    strict = load_strict_pcm16_wav(output)
    assert strict.sample_rate == converted.sample_rate == 16_000
    assert strict.samples.size == converted.samples.size
    with pytest.raises(ArtifactExistsError):
        convert_to_pcm16_wav(source, output)


def test_auto_resample_policy_rejects_non_wav_suffix(tmp_path: Path) -> None:
    source = tmp_path / "audio.bin"
    source.write_bytes(b"not audio")
    with pytest.raises(AudioFormatError, match="WAV input only"):
        load_audio_with_av(source, require_wav_container=True)


def test_audio_convert_cli_prints_stable_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "stereo-8k.wav"
    output = tmp_path / "canonical.wav"
    _write_stereo_8k(source)
    assert cli.main(["audio", "convert", str(source), str(output)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["output_path"] == str(output)
    assert payload["sample_rate"] == 16_000
    assert payload["schema_version"] == "1"


def test_missing_audio_extra_has_actionable_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "stereo-8k.wav"
    _write_stereo_8k(source)
    monkeypatch.setitem(sys.modules, "av", None)
    with pytest.raises(OptionalDependencyError) as caught:
        load_audio_with_av(source)
    assert caught.value.context["extra"] == "audio"
    assert "flexaligner[audio]" in str(caught.value.context["suggested_command"])
