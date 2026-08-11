"""Model-free characterization of strict reference input behavior."""

from __future__ import annotations

import struct
import wave
from pathlib import Path

import pytest

from tests.characterization.reference_loader import load_reference_module


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


def test_read_input_words_requires_exactly_one_text_source(tmp_path: Path) -> None:
    reference = load_reference_module()
    text_path = tmp_path / "words.txt"
    text_path.write_text("hello", encoding="utf-8")

    with pytest.raises(ValueError, match="Exactly one"):
        reference.read_input_words(None, None)
    with pytest.raises(ValueError, match="Exactly one"):
        reference.read_input_words("hello", text_path)
    with pytest.raises(FileNotFoundError, match="--text_path is not a file"):
        reference.read_input_words(None, tmp_path / "missing.txt")


def test_read_input_words_utf8_normalization_preserves_order_and_duplicates(
    tmp_path: Path,
) -> None:
    reference = load_reference_module()
    raw = "  Hello, HELLO! don't Café  "
    text_path = tmp_path / "words.txt"
    text_path.write_text(raw, encoding="utf-8")

    returned_raw, words = reference.read_input_words(None, text_path)

    assert returned_raw == raw
    assert words == ["hello", "hello", "don't", "café"]


def test_read_input_words_rejects_empty_punctuation_only_and_invalid_utf8(
    tmp_path: Path,
) -> None:
    reference = load_reference_module()

    with pytest.raises(ValueError, match="Input transcript is empty"):
        reference.read_input_words(" \n\t ", None)
    with pytest.raises(ValueError, match="token_index=1"):
        reference.read_input_words("hello !!! after", None)

    invalid_utf8 = tmp_path / "invalid.txt"
    invalid_utf8.write_bytes(b"hello\xff")
    with pytest.raises(UnicodeDecodeError):
        reference.read_input_words(None, invalid_utf8)


def test_load_raw_lexicon_ignores_comments_and_preserves_pronunciation_order(
    tmp_path: Path,
) -> None:
    reference = load_reference_module()
    lexicon_path = tmp_path / "lexicon.dict"
    lexicon_path.write_text(
        "\n# comment\n  # indented comment\nWORD, W ER1 D\nword W AO2 R D\ncafé K AE1 F EY0\n",
        encoding="utf-8",
    )

    dictionary = reference.load_raw_lexicon(lexicon_path)

    assert list(dictionary.lex) == ["word", "café"]
    assert dictionary.lex["word"] == [["W", "ER1", "D"], ["W", "AO2", "R", "D"]]
    assert dictionary.lex["café"] == [["K", "AE1", "F", "EY0"]]


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("word\n", "Invalid lexicon line 1"),
        ("!!! AH\n", "Lexicon word became empty"),
        ("\n# comment only\n", "No lexicon entries loaded"),
    ],
)
def test_load_raw_lexicon_rejects_invalid_or_empty_content(
    tmp_path: Path,
    contents: str,
    message: str,
) -> None:
    reference = load_reference_module()
    lexicon_path = tmp_path / "invalid.dict"
    lexicon_path.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        reference.load_raw_lexicon(lexicon_path)


def test_load_raw_lexicon_rejects_invalid_utf8_and_missing_path(tmp_path: Path) -> None:
    reference = load_reference_module()
    invalid_utf8 = tmp_path / "invalid-utf8.dict"
    invalid_utf8.write_bytes(b"word W ER1 D\xff\n")

    with pytest.raises(UnicodeDecodeError):
        reference.load_raw_lexicon(invalid_utf8)
    with pytest.raises(FileNotFoundError, match="Lexicon not found"):
        reference.load_raw_lexicon(tmp_path / "missing.dict")


def test_build_chunk_lexicon_strips_terminal_arpabet_stress_without_reordering() -> None:
    reference = load_reference_module()
    raw = reference.PronouncingDictionary(
        lex={
            "word": [["W", "ER1", "D"], ["W", "AO2", "R", "D"]],
            "odd": [["B2", "IY0"]],
        }
    )

    chunk_lexicon = reference.build_chunk_lexicon(raw)

    assert chunk_lexicon == {
        "word": [["W", "ER", "D"], ["W", "AO", "R", "D"]],
        "odd": [["B", "IY"]],
    }
    assert raw.lex["word"][0] == ["W", "ER1", "D"]


def test_validate_transcript_lexicon_reports_oov_and_empty_pronunciation_indices() -> None:
    reference = load_reference_module()
    valid = reference.PronouncingDictionary(lex={"known": [["N", "OW1", "N"]]})

    reference.validate_transcript_lexicon(["known", "known"], valid)

    with pytest.raises(KeyError) as oov_error:
        reference.validate_transcript_lexicon(["known", "missing"], valid)
    assert "word_index=1" in oov_error.value.args[0]
    assert "word='missing'" in oov_error.value.args[0]

    empty = reference.PronouncingDictionary(lex={"known": []})
    with pytest.raises(RuntimeError, match="word_index=0"):
        reference.validate_transcript_lexicon(["known"], empty)


def test_validate_align_phones_checks_every_pronunciation() -> None:
    reference = load_reference_module()
    dictionary = reference.PronouncingDictionary(
        lex={"word": [["W", "ER1", "D"], ["W", "UNKNOWN", "D"]]}
    )

    with pytest.raises(KeyError) as unknown_error:
        reference.validate_align_phones(
            ["word"],
            dictionary,
            align_vocab={"W": 0, "ER1": 1, "D": 2},
            align_vocab_size=3,
        )

    message = unknown_error.value.args[0]
    assert "phone='UNKNOWN'" in message
    assert "word_index=0" in message
    assert "pronunciation_index=1" in message


def test_validate_align_phones_rejects_empty_pronunciation_and_out_of_range_id() -> None:
    reference = load_reference_module()
    empty_pronunciation = reference.PronouncingDictionary(lex={"word": [[]]})
    with pytest.raises(RuntimeError, match="pronunciation_index=0"):
        reference.validate_align_phones(
            ["word"], empty_pronunciation, align_vocab={}, align_vocab_size=0
        )

    out_of_range = reference.PronouncingDictionary(lex={"word": [["W"]]})
    with pytest.raises(KeyError) as range_error:
        reference.validate_align_phones(
            ["word"], out_of_range, align_vocab={"W": 3}, align_vocab_size=3
        )
    assert "phone_id=3" in range_error.value.args[0]
    assert "model_vocab_size=3" in range_error.value.args[0]


def test_validate_align_phones_accepts_all_known_pronunciations() -> None:
    reference = load_reference_module()
    dictionary = reference.PronouncingDictionary(
        lex={"word": [["W", "ER1", "D"], ["W", "AO2", "R", "D"]]}
    )

    reference.validate_align_phones(
        ["word"],
        dictionary,
        align_vocab={"W": 0, "ER1": 1, "D": 2, "AO2": 3, "R": 4},
        align_vocab_size=5,
    )


def test_load_pcm16_mono_wav_decodes_strict_16khz_audio(tmp_path: Path) -> None:
    reference = load_reference_module()
    wav_path = tmp_path / "valid.wav"
    _write_wav(wav_path, frames=struct.pack("<hhh", -32768, 0, 32767))

    audio, sample_rate = reference.load_pcm16_mono_wav(wav_path)

    assert sample_rate == 16_000
    assert audio.dtype.name == "float32"
    assert audio.flags.c_contiguous
    assert audio.tolist() == pytest.approx([-1.0, 0.0, 32767 / 32768])


@pytest.mark.parametrize(
    ("name", "channels", "sample_width", "sample_rate", "frames", "message"),
    [
        ("stereo.wav", 2, 2, 16_000, struct.pack("<hhhh", 0, 0, 1, 1), "Expected mono"),
        ("rate.wav", 1, 2, 8_000, struct.pack("<hh", 0, 1), "Expected 16000 Hz"),
        ("width.wav", 1, 1, 16_000, b"\x00\x01", "Expected PCM16"),
        ("empty.wav", 1, 2, 16_000, b"", "Empty WAV"),
    ],
)
def test_load_pcm16_mono_wav_rejects_contract_violations(
    tmp_path: Path,
    name: str,
    channels: int,
    sample_width: int,
    sample_rate: int,
    frames: bytes,
    message: str,
) -> None:
    reference = load_reference_module()
    wav_path = tmp_path / name
    _write_wav(
        wav_path,
        channels=channels,
        sample_width=sample_width,
        sample_rate=sample_rate,
        frames=frames,
    )

    with pytest.raises(ValueError, match=message):
        reference.load_pcm16_mono_wav(wav_path)


def test_load_pcm16_mono_wav_rejects_invalid_and_missing_files(tmp_path: Path) -> None:
    reference = load_reference_module()
    invalid = tmp_path / "invalid.wav"
    invalid.write_bytes(b"not a RIFF/WAVE file")

    with pytest.raises(ValueError, match="Invalid PCM WAV file"):
        reference.load_pcm16_mono_wav(invalid)
    with pytest.raises(FileNotFoundError, match="--wav_path is not a file"):
        reference.load_pcm16_mono_wav(tmp_path / "missing.wav")
