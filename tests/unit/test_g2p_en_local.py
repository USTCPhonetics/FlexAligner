from __future__ import annotations

import sys

import numpy as np
import pytest

import flexaligner.adapters.g2p_en_local as g2p_module
from flexaligner.adapters.g2p_en_local import (
    CHECKPOINT_SHA256,
    ENGINE_ID,
    ENGINE_VERSION,
    LocalEnglishG2P,
)
from flexaligner.errors import OptionalDependencyError, PronunciationGenerationError


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        ("activationist", ("AE2", "K", "T", "IH0", "V", "EY1", "SH", "AH0", "N", "IH0", "S", "T")),
        ("openphonetics", ("AA2", "P", "AH0", "N", "F", "AA1", "N", "IH0", "T", "IH0", "S", "K")),
        ("codex", ("K", "OW1", "D", "AH0", "K", "S")),
        ("hello", ("HH", "EH1", "L", "OW0")),
        ("foobar", ("F", "UW1", "B", "AA1", "R")),
    ],
)
def test_local_g2p_has_deterministic_golden_pronunciations(
    word: str,
    expected: tuple[str, ...],
) -> None:
    backend = LocalEnglishG2P()

    assert backend.pronounce(word) == expected
    assert backend.pronounce(word) == expected
    assert backend.engine_id == ENGINE_ID
    assert backend.engine_version == ENGINE_VERSION
    assert len(CHECKPOINT_SHA256) == 64
    assert "nltk" not in sys.modules
    assert "g2p_en" not in sys.modules


@pytest.mark.parametrize("word", ["Hello", "hello2", "naïve", "-hello", "hello-", "a" * 65])
def test_local_g2p_rejects_unsupported_word_shapes(word: str) -> None:
    with pytest.raises(PronunciationGenerationError):
        LocalEnglishG2P().pronounce(word)


def test_local_g2p_rejects_non_finite_recurrent_state() -> None:
    backend = LocalEnglishG2P()
    backend._arrays["enc_emb"].fill(np.nan)

    with pytest.raises(PronunciationGenerationError, match="non-finite recurrent state"):
        backend.pronounce("hello")


def test_local_g2p_rejects_decoder_without_end_of_sequence() -> None:
    backend = LocalEnglishG2P()
    backend._arrays["fc_w"].fill(0.0)
    backend._arrays["fc_b"].fill(-1.0)
    backend._arrays["fc_b"][4] = 1.0

    with pytest.raises(PronunciationGenerationError, match="end-of-sequence"):
        backend.pronounce("hello")


def test_local_g2p_requires_the_optional_english_language_pack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_pack(name: str) -> object:
        assert name == "flexaligner_g2p_en"
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(g2p_module, "import_module", missing_pack)
    with pytest.raises(OptionalDependencyError) as caught:
        LocalEnglishG2P()
    assert caught.value.code == "optional_dependency_missing"
    assert caught.value.context["extra"] == "en"
    assert caught.value.context["dependency"] == "flexaligner-g2p-en==0.3.0a1"


def test_local_g2p_rejects_an_incompatible_language_pack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class IncompatiblePack:
        __version__ = "9.9.9"

        @staticmethod
        def checkpoint_bytes() -> bytes:
            return b"not-used"

    monkeypatch.setattr(g2p_module, "import_module", lambda name: IncompatiblePack())
    with pytest.raises(OptionalDependencyError, match="incompatible") as caught:
        LocalEnglishG2P()
    assert caught.value.context["actual_version"] == "9.9.9"
