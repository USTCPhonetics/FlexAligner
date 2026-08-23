from __future__ import annotations

import sys

import numpy as np
import pytest

from flexaligner.adapters.g2p_en_local import (
    CHECKPOINT_SHA256,
    ENGINE_ID,
    ENGINE_VERSION,
    LocalEnglishG2P,
)
from flexaligner.errors import PronunciationGenerationError


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
