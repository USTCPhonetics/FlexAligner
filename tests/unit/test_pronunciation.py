from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from flexaligner.adapters.lexicon_file import PronouncingLexicon, TokenVocabulary
from flexaligner.contracts import PronunciationMode
from flexaligner.errors import PronunciationGenerationError
from flexaligner.pronunciation import oov_words, resolve_effective_lexicon


@dataclass
class FakeG2P:
    outputs: dict[str, tuple[str, ...]]
    calls: list[str] = field(default_factory=list)
    engine_id: str = "fake-local"
    engine_version: str = "test"

    def pronounce(self, word: str) -> tuple[str, ...]:
        self.calls.append(word)
        return self.outputs[word]


def _vocabularies() -> tuple[TokenVocabulary, TokenVocabulary]:
    return (
        TokenVocabulary(token_to_id={"<pad>": 0, "AH": 1, "B": 2}),
        TokenVocabulary(token_to_id={"AH0": 0, "B": 1, "sil": 2, "sph": 3}),
    )


def test_oov_words_are_unique_and_in_first_occurrence_order() -> None:
    lexicon = PronouncingLexicon(entries={"known": (("B",),)})
    assert oov_words(("first", "known", "second", "first"), lexicon) == (
        "first",
        "second",
    )


def test_g2p_fills_only_true_oovs_once_and_records_every_occurrence() -> None:
    chunker, aligner = _vocabularies()
    lexicon = PronouncingLexicon(entries={"known": (("B",), ("AH0",))})
    g2p = FakeG2P(outputs={"missing": ("AH0", "B")})

    effective, notices = resolve_effective_lexicon(
        words=("known", "missing", "missing"),
        lexicon=lexicon,
        mode=PronunciationMode.G2P,
        g2p=g2p,
        chunker_vocabulary=chunker,
        aligner_vocabulary=aligner,
    )

    assert g2p.calls == ["missing"]
    assert effective.entries["known"] == (("B",), ("AH0",))
    assert effective.entries["missing"] == (("AH0", "B"),)
    assert lexicon.entries == {"known": (("B",), ("AH0",))}
    assert notices[0].word_indices == (1, 2)
    assert notices[0].pronunciation == ("AH0", "B")
    assert notices[0].to_dict()["code"] == "oov_g2p_fallback"


def test_lexicon_only_mode_does_not_call_g2p_or_modify_lexicon() -> None:
    chunker, aligner = _vocabularies()
    lexicon = PronouncingLexicon(entries={"known": (("B",),)})
    g2p = FakeG2P(outputs={"missing": ("AH0",)})

    effective, notices = resolve_effective_lexicon(
        words=("known", "missing"),
        lexicon=lexicon,
        mode=PronunciationMode.LEXICON_ONLY,
        g2p=g2p,
        chunker_vocabulary=chunker,
        aligner_vocabulary=aligner,
    )

    assert effective is lexicon
    assert notices == ()
    assert g2p.calls == []


@pytest.mark.parametrize(
    ("pronunciation", "message"),
    [
        ((), "empty or invalid"),
        (("sil",), "invalid or reserved"),
        (("ZZ1",), "Aligner vocabulary"),
        (("AH0", "Z"), "Aligner vocabulary"),
    ],
)
def test_invalid_generated_pronunciations_fail_closed(
    pronunciation: tuple[str, ...],
    message: str,
) -> None:
    chunker, aligner = _vocabularies()
    g2p = FakeG2P(outputs={"missing": pronunciation})

    with pytest.raises(PronunciationGenerationError, match=message):
        resolve_effective_lexicon(
            words=("missing",),
            lexicon=PronouncingLexicon(entries={}),
            mode=PronunciationMode.G2P,
            g2p=g2p,
            chunker_vocabulary=chunker,
            aligner_vocabulary=aligner,
        )


def test_generated_phone_must_match_chunker_after_stress_removal() -> None:
    with pytest.raises(PronunciationGenerationError, match="Chunker vocabulary"):
        resolve_effective_lexicon(
            words=("missing",),
            lexicon=PronouncingLexicon(entries={}),
            mode=PronunciationMode.G2P,
            g2p=FakeG2P(outputs={"missing": ("AH0",)}),
            chunker_vocabulary=TokenVocabulary(token_to_id={"<pad>": 0}),
            aligner_vocabulary=TokenVocabulary(token_to_id={"AH0": 0}),
        )
