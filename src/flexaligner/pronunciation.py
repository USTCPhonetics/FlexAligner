"""English OOV pronunciation resolution with lexicon-first semantics."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Protocol

from .adapters.lexicon_file import PronouncingLexicon, TokenVocabulary
from .contracts import Language, PronunciationMode, PronunciationNotice
from .errors import PronunciationGenerationError

_STRESS_SUFFIX = re.compile(r"[012]$")
_RESERVED = {"sil", "sph", "spn", "null"}


class G2PPort(Protocol):
    engine_id: str
    engine_version: str

    def pronounce(self, word: str) -> tuple[str, ...]: ...


def oov_words(words: Sequence[str], lexicon: PronouncingLexicon) -> tuple[str, ...]:
    """Return unique OOVs in first-occurrence order."""

    return tuple(dict.fromkeys(word for word in words if word not in lexicon.entries))


def resolve_effective_lexicon(
    *,
    words: Sequence[str],
    lexicon: PronouncingLexicon,
    mode: PronunciationMode,
    g2p: G2PPort | None,
    chunker_vocabulary: TokenVocabulary,
    aligner_vocabulary: TokenVocabulary,
    language: Language = Language.EN,
) -> tuple[PronouncingLexicon, tuple[PronunciationNotice, ...]]:
    """Fill only true OOVs in memory and preserve every explicit lexicon entry."""

    missing = oov_words(words, lexicon)
    if not missing or mode is PronunciationMode.LEXICON_ONLY:
        return lexicon, ()
    if g2p is None:
        raise PronunciationGenerationError(f"{_language_name(language)} G2P backend is unavailable")

    generated: dict[str, tuple[str, ...]] = {}
    for word in missing:
        pronunciation = g2p.pronounce(word)
        _validate_generated_pronunciation(
            word=word,
            pronunciation=pronunciation,
            engine_id=g2p.engine_id,
            language=language,
            chunker_vocabulary=chunker_vocabulary.token_to_id,
            aligner_vocabulary=aligner_vocabulary.token_to_id,
        )
        generated[word] = pronunciation

    merged = dict(lexicon.entries)
    for word in missing:
        merged[word] = (generated[word],)
    notices = tuple(
        PronunciationNotice(
            code="oov_g2p_fallback",
            word=word,
            word_indices=tuple(index for index, value in enumerate(words) if value == word),
            pronunciation=generated[word],
            engine_id=g2p.engine_id,
            engine_version=g2p.engine_version,
        )
        for word in missing
    )
    return PronouncingLexicon(entries=merged), notices


def _validate_generated_pronunciation(
    *,
    word: str,
    pronunciation: tuple[str, ...],
    engine_id: str,
    language: Language,
    chunker_vocabulary: Mapping[str, int],
    aligner_vocabulary: Mapping[str, int],
) -> None:
    if not isinstance(pronunciation, tuple) or not pronunciation:
        raise PronunciationGenerationError(
            f"{_language_name(language)} G2P generated an empty or invalid pronunciation",
            context={"engine": engine_id, "word": word},
        )
    for phone_index, phone in enumerate(pronunciation):
        context: dict[str, str | int] = {
            "engine": engine_id,
            "phone": str(phone),
            "phone_index": phone_index,
            "word": word,
        }
        if not isinstance(phone, str) or not phone or phone.lower() in _RESERVED:
            raise PronunciationGenerationError(
                f"{_language_name(language)} G2P generated an invalid or reserved phone",
                context=context,
            )
        if phone not in aligner_vocabulary:
            raise PronunciationGenerationError(
                f"{_language_name(language)} G2P phone is not in the Aligner vocabulary",
                context=context,
            )
        chunker_phone = _STRESS_SUFFIX.sub("", phone) if language is Language.EN else phone
        if chunker_phone not in chunker_vocabulary:
            raise PronunciationGenerationError(
                f"{_language_name(language)} G2P phone is not in the Chunker vocabulary"
                + (" after stress removal" if language is Language.EN else ""),
                context={**context, "chunker_phone": chunker_phone},
            )


def _language_name(language: Language) -> str:
    return "English" if language is Language.EN else "Mandarin"


EnglishG2PPort = G2PPort


__all__ = ["EnglishG2PPort", "G2PPort", "oov_words", "resolve_effective_lexicon"]
