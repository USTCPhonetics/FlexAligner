"""Fail-closed language consistency checks for text, lexicon, and model vocabularies."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from .adapters.lexicon_file import PronouncingLexicon, TokenVocabulary
from .contracts import Language
from .errors import LanguageMismatchError, ModelCompatibilityError

_ENGLISH_PHONE = re.compile(r"^[A-Z]+[012]?$")


def contains_han(text: str) -> bool:
    return any(
        "\u3400" <= character <= "\u9fff" or "\uf900" <= character <= "\ufaff" for character in text
    )


def validate_transcript_language(text: str, language: Language) -> None:
    has_han = contains_han(text)
    if language is Language.EN and has_han:
        raise LanguageMismatchError(
            "Transcript contains Han characters but --language en was selected; the language may be wrong",
            context={"component": "transcript", "detected": "zh", "selected": "en"},
        )
    if language is Language.ZH and not has_han:
        raise LanguageMismatchError(
            "Transcript contains no Han characters but --language zh was selected; the language may be wrong",
            context={"component": "transcript", "detected": "non-zh", "selected": "zh"},
        )


def validate_lexicon_language(lexicon: PronouncingLexicon, language: Language) -> None:
    headwords = tuple(lexicon.entries)
    han_words = tuple(word for word in headwords if contains_han(word))
    if language is Language.EN and han_words:
        raise LanguageMismatchError(
            "Lexicon contains Han headwords but --language en was selected; the language may be wrong",
            context={
                "component": "lexicon",
                "detected": "zh",
                "selected": "en",
                "example": han_words[0],
            },
        )
    if language is Language.ZH and not han_words:
        raise LanguageMismatchError(
            "Lexicon contains no Han headwords but --language zh was selected; the language may be wrong",
            context={"component": "lexicon", "detected": "non-zh", "selected": "zh"},
        )


def infer_model_language(vocabulary: Mapping[str, int]) -> Language | None:
    tokens = set(vocabulary)
    if {"ix", "iy", "iz"}.issubset(tokens) and "sph" not in tokens:
        return Language.ZH
    lexical_tokens = {
        token for token in tokens if not token.startswith("<") and token not in {"|", " "}
    }
    if "sph" in tokens or any(_ENGLISH_PHONE.fullmatch(token) for token in lexical_tokens):
        return Language.EN
    return None


def validate_model_language(
    vocabularies: Sequence[tuple[str, TokenVocabulary]],
    language: Language,
) -> None:
    for role, vocabulary in vocabularies:
        detected = infer_model_language(vocabulary.token_to_id)
        if detected is None:
            raise ModelCompatibilityError(
                "Unable to identify model language from its phone vocabulary",
                context={"role": role, "selected_language": language.value},
            )
        if detected is not language:
            raise LanguageMismatchError(
                "Model phone vocabulary does not match the selected language; the language may be wrong",
                context={
                    "component": role,
                    "detected": detected.value,
                    "selected": language.value,
                },
            )


__all__ = [
    "contains_han",
    "infer_model_language",
    "validate_lexicon_language",
    "validate_model_language",
    "validate_transcript_language",
]
