"""Optional local Mandarin segmentation and G2P adapters."""

from __future__ import annotations

import logging
import unicodedata
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version

from ..errors import InputValidationError, OptionalDependencyError, PronunciationGenerationError


def _is_discardable_punctuation(token: str) -> bool:
    return bool(token) and all(
        unicodedata.category(character)[0] in {"P", "S"} for character in token
    )


def _contains_han(text: str) -> bool:
    return any(
        "\u3400" <= character <= "\u9fff" or "\uf900" <= character <= "\ufaff" for character in text
    )


def _require_jieba() -> object:
    try:
        jieba = import_module("jieba")
    except ImportError as error:
        raise OptionalDependencyError(
            "Mandarin segmentation requires the optional zh language pack",
            context={
                "dependency": "jieba==0.42.1",
                "extra": "zh",
                "suggested_command": "python -m pip install 'flexaligner[zh]'",
            },
        ) from error
    set_log_level = getattr(jieba, "setLogLevel", None)
    if callable(set_log_level):
        set_log_level(logging.WARNING)
    return jieba


def segment_mandarin(text: str) -> tuple[str, ...]:
    """Segment Mandarin without crossing user-supplied whitespace boundaries."""

    if not isinstance(text, str):
        raise InputValidationError(
            "Transcript must be a string",
            context={"text_type": type(text).__name__},
        )
    if not text.strip():
        raise InputValidationError("Input transcript is empty")
    jieba = _require_jieba()
    words: list[str] = []
    for fragment_index, fragment in enumerate(text.split()):
        cut = getattr(jieba, "cut", None)
        if not callable(cut):
            raise OptionalDependencyError(
                "Installed jieba does not provide the required cut API",
                context={"dependency": "jieba==0.42.1", "extra": "zh"},
            )
        for raw_token in cut(fragment, HMM=False):
            token = str(raw_token).strip()
            if not token or _is_discardable_punctuation(token):
                continue
            if not _contains_han(token):
                raise InputValidationError(
                    "Mandarin segmentation produced a non-Han token",
                    context={
                        "fragment_index": fragment_index,
                        "raw_fragment": fragment,
                        "token": token,
                    },
                )
            words.append(token)
    if not words:
        raise InputValidationError(
            "Input transcript has no Mandarin word tokens after segmentation"
        )
    return tuple(words)


class LocalMandarinG2P:
    """Tone-free initial/final G2P compatible with the current Mandarin bundle."""

    engine_id = "pypinyin-local-initial-final"

    def __init__(self) -> None:
        try:
            from pypinyin import Style, pinyin
        except ImportError as error:
            raise OptionalDependencyError(
                "Mandarin G2P requires the optional zh language pack",
                context={
                    "dependency": "pypinyin==0.55.0",
                    "extra": "zh",
                    "suggested_command": "python -m pip install 'flexaligner[zh]'",
                },
            ) from error
        self._style = Style
        self._pinyin = pinyin
        try:
            self.engine_version = version("pypinyin")
        except PackageNotFoundError:
            self.engine_version = "unknown"

    def pronounce(self, word: str) -> tuple[str, ...]:
        if not isinstance(word, str) or not word or not _contains_han(word):
            raise PronunciationGenerationError(
                "Mandarin G2P accepts non-empty Han words only",
                context={"engine": self.engine_id, "word": str(word)},
            )
        initials = self._pinyin(
            word,
            style=self._style.INITIALS,
            heteronym=False,
            strict=True,
            errors="default",
        )
        finals = self._pinyin(
            word,
            style=self._style.FINALS,
            heteronym=False,
            strict=True,
            errors="default",
        )
        if len(initials) != len(finals) or len(initials) != len(word):
            raise PronunciationGenerationError(
                "Mandarin G2P returned an unexpected syllable count",
                context={
                    "engine": self.engine_id,
                    "word": word,
                    "characters": len(word),
                    "initials": len(initials),
                    "finals": len(finals),
                },
            )
        phones: list[str] = []
        for character_index, (initial_item, final_item) in enumerate(
            zip(initials, finals, strict=True)
        ):
            if len(initial_item) != 1 or len(final_item) != 1:
                raise PronunciationGenerationError(
                    "Mandarin G2P returned an ambiguous pronunciation",
                    context={
                        "engine": self.engine_id,
                        "word": word,
                        "character_index": character_index,
                    },
                )
            initial = initial_item[0]
            final = final_item[0]
            if not isinstance(initial, str) or not isinstance(final, str):
                raise PronunciationGenerationError(
                    "Mandarin G2P returned non-string phones",
                    context={
                        "engine": self.engine_id,
                        "word": word,
                        "character_index": character_index,
                    },
                )
            if final == "i":
                if initial in {"zh", "ch", "sh"}:
                    final = "ix"
                elif initial in {"z", "c", "s"}:
                    final = "iy"
                elif initial == "r":
                    final = "iz"
            syllable = tuple(phone for phone in (initial, final) if phone)
            if not syllable or any(
                not phone.isascii() or not phone.islower() for phone in syllable
            ):
                raise PronunciationGenerationError(
                    "Mandarin G2P could not map a character to the model phone set",
                    context={
                        "engine": self.engine_id,
                        "word": word,
                        "character": word[character_index],
                        "character_index": character_index,
                        "initial": initial,
                        "final": final,
                    },
                )
            phones.extend(syllable)
        return tuple(phones)


__all__ = ["LocalMandarinG2P", "segment_mandarin"]
