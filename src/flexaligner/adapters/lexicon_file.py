"""Strict UTF-8 transcript, pronunciation lexicon, and vocabulary adapters."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from flexaligner.errors import InputValidationError

Pronunciation = tuple[str, ...]
Pronunciations = tuple[Pronunciation, ...]
LexiconEntries = Mapping[str, Pronunciations]

_EDGE_PUNCTUATION = re.compile(r"^[^\w']+|[^\w']+$")


class _DuplicateVocabularyTokenError(ValueError):
    """Internal parse sentinel for duplicate JSON object member names."""

    def __init__(self, token: str) -> None:
        super().__init__(f"Duplicate token key: {token!r}")
        self.token = token


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Build a JSON object while rejecting keys that ``dict`` would overwrite."""

    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateVocabularyTokenError(key)
        result[key] = value
    return result


@dataclass(frozen=True, slots=True, kw_only=True)
class PronouncingLexicon:
    """Ordered, immutable pronunciation entries used by both alignment stages."""

    entries: LexiconEntries

    def __post_init__(self) -> None:
        frozen = {
            word: tuple(tuple(pronunciation) for pronunciation in pronunciations)
            for word, pronunciations in self.entries.items()
        }
        object.__setattr__(self, "entries", MappingProxyType(frozen))

    @property
    def lex(self) -> LexiconEntries:
        """Compatibility alias for the current reference lexicon shape."""

        return self.entries

    def get_prons(self, word: str) -> Pronunciations:
        """Return every pronunciation for *word* without reordering."""

        if word not in self.entries:
            raise InputValidationError(
                f"Word not in lexicon: {word!r}",
                context={"word": word},
            )
        pronunciations = self.entries[word]
        if not pronunciations:
            raise InputValidationError(
                f"Word has no pronunciations: {word!r}",
                context={"word": word},
            )
        return pronunciations


@dataclass(frozen=True, slots=True, kw_only=True)
class TokenVocabulary:
    """An immutable token-to-dense-ID mapping loaded from local JSON."""

    token_to_id: Mapping[str, int]

    def __post_init__(self) -> None:
        object.__setattr__(self, "token_to_id", MappingProxyType(dict(self.token_to_id)))


def _validate_path(path: Path, *, role: str, expected: str) -> None:
    if not isinstance(path, Path):
        raise InputValidationError(
            f"{role} path must be a pathlib.Path",
            context={"role": role, "path_type": type(path).__name__},
        )
    predicate = path.is_file if expected == "file" else path.is_dir
    if not predicate():
        raise InputValidationError(
            f"{role} path is not a {expected}: {path}",
            context={"role": role, "path": str(path), "expected": expected},
        )


def normalize_word(word: str) -> str:
    """Lowercase a word and trim non-word punctuation from both edges."""

    if not isinstance(word, str):
        raise InputValidationError(
            "Transcript token must be a string",
            context={"token_type": type(word).__name__},
        )
    return _EDGE_PUNCTUATION.sub("", word.strip().lower())


def normalize_transcript(text: str) -> tuple[str, ...]:
    """Normalize a non-empty English whitespace-tokenized transcript."""

    if not isinstance(text, str):
        raise InputValidationError(
            "Transcript must be a string",
            context={"text_type": type(text).__name__},
        )
    if not text.strip():
        raise InputValidationError("Input transcript is empty")

    words: list[str] = []
    for token_index, raw_token in enumerate(text.strip().split()):
        word = normalize_word(raw_token)
        if not word:
            raise InputValidationError(
                "Transcript token became empty after normalization",
                context={"token_index": token_index, "raw_token": raw_token},
            )
        words.append(word)
    if not words:
        raise InputValidationError("Input transcript has no word tokens after normalization")
    return tuple(words)


def read_utf8_text(path: Path) -> str:
    """Read one existing regular file as strict UTF-8, preserving the cause."""

    _validate_path(path, role="text", expected="file")
    try:
        return path.read_text(encoding="utf-8", errors="strict")
    except (OSError, UnicodeError) as exc:
        raise InputValidationError(
            f"Unable to read strict UTF-8 text file: {path}: {exc}",
            context={"path": str(path), "reason": str(exc)},
        ) from exc


def load_lexicon(path: Path) -> PronouncingLexicon:
    """Load a strict UTF-8 lexicon while preserving entry/pronunciation order."""

    _validate_path(path, role="lexicon", expected="file")
    try:
        contents = path.read_text(encoding="utf-8", errors="strict")
    except (OSError, UnicodeError) as exc:
        raise InputValidationError(
            f"Unable to read strict UTF-8 lexicon: {path}: {exc}",
            context={"path": str(path), "reason": str(exc)},
        ) from exc

    entries: dict[str, list[Pronunciation]] = {}
    for line_number, raw_line in enumerate(contents.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            raise InputValidationError(
                f"Invalid lexicon line {line_number} in {path}: {line!r}",
                context={"path": str(path), "line_number": line_number},
            )
        word = normalize_word(parts[0])
        if not word:
            raise InputValidationError(
                "Lexicon word became empty after normalization",
                context={
                    "path": str(path),
                    "line_number": line_number,
                    "raw_word": parts[0],
                },
            )
        pronunciation = tuple(parts[1:])
        if not pronunciation or any(not phone for phone in pronunciation):
            raise InputValidationError(
                f"Empty pronunciation at lexicon line {line_number} in {path}",
                context={"path": str(path), "line_number": line_number, "word": word},
            )
        entries.setdefault(word, []).append(pronunciation)

    if not entries:
        raise InputValidationError(
            f"No lexicon entries loaded: {path}",
            context={"path": str(path)},
        )
    return PronouncingLexicon(
        entries={word: tuple(pronunciations) for word, pronunciations in entries.items()}
    )


def validate_transcript_lexicon(
    words: Sequence[str],
    lexicon: PronouncingLexicon,
) -> None:
    """Require every normalized word to have at least one pronunciation."""

    if isinstance(words, (str, bytes)) or not isinstance(words, Sequence):
        raise InputValidationError(
            "words must be a sequence of strings",
            context={"words_type": type(words).__name__},
        )
    if not words:
        raise InputValidationError("Transcript word sequence is empty")
    if not isinstance(lexicon, PronouncingLexicon):
        raise InputValidationError(
            "lexicon must be PronouncingLexicon",
            context={"lexicon_type": type(lexicon).__name__},
        )

    for word_index, word in enumerate(words):
        if not isinstance(word, str) or not word:
            raise InputValidationError(
                "Transcript word must be a non-empty string",
                context={"word_index": word_index, "word_type": type(word).__name__},
            )
        if word not in lexicon.entries:
            raise InputValidationError(
                f"OOV word not found in lexicon: word_index={word_index}, word={word!r}",
                context={"word_index": word_index, "word": word, "reason": "oov"},
            )
        if not lexicon.entries[word]:
            raise InputValidationError(
                f"Word has no pronunciations: word_index={word_index}, word={word!r}",
                context={
                    "word_index": word_index,
                    "word": word,
                    "reason": "no_pronunciations",
                },
            )


def load_dense_token_vocab(path: Path) -> TokenVocabulary:
    """Load a strict UTF-8 JSON object whose integer IDs are dense 0..V-1."""

    _validate_path(path, role="token vocabulary", expected="file")
    try:
        raw_text = path.read_text(encoding="utf-8", errors="strict")
    except (OSError, UnicodeError) as exc:
        raise InputValidationError(
            f"Unable to read strict UTF-8 token vocabulary: {path}: {exc}",
            context={"path": str(path), "reason": str(exc)},
        ) from exc

    try:
        loaded: object = json.loads(raw_text, object_pairs_hook=_unique_json_object)
    except _DuplicateVocabularyTokenError as exc:
        raise InputValidationError(
            f"Duplicate token key in token vocabulary: {path}: {exc.token!r}",
            context={"path": str(path), "token": exc.token, "reason": "duplicate_token"},
        ) from exc
    except json.JSONDecodeError as exc:
        raise InputValidationError(
            f"Invalid token vocabulary JSON: {path}: {exc}",
            context={"path": str(path), "line": exc.lineno, "column": exc.colno},
        ) from exc

    if not isinstance(loaded, dict):
        raise InputValidationError(
            f"Token vocabulary must be a JSON object: {path}",
            context={"path": str(path), "json_type": type(loaded).__name__},
        )
    if not loaded:
        raise InputValidationError(
            f"Token vocabulary must not be empty: {path}",
            context={"path": str(path)},
        )

    token_to_id: dict[str, int] = {}
    for token, raw_id in loaded.items():
        if not isinstance(token, str) or token == "":
            raise InputValidationError(
                f"Token vocabulary contains an empty or non-string token: {path}",
                context={"path": str(path), "token": repr(token)},
            )
        if isinstance(raw_id, bool) or not isinstance(raw_id, int):
            raise InputValidationError(
                f"Token ID must be a non-boolean integer for token={token!r}",
                context={"path": str(path), "token": token, "token_id": repr(raw_id)},
            )
        if raw_id < 0:
            raise InputValidationError(
                f"Token ID must be non-negative for token={token!r}",
                context={"path": str(path), "token": token, "token_id": raw_id},
            )
        token_to_id[token] = raw_id

    ids = tuple(token_to_id.values())
    if len(ids) != len(set(ids)):
        raise InputValidationError(
            f"Duplicate token IDs found in vocabulary: {path}",
            context={"path": str(path)},
        )
    expected_ids = set(range(len(ids)))
    actual_ids = set(ids)
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)
        unexpected = sorted(actual_ids - expected_ids)
        raise InputValidationError(
            f"Token vocabulary IDs must be dense 0..{len(ids) - 1}: {path}",
            context={
                "path": str(path),
                "missing_ids": repr(missing[:20]),
                "unexpected_ids": repr(unexpected[:20]),
            },
        )
    return TokenVocabulary(token_to_id=token_to_id)


def validate_aligner_vocabulary(
    words: Sequence[str],
    lexicon: PronouncingLexicon,
    vocabulary: TokenVocabulary,
    model_vocab_size: int,
) -> None:
    """Validate every phone in every pronunciation used by the transcript."""

    validate_transcript_lexicon(words, lexicon)
    if not isinstance(vocabulary, TokenVocabulary):
        raise InputValidationError(
            "vocabulary must be TokenVocabulary",
            context={"vocabulary_type": type(vocabulary).__name__},
        )
    if isinstance(model_vocab_size, bool) or not isinstance(model_vocab_size, int):
        raise InputValidationError(
            "model_vocab_size must be a non-boolean integer",
            context={"model_vocab_size": repr(model_vocab_size)},
        )
    if model_vocab_size <= 0:
        raise InputValidationError(
            "model_vocab_size must be positive",
            context={"model_vocab_size": model_vocab_size},
        )

    for word_index, word in enumerate(words):
        for pronunciation_index, pronunciation in enumerate(lexicon.get_prons(word)):
            if not pronunciation:
                raise InputValidationError(
                    "Empty aligner pronunciation",
                    context={
                        "word_index": word_index,
                        "word": word,
                        "pronunciation_index": pronunciation_index,
                    },
                )
            for phone_index, phone in enumerate(pronunciation):
                if not isinstance(phone, str) or phone == "":
                    raise InputValidationError(
                        "Aligner pronunciation contains an empty or non-string phone",
                        context={
                            "word_index": word_index,
                            "word": word,
                            "pronunciation_index": pronunciation_index,
                            "phone_index": phone_index,
                        },
                    )
                if phone not in vocabulary.token_to_id:
                    raise InputValidationError(
                        f"Aligner phone not in model vocabulary: phone={phone!r}",
                        context={
                            "phone": phone,
                            "word_index": word_index,
                            "word": word,
                            "pronunciation_index": pronunciation_index,
                        },
                    )
                phone_id = vocabulary.token_to_id[phone]
                if not 0 <= phone_id < model_vocab_size:
                    raise InputValidationError(
                        "Aligner phone ID is outside model output range",
                        context={
                            "phone": phone,
                            "phone_id": phone_id,
                            "model_vocab_size": model_vocab_size,
                            "word": word,
                            "word_index": word_index,
                            "pronunciation_index": pronunciation_index,
                        },
                    )


def validate_local_model_dir(path: Path, role: str) -> Path:
    """Require one explicit local model directory without resolving remotely."""

    if not isinstance(role, str) or not role.strip():
        raise InputValidationError(
            "Model role must be a non-empty string",
            context={"role_type": type(role).__name__},
        )
    _validate_path(path, role=role, expected="directory")
    return path
