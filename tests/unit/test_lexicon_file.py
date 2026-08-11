from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from flexaligner.adapters.lexicon_file import (
    PronouncingLexicon,
    TokenVocabulary,
    load_dense_token_vocab,
    load_lexicon,
    normalize_transcript,
    normalize_word,
    read_utf8_text,
    validate_aligner_vocabulary,
    validate_local_model_dir,
    validate_transcript_lexicon,
)
from flexaligner.errors import InputValidationError


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def test_normalization_preserves_reference_order_duplicates_and_unicode() -> None:
    assert normalize_word("  ...Café?! ") == "café"
    assert normalize_word("don't") == "don't"
    assert normalize_transcript("  Hello, HELLO! don't Café  ") == (
        "hello",
        "hello",
        "don't",
        "café",
    )


@pytest.mark.parametrize("text", ["", " \n\t"])
def test_normalize_transcript_rejects_empty_text(text: str) -> None:
    with pytest.raises(InputValidationError, match="transcript is empty"):
        normalize_transcript(text)


def test_normalize_transcript_rejects_punctuation_only_token_with_index() -> None:
    with pytest.raises(InputValidationError, match="became empty") as caught:
        normalize_transcript("hello !!! after")
    assert caught.value.context == {"token_index": 1, "raw_token": "!!!"}


def test_normalization_rejects_non_string_values() -> None:
    with pytest.raises(InputValidationError, match="Transcript must be a string"):
        normalize_transcript(3)  # type: ignore[arg-type]
    with pytest.raises(InputValidationError, match="token must be a string"):
        normalize_word(None)  # type: ignore[arg-type]


def test_read_utf8_text_preserves_contents(tmp_path: Path) -> None:
    path = tmp_path / "words.txt"
    raw = " Café\nsecond "
    path.write_text(raw, encoding="utf-8")

    assert read_utf8_text(path) == raw


def test_read_utf8_text_rejects_invalid_utf8_and_preserves_cause(tmp_path: Path) -> None:
    path = tmp_path / "invalid.txt"
    path.write_bytes(b"hello\xff")

    with pytest.raises(InputValidationError, match="strict UTF-8") as caught:
        read_utf8_text(path)
    assert isinstance(caught.value.__cause__, UnicodeDecodeError)


def test_read_utf8_text_rejects_missing_directory_and_non_path(tmp_path: Path) -> None:
    with pytest.raises(InputValidationError, match="not a file"):
        read_utf8_text(tmp_path / "missing.txt")
    with pytest.raises(InputValidationError, match="not a file"):
        read_utf8_text(tmp_path)
    with pytest.raises(InputValidationError, match=r"pathlib\.Path"):
        read_utf8_text("words.txt")  # type: ignore[arg-type]


def test_load_lexicon_preserves_entry_and_all_pronunciation_order(tmp_path: Path) -> None:
    path = tmp_path / "lexicon.dict"
    path.write_text(
        "\n# comment\n  # indented comment\nWORD, W ER1 D\nword W AO2 R D\ncafé K AE1 F EY0\n",
        encoding="utf-8",
    )

    lexicon = load_lexicon(path)

    assert list(lexicon.entries) == ["word", "café"]
    assert lexicon.entries["word"] == (("W", "ER1", "D"), ("W", "AO2", "R", "D"))
    assert lexicon.get_prons("word") == lexicon.entries["word"]
    assert lexicon.lex is lexicon.entries
    with pytest.raises(TypeError):
        cast(Any, lexicon.entries)["new"] = (("N",),)


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("word\n", "Invalid lexicon line 1"),
        ("!!! AH\n", "word became empty"),
        ("\n# comment only\n", "No lexicon entries loaded"),
    ],
)
def test_load_lexicon_rejects_invalid_or_empty_content(
    tmp_path: Path, contents: str, message: str
) -> None:
    path = tmp_path / "invalid.dict"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(InputValidationError, match=message):
        load_lexicon(path)


def test_load_lexicon_rejects_bad_path_and_strict_utf8(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.dict"
    invalid.write_bytes(b"word W ER1 D\xff\n")
    with pytest.raises(InputValidationError, match="strict UTF-8") as caught:
        load_lexicon(invalid)
    assert isinstance(caught.value.__cause__, UnicodeDecodeError)

    with pytest.raises(InputValidationError, match="not a file"):
        load_lexicon(tmp_path / "missing.dict")
    with pytest.raises(InputValidationError, match=r"pathlib\.Path"):
        load_lexicon("lexicon.dict")  # type: ignore[arg-type]


def test_pronouncing_lexicon_copies_input_and_get_prons_is_typed() -> None:
    source = {"known": (("N", "OW1", "N"),)}
    lexicon = PronouncingLexicon(entries=source)
    source["changed"] = (("CH",),)

    assert "changed" not in lexicon.entries
    with pytest.raises(InputValidationError, match="not in lexicon"):
        lexicon.get_prons("missing")

    empty = PronouncingLexicon(entries={"known": ()})
    with pytest.raises(InputValidationError, match="no pronunciations"):
        empty.get_prons("known")


def test_validate_transcript_lexicon_accepts_duplicates_and_rejects_oov() -> None:
    lexicon = PronouncingLexicon(entries={"known": (("N", "OW1", "N"),)})
    validate_transcript_lexicon(("known", "known"), lexicon)

    with pytest.raises(InputValidationError, match="OOV") as caught:
        validate_transcript_lexicon(("known", "missing"), lexicon)
    assert caught.value.context == {
        "word_index": 1,
        "word": "missing",
        "reason": "oov",
    }


def test_validate_transcript_lexicon_rejects_invalid_shapes() -> None:
    lexicon = PronouncingLexicon(entries={"known": (("N",),)})
    with pytest.raises(InputValidationError, match="sequence"):
        validate_transcript_lexicon("known", lexicon)
    with pytest.raises(InputValidationError, match="sequence is empty"):
        validate_transcript_lexicon((), lexicon)
    with pytest.raises(InputValidationError, match="non-empty string"):
        validate_transcript_lexicon(("",), lexicon)
    with pytest.raises(InputValidationError, match="PronouncingLexicon"):
        validate_transcript_lexicon(("known",), cast(Any, {"known": (("N",),)}))
    with pytest.raises(InputValidationError, match="no pronunciations"):
        validate_transcript_lexicon(("known",), PronouncingLexicon(entries={"known": ()}))


def test_load_dense_token_vocab_returns_immutable_ordered_mapping(tmp_path: Path) -> None:
    path = tmp_path / "vocab.json"
    path.write_text('{"<pad>": 0, " ": 1, "W": 2}', encoding="utf-8")

    vocabulary = load_dense_token_vocab(path)

    assert isinstance(vocabulary, TokenVocabulary)
    assert list(vocabulary.token_to_id.items()) == [("<pad>", 0), (" ", 1), ("W", 2)]
    with pytest.raises(TypeError):
        cast(Any, vocabulary.token_to_id)["N"] = 3


@pytest.mark.parametrize("value", [[], None, "vocab", 7, True, {}])
def test_load_dense_token_vocab_requires_nonempty_json_object(
    tmp_path: Path, value: object
) -> None:
    path = tmp_path / "vocab.json"
    _write_json(path, value)
    with pytest.raises(InputValidationError, match=r"JSON object|must not be empty"):
        load_dense_token_vocab(path)


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ('{"": 0}', "empty or non-string token"),
        ('{"A": true}', "non-boolean integer"),
        ('{"A": 0.0}', "non-boolean integer"),
        ('{"A": "0"}', "non-boolean integer"),
        ('{"A": null}', "non-boolean integer"),
        ('{"A": -1}', "non-negative"),
        ('{"A": 0, "B": 0}', "Duplicate token IDs"),
        ('{"A": 0, "B": 2}', "dense"),
    ],
)
def test_load_dense_token_vocab_rejects_invalid_tokens_and_ids(
    tmp_path: Path, contents: str, message: str
) -> None:
    path = tmp_path / "vocab.json"
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(InputValidationError, match=message):
        load_dense_token_vocab(path)


def test_load_dense_token_vocab_preserves_json_and_utf8_causes(tmp_path: Path) -> None:
    invalid_json = tmp_path / "invalid.json"
    invalid_json.write_text("{", encoding="utf-8")
    with pytest.raises(InputValidationError, match="Invalid token vocabulary JSON") as caught:
        load_dense_token_vocab(invalid_json)
    assert isinstance(caught.value.__cause__, json.JSONDecodeError)

    invalid_utf8 = tmp_path / "invalid-utf8.json"
    invalid_utf8.write_bytes(b'{"A": 0}\xff')
    with pytest.raises(InputValidationError, match="strict UTF-8") as caught:
        load_dense_token_vocab(invalid_utf8)
    assert isinstance(caught.value.__cause__, UnicodeDecodeError)


def test_load_dense_token_vocab_rejects_duplicate_json_token_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"A": 1, "A": 0}', encoding="utf-8")

    with pytest.raises(InputValidationError, match="Duplicate token key") as caught:
        load_dense_token_vocab(path)

    assert caught.value.context == {
        "path": str(path),
        "token": "A",
        "reason": "duplicate_token",
    }
    assert isinstance(caught.value.__cause__, ValueError)


def test_load_dense_token_vocab_rejects_bad_path(tmp_path: Path) -> None:
    with pytest.raises(InputValidationError, match="not a file"):
        load_dense_token_vocab(tmp_path / "missing.json")
    with pytest.raises(InputValidationError, match=r"pathlib\.Path"):
        load_dense_token_vocab("vocab.json")  # type: ignore[arg-type]


def test_validate_aligner_vocabulary_checks_every_pronunciation() -> None:
    lexicon = PronouncingLexicon(entries={"word": (("W", "ER1", "D"), ("W", "AO2", "R", "D"))})
    valid = TokenVocabulary(token_to_id={"W": 0, "ER1": 1, "D": 2, "AO2": 3, "R": 4})
    validate_aligner_vocabulary(("word",), lexicon, valid, 5)

    missing = TokenVocabulary(token_to_id={"W": 0, "ER1": 1, "D": 2, "R": 3})
    with pytest.raises(InputValidationError, match="not in model vocabulary") as caught:
        validate_aligner_vocabulary(("word",), lexicon, missing, 4)
    assert caught.value.context["phone"] == "AO2"
    assert caught.value.context["pronunciation_index"] == 1


def test_validate_aligner_vocabulary_checks_phone_id_range() -> None:
    lexicon = PronouncingLexicon(entries={"word": (("W",), ("OUT",))})
    vocabulary = TokenVocabulary(token_to_id={"W": 0, "X": 1, "OUT": 2})

    with pytest.raises(InputValidationError, match="outside model output") as caught:
        validate_aligner_vocabulary(("word",), lexicon, vocabulary, 2)
    assert caught.value.context["phone_id"] == 2
    assert caught.value.context["model_vocab_size"] == 2
    assert caught.value.context["pronunciation_index"] == 1


def test_validate_aligner_vocabulary_rejects_malformed_inputs() -> None:
    valid_lexicon = PronouncingLexicon(entries={"word": (("W",),)})
    valid_vocab = TokenVocabulary(token_to_id={"W": 0})
    for invalid_size in (True, 1.5, 0, -1):
        with pytest.raises(InputValidationError, match="model_vocab_size"):
            validate_aligner_vocabulary(
                ("word",), valid_lexicon, valid_vocab, cast(Any, invalid_size)
            )
    with pytest.raises(InputValidationError, match="TokenVocabulary"):
        validate_aligner_vocabulary(("word",), valid_lexicon, cast(Any, {"W": 0}), 1)

    empty_pron = PronouncingLexicon(entries={"word": ((),)})
    with pytest.raises(InputValidationError, match="Empty aligner pronunciation"):
        validate_aligner_vocabulary(("word",), empty_pron, valid_vocab, 1)

    invalid_phone = PronouncingLexicon(entries={"word": ((cast(Any, 7),),)})
    with pytest.raises(InputValidationError, match="empty or non-string phone"):
        validate_aligner_vocabulary(("word",), invalid_phone, valid_vocab, 1)


def test_validate_local_model_dir_is_local_directory_only(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    assert validate_local_model_dir(model_dir, "aligner") == model_dir

    file_path = tmp_path / "model.bin"
    file_path.write_bytes(b"weights")
    with pytest.raises(InputValidationError, match="not a directory"):
        validate_local_model_dir(file_path, "chunker")
    with pytest.raises(InputValidationError, match="not a directory"):
        validate_local_model_dir(tmp_path / "missing", "aligner")
    with pytest.raises(InputValidationError, match="non-empty string"):
        validate_local_model_dir(model_dir, " ")
    with pytest.raises(InputValidationError, match=r"pathlib\.Path"):
        validate_local_model_dir("model", "aligner")  # type: ignore[arg-type]
