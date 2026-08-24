from __future__ import annotations

import pytest

from flexaligner.adapters.lexicon_file import PronouncingLexicon, TokenVocabulary
from flexaligner.contracts import Language
from flexaligner.errors import LanguageMismatchError
from flexaligner.language_validation import (
    infer_model_language,
    validate_lexicon_language,
    validate_model_language,
    validate_transcript_language,
)


def test_transcript_and_lexicon_language_mismatch_are_typed() -> None:
    with pytest.raises(LanguageMismatchError) as caught:
        validate_transcript_language("这是中文", Language.EN)
    assert caught.value.code == "language_mismatch"
    assert caught.value.context["component"] == "transcript"

    english_lexicon = PronouncingLexicon(entries={"hello": (("HH",),)})
    with pytest.raises(LanguageMismatchError) as caught:
        validate_lexicon_language(english_lexicon, Language.ZH)
    assert caught.value.context["component"] == "lexicon"


def test_model_language_detection_distinguishes_current_phone_sets() -> None:
    english = {"<pad>": 0, "AH": 1, "sph": 2}
    mandarin = {"<pad>": 0, "ix": 1, "iy": 2, "iz": 3, "sil": 4}
    assert infer_model_language(english) is Language.EN
    assert infer_model_language(mandarin) is Language.ZH

    with pytest.raises(LanguageMismatchError) as caught:
        validate_model_language(
            (("chunker model", TokenVocabulary(token_to_id=english)),),
            Language.ZH,
        )
    assert caught.value.context == {
        "component": "chunker model",
        "detected": "en",
        "selected": "zh",
    }
