from __future__ import annotations

import sys

import pytest

import flexaligner.adapters.zh_local as zh_local
from flexaligner.adapters.lexicon_file import PronouncingLexicon, TokenVocabulary
from flexaligner.adapters.zh_local import LocalMandarinG2P, segment_mandarin
from flexaligner.contracts import Language, PronunciationMode
from flexaligner.errors import OptionalDependencyError, PronunciationGenerationError
from flexaligner.pronunciation import resolve_effective_lexicon


def test_jieba_segmentation_preserves_whitespace_boundaries_and_discards_punctuation() -> None:
    expected = ("甚至", "出现", "交易", "几乎", "停滞", "的", "情况")
    assert segment_mandarin("甚至出现交易几乎停滞的情况") == expected
    assert segment_mandarin("甚至 出现，交易 几乎停滞 的 情况。") == expected  # noqa: RUF001


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        ("甚至", ("sh", "en", "zh", "ix")),
        ("交易", ("j", "iao", "i")),
        ("知识", ("zh", "ix", "sh", "ix")),
        ("日子", ("r", "iz", "z", "iy")),
        ("月鱼无一", ("ve", "v", "u", "i")),
    ],
)
def test_local_mandarin_g2p_matches_current_model_phone_conventions(
    word: str, expected: tuple[str, ...]
) -> None:
    assert LocalMandarinG2P().pronounce(word) == expected


def test_mandarin_g2p_rejects_non_han_input() -> None:
    with pytest.raises(PronunciationGenerationError, match="Han words only"):
        LocalMandarinG2P().pronounce("english")


def test_mandarin_g2p_fills_oov_only_and_validates_both_vocabularies() -> None:
    lexicon = PronouncingLexicon(entries={"甚至": (("sh", "en", "zh", "ix"),)})
    phones = {"sh", "en", "zh", "ix", "j", "iao", "i"}
    vocabulary = TokenVocabulary(token_to_id={phone: index for index, phone in enumerate(phones)})
    effective, notices = resolve_effective_lexicon(
        words=("甚至", "交易"),
        lexicon=lexicon,
        mode=PronunciationMode.G2P,
        g2p=LocalMandarinG2P(),
        chunker_vocabulary=vocabulary,
        aligner_vocabulary=vocabulary,
        language=Language.ZH,
    )
    assert effective.get_prons("甚至") == (("sh", "en", "zh", "ix"),)
    assert effective.get_prons("交易") == (("j", "iao", "i"),)
    assert len(notices) == 1
    assert notices[0].word == "交易"
    assert notices[0].code == "oov_g2p_fallback"


def test_missing_zh_dependencies_have_actionable_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "pypinyin", None)
    with pytest.raises(OptionalDependencyError) as caught:
        LocalMandarinG2P()
    assert caught.value.context["extra"] == "zh"
    assert "flexaligner[zh]" in str(caught.value.context["suggested_command"])


def test_missing_jieba_has_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(name: str) -> object:
        assert name == "jieba"
        raise ImportError(name)

    monkeypatch.setattr(zh_local, "import_module", missing)
    with pytest.raises(OptionalDependencyError) as caught:
        segment_mandarin("中文")
    assert caught.value.context["extra"] == "zh"
