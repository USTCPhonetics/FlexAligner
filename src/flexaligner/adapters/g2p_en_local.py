"""Fully local, word-only English G2P using the pinned g2p-en neural checkpoint.

The GRU inference equations are adapted from g2p-en 2.1.0 (Apache-2.0). Unlike the upstream
runtime, this adapter never imports NLTK and never downloads dictionaries or taggers.
"""

from __future__ import annotations

import hashlib
import io
import re
from importlib.resources import files
from typing import cast

import numpy as np
from numpy.typing import NDArray

from ..errors import PronunciationGenerationError

FloatArray = NDArray[np.float32]

ENGINE_ID = "g2p-en-local-neural"
ENGINE_VERSION = "2.1.0-checkpoint20"
CHECKPOINT_SHA256 = "b8af35e4596d8dd5836dfd3fe9b2ba4f97b9c311efe8879544cbcfcbd566d8c6"
_SUPPORTED_WORD = re.compile(r"^[a-z]+(?:['-][a-z]+)*$")
_GRAPHEMES = ("<pad>", "<unk>", "</s>", *tuple("abcdefghijklmnopqrstuvwxyz"))
_PHONEMES = (
    "<pad>",
    "<unk>",
    "<s>",
    "</s>",
    "AA0",
    "AA1",
    "AA2",
    "AE0",
    "AE1",
    "AE2",
    "AH0",
    "AH1",
    "AH2",
    "AO0",
    "AO1",
    "AO2",
    "AW0",
    "AW1",
    "AW2",
    "AY0",
    "AY1",
    "AY2",
    "B",
    "CH",
    "D",
    "DH",
    "EH0",
    "EH1",
    "EH2",
    "ER0",
    "ER1",
    "ER2",
    "EY0",
    "EY1",
    "EY2",
    "F",
    "G",
    "HH",
    "IH0",
    "IH1",
    "IH2",
    "IY0",
    "IY1",
    "IY2",
    "JH",
    "K",
    "L",
    "M",
    "N",
    "NG",
    "OW0",
    "OW1",
    "OW2",
    "OY0",
    "OY1",
    "OY2",
    "P",
    "R",
    "S",
    "SH",
    "T",
    "TH",
    "UH0",
    "UH1",
    "UH2",
    "UW",
    "UW0",
    "UW1",
    "UW2",
    "V",
    "W",
    "Y",
    "Z",
    "ZH",
)


class LocalEnglishG2P:
    """Deterministic local neural pronunciation generator for one normalized word."""

    engine_id = ENGINE_ID
    engine_version = ENGINE_VERSION

    def __init__(self) -> None:
        checkpoint = files("flexaligner").joinpath("_vendor/g2p_en/checkpoint20.npz")
        payload = checkpoint.read_bytes()
        actual_digest = hashlib.sha256(payload).hexdigest()
        if actual_digest != CHECKPOINT_SHA256:
            raise PronunciationGenerationError(
                "Bundled English G2P checkpoint failed integrity validation",
                context={"actual": actual_digest, "expected": CHECKPOINT_SHA256},
            )
        try:
            with np.load(io.BytesIO(payload), allow_pickle=False) as variables:
                self._arrays = {
                    name: variables[name].astype(np.float32, copy=True) for name in variables
                }
        except (OSError, ValueError, KeyError) as error:
            raise PronunciationGenerationError(
                "Unable to load the bundled English G2P checkpoint",
                context={"exception_type": type(error).__name__},
            ) from error
        required = {
            "enc_emb",
            "enc_w_ih",
            "enc_w_hh",
            "enc_b_ih",
            "enc_b_hh",
            "dec_emb",
            "dec_w_ih",
            "dec_w_hh",
            "dec_b_ih",
            "dec_b_hh",
            "fc_w",
            "fc_b",
        }
        if set(self._arrays) != required:
            raise PronunciationGenerationError("English G2P checkpoint has unexpected variables")
        self._grapheme_to_id = {value: index for index, value in enumerate(_GRAPHEMES)}

    def pronounce(self, word: str) -> tuple[str, ...]:
        if not isinstance(word, str) or _SUPPORTED_WORD.fullmatch(word) is None:
            raise PronunciationGenerationError(
                "Local English G2P supports normalized ASCII words with internal apostrophes or hyphens",
                context={"word": str(word)},
            )
        if len(word) > 64:
            raise PronunciationGenerationError(
                "Local English G2P word exceeds the 64-character limit",
                context={"length": len(word), "word": word},
            )
        encoded_ids = [
            self._grapheme_to_id.get(character, self._grapheme_to_id["<unk>"])
            for character in (*word, "</s>")
        ]
        encoded = np.take(
            self._arrays["enc_emb"],
            np.expand_dims(np.asarray(encoded_ids, dtype=np.int64), 0),
            axis=0,
        )
        hidden = np.zeros((1, self._arrays["enc_w_hh"].shape[1]), dtype=np.float32)
        for step in range(encoded.shape[1]):
            hidden = self._gru_cell(
                encoded[:, step, :],
                hidden,
                prefix="enc",
            )

        decoder_input = np.take(self._arrays["dec_emb"], [2], axis=0)
        predicted: list[str] = []
        reached_end = False
        for _ in range(20):
            hidden = self._gru_cell(decoder_input, hidden, prefix="dec")
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                logits = np.matmul(hidden, self._arrays["fc_w"].T) + self._arrays["fc_b"]
            if not np.isfinite(logits).all():
                raise PronunciationGenerationError(
                    "Local English G2P produced non-finite decoder logits",
                    context={"word": word},
                )
            predicted_id = int(logits.argmax())
            if predicted_id == 3:
                reached_end = True
                break
            phone = _PHONEMES[predicted_id] if predicted_id < len(_PHONEMES) else "<unk>"
            if phone.startswith("<"):
                raise PronunciationGenerationError(
                    "Local English G2P generated an invalid control phone",
                    context={"phone": phone, "word": word},
                )
            predicted.append(phone)
            decoder_input = np.take(self._arrays["dec_emb"], [predicted_id], axis=0)
        if not reached_end:
            raise PronunciationGenerationError(
                "Local English G2P did not reach end-of-sequence within 20 phones",
                context={"word": word},
            )
        if not predicted:
            raise PronunciationGenerationError(
                "Local English G2P generated an empty pronunciation",
                context={"word": word},
            )
        return tuple(predicted)

    def _gru_cell(self, value: FloatArray, hidden: FloatArray, *, prefix: str) -> FloatArray:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            input_projection = (
                np.matmul(value, self._arrays[f"{prefix}_w_ih"].T) + self._arrays[f"{prefix}_b_ih"]
            )
            hidden_projection = (
                np.matmul(hidden, self._arrays[f"{prefix}_w_hh"].T) + self._arrays[f"{prefix}_b_hh"]
            )
            split = input_projection.shape[-1] * 2 // 3
            gates = _sigmoid(input_projection[:, :split] + hidden_projection[:, :split])
            reset, update = np.split(gates, 2, axis=-1)
            candidate = np.tanh(input_projection[:, split:] + reset * hidden_projection[:, split:])
            result = (np.float32(1.0) - update) * candidate + update * hidden
        if not np.isfinite(result).all():
            raise PronunciationGenerationError(
                "Local English G2P produced a non-finite recurrent state",
                context={"stage": prefix},
            )
        return cast(FloatArray, np.asarray(result, dtype=np.float32))


def _sigmoid(value: FloatArray) -> FloatArray:
    result = np.float32(1.0) / (np.float32(1.0) + np.exp(-value))
    return cast(FloatArray, np.asarray(result, dtype=np.float32))


__all__ = ["CHECKPOINT_SHA256", "ENGINE_ID", "ENGINE_VERSION", "LocalEnglishG2P"]
