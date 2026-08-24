"""Lazy boundary for the separately packaged offline English G2P backend."""

from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module
from typing import Any

from ..errors import OptionalDependencyError, PronunciationGenerationError

ENGINE_ID = "g2p-en-local-neural"
ENGINE_VERSION = "2.1.0-checkpoint20"
LANGUAGE_PACK_VERSION = "0.3.0a1"
CHECKPOINT_SHA256 = "b8af35e4596d8dd5836dfd3fe9b2ba4f97b9c311efe8879544cbcfcbd566d8c6"


class LocalEnglishG2P:
    """Load English G2P only when the optional ``en`` language pack is installed."""

    engine_id = ENGINE_ID
    engine_version = ENGINE_VERSION

    def __init__(self) -> None:
        try:
            language_pack = import_module("flexaligner_g2p_en")
        except ImportError as error:
            raise _missing_language_pack() from error
        actual_version = getattr(language_pack, "__version__", None)
        if actual_version != LANGUAGE_PACK_VERSION:
            raise _incompatible_language_pack(actual_version)
        try:
            backend_module = import_module("flexaligner_g2p_en.backend")
        except ImportError as error:
            raise _missing_language_pack() from error
        backend_class = getattr(backend_module, "LocalEnglishG2P", None)
        backend_error = getattr(backend_module, "EnglishG2PError", None)
        if not callable(backend_class) or not isinstance(backend_error, type):
            raise _incompatible_language_pack(actual_version)
        self._backend_error: type[Exception] = backend_error
        try:
            self._backend = backend_class()
        except Exception as error:
            self._translate_backend_error(error)

    @property
    def _arrays(self) -> Any:
        """Expose mutable arrays only for deterministic fault-injection tests."""

        return self._backend._arrays

    def pronounce(self, word: str) -> tuple[str, ...]:
        try:
            result = self._backend.pronounce(word)
        except Exception as error:
            self._translate_backend_error(error)
        if not isinstance(result, tuple) or any(not isinstance(phone, str) for phone in result):
            raise PronunciationGenerationError(
                "English G2P language pack returned an invalid pronunciation",
                context={"word": word},
            )
        return result

    def _translate_backend_error(self, error: Exception) -> None:
        if not isinstance(error, self._backend_error):
            raise error
        raw_context = getattr(error, "context", {})
        context = dict(raw_context) if isinstance(raw_context, Mapping) else {}
        raise PronunciationGenerationError(str(error), context=context) from error


def _missing_language_pack() -> OptionalDependencyError:
    return OptionalDependencyError(
        "English G2P requires the optional en language pack",
        context={
            "dependency": f"flexaligner-g2p-en=={LANGUAGE_PACK_VERSION}",
            "extra": "en",
            "suggested_command": "python -m pip install 'flexaligner[en]'",
        },
    )


def _incompatible_language_pack(actual_version: object) -> OptionalDependencyError:
    return OptionalDependencyError(
        "Installed English G2P language pack is incompatible with FlexAligner",
        context={
            "actual_version": str(actual_version),
            "dependency": f"flexaligner-g2p-en=={LANGUAGE_PACK_VERSION}",
            "extra": "en",
            "suggested_command": "python -m pip install --upgrade 'flexaligner[en]'",
        },
    )


__all__ = [
    "CHECKPOINT_SHA256",
    "ENGINE_ID",
    "ENGINE_VERSION",
    "LANGUAGE_PACK_VERSION",
    "LocalEnglishG2P",
]
