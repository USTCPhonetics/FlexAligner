"""Shared classification for per-file batch failures."""

from __future__ import annotations

from .errors import FlexAlignerError


def is_fatal_batch_error(error: FlexAlignerError) -> bool:
    """Shared model/configuration/runtime failures stop a batch."""

    return error.code in {
        "configuration_error",
        "optional_dependency_missing",
        "model_validation_error",
        "model_incompatible",
        "model_cache_miss",
        "model_download_error",
        "runtime_recovery_failed",
        "engine_closed",
        "internal_error",
    }
