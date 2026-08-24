"""Validated Hugging Face model-cache resolution for the command-line interface."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from .contracts import Language, LocalModelBundle
from .errors import ModelCacheMissError, ModelDownloadError, ModelValidationError

DEFAULT_MODEL_REPO = "USTCPhonetics/FlexAligner"
DEFAULT_MODEL_RELEASE = "v0.2.0a1"
DEFAULT_MODEL_REVISION = "f9ca09d445e5e8981e43eca6a2f5421526ddc59e"
DEFAULT_MODEL_LANGUAGE = "en"
MIRROR_ENDPOINT = "https://hf-mirror.com"
OFFICIAL_ENDPOINT = "https://huggingface.co"
MODEL_MANIFEST = "model_manifest.json"
MODEL_MANIFEST_SHA256 = "d26a82707d7ed5cd4b843046c4f9fec25b08bb6b9bfc25b67a047a835273ab7a"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MODEL_FILENAMES = (
    "config.json",
    "model.safetensors",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "vocab.json",
)
_ENGLISH_MODEL_FILES = tuple(
    f"{DEFAULT_MODEL_LANGUAGE}/{role}/{filename}"
    for role in ("chunker", "aligner")
    for filename in _MODEL_FILENAMES
)
_ALLOW_PATTERNS = (MODEL_MANIFEST, *_ENGLISH_MODEL_FILES)


def _language_code(language: Language | str) -> str:
    code = language.value if isinstance(language, Language) else str(language)
    if code not in {Language.EN.value, Language.ZH.value}:
        raise ModelDownloadError(
            "Unsupported model language",
            context={"language": code},
        )
    return code


def _model_files(language: Language | str) -> tuple[str, ...]:
    code = _language_code(language)
    return tuple(
        f"{code}/{role}/{filename}"
        for role in ("chunker", "aligner")
        for filename in _MODEL_FILENAMES
    )


def _allow_patterns(language: Language | str) -> tuple[str, ...]:
    return (MODEL_MANIFEST, *_model_files(language))


def default_model_cache_dir(environ: Mapping[str, str] | None = None) -> Path:
    """Return the Hugging Face Hub cache path without importing Hub code."""

    selected = os.environ if environ is None else environ
    explicit_hub = selected.get("HF_HUB_CACHE")
    if explicit_hub:
        return Path(explicit_hub).expanduser()
    explicit_home = selected.get("HF_HOME")
    if explicit_home:
        return Path(explicit_home).expanduser() / "hub"
    xdg_cache = selected.get("XDG_CACHE_HOME")
    cache_home = Path(xdg_cache).expanduser() if xdg_cache else Path.home() / ".cache"
    return cache_home / "huggingface" / "hub"


def endpoint_for_source(source: str) -> str:
    """Map the stable CLI source name to one HTTPS endpoint."""

    if source == "mirror":
        return MIRROR_ENDPOINT
    if source == "official":
        return OFFICIAL_ENDPOINT
    raise ModelDownloadError(
        "Unknown model download source",
        context={"source": source},
    )


def find_cached_english_models(*, cache_dir: Path | None = None) -> LocalModelBundle | None:
    """Resolve and validate the pinned English bundle without network access."""

    return find_cached_models(language=Language.EN, cache_dir=cache_dir)


def find_cached_models(
    *, language: Language | str, cache_dir: Path | None = None
) -> LocalModelBundle | None:
    """Resolve and validate one pinned language bundle without network access."""

    code = _language_code(language)
    selected_cache = default_model_cache_dir() if cache_dir is None else cache_dir.expanduser()
    try:
        snapshot = _snapshot_download(
            repo_id=DEFAULT_MODEL_REPO,
            revision=DEFAULT_MODEL_REVISION,
            cache_dir=str(selected_cache),
            local_files_only=True,
            allow_patterns=list(_allow_patterns(code)),
            token=False,
        )
    except Exception as error:
        if _is_local_cache_miss(error):
            return None
        raise ModelDownloadError(
            "Unable to inspect the local model cache",
            context={
                "cache_dir": str(selected_cache),
                "exception_type": type(error).__name__,
            },
        ) from error
    return validate_model_snapshot(Path(snapshot), language=code)


def download_english_models(
    *,
    cache_dir: Path | None = None,
    source: str = "mirror",
    force_download: bool = False,
) -> LocalModelBundle:
    """Download the pinned public English bundle and validate every manifest entry.

    ``force_download`` is reserved for an explicitly authorized repair of a cache that
    already failed integrity validation. A normal cache miss keeps Hub's incremental
    download behavior.
    """

    return download_models(
        language=Language.EN,
        cache_dir=cache_dir,
        source=source,
        force_download=force_download,
    )


def download_models(
    *,
    language: Language | str,
    cache_dir: Path | None = None,
    source: str = "mirror",
    force_download: bool = False,
) -> LocalModelBundle:
    """Download and validate one pinned language bundle."""

    code = _language_code(language)
    selected_cache = default_model_cache_dir() if cache_dir is None else cache_dir.expanduser()
    endpoint = endpoint_for_source(source)
    try:
        snapshot = _snapshot_download(
            repo_id=DEFAULT_MODEL_REPO,
            revision=DEFAULT_MODEL_REVISION,
            cache_dir=str(selected_cache),
            local_files_only=False,
            allow_patterns=list(_allow_patterns(code)),
            token=False,
            endpoint=endpoint,
            force_download=force_download,
        )
    except Exception as error:
        raise ModelDownloadError(
            "Model download failed; no alternate endpoint was tried",
            context={
                "cache_dir": str(selected_cache),
                "endpoint": endpoint,
                "exception_type": type(error).__name__,
                "repo_id": DEFAULT_MODEL_REPO,
                "revision": DEFAULT_MODEL_REVISION,
                "language": code,
            },
        ) from error
    return validate_model_snapshot(Path(snapshot), language=code)


def require_cached_english_models(*, cache_dir: Path | None = None) -> LocalModelBundle:
    """Return the pinned cached bundle or a typed, actionable cache-miss error."""

    return require_cached_models(language=Language.EN, cache_dir=cache_dir)


def require_cached_models(
    *, language: Language | str, cache_dir: Path | None = None
) -> LocalModelBundle:
    """Return one pinned cached language bundle or a typed cache-miss error."""

    code = _language_code(language)
    selected_cache = default_model_cache_dir() if cache_dir is None else cache_dir.expanduser()
    cached = find_cached_models(language=code, cache_dir=selected_cache)
    if cached is None:
        raise ModelCacheMissError(
            f"The pinned {code} models are not present in the selected cache",
            context={
                "cache_dir": str(selected_cache),
                "repo_id": DEFAULT_MODEL_REPO,
                "revision": DEFAULT_MODEL_REVISION,
                "language": code,
                "suggested_command": f"flexaligner models fetch --language {code}",
            },
        )
    return cached


def validate_english_snapshot(snapshot: Path) -> LocalModelBundle:
    """Validate the English subset of a Hub snapshot against its signed-off manifest."""

    return validate_model_snapshot(snapshot, language=Language.EN)


def validate_model_snapshot(snapshot: Path, *, language: Language | str) -> LocalModelBundle:
    """Validate one language subset of a Hub snapshot against the built-in manifest."""

    code = _language_code(language)
    if not snapshot.is_dir():
        raise ModelValidationError(
            "Model snapshot is not a directory",
            context={"path": str(snapshot)},
        )
    manifest_path = snapshot / MODEL_MANIFEST
    actual_manifest_digest = _sha256(manifest_path) if manifest_path.is_file() else None
    if actual_manifest_digest != MODEL_MANIFEST_SHA256:
        raise ModelValidationError(
            "Model manifest does not match the built-in trust root",
            context={
                "actual": actual_manifest_digest,
                "expected": MODEL_MANIFEST_SHA256,
                "path": str(manifest_path),
            },
        )
    manifest = _read_manifest(manifest_path)
    if manifest.get("schema_version") != 1:
        raise ModelValidationError(
            "Unsupported model manifest schema",
            context={"path": str(manifest_path), "schema_version": manifest.get("schema_version")},
        )
    if manifest.get("bundle_version") != "0.2.0a1":
        raise ModelValidationError(
            "Unexpected model bundle version",
            context={"path": str(manifest_path), "bundle_version": manifest.get("bundle_version")},
        )

    languages = manifest.get("languages")
    selected_language = languages.get(code) if isinstance(languages, dict) else None
    if not isinstance(selected_language, dict):
        raise ModelValidationError(
            "Model manifest has no requested language bundle",
            context={"language": code},
        )
    if (
        selected_language.get("chunker_path") != f"{code}/chunker"
        or selected_language.get("aligner_path") != f"{code}/aligner"
    ):
        raise ModelValidationError(
            "Model manifest has unexpected language model paths",
            context={"language": code},
        )

    files = manifest.get("files")
    if not isinstance(files, list):
        raise ModelValidationError("Model manifest files must be a list")
    expected: dict[str, tuple[int, str]] = {}
    for index, raw_entry in enumerate(files):
        if not isinstance(raw_entry, dict):
            raise ModelValidationError(
                "Model manifest file entry must be an object",
                context={"index": index},
            )
        raw_path = raw_entry.get("path")
        if not isinstance(raw_path, str) or not raw_path.startswith(f"{code}/"):
            continue
        manifest_relative = PurePosixPath(raw_path)
        if (
            manifest_relative.is_absolute()
            or ".." in manifest_relative.parts
            or "." in manifest_relative.parts
        ):
            raise ModelValidationError(
                "Model manifest contains an unsafe path",
                context={"index": index, "path": raw_path},
            )
        size = raw_entry.get("size")
        digest = raw_entry.get("sha256")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ModelValidationError(
                "Model manifest contains an invalid size",
                context={"index": index, "path": raw_path},
            )
        if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            raise ModelValidationError(
                "Model manifest contains an invalid SHA-256",
                context={"index": index, "path": raw_path},
            )
        if raw_path in expected:
            raise ModelValidationError(
                "Model manifest contains a duplicate path",
                context={"path": raw_path},
            )
        expected[raw_path] = (size, digest)

    expected_files = set(_model_files(code))
    if set(expected) != expected_files:
        raise ModelValidationError(
            "Language model manifest does not contain the exact built-in file set",
            context={"actual_count": len(expected), "language": code},
        )
    actual = {
        path.relative_to(snapshot).as_posix()
        for path in (snapshot / code).rglob("*")
        if path.is_file()
    }
    if actual != set(expected):
        raise ModelValidationError(
            "Language model snapshot file set does not match the manifest",
            context={
                "actual_count": len(actual),
                "expected_count": len(expected),
                "language": code,
            },
        )
    for relative, (expected_size, expected_digest) in sorted(expected.items()):
        path = snapshot / relative
        stat = path.stat()
        if stat.st_size != expected_size:
            raise ModelValidationError(
                "Model file size does not match the manifest",
                context={"path": relative, "actual": stat.st_size, "expected": expected_size},
            )
        actual_digest = _sha256(path)
        if actual_digest != expected_digest:
            raise ModelValidationError(
                "Model file SHA-256 does not match the manifest",
                context={"path": relative, "actual": actual_digest, "expected": expected_digest},
            )
    return LocalModelBundle(
        chunker_dir=snapshot / code / "chunker",
        aligner_dir=snapshot / code / "aligner",
        manifest_path=manifest_path,
    )


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        payload: object = json.loads(path.read_text(encoding="utf-8", errors="strict"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ModelValidationError(
            "Unable to read model manifest",
            context={"path": str(path), "exception_type": type(error).__name__},
        ) from error
    if not isinstance(payload, dict):
        raise ModelValidationError("Model manifest root must be an object")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _snapshot_download(**kwargs: object) -> str:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise ModelDownloadError(
            "huggingface-hub is required for model cache resolution",
            context={"dependency": "huggingface-hub==0.36.2"},
        ) from error
    return snapshot_download(**kwargs)  # type: ignore[arg-type]


def _is_local_cache_miss(error: Exception) -> bool:
    try:
        from huggingface_hub.errors import LocalEntryNotFoundError
    except ImportError:
        return False
    return isinstance(error, LocalEntryNotFoundError)


__all__ = [
    "DEFAULT_MODEL_RELEASE",
    "DEFAULT_MODEL_REPO",
    "DEFAULT_MODEL_REVISION",
    "MIRROR_ENDPOINT",
    "OFFICIAL_ENDPOINT",
    "default_model_cache_dir",
    "download_english_models",
    "download_models",
    "endpoint_for_source",
    "find_cached_english_models",
    "find_cached_models",
    "require_cached_english_models",
    "require_cached_models",
    "validate_english_snapshot",
    "validate_model_snapshot",
]
