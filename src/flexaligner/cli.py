"""Command-line interface for strict alignment and guarded placeholders."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from . import __version__
from .adapters.lexicon_file import read_utf8_text
from .api import FlexAligner, require_supported_options
from .capabilities import CapabilityId, get_capabilities
from .contracts import (
    AlignmentOptions,
    AlignmentRequest,
    AudioPolicy,
    Device,
    Language,
    LocalModelBundle,
    PronunciationMode,
    TextGridOutput,
)
from .errors import (
    ConfigurationError,
    FlexAlignerError,
    ModelCacheMissError,
    ModelValidationError,
)
from .model_download import (
    DEFAULT_MODEL_RELEASE,
    DEFAULT_MODEL_REPO,
    DEFAULT_MODEL_REVISION,
    MIRROR_ENDPOINT,
    OFFICIAL_ENDPOINT,
    default_model_cache_dir,
    download_english_models,
    download_models,
    find_cached_english_models,
    find_cached_models,
)

PLACEHOLDER_EXIT_STATUS = 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="flexaligner",
        description="Strict two-stage forced alignment contracts.",
    )
    parser.add_argument("--version", action="version", version=f"flexaligner {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    capability_parser = subparsers.add_parser(
        "capabilities", help="Show versioned package capabilities."
    )
    capability_parser.add_argument("--json", action="store_true", dest="as_json")

    align_parser = subparsers.add_parser(
        "align", help="Align one English or Mandarin file with local or cached models."
    )
    align_parser.add_argument("--audio", type=Path, required=True)
    transcript_group = align_parser.add_mutually_exclusive_group(required=True)
    transcript_group.add_argument("--text")
    transcript_group.add_argument("--text-file", type=Path)
    align_parser.add_argument("--lexicon", type=Path, required=True)
    align_parser.add_argument("--chunker-model", type=Path)
    align_parser.add_argument("--aligner-model", type=Path)
    align_parser.add_argument("--model-cache-dir", type=Path)
    align_parser.add_argument(
        "--model-source",
        choices=("mirror", "official"),
        help="Download from hf-mirror.com or huggingface.co after explicit confirmation.",
    )
    align_parser.add_argument(
        "--yes",
        action="store_true",
        help="Authorize a required model download without an interactive prompt.",
    )
    align_parser.add_argument("--output", type=Path, required=True)
    align_parser.add_argument("--chunk-metadata", type=Path)
    align_parser.add_argument("--utterance-id")
    align_parser.add_argument("--num-threads", type=int, default=1)
    align_parser.add_argument(
        "--audio-policy",
        choices=[item.value for item in AudioPolicy],
        default=AudioPolicy.STRICT_PCM16_WAV.value,
        help="Keep strict WAV input (default) or explicitly enable the optional audio extra.",
    )
    align_parser.add_argument(
        "--pronunciation-mode",
        choices=[item.value for item in PronunciationMode],
        default=PronunciationMode.G2P.value,
        help="Use the selected language's local G2P for OOVs or require strict lexicon coverage.",
    )
    align_parser.add_argument(
        "--language",
        choices=[item.value for item in Language],
        default="en",
    )
    align_parser.add_argument(
        "--device",
        choices=[item.value for item in Device],
        default="cpu",
    )

    subparsers.add_parser("batch", help="Declared batch placeholder.")
    subparsers.add_parser("serve", help="Declared Web-service placeholder.")

    models_parser = subparsers.add_parser("models", help="Model cache management.")
    model_subparsers = models_parser.add_subparsers(dest="models_command")
    fetch_parser = model_subparsers.add_parser(
        "fetch", help="Fetch and validate one pinned public language model bundle."
    )
    fetch_parser.add_argument(
        "--language",
        choices=[item.value for item in Language],
        default=Language.EN.value,
    )
    fetch_parser.add_argument("--model-cache-dir", type=Path)
    fetch_parser.add_argument(
        "--model-source",
        choices=("mirror", "official"),
        help="Download source; default is the China-accessible mirror.",
    )
    fetch_parser.add_argument(
        "--yes",
        action="store_true",
        help="Authorize download without an interactive prompt.",
    )

    audio_parser = subparsers.add_parser("audio", help="Optional explicit audio conversion.")
    audio_subparsers = audio_parser.add_subparsers(dest="audio_command")
    convert_parser = audio_subparsers.add_parser(
        "convert", help="Convert one audio file to 16 kHz mono PCM16 WAV."
    )
    convert_parser.add_argument("input", type=Path)
    convert_parser.add_argument("output", type=Path)
    convert_parser.add_argument("--sample-rate", type=int, default=16_000)
    return parser


def _print_capabilities(*, as_json: bool) -> None:
    report = get_capabilities()
    if as_json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, sort_keys=True))
        return
    print(f"FlexAligner capabilities schema {report.schema_version}")
    for capability in report.capabilities:
        reason = f" - {capability.reason}" if capability.reason else ""
        print(f"{capability.id.value}\t{capability.status.value}\t{capability.summary}{reason}")


def _run_align(args: argparse.Namespace) -> None:
    options = AlignmentOptions(
        language=Language(args.language),
        device=Device(args.device),
        num_threads=args.num_threads,
        audio_policy=AudioPolicy(args.audio_policy),
        pronunciation_mode=PronunciationMode(args.pronunciation_mode),
    )
    require_supported_options(options)
    models = _resolve_cli_models(args)
    transcript = args.text if args.text is not None else read_utf8_text(args.text_file)
    request = AlignmentRequest(
        audio_path=args.audio,
        transcript=transcript,
        output=TextGridOutput(
            path=args.output,
            chunk_metadata_path=args.chunk_metadata,
        ),
        utterance_id=args.utterance_id,
    )
    with FlexAligner(
        models=models,
        lexicon_path=args.lexicon,
        options=options,
    ) as engine:
        result = engine.align(request)
    for notice in result.pronunciation_notices:
        print(
            "WARNING "
            + json.dumps(
                notice.to_dict(),
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
    print(
        json.dumps(
            {
                "schema_version": result.schema_version,
                "utterance_id": result.utterance_id,
                "output_path": str(result.output_path),
                "output_sha256": result.output_sha256,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def _resolve_cli_models(args: argparse.Namespace) -> LocalModelBundle:
    chunker = args.chunker_model
    aligner = args.aligner_model
    if (chunker is None) != (aligner is None):
        raise ConfigurationError(
            "--chunker-model and --aligner-model must be provided together",
            context={
                "aligner_model_provided": aligner is not None,
                "chunker_model_provided": chunker is not None,
            },
        )
    if chunker is not None and aligner is not None:
        return LocalModelBundle(chunker_dir=chunker, aligner_dir=aligner)
    return _resolve_or_download_models(
        cache_dir=args.model_cache_dir,
        source=args.model_source,
        assume_yes=args.yes,
        language=Language(args.language),
    )


def _run_models_fetch(args: argparse.Namespace) -> None:
    language = Language(args.language)
    models = _resolve_or_download_models(
        cache_dir=args.model_cache_dir,
        source=args.model_source,
        assume_yes=args.yes,
        language=language,
    )
    print(
        json.dumps(
            {
                "aligner_model": str(models.aligner_dir),
                "bundle_release": DEFAULT_MODEL_RELEASE,
                "chunker_model": str(models.chunker_dir),
                "manifest": str(models.manifest_path),
                "repo_id": DEFAULT_MODEL_REPO,
                "revision": DEFAULT_MODEL_REVISION,
                "language": language.value,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def _resolve_or_download_models(
    *,
    cache_dir: Path | None,
    source: str | None,
    assume_yes: bool,
    language: Language = Language.EN,
) -> LocalModelBundle:
    selected_cache = default_model_cache_dir() if cache_dir is None else cache_dir.expanduser()
    try:
        cached = _find_cached_models(language=language, cache_dir=selected_cache)
    except ModelValidationError:
        if not assume_yes:
            raise
        selected_source = _default_model_source() if source is None else source
        return _download_models(
            language=language,
            cache_dir=selected_cache,
            source=selected_source,
            force_download=True,
        )
    if cached is not None:
        return cached

    selected_source = _default_model_source() if source is None else source
    if assume_yes:
        return _download_models(
            language=language,
            cache_dir=selected_cache,
            source=selected_source,
        )
    if not sys.stdin.isatty():
        raise _cache_miss(selected_cache, language=language)
    if not _prompt_yes_no(
        f"Default {language.value} models were not found. Download about 2.4 GiB from "
        f"{DEFAULT_MODEL_REPO}@{DEFAULT_MODEL_RELEASE}? [y/N] "
    ):
        raise _cache_miss(selected_cache, language=language)

    if cache_dir is None:
        entered_cache = _prompt_line(f"Cache directory [{selected_cache}]: ")
        if entered_cache:
            selected_cache = Path(entered_cache).expanduser()
            cached = _find_cached_models(language=language, cache_dir=selected_cache)
            if cached is not None:
                return cached
    if source is None:
        entered_source = _prompt_line(
            "Download source [mirror/official] (default: mirror, https://hf-mirror.com): "
        ).lower()
        if entered_source in ("", "mirror", "m"):
            selected_source = "mirror"
        elif entered_source in ("official", "o"):
            selected_source = "official"
        else:
            raise ConfigurationError(
                "Model source must be 'mirror' or 'official'",
                context={"source": entered_source},
            )
    endpoint = MIRROR_ENDPOINT if selected_source == "mirror" else OFFICIAL_ENDPOINT
    print(
        f"Downloading {DEFAULT_MODEL_REPO}@{DEFAULT_MODEL_RELEASE} "
        f"({DEFAULT_MODEL_REVISION}) from {endpoint} into {selected_cache}",
        file=sys.stderr,
    )
    return _download_models(
        language=language,
        cache_dir=selected_cache,
        source=selected_source,
    )


def _find_cached_models(*, language: Language, cache_dir: Path) -> LocalModelBundle | None:
    if language is Language.EN:
        return find_cached_english_models(cache_dir=cache_dir)
    return find_cached_models(language=language, cache_dir=cache_dir)


def _download_models(
    *,
    language: Language,
    cache_dir: Path,
    source: str,
    force_download: bool = False,
) -> LocalModelBundle:
    if language is Language.EN:
        if not force_download:
            return download_english_models(cache_dir=cache_dir, source=source)
        return download_english_models(
            cache_dir=cache_dir,
            source=source,
            force_download=force_download,
        )
    return download_models(
        language=language,
        cache_dir=cache_dir,
        source=source,
        force_download=force_download,
    )


def _default_model_source() -> str:
    endpoint = os.environ.get("HF_ENDPOINT", "").rstrip("/")
    if endpoint == OFFICIAL_ENDPOINT:
        return "official"
    return "mirror"


def _prompt_yes_no(prompt: str) -> bool:
    return _prompt_line(prompt).lower() in ("y", "yes")


def _prompt_line(prompt: str) -> str:
    print(prompt, end="", file=sys.stderr, flush=True)
    return sys.stdin.readline().strip()


def _cache_miss(cache_dir: Path, *, language: Language = Language.EN) -> ModelCacheMissError:
    suggested_command = "flexaligner models fetch --yes"
    if language is Language.ZH:
        suggested_command += " --language zh"
    return ModelCacheMissError(
        f"Default {language.value} models are not cached and no download was authorized",
        context={
            "cache_dir": str(cache_dir),
            "repo_id": DEFAULT_MODEL_REPO,
            "revision": DEFAULT_MODEL_REVISION,
            "language": language.value,
            "suggested_command": suggested_command,
        },
    )


def _dispatch(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.command is None:
        parser.print_help()
        return
    if args.command == "capabilities":
        _print_capabilities(as_json=args.as_json)
        return
    if args.command == "align":
        _run_align(args)
        return
    report = get_capabilities()
    if args.command == "batch":
        report.require(CapabilityId.BATCH)
        return
    if args.command == "serve":
        report.require(CapabilityId.WEB)
        return
    if args.command == "audio":
        if args.audio_command != "convert":
            parser.error("audio requires the convert subcommand")
        report.require(CapabilityId.AUDIO_TRANSCODE)
        from .adapters.audio_av import convert_to_pcm16_wav

        converted = convert_to_pcm16_wav(
            args.input,
            args.output,
            sample_rate=args.sample_rate,
        )
        print(
            json.dumps(
                {
                    "audio_duration_s": converted.duration_s,
                    "output_path": str(args.output),
                    "sample_rate": converted.sample_rate,
                    "schema_version": "1",
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return
    if args.command == "models" and args.models_command == "fetch":
        report.require(CapabilityId.AUTO_MODEL_DOWNLOAD)
        _run_models_fetch(args)
        return
    parser.error("a models subcommand is required")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        _dispatch(args, parser)
    except FlexAlignerError as error:
        print(
            json.dumps(error.to_dict(), ensure_ascii=False, sort_keys=True),
            file=sys.stderr,
        )
        return PLACEHOLDER_EXIT_STATUS
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through python -m
    raise SystemExit(main())
