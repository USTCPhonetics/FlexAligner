"""Command-line interface for capability discovery and guarded placeholders."""

from __future__ import annotations

import argparse
import json
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
    Device,
    Language,
    LocalModelBundle,
    TextGridOutput,
)
from .errors import FlexAlignerError

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
        "align", help="Align one English PCM16 WAV once the core is available."
    )
    align_parser.add_argument("--audio", type=Path, required=True)
    transcript_group = align_parser.add_mutually_exclusive_group(required=True)
    transcript_group.add_argument("--text")
    transcript_group.add_argument("--text-file", type=Path)
    align_parser.add_argument("--lexicon", type=Path, required=True)
    align_parser.add_argument("--chunker-model", type=Path, required=True)
    align_parser.add_argument("--aligner-model", type=Path, required=True)
    align_parser.add_argument("--output", type=Path, required=True)
    align_parser.add_argument("--chunk-metadata", type=Path)
    align_parser.add_argument("--utterance-id")
    align_parser.add_argument("--num-threads", type=int, default=1)
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

    models_parser = subparsers.add_parser("models", help="Model-management placeholders.")
    model_subparsers = models_parser.add_subparsers(dest="models_command")
    model_subparsers.add_parser("fetch", help="Declared automatic-download placeholder.")
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
    )
    require_supported_options(options)
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
        models=LocalModelBundle(
            chunker_dir=args.chunker_model,
            aligner_dir=args.aligner_model,
        ),
        lexicon_path=args.lexicon,
        options=options,
    ) as engine:
        result = engine.align(request)
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
    if args.command == "models" and args.models_command == "fetch":
        report.require(CapabilityId.AUTO_MODEL_DOWNLOAD)
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
