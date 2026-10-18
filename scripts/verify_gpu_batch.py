#!/usr/bin/env python3
"""Offline CPU/CUDA single/batch acceptance; absent assets or CUDA are failures.

This supplements (never replaces or updates) the frozen CPU release E2E tests.
Only a new output directory is written. No model download is attempted.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import sys
import time
import wave
from collections.abc import Callable, Sequence
from pathlib import Path, PurePosixPath
from typing import Any
from unittest.mock import patch

from flexaligner import (
    AlignmentOptions,
    AlignmentRequest,
    AlignmentResult,
    Device,
    FlexAligner,
    Language,
    LocalModelBundle,
    PronunciationMode,
    TextGridOutput,
)
from flexaligner.model_download import validate_model_snapshot
from flexaligner.textgrid import parse_textgrid_long, validate_textgrid_structure

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_ROOT = REPO_ROOT / "tests" / "fixtures" / "e2e"
MAX_BOUNDARY_DELTA = 0.020


class AcceptanceError(RuntimeError):
    """A failed prerequisite or acceptance gate, never an optional skip."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_assets(manifest: Path, asset_root: Path) -> dict[str, Path]:
    """Validate every frozen entry and return role-based, read-only paths."""
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("status") != "approved":
        raise AcceptanceError(f"GPU_E2E_BLOCKED: unapproved/invalid manifest: {manifest}")
    entries = payload.get("files")
    if not isinstance(entries, list) or not entries:
        raise AcceptanceError(f"GPU_E2E_BLOCKED: manifest has no assets: {manifest}")
    assets: dict[str, Path] = {}
    for entry in entries:
        relative = PurePosixPath(entry["path"])
        role = entry["role"]
        if relative.is_absolute() or ".." in relative.parts or role in assets:
            raise AcceptanceError(f"GPU_E2E_BLOCKED: unsafe path/duplicate role: {entry}")
        path = asset_root.joinpath(*relative.parts).resolve()
        if not path.is_relative_to(asset_root.resolve()):
            raise AcceptanceError(f"GPU_E2E_BLOCKED: asset escapes root: {path}")
        if not path.is_file() or sha256(path) != entry["sha256"]:
            raise AcceptanceError(f"GPU_E2E_BLOCKED: asset absent or hash changed: {path}")
        assets[role] = path
    return assets


def compare_textgrids(reference: Path, candidate: Path, *, duration: float) -> dict[str, object]:
    """Require identical labels, complete audio coverage and <=20ms non-NULL deltas."""
    baseline = parse_textgrid_long(reference)
    observed = parse_textgrid_long(candidate)
    for path, document in ((reference, baseline), (candidate, observed)):
        validate_textgrid_structure(document, context=str(path), require_full_coverage=True)
        if abs(document.xmin) > 1e-6 or abs(document.xmax - duration) > 1e-6:
            raise AcceptanceError(f"TextGrid does not cover complete audio: {path}")
    deltas: dict[str, float] = {}
    for first, second in zip(baseline.tiers, observed.tiers, strict=True):
        if tuple(i.text for i in first.intervals) != tuple(i.text for i in second.intervals):
            raise AcceptanceError(f"Label sequence differs in tier {first.name}: {candidate}")
        paired = [
            (a, b)
            for a, b in zip(first.intervals, second.intervals, strict=True)
            if a.text != "NULL"
        ]
        if not paired or any(not a.text.strip() for a, _ in paired):
            raise AcceptanceError(f"No valid non-NULL labels in tier {first.name}: {candidate}")
        delta = max(max(abs(a.xmin - b.xmin), abs(a.xmax - b.xmax)) for a, b in paired)
        if delta > MAX_BOUNDARY_DELTA + 1e-9:
            raise AcceptanceError(f"Non-NULL boundary delta {delta:.6f}s > 0.020s: {candidate}")
        deltas[first.name] = delta
    return {
        "reference": str(reference),
        "candidate": str(candidate),
        "max_non_null_boundary_delta_seconds": deltas,
    }


def runtime_metadata(torch: Any, cuda_index: int) -> dict[str, object]:
    """Check the supported inference runtime and selected real CUDA device."""
    if not (3, 10) <= sys.version_info[:2] <= (3, 12):
        raise AcceptanceError("GPU_E2E_BLOCKED: inference requires Python 3.10-3.12")
    for package, expected in (("torch", "2.3.1"), ("transformers", "4.41.2")):
        actual = importlib.metadata.version(package)
        if actual.split("+", 1)[0] != expected:
            raise AcceptanceError(f"GPU_E2E_BLOCKED: expected {package}=={expected}, got {actual}")
    if not torch.cuda.is_available() or cuda_index >= torch.cuda.device_count():
        raise AcceptanceError(f"GPU_E2E_BLOCKED: CUDA device cuda:{cuda_index} is unavailable")
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "flexaligner": importlib.metadata.version("flexaligner"),
        "numpy": importlib.metadata.version("numpy"),
        "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "transformers": importlib.metadata.version("transformers"),
        "cuda_device_index": cuda_index,
        "cuda_device_name": torch.cuda.get_device_name(cuda_index),
        "cuda_device_count": torch.cuda.device_count(),
        "runtime_policy": "Python 3.10-3.12; Torch 2.3.1; Transformers 4.41.2; not frozen CPU hash E2E",
    }


def measure(
    operation: Callable[[], list[AlignmentResult]],
    *,
    torch: Any,
    cuda_index: int,
    use_cuda: bool,
) -> tuple[list[AlignmentResult], dict[str, object]]:
    transformers = importlib.import_module("transformers")
    auto_model = transformers.AutoModelForCTC
    original_load = auto_model.from_pretrained
    loads: list[str] = []

    def counted_load(*args: Any, **kwargs: Any) -> Any:
        loads.append(str(args[0] if args else kwargs.get("pretrained_model_name_or_path")))
        return original_load(*args, **kwargs)

    if use_cuda:
        torch.cuda.synchronize(cuda_index)
        torch.cuda.reset_peak_memory_stats(cuda_index)
    started = time.perf_counter()
    with patch.object(auto_model, "from_pretrained", side_effect=counted_load):
        results = operation()
    if use_cuda:
        torch.cuda.synchronize(cuda_index)
    elapsed = time.perf_counter() - started
    return results, {
        "elapsed_seconds": elapsed,
        "model_load_count": len(loads),
        "model_load_paths": loads,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(cuda_index))
        if use_cuda
        else None,
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(cuda_index))
        if use_cuda
        else None,
    }


def verify(asset_root: Path, output_dir: Path, *, cuda_index: int, report: dict[str, Any]) -> None:
    # Local-only loaders plus offline flags prevent model resolution/download.
    os.environ.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    torch = importlib.import_module("torch")
    report["runtime"] = runtime_metadata(torch, cuda_index)
    english = load_assets(MANIFEST_ROOT / "asset_manifest.json", asset_root)
    mandarin = load_assets(MANIFEST_ROOT / "mandarin_asset_manifest.json", asset_root)
    report["asset_manifests"] = {
        name: sha256(MANIFEST_ROOT / name)
        for name in ("asset_manifest.json", "mandarin_asset_manifest.json")
    }
    models = {
        "en": LocalModelBundle(
            chunker_dir=english["stage1_model_config"].parent,
            aligner_dir=english["stage2_model_config"].parent,
        ),
        "zh": validate_model_snapshot(mandarin["zh_model_manifest"].parent, language=Language.ZH),
    }
    fixtures = {
        "en": [("english", english["audio"], english["transcript"])],
        "zh": [
            ("mandarin_system", mandarin["zh_audio"], mandarin["zh_transcript"]),
            ("mandarin_g2p", mandarin["zh_g2p_audio"], mandarin["zh_g2p_transcript"]),
        ],
    }
    durations: dict[str, float] = {}
    paths: dict[tuple[str, str, str], Path] = {}
    report["runs"] = runs = []
    report["comparisons"] = comparisons = []
    for language in ("en", "zh"):
        for device_name, device in (("cpu", Device.CPU), ("cuda", Device.CUDA)):
            options = AlignmentOptions(
                language=Language(language),
                device=device,
                device_index=cuda_index if device is Device.CUDA else 0,
                pronunciation_mode=PronunciationMode.G2P
                if language == "zh"
                else PronunciationMode.LEXICON_ONLY,
                num_threads=1,
            )
            for mode in ("single", "batch"):
                # Duplicate English requests make cache reuse observable even with one frozen WAV.
                cases = fixtures[language] * (2 if mode == "batch" and language == "en" else 1)
                requests = []
                for index, (name, audio, transcript) in enumerate(cases):
                    with wave.open(str(audio), "rb") as handle:
                        durations[name] = handle.getnframes() / handle.getframerate()
                    path = output_dir / f"{device_name}-{mode}-{name}-{index}.TextGrid"
                    requests.append(
                        AlignmentRequest(
                            audio_path=audio,
                            transcript=transcript.read_text(encoding="utf-8"),
                            output=TextGridOutput(path=path),
                            utterance_id=name,
                        )
                    )

                def operation(
                    selected_requests: list[AlignmentRequest] = requests,
                    selected_models: LocalModelBundle = models[language],
                    selected_options: AlignmentOptions = options,
                    selected_mode: str = mode,
                    selected_lexicon: Path | None = english["effective_fixture_lexicon"]
                    if language == "en"
                    else None,
                ) -> list[AlignmentResult]:
                    with FlexAligner(
                        models=selected_models,
                        options=selected_options,
                        lexicon_path=selected_lexicon,
                    ) as engine:
                        if selected_mode == "single":
                            return [engine.align(request) for request in selected_requests]
                        batch = engine.align_batch(selected_requests)
                        if len(batch.items) != len(selected_requests) or any(
                            item.status != "succeeded" or item.result is None
                            for item in batch.items
                        ):
                            raise AcceptanceError(
                                f"Batch failed: {[item.error for item in batch.items]}"
                            )
                        return [item.result for item in batch.items if item.result is not None]

                results, metrics = measure(
                    operation, torch=torch, cuda_index=cuda_index, use_cuda=device is Device.CUDA
                )
                for index, result in enumerate(results):
                    if result.provenance.device is not device or (
                        device is Device.CUDA and result.provenance.device_index != cuda_index
                    ):
                        raise AcceptanceError(f"Unexpected actual device: {result.provenance}")
                    name = cases[index][0]
                    key = (device_name, mode, name)
                    comparisons.append(
                        compare_textgrids(
                            result.output_path, result.output_path, duration=durations[name]
                        )
                    )
                    if key in paths:
                        comparisons.append(
                            compare_textgrids(
                                paths[key], result.output_path, duration=durations[name]
                            )
                        )
                    paths[key] = result.output_path
                runs.append(
                    {
                        "language": language,
                        "device": device_name,
                        "mode": mode,
                        "item_count": len(results),
                        **metrics,
                    }
                )
        for name, _, _ in fixtures[language]:
            baseline = paths[("cpu", "single", name)]
            for device_name, mode in (("cuda", "single"), ("cpu", "batch"), ("cuda", "batch")):
                comparisons.append(
                    compare_textgrids(
                        baseline, paths[(device_name, mode, name)], duration=durations[name]
                    )
                )
    report["long_input_acceptance"] = "NOT_RUN: frozen short fixtures only; no long-input claim"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="new directory; never overwritten"
    )
    parser.add_argument("--cuda-index", type=int, default=0)
    args = parser.parse_args(argv)
    if args.cuda_index < 0:
        parser.error("--cuda-index must be non-negative")
    root, output = args.asset_root.resolve(), args.output_dir.resolve()
    if output.is_relative_to(root) or root.is_relative_to(output):
        parser.error("asset-root and output-dir must not contain one another")
    report: dict[str, Any] = {
        "status": "FAILED",
        "asset_root": str(root),
        "output_dir": str(output),
        "boundary_tolerance_seconds": MAX_BOUNDARY_DELTA,
        "requested_cuda_device": f"cuda:{args.cuda_index}",
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    started = time.perf_counter()
    try:
        output.mkdir(parents=True, exist_ok=False)
    except OSError as error:
        print(f"GPU_E2E_BLOCKED: cannot create new output directory: {error}", file=sys.stderr)
        return 2
    try:
        verify(root, output, cuda_index=args.cuda_index, report=report)
        report["status"] = "PASSED"
        exit_code = 0
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        print(f"GPU_E2E_FAILED: {error}", file=sys.stderr)
        exit_code = 1
    report["total_elapsed_seconds"] = time.perf_counter() - started
    report_path = output / "gpu_batch_acceptance.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"GPU_E2E_{report['status']} report={report_path}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
