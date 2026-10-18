"""GPU 验收脚本的模拟边界/失败路径, 不代表真实 CUDA 验收。"""

from __future__ import annotations

import hashlib
import json
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest
from scripts import verify_gpu_batch as verifier

from flexaligner.textgrid import Interval, IntervalTier, TextGridDocument, serialize_textgrid_long


def grid(path: Path, *, start: float = 0.2, label: str = "a", gap: bool = False) -> Path:
    intervals = (
        Interval(0, start - 0.01 if gap else start, "NULL"),
        Interval(start, 0.8, label),
        Interval(0.8, 1, "NULL"),
    )
    document = TextGridDocument(
        0, 1, tuple(IntervalTier(name, 0, 1, intervals) for name in ("phones", "words"))
    )
    path.write_text(serialize_textgrid_long(document), encoding="utf-8")
    return path


@pytest.mark.parametrize("delta", [0, 0.019999, 0.02])
def test_compare_accepts_inclusive_20ms_threshold(tmp_path: Path, delta: float) -> None:
    baseline = grid(tmp_path / "cpu.TextGrid")
    candidate = grid(tmp_path / "gpu.TextGrid", start=0.2 + delta)
    result = verifier.compare_textgrids(baseline, candidate, duration=1)
    assert result["max_non_null_boundary_delta_seconds"] == pytest.approx(
        {"words": delta, "phones": delta}
    )


@pytest.mark.parametrize("change", ["boundary", "labels", "gap", "duration", "all_null"])
def test_compare_rejects_false_acceptance(tmp_path: Path, change: str) -> None:
    baseline = grid(tmp_path / "cpu.TextGrid", label="NULL" if change == "all_null" else "a")
    candidate = grid(
        tmp_path / "gpu.TextGrid",
        start=0.220001 if change == "boundary" else 0.2,
        label="b" if change == "labels" else "NULL" if change == "all_null" else "a",
        gap=change == "gap",
    )
    with pytest.raises((verifier.AcceptanceError, ValueError)):
        verifier.compare_textgrids(baseline, candidate, duration=2 if change == "duration" else 1)


def manifest(
    tmp_path: Path, *, relative: str = "model.bin", role: str = "stage1_model_weights"
) -> Path:
    source = tmp_path / "model.bin"
    source.write_bytes(b"frozen")
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "approved",
                "files": [
                    {
                        "path": relative,
                        "role": role,
                        "sha256": hashlib.sha256(b"frozen").hexdigest(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_assets_are_validated_without_writing_inputs(tmp_path: Path) -> None:
    path = manifest(tmp_path)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    assert verifier.load_assets(path, tmp_path) == {"stage1_model_weights": tmp_path / "model.bin"}
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


@pytest.mark.parametrize("issue", ["missing", "changed", "traversal", "symlink", "duplicate"])
def test_assets_fail_closed(tmp_path: Path, issue: str) -> None:
    path = manifest(tmp_path, relative="../model.bin" if issue == "traversal" else "model.bin")
    if issue == "missing":
        path = manifest(tmp_path, relative="absent.bin")
    elif issue == "changed":
        (tmp_path / "model.bin").write_bytes(b"changed")
    elif issue == "symlink":
        external = tmp_path.parent / f"{tmp_path.name}-external.bin"
        external.write_bytes(b"frozen")
        (tmp_path / "link.bin").symlink_to(external)
        path = manifest(tmp_path, relative="link.bin")
    elif issue == "duplicate":
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["files"] *= 2
        path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(verifier.AcceptanceError):
        verifier.load_assets(path, tmp_path)


@pytest.mark.parametrize("python_version", [(3, 10), (3, 11), (3, 12)])
def test_cuda_absence_is_failure_not_skip(
    monkeypatch: pytest.MonkeyPatch, python_version: tuple[int, int]
) -> None:
    monkeypatch.setattr(verifier, "sys", SimpleNamespace(version_info=python_version))
    monkeypatch.setattr(
        verifier.importlib.metadata,
        "version",
        lambda name: {"torch": "2.3.1+cu121", "transformers": "4.41.2"}[name],
    )
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    with pytest.raises(verifier.AcceptanceError, match=r"CUDA device.*unavailable"):
        verifier.runtime_metadata(fake_torch, 0)


@pytest.mark.parametrize("python_version", [(3, 9), (3, 13), (3, 14)])
def test_unsupported_python_rejected_before_runtime_checks(
    monkeypatch: pytest.MonkeyPatch, python_version: tuple[int, int]
) -> None:
    monkeypatch.setattr(verifier, "sys", SimpleNamespace(version_info=python_version))

    def unexpected_metadata_lookup(_name: str) -> str:
        pytest.fail("package metadata must not be accessed for unsupported Python")

    monkeypatch.setattr(verifier.importlib.metadata, "version", unexpected_metadata_lookup)
    with pytest.raises(verifier.AcceptanceError, match=r"inference requires Python 3\.10-3\.12"):
        verifier.runtime_metadata(object(), 0)


def test_failed_run_has_nonzero_exit_and_durable_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise verifier.AcceptanceError("GPU_E2E_BLOCKED: CUDA unavailable")

    monkeypatch.setattr(verifier, "verify", fail)
    output = tmp_path / "results"
    code = verifier.main(["--asset-root", str(tmp_path / "assets"), "--output-dir", str(output)])
    assert code == 1
    report = json.loads((output / "gpu_batch_acceptance.json").read_text(encoding="utf-8"))
    assert report["status"] == "FAILED"
    assert "CUDA unavailable" in report["error"]["message"]
    assert "GPU_E2E_FAILED" in capsys.readouterr().err


def test_existing_output_is_never_overwritten(tmp_path: Path) -> None:
    output = tmp_path / "results"
    output.mkdir()
    report = output / "gpu_batch_acceptance.json"
    report.write_text("existing evidence", encoding="utf-8")
    assert (
        verifier.main(["--asset-root", str(tmp_path / "assets"), "--output-dir", str(output)]) == 2
    )
    assert report.read_text(encoding="utf-8") == "existing evidence"


def test_output_cannot_modify_asset_tree(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as caught:
        verifier.main(["--asset-root", str(tmp_path), "--output-dir", str(tmp_path / "new")])
    assert caught.value.code == 2
    assert not (tmp_path / "new").exists()


def test_measure_records_load_count_and_cuda_peaks(monkeypatch: pytest.MonkeyPatch) -> None:
    auto_model = SimpleNamespace(from_pretrained=lambda path: object())
    monkeypatch.setattr(
        verifier.importlib,
        "import_module",
        lambda name: SimpleNamespace(AutoModelForCTC=auto_model),
    )
    events = []
    torch = SimpleNamespace(
        cuda=SimpleNamespace(
            synchronize=lambda index: events.append(("synchronize", index)),
            reset_peak_memory_stats=lambda index: events.append(("reset", index)),
            max_memory_allocated=lambda index: 123,
            max_memory_reserved=lambda index: 456,
        )
    )

    def operation() -> list[object]:
        auto_model.from_pretrained("chunker")
        auto_model.from_pretrained("aligner")
        return []

    results, metrics = verifier.measure(operation, torch=torch, cuda_index=1, use_cuda=True)
    assert results == []
    assert metrics["model_load_count"] == 2
    assert metrics["model_load_paths"] == ["chunker", "aligner"]
    assert metrics["peak_allocated_bytes"] == 123
    assert metrics["peak_reserved_bytes"] == 456
    assert metrics["elapsed_seconds"] >= 0
    assert events == [("synchronize", 1), ("reset", 1), ("synchronize", 1)]


@pytest.mark.parametrize("batch_failure", [False, True])
def test_simulated_public_api_matrix_never_hides_batch_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    batch_failure: bool,
) -> None:
    # This test deliberately simulates outputs, not actual inference or CUDA.
    audio = tmp_path / "audio.wav"
    with wave.open(str(audio), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\0\0" * 16000)
    transcript = tmp_path / "transcript.txt"
    transcript.write_text("fixture", encoding="utf-8")
    assets = dict.fromkeys(
        (
            "stage1_model_config",
            "stage2_model_config",
            "zh_model_manifest",
            "effective_fixture_lexicon",
        ),
        tmp_path / "model.json",
    )
    assets.update(dict.fromkeys(("audio", "zh_audio", "zh_g2p_audio"), audio))
    assets.update(dict.fromkeys(("transcript", "zh_transcript", "zh_g2p_transcript"), transcript))
    monkeypatch.setattr(verifier, "load_assets", lambda *args: assets)
    bundle = verifier.LocalModelBundle(chunker_dir=tmp_path, aligner_dir=tmp_path)
    monkeypatch.setattr(verifier, "validate_model_snapshot", lambda *args, **kwargs: bundle)
    monkeypatch.setattr(verifier, "runtime_metadata", lambda *args: {"simulated": True})
    monkeypatch.setattr(verifier.importlib, "import_module", lambda name: object())
    monkeypatch.setattr(verifier, "measure", lambda op, **kwargs: (op(), {"simulated": True}))

    class Engine:
        def __init__(self, **kwargs: object) -> None:
            self.options = kwargs["options"]

        def __enter__(self) -> Engine:
            return self

        def __exit__(self, *args: object) -> None:
            pass

        def align(self, request: object) -> SimpleNamespace:
            path = grid(request.output.path)
            return SimpleNamespace(
                output_path=path,
                provenance=SimpleNamespace(
                    device=self.options.device, device_index=self.options.device_index
                ),
            )

        def align_batch(self, requests: object) -> SimpleNamespace:
            return SimpleNamespace(
                items=tuple(
                    SimpleNamespace(
                        status="failed" if batch_failure else "succeeded",
                        result=None if batch_failure else self.align(request),
                        error="simulated failure",
                    )
                    for request in requests
                )
            )

    monkeypatch.setattr(verifier, "FlexAligner", Engine)
    output = tmp_path / "output"
    output.mkdir()
    report = {}
    if batch_failure:
        with pytest.raises(verifier.AcceptanceError, match="Batch failed"):
            verifier.verify(tmp_path, output, cuda_index=1, report=report)
    else:
        verifier.verify(tmp_path, output, cuda_index=1, report=report)
        assert len(report["runs"]) == 8
        assert sum(run["item_count"] for run in report["runs"]) == 14
        assert report["long_input_acceptance"].startswith("NOT_RUN")
