from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
from tests._support import SRC_ROOT

from flexaligner.adapters import hf_local
from flexaligner.adapters.hf_local import LocalHuggingFaceInferenceFactory
from flexaligner.errors import AlignmentError, ModelCompatibilityError, ModelValidationError
from flexaligner.ports import LocalInferenceFactoryPort

_UNSET = object()


class _FakeTensor:
    def __init__(self, value: Any, runtime: _FakeRuntime) -> None:
        self.array = np.asarray(value)
        self.runtime = runtime

    def to(self, device: object) -> _FakeTensor:
        self.runtime.tensor_devices.append(device)
        return self

    def detach(self) -> _FakeTensor:
        return self

    def cpu(self) -> _FakeTensor:
        self.runtime.tensor_cpu_calls += 1
        return self

    def numpy(self) -> np.ndarray[Any, Any]:
        return self.array


class _BrokenTensor(_FakeTensor):
    def detach(self) -> _FakeTensor:
        raise RuntimeError("cannot detach")


class _Tokenizer:
    def __init__(self, runtime: _FakeRuntime) -> None:
        self.runtime = runtime
        self.pad_token: object = "<pad>"

    def get_vocab(self) -> object:
        if self.runtime.vocab_error is not None:
            raise self.runtime.vocab_error
        return self.runtime.vocabulary


class _Processor:
    def __init__(self, runtime: _FakeRuntime) -> None:
        self.runtime = runtime
        self.feature_extractor = SimpleNamespace(sampling_rate=runtime.processor_sample_rate)
        self.tokenizer = _Tokenizer(runtime)

    def __call__(
        self,
        audio: np.ndarray[Any, Any],
        *,
        sampling_rate: int,
        return_tensors: str,
    ) -> object:
        self.runtime.processor_calls.append(
            {
                "audio": audio.copy(),
                "sampling_rate": sampling_rate,
                "return_tensors": return_tensors,
            }
        )
        if self.runtime.processor_error is not None:
            raise self.runtime.processor_error
        if self.runtime.encoded_override is not _UNSET:
            return self.runtime.encoded_override
        return {"input_values": _FakeTensor(audio[None, :], self.runtime)}


class _Model:
    def __init__(self, runtime: _FakeRuntime) -> None:
        self.runtime = runtime
        self.config = SimpleNamespace(vocab_size=runtime.model_vocab_size)

    def to(self, device: object) -> _Model:
        self.runtime.model_devices.append(device)
        return self

    def eval(self) -> _Model:
        self.runtime.eval_calls += 1
        return self

    def __call__(self, **inputs: object) -> object:
        self.runtime.model_inputs.append(inputs)
        if self.runtime.inference_depth != 1:
            raise AssertionError("model call was not inside torch.inference_mode")
        if self.runtime.model_error is not None:
            raise self.runtime.model_error
        tensor_type = _BrokenTensor if self.runtime.broken_logits_tensor else _FakeTensor
        return SimpleNamespace(logits=tensor_type(self.runtime.logits, self.runtime))


class _Loader:
    def __init__(self, runtime: _FakeRuntime, target: str) -> None:
        self.runtime = runtime
        self.target = target

    def from_pretrained(self, path: str, **kwargs: object) -> object:
        self.runtime.load_calls.append((self.target, path, kwargs))
        if self.runtime.load_error_target == self.target:
            raise RuntimeError(f"{self.target} load failed")
        if self.target == "processor":
            return self.runtime.processor
        return self.runtime.model


class _FakeRuntime:
    def __init__(self) -> None:
        self.processor_sample_rate: object = 16_000
        self.vocabulary: object = {"<pad>": 0, "a": 1, "b": 2}
        self.model_vocab_size: object = 3
        self.logits: Any = np.array(
            [[[1.0, 2.0, -1.0], [0.5, -0.5, 1.5]]],
            dtype=np.float64,
        )
        self.vocab_error: Exception | None = None
        self.processor_error: Exception | None = None
        self.model_error: Exception | None = None
        self.load_error_target: str | None = None
        self.encoded_override: object = _UNSET
        self.broken_logits_tensor = False

        self.processor = _Processor(self)
        self.model = _Model(self)
        self.load_calls: list[tuple[str, str, dict[str, object]]] = []
        self.thread_calls: list[int] = []
        self.device_calls: list[str] = []
        self.model_devices: list[object] = []
        self.tensor_devices: list[object] = []
        self.processor_calls: list[dict[str, object]] = []
        self.model_inputs: list[dict[str, object]] = []
        self.eval_calls = 0
        self.inference_depth = 0
        self.inference_entries = 0
        self.tensor_cpu_calls = 0
        self.gc_calls = 0

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        torch_module = ModuleType("torch")

        def set_num_threads(value: int) -> None:
            self.thread_calls.append(value)

        def device(value: str) -> str:
            self.device_calls.append(value)
            return f"fake-device:{value}"

        @contextmanager
        def inference_mode() -> Iterator[None]:
            self.inference_depth += 1
            self.inference_entries += 1
            try:
                yield
            finally:
                self.inference_depth -= 1

        def log_softmax(value: _FakeTensor, *, dim: int) -> _FakeTensor:
            assert dim == -1
            array = np.asarray(value.array, dtype=np.float64)
            shifted = array - np.max(array, axis=dim, keepdims=True)
            result = shifted - np.log(np.exp(shifted).sum(axis=dim, keepdims=True))
            return _FakeTensor(result, self)

        torch_module.set_num_threads = set_num_threads
        torch_module.device = device
        torch_module.inference_mode = inference_mode
        torch_module.log_softmax = log_softmax

        transformers_module = ModuleType("transformers")
        transformers_module.AutoProcessor = _Loader(self, "processor")
        transformers_module.AutoModelForCTC = _Loader(self, "model")

        monkeypatch.setitem(sys.modules, "torch", torch_module)
        monkeypatch.setitem(sys.modules, "transformers", transformers_module)

        def collect() -> int:
            self.gc_calls += 1
            return 0

        monkeypatch.setattr(hf_local.gc, "collect", collect)


def _installed_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[_FakeRuntime, Path]:
    runtime = _FakeRuntime()
    runtime.install(monkeypatch)
    model_dir = tmp_path / "local-model"
    model_dir.mkdir()
    return runtime, model_dir


def _expected_log_softmax(logits: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def test_module_import_is_lazy_and_does_not_load_optional_dependencies(tmp_path: Path) -> None:
    probe = r"""
import json
import sys
import flexaligner.adapters.hf_local
print(json.dumps({
    "torch_loaded": "torch" in sys.modules,
    "transformers_loaded": "transformers" in sys.modules,
}))
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC_ROOT)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "torch_loaded": False,
        "transformers_loaded": False,
    }


@pytest.mark.parametrize("session_method", ["chunker_session", "aligner_session"])
def test_successful_session_is_offline_cpu_only_and_returns_valid_posterior(
    session_method: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime, model_dir = _installed_runtime(monkeypatch, tmp_path)
    factory = LocalHuggingFaceInferenceFactory()
    assert isinstance(factory, LocalInferenceFactoryPort)
    context = getattr(factory, session_method)(model_dir, num_threads=3)
    audio = np.linspace(-0.75, 0.75, 1_600, dtype=np.float32)[::2]
    assert not audio.flags.c_contiguous

    with context as session:
        assert dict(session.vocabulary) == {"<pad>": 0, "a": 1, "b": 2}
        assert session.model_vocab_size == 3
        assert session.sample_rate == 16_000
        assert session.pad_token == "<pad>"
        with pytest.raises(TypeError):
            session.vocabulary["new"] = 3  # type: ignore[index]

        posterior = session.infer(audio, 16_000)
        expected = _expected_log_softmax(runtime.logits)[0].astype(np.float32)
        np.testing.assert_allclose(posterior.log_probs, expected, rtol=1e-6, atol=1e-6)
        assert posterior.log_probs.dtype == np.float32
        assert posterior.log_probs.flags.c_contiguous
        assert posterior.seconds_per_frame == pytest.approx((audio.size / 16_000) / 2)

    expected_path = str(model_dir)
    assert runtime.load_calls == [
        (
            "processor",
            expected_path,
            {"local_files_only": True, "trust_remote_code": False},
        ),
        ("model", expected_path, {"local_files_only": True, "trust_remote_code": False}),
    ]
    assert runtime.thread_calls == [3]
    assert runtime.device_calls == ["cpu", "cpu"]
    assert runtime.model_devices == ["fake-device:cpu"]
    assert runtime.tensor_devices == ["fake-device:cpu"]
    assert runtime.eval_calls == 1
    assert runtime.inference_entries == 1
    assert runtime.processor_calls[0]["sampling_rate"] == 16_000
    assert runtime.processor_calls[0]["return_tensors"] == "pt"
    assert np.asarray(runtime.processor_calls[0]["audio"]).flags.c_contiguous
    assert runtime.gc_calls == 1
    with pytest.raises(AlignmentError, match="closed"):
        _ = session.vocabulary
    with pytest.raises(AlignmentError, match="closed"):
        session.infer(np.ones(8, dtype=np.float32), 16_000)


def test_context_body_exception_still_closes_and_factory_can_be_reused(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime, model_dir = _installed_runtime(monkeypatch, tmp_path)
    factory = LocalHuggingFaceInferenceFactory()
    sentinel = LookupError("caller failed")

    with (
        pytest.raises(LookupError, match="caller failed") as caught,
        factory.chunker_session(model_dir, num_threads=1) as first_session,
    ):
        raise sentinel

    assert caught.value is sentinel
    with pytest.raises(AlignmentError, match="closed"):
        _ = first_session.sample_rate
    with factory.aligner_session(model_dir, num_threads=2) as second_session:
        assert second_session.sample_rate == 16_000
    assert runtime.thread_calls == [1, 2]
    assert runtime.gc_calls == 2


def test_chunker_and_aligner_sessions_cannot_overlap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime, model_dir = _installed_runtime(monkeypatch, tmp_path)
    factory = LocalHuggingFaceInferenceFactory()

    with factory.chunker_session(model_dir, num_threads=1) as chunker:
        with (
            pytest.raises(ModelValidationError, match="may not overlap") as error,
            factory.aligner_session(model_dir, num_threads=1),
        ):
            raise AssertionError("unreachable")
        assert error.value.context == {
            "active_session": "chunker",
            "requested_session": "aligner",
        }
        assert chunker.sample_rate == 16_000

    with factory.aligner_session(model_dir, num_threads=1) as aligner:
        assert aligner.sample_rate == 16_000
    assert runtime.thread_calls == [1, 1]
    assert runtime.gc_calls == 3


@pytest.mark.parametrize("num_threads", [0, -1, True, 1.5])
def test_num_threads_must_be_a_positive_real_integer(
    num_threads: object,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime, model_dir = _installed_runtime(monkeypatch, tmp_path)
    factory = LocalHuggingFaceInferenceFactory()
    with (
        pytest.raises(ModelValidationError, match="positive integer"),
        factory.chunker_session(
            model_dir,
            num_threads=num_threads,  # type: ignore[arg-type]
        ),
    ):
        raise AssertionError("unreachable")
    assert runtime.thread_calls == []
    assert runtime.load_calls == []
    assert runtime.gc_calls == 1


def test_model_path_must_be_an_existing_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime, _ = _installed_runtime(monkeypatch, tmp_path)
    missing = tmp_path / "missing"
    with (
        pytest.raises(ModelValidationError, match="existing local directory"),
        LocalHuggingFaceInferenceFactory().chunker_session(missing, num_threads=1),
    ):
        raise AssertionError("unreachable")
    assert runtime.load_calls == []


def test_missing_optional_dependency_is_typed_chained_and_cleanup_allows_retry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime, model_dir = _installed_runtime(monkeypatch, tmp_path)
    real_import = hf_local.importlib.import_module

    def missing(name: str) -> ModuleType:
        if name == "transformers":
            raise ModuleNotFoundError("no transformers")
        return real_import(name)

    monkeypatch.setattr(hf_local.importlib, "import_module", missing)
    factory = LocalHuggingFaceInferenceFactory()
    with (
        pytest.raises(ModelValidationError, match="optional dependencies") as error,
        factory.chunker_session(model_dir, num_threads=1),
    ):
        raise AssertionError("unreachable")
    assert isinstance(error.value.__cause__, ModuleNotFoundError)
    assert error.value.context["extra"] == "inference"
    assert runtime.gc_calls == 1

    monkeypatch.setattr(hf_local.importlib, "import_module", real_import)
    with factory.aligner_session(model_dir, num_threads=1) as session:
        assert session.sample_rate == 16_000


def test_local_loader_failure_is_typed_chained_and_releases_factory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime, model_dir = _installed_runtime(monkeypatch, tmp_path)
    runtime.load_error_target = "model"
    factory = LocalHuggingFaceInferenceFactory()
    with (
        pytest.raises(ModelValidationError, match="local chunker") as error,
        factory.chunker_session(model_dir, num_threads=4),
    ):
        raise AssertionError("unreachable")
    assert isinstance(error.value.__cause__, RuntimeError)
    assert runtime.gc_calls == 1

    runtime.load_error_target = None
    with factory.aligner_session(model_dir, num_threads=4) as session:
        assert session.model_vocab_size == 3


@pytest.mark.parametrize(
    ("sample_rate", "error_type"),
    [
        (8_000, ModelCompatibilityError),
        ("16000", ModelValidationError),
        (True, ModelValidationError),
    ],
)
def test_processor_sample_rate_is_strictly_16khz(
    sample_rate: object,
    error_type: type[Exception],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime, model_dir = _installed_runtime(monkeypatch, tmp_path)
    runtime.processor.feature_extractor.sampling_rate = sample_rate
    with (
        pytest.raises(error_type),
        LocalHuggingFaceInferenceFactory().aligner_session(model_dir, num_threads=1),
    ):
        raise AssertionError("unreachable")
    assert runtime.gc_calls == 1


def test_compatibility_error_traceback_does_not_retain_loaded_resources(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime, model_dir = _installed_runtime(monkeypatch, tmp_path)
    runtime.processor.feature_extractor.sampling_rate = 8_000
    torch_module = sys.modules["torch"]
    transformers_module = sys.modules["transformers"]

    with (
        pytest.raises(ModelCompatibilityError) as caught,
        LocalHuggingFaceInferenceFactory().aligner_session(model_dir, num_threads=1),
    ):
        raise AssertionError("unreachable")

    forbidden = {
        id(runtime.model),
        id(runtime.processor),
        id(torch_module),
        id(transformers_module),
    }
    traceback_cursor = caught.value.__traceback__
    while traceback_cursor is not None:
        frame = traceback_cursor.tb_frame
        if Path(frame.f_code.co_filename).resolve() == Path(hf_local.__file__).resolve():
            retained = {id(value) for value in frame.f_locals.values()}
            assert retained.isdisjoint(forbidden), frame.f_code.co_name
        traceback_cursor = traceback_cursor.tb_next
    assert runtime.gc_calls == 1


@pytest.mark.parametrize(
    ("vocabulary", "model_vocab_size", "error_type"),
    [
        ({}, 3, ModelValidationError),
        ({"a": -1}, 3, ModelValidationError),
        ({"a": True}, 3, ModelValidationError),
        ({"a": 0, "b": 0}, 3, ModelValidationError),
        ({"a": 3}, 3, ModelCompatibilityError),
        ({"a": 0}, 0, ModelValidationError),
        ({"a": 0}, True, ModelValidationError),
    ],
)
def test_tokenizer_and_model_vocabularies_are_validated(
    vocabulary: object,
    model_vocab_size: object,
    error_type: type[Exception],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime, model_dir = _installed_runtime(monkeypatch, tmp_path)
    runtime.vocabulary = vocabulary
    runtime.model.config.vocab_size = model_vocab_size
    with (
        pytest.raises(error_type),
        LocalHuggingFaceInferenceFactory().aligner_session(model_dir, num_threads=1),
    ):
        raise AssertionError("unreachable")


def test_tokenizer_get_vocab_and_pad_token_are_validated(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime, model_dir = _installed_runtime(monkeypatch, tmp_path)
    runtime.vocab_error = ValueError("bad vocabulary")
    with (
        pytest.raises(ModelValidationError, match=r"get_vocab\(\) failed") as error,
        LocalHuggingFaceInferenceFactory().aligner_session(model_dir, num_threads=1),
    ):
        raise AssertionError("unreachable")
    assert error.value.__cause__ is runtime.vocab_error

    runtime.vocab_error = None
    runtime.processor.tokenizer.pad_token = 42
    with (
        pytest.raises(ModelValidationError, match="pad_token"),
        LocalHuggingFaceInferenceFactory().aligner_session(model_dir, num_threads=1),
    ):
        raise AssertionError("unreachable")


@pytest.mark.parametrize(
    ("audio", "sample_rate", "message"),
    [
        ([0.0], 16_000, "ndarray"),
        (np.ones(4, dtype=np.float64), 16_000, "float32"),
        (np.ones((1, 4), dtype=np.float32), 16_000, "one-dimensional"),
        (np.array([], dtype=np.float32), 16_000, "empty"),
        (np.array([0.0, np.nan], dtype=np.float32), 16_000, "NaN"),
        (np.ones(4, dtype=np.float32), 8_000, "16000 Hz"),
        (np.ones(4, dtype=np.float32), True, "16000 Hz"),
    ],
)
def test_infer_rejects_invalid_waveform_contract(
    audio: object,
    sample_rate: object,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime, model_dir = _installed_runtime(monkeypatch, tmp_path)
    with (
        LocalHuggingFaceInferenceFactory().aligner_session(model_dir, num_threads=1) as session,
        pytest.raises(AlignmentError, match=message),
    ):
        session.infer(audio, sample_rate)  # type: ignore[arg-type]
    assert runtime.processor_calls == []


@pytest.mark.parametrize(
    ("logits", "error_type", "message"),
    [
        (np.zeros((2, 3), dtype=np.float32), AlignmentError, r"\[1, T, V\]"),
        (np.zeros((2, 1, 3), dtype=np.float32), AlignmentError, r"\[1, T, V\]"),
        (np.zeros((1, 0, 3), dtype=np.float32), AlignmentError, "positive time"),
        (np.zeros((1, 2, 4), dtype=np.float32), ModelCompatibilityError, "vocabulary"),
        (
            np.array([[[0.0, np.inf, 1.0]]], dtype=np.float32),
            AlignmentError,
            "NaN or infinity",
        ),
    ],
)
def test_infer_validates_model_logits(
    logits: np.ndarray[Any, Any],
    error_type: type[Exception],
    message: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime, model_dir = _installed_runtime(monkeypatch, tmp_path)
    runtime.logits = logits
    with (
        LocalHuggingFaceInferenceFactory().aligner_session(model_dir, num_threads=1) as session,
        pytest.raises(error_type, match=message),
    ):
        session.infer(np.ones(32, dtype=np.float32), 16_000)


@pytest.mark.parametrize("failure", ["processor", "model", "tensor"])
def test_inference_runtime_failures_are_typed_and_chained(
    failure: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime, model_dir = _installed_runtime(monkeypatch, tmp_path)
    if failure == "processor":
        runtime.processor_error = RuntimeError("processor failed")
    elif failure == "model":
        runtime.model_error = RuntimeError("model failed")
    else:
        runtime.broken_logits_tensor = True

    with LocalHuggingFaceInferenceFactory().chunker_session(model_dir, num_threads=1) as session:
        with pytest.raises(AlignmentError) as error:
            session.infer(np.ones(32, dtype=np.float32), 16_000)
        assert error.value.__cause__ is not None


@pytest.mark.parametrize("encoded", [None, {}, {1: object()}, {"input_values": object()}])
def test_processor_output_mapping_is_strictly_validated(
    encoded: object,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime, model_dir = _installed_runtime(monkeypatch, tmp_path)
    runtime.encoded_override = encoded
    with (
        LocalHuggingFaceInferenceFactory().chunker_session(model_dir, num_threads=1) as session,
        pytest.raises(AlignmentError),
    ):
        session.infer(np.ones(32, dtype=np.float32), 16_000)
