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
from flexaligner.contracts import Device
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
        self.config = SimpleNamespace(
            vocab_size=runtime.model_vocab_size,
            conv_stride=runtime.model_conv_stride,
        )

    def to(self, device: object) -> _Model:
        self.runtime.model_devices.append(device)
        if self.runtime.move_errors:
            error = self.runtime.move_errors.pop(0)
            if error is not None:
                raise error
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
        self.model_conv_stride: object = [5, 2, 2, 2, 2, 2, 1]
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
        self.cuda_available = False
        self.cuda_count = 0
        self.empty_cache_calls = 0
        self.move_errors: list[Exception | None] = []

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

        class OutOfMemoryError(RuntimeError):
            pass

        def empty_cache() -> None:
            self.empty_cache_calls += 1

        @contextmanager
        def cuda_device(index: int) -> Iterator[None]:
            assert 0 <= index < self.cuda_count
            yield

        torch_module.cuda = SimpleNamespace(
            is_available=lambda: self.cuda_available,
            device_count=lambda: self.cuda_count,
            empty_cache=empty_cache,
            OutOfMemoryError=OutOfMemoryError,
            device=cuda_device,
        )

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


@pytest.mark.parametrize(
    "conv_stride",
    [None, [], "5,2,2,2,2,2,1", [5, 2, True, 2, 2, 2, 1], [5, 2, 0, 2, 2, 2, 1]],
)
def test_d037_aligner_nominal_conv_stride_is_required_and_strictly_validated(
    conv_stride: object,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime, model_dir = _installed_runtime(monkeypatch, tmp_path)
    runtime.model.config.conv_stride = conv_stride

    with (
        pytest.raises(ModelValidationError, match=r"config\.conv_stride") as error,
        LocalHuggingFaceInferenceFactory().aligner_session(model_dir, num_threads=1),
    ):
        raise AssertionError("unreachable")

    assert error.value.context == {
        "session": "aligner",
        "conv_stride": str(conv_stride),
    }


def test_d037_aligner_nominal_conv_stride_must_equal_160_samples(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime, model_dir = _installed_runtime(monkeypatch, tmp_path)
    runtime.model.config.conv_stride = [5, 2, 2, 2, 2, 2, 2]

    with (
        pytest.raises(ModelCompatibilityError, match="160 samples") as error,
        LocalHuggingFaceInferenceFactory().aligner_session(model_dir, num_threads=1),
    ):
        raise AssertionError("unreachable")

    assert error.value.context == {
        "session": "aligner",
        "conv_stride": "[5, 2, 2, 2, 2, 2, 2]",
        "nominal_stride_samples": 320,
        "required_stride_samples": 160,
        "sample_rate": 16_000,
    }


def test_d037_chunker_does_not_use_the_aligner_stride_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime, model_dir = _installed_runtime(monkeypatch, tmp_path)
    runtime.model.config.conv_stride = None

    with LocalHuggingFaceInferenceFactory().chunker_session(
        model_dir,
        num_threads=1,
    ) as session:
        assert session.sample_rate == 16_000


def test_d037_aligner_accepts_observed_frame_duration_that_is_not_exactly_10ms(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _, model_dir = _installed_runtime(monkeypatch, tmp_path)
    audio = np.ones(800, dtype=np.float32)

    with LocalHuggingFaceInferenceFactory().aligner_session(
        model_dir,
        num_threads=1,
    ) as session:
        posterior = session.infer(audio, 16_000)

    assert posterior.log_probs.shape[0] == 2
    assert posterior.seconds_per_frame == pytest.approx(0.025)
    assert posterior.seconds_per_frame != pytest.approx(0.01)


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


def test_device_resolution_and_explicit_cuda_validation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from flexaligner.errors import ConfigurationError

    runtime, _ = _installed_runtime(monkeypatch, tmp_path)
    factory = LocalHuggingFaceInferenceFactory()
    assert factory.configure_device(Device.AUTO) == (Device.CPU, None)
    runtime.cuda_available = True
    runtime.cuda_count = 2
    assert factory.configure_device(Device.AUTO) == (Device.CUDA, 0)
    assert factory.configure_device(Device.CUDA, 1) == (Device.CUDA, 1)
    for device, index in [(Device.CUDA, 2), (Device.CPU, 1), (Device.AUTO, 1), (Device.MPS, 0)]:
        with pytest.raises(ConfigurationError):
            factory.configure_device(device, index)


def test_batch_cache_loads_each_kind_once_and_closes_sessions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runtime, model_dir = _installed_runtime(monkeypatch, tmp_path)
    factory = LocalHuggingFaceInferenceFactory()
    sessions = []
    with factory.batch_session():
        for _ in range(2):
            for method in [factory.chunker_session, factory.aligner_session]:
                with method(model_dir, num_threads=1) as session:
                    sessions.append(session)
                    session.infer(np.ones(32, dtype=np.float32), 16_000)
                with pytest.raises(AlignmentError, match="closed"):
                    _ = session.sample_rate
        assert len(runtime.load_calls) == 4
        assert factory.load_counts == {"chunker": 1, "aligner": 1}
    assert not factory._cache
    with factory.chunker_session(model_dir, num_threads=1):
        pass
    assert len(runtime.load_calls) == 6


def test_cuda_sessions_move_only_active_model_and_restore_cpu(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runtime, model_dir = _installed_runtime(monkeypatch, tmp_path)
    runtime.cuda_available, runtime.cuda_count = True, 2
    factory = LocalHuggingFaceInferenceFactory()
    factory.configure_device(Device.CUDA, 1)
    with factory.batch_session():
        for method in [factory.chunker_session, factory.aligner_session, factory.chunker_session]:
            with method(model_dir, num_threads=1) as session:
                session.infer(np.ones(32, dtype=np.float32), 16_000)
    assert runtime.model_devices == [
        "fake-device:cpu",
        "fake-device:cuda:1",
        "fake-device:cpu",
        "fake-device:cpu",
        "fake-device:cuda:1",
        "fake-device:cpu",
        "fake-device:cuda:1",
        "fake-device:cpu",
    ]
    assert runtime.tensor_devices == ["fake-device:cuda:1"] * 3


@pytest.mark.parametrize("failure", ["forward", "move"])
def test_cuda_oom_evicts_resource_and_next_item_reloads(
    failure: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from flexaligner.errors import ResourceLimitError

    runtime, model_dir = _installed_runtime(monkeypatch, tmp_path)
    runtime.cuda_available, runtime.cuda_count = True, 1
    factory = LocalHuggingFaceInferenceFactory()
    factory.configure_device(Device.CUDA)
    oom = sys.modules["torch"].cuda.OutOfMemoryError("oom")
    if failure == "forward":
        runtime.model_error = oom
    else:
        runtime.move_errors = [None, oom]
    with factory.batch_session():
        with (
            pytest.raises(ResourceLimitError),
            factory.chunker_session(model_dir, num_threads=1) as session,
        ):
            session.infer(np.ones(32, dtype=np.float32), 16_000)
        assert not factory._cache
        assert factory._active_kind is None
        runtime.model_error = None
        with factory.chunker_session(model_dir, num_threads=1) as session:
            session.infer(np.ones(32, dtype=np.float32), 16_000)
    assert len(runtime.load_calls) == 4
    assert runtime.model_devices[-1] == "fake-device:cpu"


def test_cuda_restore_failure_is_fatal_and_clears_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from flexaligner.errors import RuntimeRecoveryError

    runtime, model_dir = _installed_runtime(monkeypatch, tmp_path)
    runtime.cuda_available, runtime.cuda_count = True, 1
    runtime.move_errors = [None, None, RuntimeError("cannot return to CPU")]
    factory = LocalHuggingFaceInferenceFactory()
    factory.configure_device(Device.CUDA)
    with (
        pytest.raises(RuntimeRecoveryError),
        factory.batch_session(),
        factory.chunker_session(model_dir, num_threads=1),
    ):
        pass
    assert factory._active_kind is None
    assert not factory._cache


def test_batch_context_is_exclusive(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _, model_dir = _installed_runtime(monkeypatch, tmp_path)
    factory = LocalHuggingFaceInferenceFactory()
    with factory.batch_session():
        with pytest.raises(ModelValidationError), factory.batch_session():
            pass
        with (
            factory.chunker_session(model_dir, num_threads=1),
            pytest.raises(ModelValidationError),
        ):
            factory.configure_device(Device.CPU)


def test_batch_device_configuration_is_idempotent_without_reprobing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runtime, model_dir = _installed_runtime(monkeypatch, tmp_path)
    factory = LocalHuggingFaceInferenceFactory()
    with factory.batch_session():
        assert factory.configure_device(Device.AUTO) == (Device.CPU, None)
        with factory.chunker_session(model_dir, num_threads=1):
            pass
        runtime.cuda_available, runtime.cuda_count = True, 1
        assert factory.configure_device(Device.AUTO) == (Device.CPU, None)
        with factory.chunker_session(model_dir, num_threads=1):
            pass
    assert factory.load_counts == {"chunker": 1, "aligner": 0}


def test_forward_exception_chain_does_not_retain_heavy_locals(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runtime, model_dir = _installed_runtime(monkeypatch, tmp_path)
    runtime.model_error = RuntimeError("forward failed")
    factory = LocalHuggingFaceInferenceFactory()
    with factory.batch_session():
        with (
            pytest.raises(AlignmentError) as caught,
            factory.chunker_session(model_dir, num_threads=1) as session,
        ):
            session.infer(np.ones(32, dtype=np.float32), 16_000)
        assert not factory._cache
    forbidden = {id(runtime.model), id(runtime.processor), id(sys.modules["torch"])}
    error: BaseException | None = caught.value
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        cursor = error.__traceback__
        while cursor is not None:
            frame = cursor.tb_frame
            if Path(frame.f_code.co_filename).resolve() == Path(hf_local.__file__).resolve():
                assert {id(value) for value in frame.f_locals.values()}.isdisjoint(forbidden)
            cursor = cursor.tb_next
        error = error.__cause__ if error.__cause__ is not None else error.__context__


def test_finished_batch_does_not_retain_model_resource_in_session_closure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import weakref

    _, model_dir = _installed_runtime(monkeypatch, tmp_path)
    factory = LocalHuggingFaceInferenceFactory()
    with factory.batch_session():
        context = factory.chunker_session(model_dir, num_threads=1)
        with context as session:
            resource_reference = weakref.ref(next(iter(factory._cache.values())))
        assert resource_reference() is not None
    assert resource_reference() is None
    assert session._model is None
    assert session._processor is None


def test_non_oom_forward_failure_evicts_even_if_caller_catches_inside_session(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runtime, model_dir = _installed_runtime(monkeypatch, tmp_path)
    runtime.model_error = RuntimeError("bad forward")
    factory = LocalHuggingFaceInferenceFactory()
    with factory.batch_session():
        with (
            factory.chunker_session(model_dir, num_threads=1) as session,
            pytest.raises(AlignmentError),
        ):
            session.infer(np.ones(32, dtype=np.float32), 16_000)
        assert not factory._cache


def test_cuda_cache_cleanup_failure_is_fatal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from flexaligner.errors import RuntimeRecoveryError

    runtime, model_dir = _installed_runtime(monkeypatch, tmp_path)
    runtime.cuda_available, runtime.cuda_count = True, 1
    factory = LocalHuggingFaceInferenceFactory()
    factory.configure_device(Device.CUDA)

    def broken_cleanup() -> None:
        raise RuntimeError("allocator failed")

    monkeypatch.setattr(sys.modules["torch"].cuda, "empty_cache", broken_cleanup)
    with (
        pytest.raises(RuntimeRecoveryError, match="allocator"),
        factory.batch_session(),
        factory.chunker_session(model_dir, num_threads=1),
    ):
        pass
    assert factory._active_kind is None
    assert not factory._cache


def test_cache_capacity_is_one_resource_per_stage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _, model_dir = _installed_runtime(monkeypatch, tmp_path)
    second_dir = tmp_path / "second-model"
    second_dir.mkdir()
    factory = LocalHuggingFaceInferenceFactory()
    with factory.batch_session():
        for path in [model_dir, second_dir, model_dir]:
            with factory.chunker_session(path, num_threads=1):
                pass
            assert len(factory._cache) == 1
    assert factory.load_counts == {"chunker": 3, "aligner": 0}


def test_batch_owner_excludes_sessions_from_other_threads(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import threading

    _, model_dir = _installed_runtime(monkeypatch, tmp_path)
    factory = LocalHuggingFaceInferenceFactory()
    errors: list[BaseException] = []

    def other_thread() -> None:
        try:
            with factory.chunker_session(model_dir, num_threads=1):
                raise AssertionError("session should not open")
        except BaseException as exc:
            errors.append(exc)

    with factory.batch_session():
        thread = threading.Thread(target=other_thread)
        thread.start()
        thread.join(timeout=5)
        assert not thread.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], ModelValidationError)
    assert factory.load_counts == {"chunker": 0, "aligner": 0}


@pytest.mark.parametrize("outcome", ["succeeded", "failed"])
def test_engine_close_releases_suspended_batch_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, outcome: str
) -> None:
    import weakref

    from tests.integration._support import FakeInferenceFactory, make_integration_fixture

    from flexaligner import AlignmentOptions, EngineClosedError, FlexAligner, InputValidationError
    from flexaligner.pipeline import AlignmentPipeline

    fixture = make_integration_fixture(tmp_path, metadata=False)
    result = AlignmentPipeline(inference_factory=FakeInferenceFactory()).align(
        request=fixture.request,
        models=fixture.models,
        lexicon_path=fixture.lexicon_path,
        options=AlignmentOptions(device=Device.CPU),
    )
    _, model_dir = _installed_runtime(monkeypatch, tmp_path)
    factory = LocalHuggingFaceInferenceFactory()
    engine = FlexAligner(
        models=fixture.models,
        lexicon_path=fixture.lexicon_path,
        options=AlignmentOptions(device=Device.CPU),
    )
    engine._pipeline = AlignmentPipeline(inference_factory=factory)

    def align_after_loading(request):
        with factory.chunker_session(model_dir, num_threads=1):
            pass
        if outcome == "failed":
            raise InputValidationError("item failed after model use")
        return result

    monkeypatch.setattr(engine, "align", align_after_loading)
    iterator = engine.iter_batch([fixture.request, fixture.request])
    assert next(iterator).status == outcome
    reference = weakref.ref(next(iter(factory._cache.values())))
    assert reference() is not None and factory._batch_owner is not None
    engine.close()
    engine.close()
    assert not factory._cache
    assert factory._batch_owner is None
    assert reference() is None
    # Even an injected align implementation must not run after engine closure.
    try:
        with pytest.raises(EngineClosedError):
            next(iterator)
    finally:
        iterator.close()
    assert not engine._batch_active


def test_factory_close_during_active_session_keeps_exclusion_until_exit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from flexaligner.errors import EngineClosedError

    runtime, model_dir = _installed_runtime(monkeypatch, tmp_path)
    runtime.cuda_available, runtime.cuda_count = True, 1
    factory = LocalHuggingFaceInferenceFactory()
    factory.configure_device(Device.CUDA)
    with factory.batch_session():
        with factory.chunker_session(model_dir, num_threads=1) as session:
            factory.close()
            factory.close()
            assert factory._active_kind == "chunker"
            assert factory._batch_owner is not None
            assert not factory._cache
            with pytest.raises(EngineClosedError):
                factory.configure_device(Device.CPU)
            with pytest.raises(EngineClosedError), factory.batch_session():
                pass
            with (
                pytest.raises(EngineClosedError),
                factory.aligner_session(model_dir, num_threads=1),
            ):
                pass
            assert session._model is not None
        assert factory._active_kind is None
        assert factory._batch_owner is None
        assert session._model is None
    assert runtime.model_devices[-1] == "fake-device:cpu"


@pytest.mark.parametrize("failure", ["restore", "allocator"])
def test_recovery_failure_poison_prevents_factory_reuse(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure: str
) -> None:
    from flexaligner.errors import RuntimeRecoveryError

    runtime, model_dir = _installed_runtime(monkeypatch, tmp_path)
    runtime.cuda_available, runtime.cuda_count = True, 1
    factory = LocalHuggingFaceInferenceFactory()
    factory.configure_device(Device.CUDA)
    if failure == "restore":
        runtime.move_errors = [None, None, RuntimeError("cannot restore")]
    else:

        def fail_cleanup():
            raise RuntimeError("cannot clear allocator")

        monkeypatch.setattr(sys.modules["torch"].cuda, "empty_cache", fail_cleanup)
    with (
        pytest.raises(RuntimeRecoveryError),
        factory.batch_session(),
        factory.chunker_session(model_dir, num_threads=1),
    ):
        pass
    loads = len(runtime.load_calls)
    with pytest.raises(RuntimeRecoveryError, match="new engine"):
        factory.configure_device(Device.CPU)
    with pytest.raises(RuntimeRecoveryError), factory.batch_session():
        pass
    with pytest.raises(RuntimeRecoveryError), factory.chunker_session(model_dir, num_threads=1):
        pass
    assert len(runtime.load_calls) == loads
    factory.close()
    factory.close()
