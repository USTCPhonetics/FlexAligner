"""Offline Hugging Face CTC inference with exclusive CPU/CUDA model sessions.

The optional inference dependencies are deliberately imported only while a
session context is being entered.  Importing :mod:`flexaligner` therefore does
not import PyTorch or Transformers and cannot trigger model discovery.
"""

from __future__ import annotations

import gc
import importlib
import math
import threading
import traceback
from collections.abc import Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType, ModuleType
from typing import Any, cast

import numpy as np

from flexaligner.contracts import Device
from flexaligner.errors import (
    AlignmentError,
    ConfigurationError,
    EngineClosedError,
    ModelCompatibilityError,
    ModelValidationError,
    ResourceLimitError,
    RuntimeRecoveryError,
)
from flexaligner.ports import CtcPosterior, CtcSessionPort, Float32Array

_TARGET_SAMPLE_RATE = 16_000
_ALIGNER_NOMINAL_STRIDE_SAMPLES = 160


@dataclass
class _LocalCtcResource:
    """Validated CPU-owned resources; never exposed directly to callers."""

    kind: str
    torch_module: ModuleType
    processor: Any
    tokenizer: Any
    model: Any
    vocabulary: Mapping[str, int]
    model_vocab_size: int
    sample_rate: int
    pad_token: str | None


class _LocalCtcSession:
    """One validated, explicitly disposable CTC model session."""

    def __init__(
        self,
        resource: _LocalCtcResource,
        device: str,
    ) -> None:
        self._kind = resource.kind
        self._torch: ModuleType | None = resource.torch_module
        self._processor: Any | None = resource.processor
        self._tokenizer: Any | None = resource.tokenizer
        self._model: Any | None = resource.model
        self._vocabulary: Mapping[str, int] | None = resource.vocabulary
        self._model_vocab_size: int | None = resource.model_vocab_size
        self._sample_rate: int | None = resource.sample_rate
        self._pad_token = resource.pad_token
        self._device = device
        self.failed = False

    def _require_open(self) -> None:
        if self._model is None or self._processor is None or self._torch is None:
            raise AlignmentError(
                f"The {self._kind} inference session is closed.",
                context={"session": self._kind},
            )

    @property
    def vocabulary(self) -> Mapping[str, int]:
        self._require_open()
        assert self._vocabulary is not None
        return self._vocabulary

    @property
    def model_vocab_size(self) -> int:
        self._require_open()
        assert self._model_vocab_size is not None
        return self._model_vocab_size

    @property
    def sample_rate(self) -> int:
        self._require_open()
        assert self._sample_rate is not None
        return self._sample_rate

    @property
    def pad_token(self) -> str | None:
        self._require_open()
        return self._pad_token

    def infer(self, audio: Float32Array, sample_rate: int) -> CtcPosterior:
        """Do not let saved inference exceptions keep tensors or models alive."""

        try:
            return self._infer(audio, sample_rate)
        except BaseException as exc:
            self.failed = True
            oom = _is_cuda_oom(exc, self._torch) and self._device.startswith("cuda")
            _clear_exception_traceback_frames(exc)
            if oom:
                raise ResourceLimitError(
                    "CUDA out of memory during inference.",
                    context={"session": self._kind, "device": self._device},
                ) from exc
            raise

    def _infer(self, audio: Float32Array, sample_rate: int) -> CtcPosterior:
        """Return finite contiguous float32 log probabilities for one waveform."""

        self._require_open()
        if not isinstance(audio, np.ndarray):
            raise AlignmentError(
                "Audio must be a NumPy ndarray.",
                context={"session": self._kind},
            )
        if audio.dtype != np.dtype(np.float32):
            raise AlignmentError(
                "Audio must have dtype float32.",
                context={"session": self._kind, "dtype": str(audio.dtype)},
            )
        if audio.ndim != 1:
            raise AlignmentError(
                "Audio must be one-dimensional.",
                context={"session": self._kind, "ndim": int(audio.ndim)},
            )
        if audio.size <= 0:
            raise AlignmentError(
                "Audio must not be empty.",
                context={"session": self._kind},
            )
        if not bool(np.isfinite(audio).all()):
            raise AlignmentError(
                "Audio contains NaN or infinity.",
                context={"session": self._kind},
            )
        if type(sample_rate) is not int or sample_rate != _TARGET_SAMPLE_RATE:
            raise AlignmentError(
                f"Inference requires {_TARGET_SAMPLE_RATE} Hz audio.",
                context={"session": self._kind, "sample_rate": sample_rate},
            )

        torch_module = cast(Any, self._torch)
        processor = cast(Any, self._processor)
        model = cast(Any, self._model)
        device = torch_module.device(self._device)
        waveform = np.ascontiguousarray(audio, dtype=np.float32)

        try:
            with torch_module.inference_mode():
                encoded = processor(
                    waveform,
                    sampling_rate=sample_rate,
                    return_tensors="pt",
                )
                if not isinstance(encoded, Mapping) or not encoded:
                    raise AlignmentError(
                        "Processor returned an empty or invalid input mapping.",
                        context={"session": self._kind},
                    )

                inputs: dict[str, Any] = {}
                for name, value in encoded.items():
                    if not isinstance(name, str):
                        raise AlignmentError(
                            "Processor input names must be strings.",
                            context={"session": self._kind},
                        )
                    move_to_device = getattr(value, "to", None)
                    if not callable(move_to_device):
                        raise AlignmentError(
                            "Processor returned a value that cannot be moved to the inference device.",
                            context={"session": self._kind, "input": name},
                        )
                    inputs[name] = move_to_device(device)

                output = model(**inputs)
                logits = getattr(output, "logits", None)
                logits_array = _tensor_to_numpy(logits, session_kind=self._kind)
                if logits_array.ndim != 3 or logits_array.shape[0] != 1:
                    raise AlignmentError(
                        "Model logits must have shape [1, T, V].",
                        context={"session": self._kind, "shape": str(logits_array.shape)},
                    )
                if logits_array.shape[1] <= 0 or logits_array.shape[2] <= 0:
                    raise AlignmentError(
                        "Model logits must have positive time and vocabulary dimensions.",
                        context={"session": self._kind, "shape": str(logits_array.shape)},
                    )
                if logits_array.shape[2] != self.model_vocab_size:
                    raise ModelCompatibilityError(
                        "Model output vocabulary does not match config.vocab_size.",
                        context={
                            "session": self._kind,
                            "logits_vocab_size": int(logits_array.shape[2]),
                            "model_vocab_size": self.model_vocab_size,
                        },
                    )
                if not bool(np.isfinite(logits_array).all()):
                    raise AlignmentError(
                        "Model logits contain NaN or infinity.",
                        context={"session": self._kind},
                    )

                log_probs_tensor = torch_module.log_softmax(logits, dim=-1)
                log_probs_array = _tensor_to_numpy(
                    log_probs_tensor,
                    session_kind=self._kind,
                )[0]
        except (AlignmentError, ModelCompatibilityError):
            raise
        except Exception as exc:
            raise AlignmentError(
                f"{self._kind.capitalize()} inference failed.",
                context={"session": self._kind},
            ) from exc

        log_probs = np.ascontiguousarray(log_probs_array, dtype=np.float32)
        if log_probs.ndim != 2 or log_probs.shape[0] <= 0:
            raise AlignmentError(
                "Log probabilities must have shape [T, V] with T > 0.",
                context={"session": self._kind, "shape": str(log_probs.shape)},
            )
        if log_probs.shape[1] != self.model_vocab_size:
            raise ModelCompatibilityError(
                "Log-probability vocabulary does not match config.vocab_size.",
                context={
                    "session": self._kind,
                    "log_probs_vocab_size": int(log_probs.shape[1]),
                    "model_vocab_size": self.model_vocab_size,
                },
            )
        if not bool(np.isfinite(log_probs).all()):
            raise AlignmentError(
                "Model produced non-finite log probabilities.",
                context={"session": self._kind},
            )

        seconds_per_frame = (float(audio.size) / float(sample_rate)) / float(log_probs.shape[0])
        if seconds_per_frame <= 0.0 or not math.isfinite(seconds_per_frame):
            raise AlignmentError(
                "Could not derive a finite positive frame duration.",
                context={"session": self._kind},
            )
        return CtcPosterior(log_probs=log_probs, seconds_per_frame=seconds_per_frame)

    def close(self) -> None:
        """Sever every heavyweight reference, including on exceptional exits."""

        self._model = None
        self._processor = None
        self._tokenizer = None
        self._torch = None
        self._vocabulary = None
        self._model_vocab_size = None
        self._sample_rate = None
        self._pad_token = None


def _tensor_to_numpy(value: Any, *, session_kind: str) -> np.ndarray[Any, Any]:
    if value is None:
        raise AlignmentError(
            "Model output has no logits tensor.",
            context={"session": session_kind},
        )
    try:
        detached = value.detach()
        on_cpu = detached.cpu()
        return np.asarray(on_cpu.numpy())
    except Exception as exc:
        raise AlignmentError(
            "Model output could not be converted to a CPU NumPy array.",
            context={"session": session_kind},
        ) from exc


class LocalHuggingFaceInferenceFactory:
    """Create mutually exclusive, local-only Chunker and Aligner sessions."""

    def __init__(self) -> None:
        self._state_lock = threading.Lock()
        self._active_kind: str | None = None
        self._active_session: _LocalCtcSession | None = None
        self._device = Device.CPU
        self._device_index: int | None = None
        self._requested_device: tuple[Device, int] | None = None
        self._batch_owner: int | None = None
        self._cache: dict[tuple[str, Path], _LocalCtcResource] = {}
        self._closed = False
        self._recovery_failure: str | None = None
        self.load_counts: dict[str, int] = {"chunker": 0, "aligner": 0}

    def _require_usable(self) -> None:
        """Check lifecycle under the state lock, without retaining error objects."""

        if self._closed:
            raise EngineClosedError("Local inference factory is closed.")
        if self._recovery_failure is not None:
            raise RuntimeRecoveryError(
                "Inference runtime recovery failed; create a new engine before retrying.",
                context={"reason": self._recovery_failure},
            )

    def close(self) -> None:
        """Release idle resources; an active session retains exclusion until cleanup."""

        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            self._cache.clear()
            if self._active_kind is None:
                self._batch_owner = None
        gc.collect()

    def configure_device(
        self,
        device: Device,
        device_index: int = 0,
    ) -> tuple[Device, int | None]:
        """Resolve one execution device before entering model sessions."""

        with self._state_lock:
            self._require_usable()
            if self._active_kind is not None:
                raise ModelValidationError("Cannot configure devices during an active session.")
            if not isinstance(device, Device) or type(device_index) is not int or device_index < 0:
                raise ConfigurationError("Invalid device or device index.")
            if device in (Device.CPU, Device.AUTO) and device_index != 0:
                raise ConfigurationError("CPU and AUTO only accept device_index=0.")
            if device is Device.MPS:
                raise ConfigurationError("MPS inference is unsupported.")
            if self._batch_owner is not None:
                if self._batch_owner != threading.get_ident():
                    raise ModelValidationError("Cannot configure another thread's active batch.")
                if self._requested_device == (device, device_index):
                    return self._device, self._device_index
                if self._requested_device is not None or self._cache:
                    raise ModelValidationError("Cannot change devices during an active batch.")
            if device is Device.CPU:
                resolved, index = Device.CPU, None
            else:
                try:
                    torch_api = cast(Any, importlib.import_module("torch"))
                    available = bool(torch_api.cuda.is_available())
                    count = int(torch_api.cuda.device_count()) if available else 0
                except Exception as exc:
                    _clear_exception_traceback_frames(exc)
                    raise ModelValidationError(
                        "Device resolution requires the 'inference' optional dependencies.",
                        context={"extra": "inference"},
                    ) from exc
                finally:
                    torch_api = None
                if device is Device.CUDA and (not available or device_index >= count):
                    raise ConfigurationError(
                        "Requested CUDA device is unavailable.",
                        context={"device_index": device_index, "cuda_device_count": count},
                    )
                resolved = Device.CUDA if available and count > 0 else Device.CPU
                index = device_index if resolved is Device.CUDA else None
            self._device, self._device_index = resolved, index
            self._requested_device = device, device_index
            return resolved, index

    @contextmanager
    def batch_session(self) -> Iterator[None]:
        """Cache two CPU model bundles only within an exclusive batch."""

        with self._state_lock:
            self._require_usable()
            if self._batch_owner is not None or self._active_kind is not None:
                raise ModelValidationError(
                    "Batch and single-file inference contexts may not overlap."
                )
            self._batch_owner = threading.get_ident()
        try:
            yield
        finally:
            with self._state_lock:
                self._cache.clear()
                self._batch_owner = None
            gc.collect()

    def chunker_session(
        self,
        model_dir: Path,
        *,
        num_threads: int,
    ) -> AbstractContextManager[CtcSessionPort]:
        return self._open_session("chunker", model_dir, num_threads=num_threads)

    def aligner_session(
        self,
        model_dir: Path,
        *,
        num_threads: int,
    ) -> AbstractContextManager[CtcSessionPort]:
        return self._open_session("aligner", model_dir, num_threads=num_threads)

    @contextmanager
    def _open_session(
        self,
        kind: str,
        model_dir: Path,
        *,
        num_threads: int,
    ) -> Iterator[CtcSessionPort]:
        session: _LocalCtcSession | None = None
        resource: _LocalCtcResource | None = None
        claimed = False
        cache_key: tuple[str, Path] | None = None
        on_cuda = False
        restore_required = False
        try:
            with self._state_lock:
                self._require_usable()
            if type(num_threads) is not int or num_threads <= 0:
                raise ModelValidationError(
                    "num_threads must be a positive integer.",
                    context={"session": kind, "num_threads": num_threads},
                )
            try:
                path = Path(model_dir)
            except Exception as exc:
                raise ModelValidationError(
                    "Model path must be path-like.",
                    context={"session": kind},
                ) from exc
            if not path.is_dir():
                raise ModelValidationError(
                    "Model path must be an existing local directory.",
                    context={"session": kind, "model_dir": str(path)},
                )

            with self._state_lock:
                self._require_usable()
                if self._active_kind is not None or (
                    self._batch_owner is not None and self._batch_owner != threading.get_ident()
                ):
                    raise ModelValidationError(
                        "Chunker and Aligner sessions may not overlap.",
                        context={"active_session": self._active_kind, "requested_session": kind},
                    )
                self._active_kind = kind
                claimed = True

            cache_key = (kind, path.resolve())
            resource = self._cache.get(cache_key)

            try:
                if resource is None:
                    torch_module, transformers_module = _import_inference_dependencies(kind)
                    try:
                        cast(Any, torch_module).set_num_threads(num_threads)
                        resource = _load_local_session(
                            kind=kind,
                            model_dir=path,
                            torch_module=torch_module,
                            transformers_module=transformers_module,
                        )
                        self.load_counts[kind] += 1
                    finally:
                        del torch_module, transformers_module
                    with self._state_lock:
                        self._require_usable()
                        if self._batch_owner is not None:
                            # Bound host RAM to one validated bundle per stage.
                            for previous_key in tuple(self._cache):
                                if previous_key[0] == kind and previous_key != cache_key:
                                    del self._cache[previous_key]
                            self._cache[cache_key] = resource
                else:
                    cast(Any, resource.torch_module).set_num_threads(num_threads)
            except (
                ModelValidationError,
                ModelCompatibilityError,
                AlignmentError,
                EngineClosedError,
                RuntimeRecoveryError,
            ):
                raise
            except Exception as exc:
                raise ModelValidationError(
                    f"Could not initialize the local {kind} model.",
                    context={"session": kind, "model_dir": str(model_dir)},
                ) from exc
            on_cuda = self._device is Device.CUDA
            device_spec = f"cuda:{self._device_index}" if on_cuda else "cpu"
            if on_cuda:
                restore_required = True
                try:
                    _move_resource(resource, device_spec)
                except Exception as exc:
                    self._cache.pop(cache_key, None)
                    _clear_exception_traceback_frames(exc)
                    if _is_cuda_oom(exc, resource.torch_module):
                        raise ResourceLimitError(
                            "CUDA out of memory while moving the model.",
                            context={"session": kind, "device": device_spec},
                        ) from exc
                    raise ModelValidationError(
                        "Could not move the model to CUDA.",
                        context={"session": kind, "device": device_spec},
                    ) from exc
            session = _LocalCtcSession(resource, device_spec)
            with self._state_lock:
                self._require_usable()
                self._active_session = session
            yield session
        except BaseException as exc:
            if cache_key is not None:
                self._cache.pop(cache_key, None)
            _clear_exception_traceback_frames(exc)
            raise
        finally:
            recovery_error: RuntimeRecoveryError | None = None
            if restore_required and resource is not None:
                try:
                    _move_resource(resource, "cpu")
                except Exception as exc:
                    self._cache.clear()
                    _clear_exception_traceback_frames(exc)
                    recovery_error = RuntimeRecoveryError(
                        "Could not restore the CUDA model to CPU; stop this batch.",
                        context={"session": kind, "reason": str(exc)},
                    )
            if session is not None:
                if session.failed and cache_key is not None:
                    self._cache.pop(cache_key, None)
                session.close()
            session = None
            # Clear locals before collection: the generator can remain in a saved traceback.
            torch_for_cleanup = resource.torch_module if resource is not None and on_cuda else None
            resource = None
            gc.collect()
            if torch_for_cleanup is not None:
                try:
                    _empty_cuda_cache(torch_for_cleanup, self._device_index)
                except Exception as exc:
                    self._cache.clear()
                    _clear_exception_traceback_frames(exc)
                    recovery_error = RuntimeRecoveryError(
                        "Could not clear the CUDA allocator; stop this batch.",
                        context={"session": kind, "reason": str(exc)},
                    )
                finally:
                    torch_for_cleanup = None
            if claimed:
                with self._state_lock:
                    if recovery_error is not None:
                        self._recovery_failure = recovery_error.message
                    self._active_session = None
                    self._active_kind = None
                    if self._closed:
                        self._cache.clear()
                        self._batch_owner = None
            if recovery_error is not None:
                raise recovery_error from None


def _move_resource(resource: _LocalCtcResource, device_spec: str) -> None:
    torch_api = cast(Any, resource.torch_module)
    moved = resource.model.to(torch_api.device(device_spec))
    if moved is not None:
        resource.model = moved


def _empty_cuda_cache(torch_module: ModuleType, device_index: int | None) -> None:
    cuda = cast(Any, torch_module).cuda
    with cuda.device(device_index):
        cuda.empty_cache()


def _is_cuda_oom(exc: BaseException, torch_module: ModuleType | None) -> bool:
    cuda = getattr(torch_module, "cuda", None)
    oom_type = getattr(cuda, "OutOfMemoryError", None)
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(oom_type, type) and isinstance(current, oom_type):
            return True
        current = current.__cause__ if current.__cause__ is not None else current.__context__
    return False


def _import_inference_dependencies(kind: str) -> tuple[ModuleType, ModuleType]:
    try:
        torch_module = importlib.import_module("torch")
        transformers_module = importlib.import_module("transformers")
    except Exception as exc:
        raise ModelValidationError(
            "Local inference requires the 'inference' optional dependencies.",
            context={"session": kind, "extra": "inference"},
        ) from exc
    return torch_module, transformers_module


def _load_local_session(
    *,
    kind: str,
    model_dir: Path,
    torch_module: ModuleType,
    transformers_module: ModuleType,
) -> _LocalCtcResource:
    """Load a session without retaining heavyweight locals on failure tracebacks."""

    try:
        return _load_local_session_impl(
            kind=kind,
            model_dir=model_dir,
            torch_module=torch_module,
            transformers_module=transformers_module,
        )
    except BaseException as exc:
        _clear_exception_traceback_frames(exc)
        raise
    finally:
        del torch_module, transformers_module


def _clear_exception_traceback_frames(exc: BaseException) -> None:
    """Clear completed traceback frames while preserving the typed cause chain."""

    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        traceback.clear_frames(current.__traceback__)
        current = current.__cause__ if current.__cause__ is not None else current.__context__


def _load_local_session_impl(
    *,
    kind: str,
    model_dir: Path,
    torch_module: ModuleType,
    transformers_module: ModuleType,
) -> _LocalCtcResource:
    transformers_api = cast(Any, transformers_module)
    torch_api = cast(Any, torch_module)
    load_options = {"local_files_only": True, "trust_remote_code": False}
    try:
        processor = transformers_api.AutoProcessor.from_pretrained(
            str(model_dir),
            **load_options,
        )
        model = transformers_api.AutoModelForCTC.from_pretrained(
            str(model_dir),
            **load_options,
        )
        cpu = torch_api.device("cpu")
        moved_model = model.to(cpu)
        if moved_model is not None:
            model = moved_model
        evaluated_model = model.eval()
        if evaluated_model is not None:
            model = evaluated_model
    except Exception as exc:
        raise ModelValidationError(
            f"Could not load the local {kind} Hugging Face bundle.",
            context={"session": kind, "model_dir": str(model_dir)},
        ) from exc

    feature_extractor = getattr(processor, "feature_extractor", None)
    sample_rate = getattr(feature_extractor, "sampling_rate", None)
    if type(sample_rate) is not int:
        raise ModelValidationError(
            "Processor feature_extractor.sampling_rate must be an integer.",
            context={"session": kind, "sample_rate": str(sample_rate)},
        )
    if sample_rate != _TARGET_SAMPLE_RATE:
        raise ModelCompatibilityError(
            f"Processor must require {_TARGET_SAMPLE_RATE} Hz audio.",
            context={"session": kind, "sample_rate": sample_rate},
        )

    if kind == "aligner":
        _validate_aligner_nominal_stride(model)

    tokenizer = getattr(processor, "tokenizer", None)
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if not callable(get_vocab):
        raise ModelValidationError(
            "Processor tokenizer must provide get_vocab().",
            context={"session": kind},
        )
    try:
        raw_vocabulary = get_vocab()
    except Exception as exc:
        raise ModelValidationError(
            "Tokenizer get_vocab() failed.",
            context={"session": kind},
        ) from exc
    vocabulary = _validate_vocabulary(raw_vocabulary, kind=kind)

    model_vocab_size = getattr(getattr(model, "config", None), "vocab_size", None)
    if type(model_vocab_size) is not int or model_vocab_size <= 0:
        raise ModelValidationError(
            "Model config.vocab_size must be a positive integer.",
            context={"session": kind, "model_vocab_size": str(model_vocab_size)},
        )
    if any(token_id >= model_vocab_size for token_id in vocabulary.values()):
        raise ModelCompatibilityError(
            "Tokenizer vocabulary contains an id outside model config.vocab_size.",
            context={"session": kind, "model_vocab_size": model_vocab_size},
        )

    pad_token = getattr(tokenizer, "pad_token", None)
    if pad_token is not None and not isinstance(pad_token, str):
        raise ModelValidationError(
            "Tokenizer pad_token must be a string or None.",
            context={"session": kind},
        )

    return _LocalCtcResource(
        kind=kind,
        torch_module=torch_module,
        processor=processor,
        tokenizer=tokenizer,
        model=model,
        vocabulary=MappingProxyType(dict(vocabulary)),
        model_vocab_size=model_vocab_size,
        sample_rate=sample_rate,
        pad_token=pad_token,
    )


def _validate_aligner_nominal_stride(model: Any) -> None:
    """Require the reviewed 10 ms Aligner convolution stride at 16 kHz."""

    raw_stride = getattr(getattr(model, "config", None), "conv_stride", None)
    if not isinstance(raw_stride, (list, tuple)) or not raw_stride:
        raise ModelValidationError(
            "Aligner model config.conv_stride must be a non-empty list or tuple.",
            context={"session": "aligner", "conv_stride": str(raw_stride)},
        )

    stride: list[int] = []
    for value in raw_stride:
        if type(value) is not int or value <= 0:
            raise ModelValidationError(
                "Aligner model config.conv_stride must contain positive integers.",
                context={"session": "aligner", "conv_stride": str(raw_stride)},
            )
        stride.append(value)

    nominal_stride_samples = math.prod(stride)
    if nominal_stride_samples != _ALIGNER_NOMINAL_STRIDE_SAMPLES:
        raise ModelCompatibilityError(
            "Aligner model nominal convolution stride must be 160 samples (10 ms at 16 kHz).",
            context={
                "session": "aligner",
                "conv_stride": str(stride),
                "nominal_stride_samples": nominal_stride_samples,
                "required_stride_samples": _ALIGNER_NOMINAL_STRIDE_SAMPLES,
                "sample_rate": _TARGET_SAMPLE_RATE,
            },
        )


def _validate_vocabulary(value: Any, *, kind: str) -> dict[str, int]:
    if not isinstance(value, Mapping) or not value:
        raise ModelValidationError(
            "Tokenizer get_vocab() must return a non-empty mapping.",
            context={"session": kind},
        )

    vocabulary: dict[str, int] = {}
    ids: set[int] = set()
    for token, token_id in value.items():
        if not isinstance(token, str) or type(token_id) is not int or token_id < 0:
            raise ModelValidationError(
                "Tokenizer vocabulary must map strings to non-negative integer ids.",
                context={"session": kind},
            )
        if token_id in ids:
            raise ModelValidationError(
                "Tokenizer vocabulary ids must be unique.",
                context={"session": kind, "token_id": token_id},
            )
        vocabulary[token] = token_id
        ids.add(token_id)
    return vocabulary
