"""Offline, CPU-only Hugging Face CTC inference adapter.

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
from pathlib import Path
from types import MappingProxyType, ModuleType
from typing import Any, cast

import numpy as np

from flexaligner.errors import AlignmentError, ModelCompatibilityError, ModelValidationError
from flexaligner.ports import CtcPosterior, CtcSessionPort, Float32Array

_TARGET_SAMPLE_RATE = 16_000
_ALIGNER_NOMINAL_STRIDE_SAMPLES = 160


class _LocalCtcSession:
    """One validated, explicitly disposable CTC model session."""

    def __init__(
        self,
        *,
        kind: str,
        torch_module: ModuleType,
        processor: Any,
        tokenizer: Any,
        model: Any,
        vocabulary: Mapping[str, int],
        model_vocab_size: int,
        sample_rate: int,
        pad_token: str | None,
    ) -> None:
        self._kind = kind
        self._torch: ModuleType | None = torch_module
        self._processor: Any | None = processor
        self._tokenizer: Any | None = tokenizer
        self._model: Any | None = model
        self._vocabulary: Mapping[str, int] | None = MappingProxyType(dict(vocabulary))
        self._model_vocab_size: int | None = model_vocab_size
        self._sample_rate: int | None = sample_rate
        self._pad_token = pad_token

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
        cpu = torch_module.device("cpu")
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
                    move_to_cpu = getattr(value, "to", None)
                    if not callable(move_to_cpu):
                        raise AlignmentError(
                            "Processor returned a value that cannot be moved to CPU.",
                            context={"session": self._kind, "input": name},
                        )
                    inputs[name] = move_to_cpu(cpu)

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
        claimed = False
        try:
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
                if self._active_kind is not None:
                    raise ModelValidationError(
                        "Chunker and Aligner sessions may not overlap.",
                        context={"active_session": self._active_kind, "requested_session": kind},
                    )
                self._active_kind = kind
                claimed = True

            try:
                torch_module, transformers_module = _import_inference_dependencies(kind)
                try:
                    torch_api = cast(Any, torch_module)
                    torch_api.set_num_threads(num_threads)
                    session = _load_local_session(
                        kind=kind,
                        model_dir=path,
                        torch_module=torch_module,
                        transformers_module=transformers_module,
                    )
                finally:
                    del torch_api, torch_module, transformers_module
            except (ModelValidationError, ModelCompatibilityError, AlignmentError):
                raise
            except Exception as exc:
                raise ModelValidationError(
                    f"Could not initialize the local {kind} model.",
                    context={"session": kind, "model_dir": str(model_dir)},
                ) from exc
            with self._state_lock:
                self._active_session = session
            yield session
        finally:
            if session is not None:
                session.close()
            session = None
            if claimed:
                with self._state_lock:
                    self._active_session = None
                    self._active_kind = None
            gc.collect()


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
) -> _LocalCtcSession:
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
) -> _LocalCtcSession:
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

    return _LocalCtcSession(
        kind=kind,
        torch_module=torch_module,
        processor=processor,
        tokenizer=tokenizer,
        model=model,
        vocabulary=vocabulary,
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
