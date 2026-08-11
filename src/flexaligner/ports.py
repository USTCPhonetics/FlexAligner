"""Import-safe ports used to keep orchestration separate from adapters."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from .contracts import (
    AlignmentOptions,
    AlignmentRequest,
    AlignmentResult,
    LocalModelBundle,
)

Float32Array = NDArray[np.float32]


@dataclass(frozen=True, slots=True, kw_only=True)
class CtcPosterior:
    """Validated CTC posterior returned by a local inference session."""

    log_probs: Float32Array
    seconds_per_frame: float


@runtime_checkable
class CtcSessionPort(Protocol):
    """One lazily loaded, CPU-only CTC model session."""

    @property
    def vocabulary(self) -> Mapping[str, int]:
        """Return the tokenizer vocabulary used by the model output."""

        ...

    @property
    def model_vocab_size(self) -> int:
        """Return the model output vocabulary dimension."""

        ...

    @property
    def sample_rate(self) -> int:
        """Return the processor's required sample rate."""

        ...

    @property
    def pad_token(self) -> str | None:
        """Return the tokenizer pad token when declared."""

        ...

    def infer(self, audio: Float32Array, sample_rate: int) -> CtcPosterior:
        """Compute finite CPU log probabilities for one waveform."""

        ...


@runtime_checkable
class LocalInferenceFactoryPort(Protocol):
    """Create non-overlapping, local-only Chunker and Aligner sessions."""

    def chunker_session(
        self,
        model_dir: Path,
        *,
        num_threads: int,
    ) -> AbstractContextManager[CtcSessionPort]:
        """Load one local Chunker session and release it on context exit."""

        ...

    def aligner_session(
        self,
        model_dir: Path,
        *,
        num_threads: int,
    ) -> AbstractContextManager[CtcSessionPort]:
        """Load one local Aligner session and release it on context exit."""

        ...


@runtime_checkable
class AlignmentPipelinePort(Protocol):
    """Future production pipeline boundary used by the lazy public engine."""

    def align(
        self,
        *,
        request: AlignmentRequest,
        models: LocalModelBundle,
        lexicon_path: Path,
        options: AlignmentOptions,
    ) -> AlignmentResult:
        """Align one request or raise a typed FlexAligner error."""

        ...

    def close(self) -> None:
        """Release any resources held by the pipeline."""

        ...


@runtime_checkable
class LocalModelResolverPort(Protocol):
    """Local-only model resolution boundary; downloading is not implemented."""

    def resolve(self, bundle: LocalModelBundle) -> LocalModelBundle:
        """Validate and return a local model bundle without network access."""

        ...
