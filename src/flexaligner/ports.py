"""Import-safe ports used to keep orchestration separate from adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from .contracts import (
    AlignmentOptions,
    AlignmentRequest,
    AlignmentResult,
    LocalModelBundle,
)


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
