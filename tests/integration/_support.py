from __future__ import annotations

import json
import wave
from collections.abc import Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from flexaligner.contracts import (
    AlignmentRequest,
    LocalModelBundle,
    TextGridOutput,
)
from flexaligner.errors import AlignmentError, ModelValidationError
from flexaligner.ports import CtcPosterior


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    maximum = np.max(logits, axis=1, keepdims=True)
    normalized = logits - maximum
    return np.ascontiguousarray(
        normalized - np.log(np.sum(np.exp(normalized), axis=1, keepdims=True)),
        dtype=np.float32,
    )


@dataclass(frozen=True, slots=True)
class IntegrationFixture:
    request: AlignmentRequest
    models: LocalModelBundle
    lexicon_path: Path


def make_integration_fixture(tmp_path: Path, *, metadata: bool = True) -> IntegrationFixture:
    audio_path = tmp_path / "sample.wav"
    samples = (np.sin(np.linspace(0.0, 30.0, 16_000)) * 8_000).astype("<i2")
    with wave.open(str(audio_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(samples.tobytes())

    lexicon_path = tmp_path / "english.dict"
    lexicon_path.write_text("alpha AH\nbeta B\n", encoding="utf-8")
    chunker_dir = tmp_path / "chunker"
    aligner_dir = tmp_path / "aligner"
    chunker_dir.mkdir()
    aligner_dir.mkdir()
    (chunker_dir / "vocab.json").write_text(
        json.dumps({"<pad>": 0, "AH": 1, "B": 2}),
        encoding="utf-8",
    )
    output_path = tmp_path / "result.TextGrid"
    metadata_path = tmp_path / "result.chunker.json" if metadata else None
    request = AlignmentRequest(
        audio_path=audio_path,
        transcript="Alpha beta",
        output=TextGridOutput(
            path=output_path,
            chunk_metadata_path=metadata_path,
        ),
        utterance_id="fixture",
    )
    return IntegrationFixture(
        request=request,
        models=LocalModelBundle(
            chunker_dir=chunker_dir,
            aligner_dir=aligner_dir,
        ),
        lexicon_path=lexicon_path,
    )


class FakeCtcSession:
    def __init__(
        self,
        *,
        kind: str,
        trace: list[str],
        fail_infer: bool,
        unreachable: bool,
        chunk_vocabulary: Mapping[str, int] | None = None,
        raw_infer_failure: Exception | None = None,
    ) -> None:
        self.kind = kind
        self.trace = trace
        self.fail_infer = fail_infer
        self.unreachable = unreachable
        self.chunk_vocabulary = chunk_vocabulary
        self.raw_infer_failure = raw_infer_failure

    @property
    def vocabulary(self) -> Mapping[str, int]:
        if self.kind == "chunk":
            return self.chunk_vocabulary or {"<pad>": 0, "AH": 1, "B": 2}
        return {"AH": 0, "B": 1, "sil": 2, "sph": 3}

    @property
    def model_vocab_size(self) -> int:
        return len(self.vocabulary)

    @property
    def sample_rate(self) -> int:
        return 16_000

    @property
    def pad_token(self) -> str | None:
        return "<pad>" if self.kind == "chunk" else None

    def infer(self, audio: np.ndarray, sample_rate: int) -> CtcPosterior:
        del sample_rate
        self.trace.append(f"{self.kind}.infer")
        if self.raw_infer_failure is not None:
            raise self.raw_infer_failure
        if self.fail_infer:
            raise AlignmentError(f"injected {self.kind} inference failure")
        if self.kind == "chunk":
            logits = np.full((6, 3), -12.0, dtype=np.float32)
            logits[:, 0] = 0.0
            logits[1, 1] = 8.0
            logits[4, 2] = 8.0
        elif self.unreachable:
            logits = np.asarray([[8.0, -12.0, -12.0, -12.0]], dtype=np.float32)
        else:
            frame_count = max(2, round((float(audio.size) / 16_000.0) / 0.01))
            split = frame_count // 2
            logits = np.full((frame_count, 4), -12.0, dtype=np.float32)
            logits[:split, 0] = 8.0
            logits[split:, 1] = 8.0
        log_probs = _log_softmax(logits)
        seconds_per_frame = (float(audio.size) / 16_000.0) / float(log_probs.shape[0])
        return CtcPosterior(
            log_probs=np.ascontiguousarray(log_probs),
            seconds_per_frame=seconds_per_frame,
        )


class FakeInferenceFactory:
    def __init__(
        self,
        *,
        fail_enter: str | None = None,
        fail_infer: str | None = None,
        unreachable: bool = False,
        chunk_vocabulary: Mapping[str, int] | None = None,
        raw_infer_failure: Exception | None = None,
    ) -> None:
        self.fail_enter = fail_enter
        self.fail_infer = fail_infer
        self.unreachable = unreachable
        self.chunk_vocabulary = chunk_vocabulary
        self.raw_infer_failure = raw_infer_failure
        self.trace: list[str] = []
        self.active: str | None = None
        self.closed = False

    def chunker_session(
        self,
        model_dir: Path,
        *,
        num_threads: int,
    ) -> AbstractContextManager[FakeCtcSession]:
        del model_dir, num_threads
        return self._session("chunk")

    def aligner_session(
        self,
        model_dir: Path,
        *,
        num_threads: int,
    ) -> AbstractContextManager[FakeCtcSession]:
        del model_dir, num_threads
        return self._session("align")

    @contextmanager
    def _session(self, kind: str) -> Iterator[FakeCtcSession]:
        self.trace.append(f"{kind}.load")
        if self.fail_enter == kind:
            raise ModelValidationError(f"injected {kind} load failure")
        if self.active is not None:
            raise AssertionError(f"overlapping sessions: active={self.active}, requested={kind}")
        self.active = kind
        try:
            yield FakeCtcSession(
                kind=kind,
                trace=self.trace,
                fail_infer=self.fail_infer == kind,
                unreachable=self.unreachable and kind == "align",
                chunk_vocabulary=self.chunk_vocabulary,
                raw_infer_failure=self.raw_infer_failure,
            )
        finally:
            self.trace.append(f"{kind}.close")
            self.active = None

    def close(self) -> None:
        self.trace.append("factory.close")
        self.closed = True
