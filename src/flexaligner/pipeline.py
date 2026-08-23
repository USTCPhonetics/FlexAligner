"""English CPU single-file alignment orchestration.

The pipeline keeps file/model adapters outside the NumPy cores and deliberately
loads the Chunker and Aligner in two non-overlapping context managers.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import replace
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from .adapters.lexicon_file import (
    PronouncingLexicon,
    TokenVocabulary,
    load_dense_token_vocab,
    load_lexicon,
    normalize_transcript,
    validate_aligner_vocabulary,
    validate_local_model_dir,
    validate_transcript_lexicon,
)
from .adapters.wav_pcm16 import DecodedAudio, load_strict_pcm16_wav
from .contracts import (
    AlignmentOptions,
    AlignmentRequest,
    AlignmentResult,
    ChunkResult,
    Device,
    Language,
    LocalModelBundle,
    PhoneInterval,
    RunProvenance,
    Score,
    ScoreKind,
    WordInterval,
)
from .core.stage1 import (
    RuntimeChunk,
    WordSpan,
    attach_phone_confidence_from_points,
    backtrace,
    build_chunk_lexicon,
    build_trellis,
    choose_greedy_pronunciations,
    emission_frames_by_token_index,
    make_word_anchors_from_emissions,
    merge_word_anchors_into_chunks,
    phones_to_word_segments_by_offsets,
    points_to_segments,
    round_chunks_to_legacy_grid,
    word_phone_token_ranges,
    word_segments_with_confidence,
)
from .core.stage2 import (
    BeamWorkBudget,
    Stage2DecodeConfig,
    align_beam_viterbi,
    build_phone_graph_optional_sil_sph,
    redecode_with_pruned_fixed_sequence,
)
from .errors import (
    AlignmentError,
    ArtifactExistsError,
    ConfigurationError,
    EngineClosedError,
    FlexAlignerError,
    InputValidationError,
    InternalError,
    ModelCompatibilityError,
    ResourceLimitError,
    UnreachableAlignmentError,
)
from .ports import CtcPosterior, LocalInferenceFactoryPort
from .textgrid import (
    Interval,
    IntervalTier,
    LocalAlignment,
    TextGridDocument,
    merge_local_alignments,
    validate_textgrid_structure,
    write_validated_artifacts,
)

CHUNK_VOCAB_FILENAME = "vocab.json"
TARGET_SAMPLE_RATE = 16_000
ANCHOR_PAD_SECONDS = 0.3
ANCHOR_MERGE_GAP_SECONDS = 0.2
SILENCE_COST = -0.5
SILENCE_ENTER_COST = -0.5
SPEECH_GAP_COST = -2.0
SPEECH_GAP_ENTER_COST = -3.0


class AlignmentPipeline:
    """Run the implemented local English CPU alignment path."""

    def __init__(
        self,
        *,
        inference_factory: LocalInferenceFactoryPort | None = None,
    ) -> None:
        self._inference_factory = inference_factory
        self._closed = False

    def align(
        self,
        *,
        request: AlignmentRequest,
        models: LocalModelBundle,
        lexicon_path: Path,
        options: AlignmentOptions,
    ) -> AlignmentResult:
        """Align one request, or leave every official artifact absent."""

        self._ensure_open()
        _validate_pipeline_contracts(request, models, lexicon_path, options)
        _preflight_output_paths(
            request.output.path,
            request.output.chunk_metadata_path,
        )

        words = normalize_transcript(request.transcript)
        _reject_reserved_words(words)
        _check_transcript_limit(words, options)
        lexicon = load_lexicon(lexicon_path)
        validate_transcript_lexicon(words, lexicon)
        audio = load_strict_pcm16_wav(request.audio_path, options.limits)

        chunker_dir = validate_local_model_dir(models.chunker_dir, "chunker model")
        aligner_dir = validate_local_model_dir(models.aligner_dir, "aligner model")
        if models.manifest_path is not None and not models.manifest_path.is_file():
            raise InputValidationError(
                f"Model manifest is not a file: {models.manifest_path}",
                context={"path": str(models.manifest_path)},
            )
        chunk_vocabulary = load_dense_token_vocab(chunker_dir / CHUNK_VOCAB_FILENAME)
        utterance_id = request.utterance_id or request.audio_path.stem or "utterance"

        try:
            chunks, word_spans = self._run_chunker(
                audio=audio,
                words=words,
                lexicon=lexicon,
                vocabulary=chunk_vocabulary,
                model_dir=chunker_dir,
                utterance_id=utterance_id,
                options=options,
            )
            local_alignments = self._run_aligner(
                audio=audio,
                words=words,
                lexicon=lexicon,
                chunks=chunks,
                model_dir=aligner_dir,
                options=options,
            )
            stage2_config = Stage2DecodeConfig()
            merged = merge_local_alignments(
                chunks=chunks,
                local_alignments=local_alignments,
                full_duration_s=audio.duration_s,
                expected_words=words,
                word_sil_label=stage2_config.word_sil_label,
                sph_word_label=stage2_config.sph_word_label,
            )
        except FlexAlignerError:
            raise
        except (KeyError, TypeError, ValueError, RuntimeError) as error:
            raise AlignmentError(
                "The alignment core rejected the validated request.",
                context={"utterance_id": utterance_id, "reason": str(error)},
            ) from error
        except Exception as error:
            raise InternalError(
                "An unexpected alignment pipeline failure occurred.",
                context={"utterance_id": utterance_id, "error_type": type(error).__name__},
            ) from error

        raw_scores = _scores_from_word_spans(word_spans)
        metadata = _chunk_metadata(words, word_spans, raw_scores)
        prepared_result = _public_result(
            request=request,
            options=options,
            utterance_id=utterance_id,
            audio=audio,
            words=words,
            chunks=chunks,
            word_spans=word_spans,
            raw_scores=raw_scores,
            merged=merged,
            output_sha256="0" * 64,
            config=stage2_config,
        )
        output_sha256 = write_validated_artifacts(
            textgrid=merged,
            output_path=request.output.path,
            expected_words=words,
            word_sil_label=stage2_config.word_sil_label,
            sph_word_label=stage2_config.sph_word_label,
            metadata_path=request.output.chunk_metadata_path,
            metadata=(metadata if request.output.chunk_metadata_path is not None else None),
        )
        return replace(prepared_result, output_sha256=output_sha256)

    def _run_chunker(
        self,
        *,
        audio: DecodedAudio,
        words: Sequence[str],
        lexicon: PronouncingLexicon,
        vocabulary: TokenVocabulary,
        model_dir: Path,
        utterance_id: str,
        options: AlignmentOptions,
    ) -> tuple[list[RuntimeChunk], list[WordSpan]]:
        factory = self._get_inference_factory()
        with factory.chunker_session(
            model_dir,
            num_threads=options.num_threads,
        ) as session:
            if session.sample_rate != audio.sample_rate:
                raise ModelCompatibilityError(
                    "Chunker processor sample rate does not match decoded audio.",
                    context={
                        "processor_sample_rate": session.sample_rate,
                        "audio_sample_rate": audio.sample_rate,
                    },
                )
            if session.model_vocab_size != len(vocabulary.token_to_id):
                raise ModelCompatibilityError(
                    "Chunker model vocabulary size does not match vocab.json.",
                    context={
                        "model_vocab_size": session.model_vocab_size,
                        "vocabulary_size": len(vocabulary.token_to_id),
                    },
                )
            session_vocabulary = dict(session.vocabulary)
            expected_vocabulary = dict(vocabulary.token_to_id)
            if session_vocabulary != expected_vocabulary:
                mismatch = _first_vocabulary_mismatch(
                    expected_vocabulary,
                    session_vocabulary,
                )
                raise ModelCompatibilityError(
                    "Chunker tokenizer vocabulary does not match vocab.json.",
                    context={
                        "token": mismatch[0],
                        "vocab_json_id": mismatch[1],
                        "tokenizer_id": mismatch[2],
                    },
                )
            blank_id = _resolve_blank_id(vocabulary.token_to_id, session.pad_token)
            posterior = session.infer(audio.samples, audio.sample_rate)
            _validate_posterior_vocabulary(
                posterior,
                expected_size=session.model_vocab_size,
                role="chunker",
            )
            return _stage1_from_posterior(
                posterior=posterior,
                audio=audio,
                words=words,
                lexicon=lexicon,
                vocabulary=vocabulary.token_to_id,
                blank_id=blank_id,
                utterance_id=utterance_id,
                options=options,
            )

    def _run_aligner(
        self,
        *,
        audio: DecodedAudio,
        words: Sequence[str],
        lexicon: PronouncingLexicon,
        chunks: Sequence[RuntimeChunk],
        model_dir: Path,
        options: AlignmentOptions,
    ) -> list[LocalAlignment]:
        del words
        factory = self._get_inference_factory()
        with factory.aligner_session(
            model_dir,
            num_threads=options.num_threads,
        ) as session:
            if session.sample_rate != audio.sample_rate:
                raise ModelCompatibilityError(
                    "Aligner processor sample rate does not match decoded audio.",
                    context={
                        "processor_sample_rate": session.sample_rate,
                        "audio_sample_rate": audio.sample_rate,
                    },
                )
            vocabulary = TokenVocabulary(token_to_id=dict(session.vocabulary))
            all_chunk_words = tuple(word for chunk in chunks for word in chunk.words)
            validate_aligner_vocabulary(
                all_chunk_words,
                lexicon,
                vocabulary,
                session.model_vocab_size,
            )
            config = Stage2DecodeConfig()
            _validate_special_phones(vocabulary, session.model_vocab_size, config)
            beam_work_budget = BeamWorkBudget(limit=options.limits.max_beam_work_units)

            local_alignments: list[LocalAlignment] = []
            for chunk in chunks:
                waveform = np.ascontiguousarray(
                    audio.samples[chunk.start_sample : chunk.end_sample],
                    dtype=np.float32,
                )
                posterior = session.infer(waveform, audio.sample_rate)
                _validate_posterior_vocabulary(
                    posterior,
                    expected_size=session.model_vocab_size,
                    role="aligner",
                )
                local_alignments.append(
                    _stage2_from_posterior(
                        posterior=posterior,
                        words=chunk.words,
                        lexicon=lexicon,
                        vocabulary=vocabulary.token_to_id,
                        config=config,
                        context=f"local chunk_id={chunk.chunk_id}",
                        beam_work_budget=beam_work_budget,
                        max_graph_states=options.limits.max_stage2_graph_states,
                    )
                )
            return local_alignments

    def _get_inference_factory(self) -> LocalInferenceFactoryPort:
        if self._inference_factory is None:
            from .adapters.hf_local import LocalHuggingFaceInferenceFactory

            self._inference_factory = LocalHuggingFaceInferenceFactory()
        return self._inference_factory

    def close(self) -> None:
        """Close the pipeline and any injectable factory hook, idempotently."""

        if self._closed:
            return
        self._closed = True
        factory = self._inference_factory
        close = getattr(factory, "close", None)
        if callable(close):
            close()
        self._inference_factory = None

    def _ensure_open(self) -> None:
        if self._closed:
            raise EngineClosedError("Alignment pipeline is closed")


def _validate_pipeline_contracts(
    request: AlignmentRequest,
    models: LocalModelBundle,
    lexicon_path: Path,
    options: AlignmentOptions,
) -> None:
    if not isinstance(request, AlignmentRequest):
        raise ConfigurationError("request must be an AlignmentRequest")
    if not isinstance(models, LocalModelBundle):
        raise ConfigurationError("models must be a LocalModelBundle")
    if not isinstance(lexicon_path, Path):
        raise ConfigurationError("lexicon_path must be a pathlib.Path")
    if not isinstance(options, AlignmentOptions):
        raise ConfigurationError("options must be AlignmentOptions")
    if options.language is not Language.EN or options.device is not Device.CPU:
        raise ConfigurationError("Pipeline accepts only the guarded English CPU profile")
    if not isinstance(request.audio_path, Path):
        raise ConfigurationError("request.audio_path must be a pathlib.Path")
    if not isinstance(request.transcript, str):
        raise ConfigurationError("request.transcript must be a string")
    if not isinstance(request.output.path, Path):
        raise ConfigurationError("request.output.path must be a pathlib.Path")
    metadata_path = request.output.chunk_metadata_path
    if metadata_path is not None and not isinstance(metadata_path, Path):
        raise ConfigurationError("chunk_metadata_path must be a pathlib.Path or None")


def _preflight_output_paths(output_path: Path, metadata_path: Path | None) -> None:
    official_paths = [output_path]
    if metadata_path is not None:
        official_paths.append(metadata_path)
    paths: list[tuple[Path, str]] = []
    for official in official_paths:
        paths.append((official, "official"))
        paths.append((official.with_name(official.name + ".tmp"), "temporary"))
    normalized = [os.path.abspath(path) for path, _role in paths]
    if len(normalized) != len(set(normalized)):
        raise ConfigurationError("Official and temporary output paths must be distinct")
    for path, role in paths:
        if os.path.lexists(path):
            raise ArtifactExistsError(
                f"{role.capitalize()} output already exists: {path}",
                context={"path": str(path), "role": role},
            )


def _check_transcript_limit(words: Sequence[str], options: AlignmentOptions) -> None:
    limits = options.limits
    if (
        limits is not None
        and limits.max_transcript_words is not None
        and len(words) > limits.max_transcript_words
    ):
        raise ResourceLimitError(
            "Transcript word limit exceeded before model loading",
            context={"words": len(words), "limit": limits.max_transcript_words},
        )


def _reject_reserved_words(words: Sequence[str]) -> None:
    reserved = {"null", Stage2DecodeConfig().word_sil_label.strip().lower()}
    for word_index, word in enumerate(words):
        if word.strip().lower() in reserved:
            raise InputValidationError(
                "Transcript contains a label reserved by the alignment tiers.",
                context={"word_index": word_index, "word": word},
            )


def _resolve_blank_id(vocabulary: Mapping[str, int], pad_token: str | None) -> int:
    candidates = ["<pad>"]
    if pad_token:
        candidates.append(pad_token)
    candidates.extend(("[PAD]", "<blank>"))
    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate in vocabulary:
            return vocabulary[candidate]
    raise ModelCompatibilityError(
        "Could not resolve the Chunker blank token.",
        context={"candidates": repr(candidates), "vocabulary_sample": repr(list(vocabulary)[:30])},
    )


def _first_vocabulary_mismatch(
    expected: Mapping[str, int],
    actual: Mapping[str, int],
) -> tuple[str, str, str]:
    for token in sorted(set(expected) | set(actual)):
        expected_id = expected.get(token)
        actual_id = actual.get(token)
        if expected_id != actual_id:
            return token, repr(expected_id), repr(actual_id)
    raise AssertionError("Vocabulary mismatch requested for equal mappings")


def _validate_posterior_vocabulary(
    posterior: CtcPosterior,
    *,
    expected_size: int,
    role: str,
) -> None:
    if not isinstance(posterior, CtcPosterior):
        raise ModelCompatibilityError(
            f"{role.capitalize()} session returned an invalid posterior record."
        )
    log_probs = posterior.log_probs
    if not isinstance(log_probs, np.ndarray):
        raise ModelCompatibilityError(
            f"{role.capitalize()} posterior log_probs must be a NumPy array."
        )
    if log_probs.ndim != 2 or log_probs.shape[0] <= 0 or log_probs.shape[1] != expected_size:
        raise ModelCompatibilityError(
            f"{role.capitalize()} posterior vocabulary mismatch.",
            context={
                "shape": str(log_probs.shape),
                "expected_vocabulary_size": expected_size,
            },
        )
    if not np.issubdtype(log_probs.dtype, np.floating):
        raise ModelCompatibilityError(
            f"{role.capitalize()} posterior log_probs must have a floating dtype.",
            context={"dtype": str(log_probs.dtype)},
        )
    if not bool(np.isfinite(log_probs).all()):
        raise ModelCompatibilityError(f"{role.capitalize()} posterior contains NaN or infinity.")
    if bool((log_probs > 1.0e-5).any()):
        raise ModelCompatibilityError(
            f"{role.capitalize()} posterior contains positive log probabilities."
        )
    row_maximum = np.max(log_probs, axis=1)
    row_logsumexp = row_maximum + np.log(np.sum(np.exp(log_probs - row_maximum[:, None]), axis=1))
    if not bool(np.allclose(row_logsumexp, 0.0, rtol=0.0, atol=1.0e-4)):
        raise ModelCompatibilityError(
            f"{role.capitalize()} posterior rows are not normalized log probabilities."
        )
    if not math.isfinite(posterior.seconds_per_frame) or posterior.seconds_per_frame <= 0:
        raise ModelCompatibilityError(
            f"{role.capitalize()} posterior has an invalid frame duration."
        )


def _stage1_from_posterior(
    *,
    posterior: CtcPosterior,
    audio: DecodedAudio,
    words: Sequence[str],
    lexicon: PronouncingLexicon,
    vocabulary: Mapping[str, int],
    blank_id: int,
    utterance_id: str,
    options: AlignmentOptions,
) -> tuple[list[RuntimeChunk], list[WordSpan]]:
    chunk_lexicon = build_chunk_lexicon(lexicon.entries)
    pronunciation = choose_greedy_pronunciations(
        words,
        chunk_lexicon,
        vocabulary,
        inter_word_token=None,
    )
    limits = options.limits
    if (
        limits is not None
        and limits.max_phone_tokens is not None
        and len(pronunciation.phones) > limits.max_phone_tokens
    ):
        raise ResourceLimitError(
            "Phone-token limit exceeded before trellis allocation",
            context={"phones": len(pronunciation.phones), "limit": limits.max_phone_tokens},
        )
    phone_ids = [vocabulary[phone] for phone in pronunciation.phones]
    trellis = build_trellis(
        posterior.log_probs,
        phone_ids,
        blank_id,
        max_trellis_cells=(limits.max_trellis_cells if limits is not None else None),
    )
    points = backtrace(trellis, posterior.log_probs, phone_ids, blank_id)
    phone_segments = points_to_segments(points, pronunciation.phones)
    phone_segments_with_confidence = attach_phone_confidence_from_points(
        phone_segments,
        points,
        pronunciation.phones,
        vocabulary,
        posterior.log_probs,
        mode="emission",
    )
    token_ranges = word_phone_token_ranges(
        phone_segments,
        words,
        pronunciation.chosen_prons,
        inter_word_token=None,
    )
    word_segments = phones_to_word_segments_by_offsets(
        phone_segments,
        words,
        pronunciation.chosen_prons,
        inter_word_token=None,
    )
    word_segments_with_scores = word_segments_with_confidence(
        word_segments,
        phone_segments_with_confidence,
    )
    if [segment.label for segment in word_segments_with_scores] != list(words):
        raise AlignmentError("Stage 1 word reconstruction changed transcript order")

    word_spans = [
        WordSpan(
            word_index=word_index,
            word=segment.label,
            start_frame=segment.start_frame,
            end_frame=segment.end_frame,
            start_s=float(segment.start_frame) * posterior.seconds_per_frame,
            end_s=float(segment.end_frame) * posterior.seconds_per_frame,
            conf_log=float(segment.conf_log),
            pron=list(pronunciation.chosen_prons[word_index]),
        )
        for word_index, segment in enumerate(word_segments_with_scores)
    ]
    _validate_word_spans(word_spans)
    emission_frames = emission_frames_by_token_index(points, len(pronunciation.phones))
    anchors = make_word_anchors_from_emissions(
        word_spans,
        token_ranges,
        emission_frames,
        spf=posterior.seconds_per_frame,
        anchor_pad_s=ANCHOR_PAD_SECONDS,
        audio_dur_s=audio.duration_s,
    )
    raw_chunks = merge_word_anchors_into_chunks(
        anchors,
        anchor_merge_gap_s=ANCHOR_MERGE_GAP_SECONDS,
    )
    chunks = round_chunks_to_legacy_grid(
        raw_chunks=raw_chunks,
        utt_id=utterance_id,
        words=words,
        num_samples=int(audio.samples.size),
        sample_rate=audio.sample_rate,
    )
    return chunks, word_spans


def _validate_word_spans(word_spans: Sequence[WordSpan]) -> None:
    previous_end = -1.0
    previous_frame = -1
    for span in word_spans:
        if span.end_s <= span.start_s or span.end_frame <= span.start_frame:
            raise AlignmentError("Stage 1 produced a non-positive word span")
        if span.start_s < previous_end - 1.0e-6 or span.start_frame < previous_frame:
            raise AlignmentError("Stage 1 produced overlapping or backward word spans")
        previous_end = span.end_s
        previous_frame = span.start_frame


def _validate_special_phones(
    vocabulary: TokenVocabulary,
    model_vocab_size: int,
    config: Stage2DecodeConfig,
) -> None:
    for role, phone in (("sil_phone", config.sil_phone), ("sph_phone", config.sph_phone)):
        if phone not in vocabulary.token_to_id:
            raise ModelCompatibilityError(
                f"{role}={phone!r} is not in the Aligner vocabulary.",
                context={"role": role, "phone": phone},
            )
        phone_id = vocabulary.token_to_id[phone]
        if phone_id < 0 or phone_id >= model_vocab_size:
            raise ModelCompatibilityError(
                f"{role} has an ID outside the Aligner output range.",
                context={"role": role, "phone": phone, "phone_id": phone_id},
            )


def _stage2_from_posterior(
    *,
    posterior: CtcPosterior,
    words: Sequence[str],
    lexicon: PronouncingLexicon,
    vocabulary: Mapping[str, int],
    config: Stage2DecodeConfig,
    context: str,
    beam_work_budget: BeamWorkBudget,
    max_graph_states: int,
) -> LocalAlignment:
    silence_id = vocabulary[config.sil_phone]
    speech_gap_id = vocabulary[config.sph_phone]
    graph, entry_bias = build_phone_graph_optional_sil_sph(
        words=words,
        lexicon=lexicon,
        phone_to_id=vocabulary,
        sil_phone=config.sil_phone,
        optional_sil_between_words=True,
        optional_sil_at_start=None,
        optional_sil_at_end=None,
        sil_cost=SILENCE_COST,
        sph_phone=config.sph_phone,
        optional_sph_between_words=True,
        optional_sph_at_start=None,
        optional_sph_at_end=None,
        sph_cost=SPEECH_GAP_COST,
        sph_word_label=config.sph_word_label,
        max_graph_states=max_graph_states,
    )
    try:
        first_pass = align_beam_viterbi(
            logp=posterior.log_probs,
            graph=graph,
            entry_bias=entry_bias,
            p_stay=config.p_stay,
            beam_size=config.beam,
            word_sil_label=config.word_sil_label,
            boundary_lambda=config.boundary_lambda,
            boundary_context_s=config.boundary_context_s,
            frame_hop_s=config.frame_hop_s,
            sil_phone_id=silence_id,
            min_sil_dur_ms=0.0,
            sil_enter_cost=SILENCE_ENTER_COST,
            sph_phone_id=speech_gap_id,
            sph_enter_cost=SPEECH_GAP_ENTER_COST,
            beam_work_budget=beam_work_budget,
        )
        aligned, stats = redecode_with_pruned_fixed_sequence(
            first_pass_ali=first_pass,
            first_pass_graph=graph,
            first_pass_entry_bias=entry_bias,
            logp=posterior.log_probs,
            sil_phone=config.sil_phone,
            sil_phone_id=silence_id,
            sph_phone=config.sph_phone,
            sph_phone_id=speech_gap_id,
            config=config,
            beam_work_budget=beam_work_budget,
        )
    except RuntimeError as error:
        if "Viterbi failed to reach any end state" in str(error):
            raise UnreachableAlignmentError(
                "Stage 2 could not reach a complete end state.",
                context={"context": context},
            ) from error
        raise

    ignored = {
        "",
        config.word_sil_label.strip().lower(),
        config.sph_word_label.strip().lower(),
        "null",
    }
    actual_words = [
        label
        for label, _start, _end in aligned.word_segments_f
        if label.strip().lower() not in ignored
    ]
    if actual_words != list(words):
        raise AlignmentError(
            "Stage 2 word sequence does not match the chunk transcript.",
            context={"context": context},
        )

    duration_s = float(posterior.log_probs.shape[0]) * config.frame_hop_s
    textgrid = TextGridDocument(
        xmin=0.0,
        xmax=duration_s,
        tiers=(
            IntervalTier(
                name="phones",
                xmin=0.0,
                xmax=duration_s,
                intervals=tuple(
                    Interval(
                        xmin=float(start) * config.frame_hop_s,
                        xmax=float(end) * config.frame_hop_s,
                        text=label,
                        word_index=word_index,
                        pronunciation_index=pronunciation_index,
                        phone_index=phone_index,
                    )
                    for (
                        label,
                        start,
                        end,
                        word_index,
                        pronunciation_index,
                        phone_index,
                    ) in aligned.phone_provenance_f
                ),
            ),
            IntervalTier(
                name="words",
                xmin=0.0,
                xmax=duration_s,
                intervals=tuple(
                    Interval(
                        xmin=float(start) * config.frame_hop_s,
                        xmax=float(end) * config.frame_hop_s,
                        text=label,
                    )
                    for label, start, end in aligned.word_segments_f
                ),
            ),
        ),
    )
    validate_textgrid_structure(textgrid, context=context)
    return LocalAlignment(textgrid=textgrid, redecode_stats=stats)


def _scores_from_word_spans(word_spans: Sequence[WordSpan]) -> tuple[Score, ...]:
    return tuple(
        Score(
            value=span.conf_prob,
            kind=ScoreKind.CHUNKER_EMISSION_GEOMETRIC_MEAN,
            calibrated=False,
        )
        for span in word_spans
    )


def _chunk_metadata(
    words: Sequence[str],
    word_spans: Sequence[WordSpan],
    scores: Sequence[Score],
) -> dict[str, object]:
    if len(words) != len(word_spans) or len(words) != len(scores):
        raise AlignmentError("Chunk metadata inputs do not have one record per word")
    return {
        "schema_version": "1",
        "score_kind": ScoreKind.CHUNKER_EMISSION_GEOMETRIC_MEAN.value,
        "calibrated": False,
        "words": [
            {
                "word_index": word_index,
                "word": word,
                "value": scores[word_index].value,
                "log_value": word_spans[word_index].conf_log,
                "chunker_pronunciation": list(word_spans[word_index].pron),
            }
            for word_index, word in enumerate(words)
        ],
    }


def _public_result(
    *,
    request: AlignmentRequest,
    options: AlignmentOptions,
    utterance_id: str,
    audio: DecodedAudio,
    words: Sequence[str],
    chunks: Sequence[RuntimeChunk],
    word_spans: Sequence[WordSpan],
    raw_scores: tuple[Score, ...],
    merged: TextGridDocument,
    output_sha256: str,
    config: Stage2DecodeConfig,
) -> AlignmentResult:
    del word_spans
    ignored = {
        "",
        "null",
        config.word_sil_label.strip().lower(),
        config.sph_word_label.strip().lower(),
    }
    public_words: list[WordInterval] = []
    next_word_index = 0
    for interval in merged.tiers[1].intervals:
        normalized_label = interval.text.strip().lower()
        word_index: int | None = None
        if normalized_label not in ignored:
            word_index = next_word_index
            next_word_index += 1
        public_words.append(
            WordInterval(
                label=interval.text,
                start_s=interval.xmin,
                end_s=interval.xmax,
                word_index=word_index,
            )
        )
    if next_word_index != len(words):
        raise AlignmentError("Public word intervals lost transcript word identity")
    public_phones = tuple(
        PhoneInterval(
            label=interval.text,
            start_s=interval.xmin,
            end_s=interval.xmax,
            word_index=interval.word_index,
            pronunciation_index=interval.pronunciation_index,
            phone_index=interval.phone_index,
        )
        for interval in merged.tiers[0].intervals
    )
    try:
        package_version = version("flexaligner")
    except PackageNotFoundError:
        package_version = "0+unknown"
    return AlignmentResult(
        utterance_id=utterance_id,
        audio_duration_s=audio.duration_s,
        normalized_words=tuple(words),
        words=tuple(public_words),
        phones=public_phones,
        chunks=tuple(
            ChunkResult(
                chunk_id=chunk.chunk_id,
                start_s=chunk.start_s,
                end_s=chunk.end_s,
                word_indices=tuple(chunk.word_indices),
            )
            for chunk in chunks
        ),
        raw_scores=raw_scores,
        calibrated_scores=None,
        output_path=request.output.path,
        output_sha256=output_sha256,
        provenance=RunProvenance(
            package_version=package_version,
            algorithm_profile=options.algorithm_profile,
            language=options.language,
            device=options.device,
            model_fingerprints=(),
        ),
    )


__all__ = ["AlignmentPipeline"]
