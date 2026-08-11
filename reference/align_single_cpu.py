#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single-utterance, CPU-only FlexAligner reference pipeline.

Direct dependencies: torch, transformers, numpy.
Input audio contract: 16 kHz, mono, uncompressed PCM16 WAV.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import re
import sys
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import torch
from transformers import AutoModelForCTC, AutoProcessor

NEG_INF = -1e30
EPS = 1e-6
TARGET_SAMPLE_RATE = 16000


def normalize_word(w: str) -> str:
    """Match chunks2.py-style normalization: lowercase and trim edge punctuation."""
    w = w.strip().lower()
    w = re.sub("^[^\\w']+|[^\\w']+$", '', w)
    return w




def strip_arpabet_stress(phone: str) -> str:
    if len(phone) >= 2 and phone[-1] in {'0', '1', '2'}:
        return phone[:-1]
    return phone




def read_phone_json(path: Path) -> Dict[str, int]:
    if not path.is_file():
        raise FileNotFoundError(f'phone_json not found: {path}')
    with path.open('r', encoding='utf-8', errors='strict') as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f'phone_json must be a dict token -> id: {path}')
    phone_to_id: Dict[str, int] = {}
    for token, idx in obj.items():
        if not isinstance(token, str) or token == '':
            raise ValueError(f'Invalid phone token in {path}: {token!r}')
        if not isinstance(idx, int):
            raise ValueError(f'Token id must be int for token={token!r}, got {idx!r}')
        phone_to_id[token] = idx
    ids = list(phone_to_id.values())
    if len(ids) != len(set(ids)):
        raise ValueError(f'Duplicate ids found in phone_json: {path}')
    expected = set(range(max(ids) + 1))
    got = set(ids)
    if got != expected:
        missing = sorted(expected - got)
        raise ValueError(f'phone_json ids must be dense 0..V-1; missing examples={missing[:20]}')
    return phone_to_id




def resolve_blank_token(requested_blank_token: Optional[str], processor, phone_to_id: Dict[str, int]) -> Tuple[str, int]:
    candidates: List[str] = []
    if requested_blank_token:
        candidates.append(requested_blank_token)
    tokenizer = getattr(processor, 'tokenizer', None)
    pad_token = getattr(tokenizer, 'pad_token', None) if tokenizer is not None else None
    if isinstance(pad_token, str) and pad_token:
        candidates.append(pad_token)
    candidates.extend(['<pad>', '[PAD]', '<blank>'])
    seen = set()
    for token in candidates:
        if token in seen:
            continue
        seen.add(token)
        if token in phone_to_id:
            return (token, phone_to_id[token])
    raise ValueError(f'Could not resolve blank token. requested={requested_blank_token!r}, tried={candidates!r}, vocab_sample={list(phone_to_id)[:30]!r}')




@dataclass(frozen=True)
class Point:
    token_index: int
    time_index: int




@dataclass(frozen=True)
class Segment:
    label: str
    start_frame: int
    end_frame: int

    @property
    def dur_frames(self) -> int:
        return self.end_frame - self.start_frame




@dataclass(frozen=True)
class SegmentWithConf:
    label: str
    start_frame: int
    end_frame: int
    conf_log: float

    @property
    def conf_prob(self) -> float:
        return math.exp(self.conf_log) if math.isfinite(self.conf_log) else -1.0




@dataclass(frozen=True)
class WordSpan:
    word_index: int
    word: str
    start_frame: int
    end_frame: int
    start_s: float
    end_s: float
    conf_log: float
    pron: List[str]

    @property
    def dur_s(self) -> float:
        return self.end_s - self.start_s

    @property
    def conf_prob(self) -> float:
        return math.exp(self.conf_log) if math.isfinite(self.conf_log) else -1.0




@dataclass(frozen=True)
class Chunk:
    start: float
    end: float
    words: List[str]
    word_indices: List[int]

    @property
    def dur(self) -> float:
        return self.end - self.start




@dataclass(frozen=True)
class WordAnchor:
    word_index: int
    word: str
    emit_start_frame: int
    emit_end_frame: int
    emit_start_s: float
    emit_end_s: float
    anchor_start_s: float
    anchor_end_s: float

    @property
    def anchor_dur_s(self) -> float:
        return self.anchor_end_s - self.anchor_start_s




@dataclass(frozen=True)
class GreedyPronResult:
    phones: List[str]
    chosen_prons: List[List[str]]
    pron_choice_idxs: List[int]




@torch.inference_mode()
def compute_log_probs(model, processor, audio: torch.Tensor, sample_rate: int, device: torch.device) -> Tuple[torch.Tensor, float]:
    inputs = processor(audio.cpu().numpy(), sampling_rate=sample_rate, return_tensors='pt')
    inputs = {k: v.to(device) for k, v in inputs.items()}
    logits = model(**inputs).logits
    if logits.ndim != 3 or logits.shape[0] != 1:
        raise RuntimeError(f'Expected logits shape [1,T,V], got {tuple(logits.shape)}')
    log_probs = torch.log_softmax(logits, dim=-1).squeeze(0).detach().cpu()
    if log_probs.ndim != 2 or log_probs.shape[0] <= 0 or log_probs.shape[1] <= 0:
        raise RuntimeError(f'Invalid log_probs shape: {tuple(log_probs.shape)}')
    if not torch.isfinite(log_probs).all():
        raise RuntimeError('Model produced NaN/Inf log_probs')
    seconds_per_frame = float(audio.numel() / sample_rate) / float(log_probs.shape[0])
    if seconds_per_frame <= 0 or not math.isfinite(seconds_per_frame):
        raise RuntimeError(f'Invalid seconds_per_frame={seconds_per_frame}')
    return (log_probs, seconds_per_frame)




def choose_greedy_pronunciations(words: List[str], lex: Dict[str, List[List[str]]], phone_to_id: Dict[str, int], inter_word_token: Optional[str]) -> GreedyPronResult:
    phones: List[str] = []
    chosen_prons: List[List[str]] = []
    choice_idxs: List[int] = []
    for wi, word in enumerate(words):
        if word not in lex:
            raise KeyError(f'OOV word not found in lexicon at word_index={wi}: {word!r}')
        prons = lex[word]
        if not prons:
            raise RuntimeError(f'Word has no pronunciations at word_index={wi}: {word!r}')
        pron = list(prons[0])
        if not pron:
            raise RuntimeError(f'Empty greedy pronunciation at word_index={wi}: {word!r}')
        for ph in pron:
            if ph not in phone_to_id:
                raise KeyError(f'Phone not in vocab: phone={ph!r}, word={word!r}, word_index={wi}, pron={pron!r}')
        if wi > 0 and inter_word_token is not None:
            if inter_word_token not in phone_to_id:
                raise KeyError(f'inter_word_token not in vocab: {inter_word_token!r}')
            phones.append(inter_word_token)
        phones.extend(pron)
        chosen_prons.append(pron)
        choice_idxs.append(0)
    if not phones:
        raise RuntimeError('Greedy pronunciation produced empty phone sequence.')
    return GreedyPronResult(phones=phones, chosen_prons=chosen_prons, pron_choice_idxs=choice_idxs)




def build_trellis(log_probs: torch.Tensor, targets: List[int], blank_id: int) -> torch.Tensor:
    T, V = log_probs.shape
    N = len(targets)
    if N <= 0:
        raise ValueError('Empty target token sequence.')
    if blank_id < 0 or blank_id >= V:
        raise ValueError(f'blank_id out of range: blank_id={blank_id}, V={V}')
    for i, tid in enumerate(targets):
        if tid < 0 or tid >= V:
            raise ValueError(f'target id out of range at target_index={i}: {tid}, V={V}')
    neg_inf = -float('inf')
    device = log_probs.device
    targets_t = torch.tensor(targets, device=device, dtype=torch.long)
    trellis = torch.full((T + 1, N + 1), neg_inf, device=device, dtype=log_probs.dtype)
    trellis[0, 0] = 0.0
    trellis[1:, 0] = torch.cumsum(log_probs[:, blank_id], dim=0)
    for t in range(1, T + 1):
        lp_t = log_probs[t - 1]
        blank_score = lp_t[blank_id]
        emit_scores = lp_t[targets_t]
        stay = trellis[t - 1, 1:] + blank_score
        emit = trellis[t - 1, :-1] + emit_scores
        trellis[t, 1:] = torch.maximum(stay, emit)
    if not torch.isfinite(torch.max(trellis[:, N])):
        raise RuntimeError(f'CTC trellis failed to consume all targets: T={T}, N={N}, blank_id={blank_id}')
    return trellis




def backtrace(trellis: torch.Tensor, log_probs: torch.Tensor, targets: List[int], blank_id: int) -> List[Point]:
    T = trellis.size(0) - 1
    N = trellis.size(1) - 1
    j = N
    t = int(torch.argmax(trellis[:, j]).item())
    path: List[Point] = []
    while t > 0 and j > 0:
        lp_t = log_probs[t - 1]
        score_stay = trellis[t - 1, j] + lp_t[blank_id]
        score_emit = trellis[t - 1, j - 1] + lp_t[targets[j - 1]]
        if score_emit > score_stay:
            path.append(Point(token_index=j - 1, time_index=t - 1))
            j -= 1
            t -= 1
        else:
            t -= 1
    path.reverse()
    if j != 0:
        raise RuntimeError(f'Backtrace did not consume all targets: remaining_targets={j}, N={N}, T={T}')
    if len(path) != N:
        raise RuntimeError(f'Backtrace length mismatch: len(path)={len(path)} != N={N}')
    return path




def points_to_segments(points: List[Point], target_labels: List[str]) -> List[Segment]:
    if len(points) != len(target_labels):
        raise ValueError(f'len(points)={len(points)} != len(target_labels)={len(target_labels)}')
    if not points:
        raise ValueError('Cannot convert empty points to segments.')
    segs: List[Segment] = []
    for i, p in enumerate(points):
        start = int(p.time_index)
        end = int(points[i + 1].time_index) if i + 1 < len(points) else int(p.time_index) + 1
        if end <= start:
            raise RuntimeError(f'Non-positive token segment at token_index={p.token_index}: start={start}, end={end}')
        segs.append(Segment(label=target_labels[p.token_index], start_frame=start, end_frame=end))
    return segs




def compute_segment_confidence(seg: Segment, label_to_id: Dict[str, int], log_probs: torch.Tensor, emission_frame: Optional[int], mode: str) -> float:
    if seg.label not in label_to_id:
        raise KeyError(f'Segment label not in vocab for confidence: {seg.label!r}')
    tid = label_to_id[seg.label]
    T = int(log_probs.size(0))
    if mode == 'emission':
        t = emission_frame
        if t is None:
            t = (seg.start_frame + seg.end_frame) // 2
        t = max(0, min(int(t), T - 1))
        return float(log_probs[t, tid].item())
    if mode == 'avg_frame':
        s = max(0, int(seg.start_frame))
        e = min(int(seg.end_frame), T)
        if e <= s:
            raise RuntimeError(f'Invalid segment for avg_frame confidence: {seg}')
        return float(log_probs[s:e, tid].mean().item())
    raise ValueError(f'Unsupported confidence_mode={mode!r}')




def attach_phone_confidence_from_points(phone_token_segs: List[Segment], points: List[Point], target_labels: List[str], phone_to_id: Dict[str, int], log_probs: torch.Tensor, mode: str) -> List[SegmentWithConf]:
    if len(phone_token_segs) != len(target_labels):
        raise ValueError(f'phone_token_segs length must equal target_labels length: {len(phone_token_segs)} != {len(target_labels)}')
    emission_frames: List[Optional[int]] = [None] * len(target_labels)
    for p in points:
        emission_frames[p.token_index] = p.time_index
    out: List[SegmentWithConf] = []
    for i, seg in enumerate(phone_token_segs):
        if seg.label != target_labels[i]:
            raise RuntimeError(f'Phone token segment label mismatch at token_index={i}: seg={seg.label!r}, target={target_labels[i]!r}')
        conf_log = compute_segment_confidence(seg=seg, label_to_id=phone_to_id, log_probs=log_probs, emission_frame=emission_frames[i], mode=mode)
        out.append(SegmentWithConf(seg.label, seg.start_frame, seg.end_frame, conf_log))
    return out




def phones_to_word_segments_by_offsets(phone_token_segs: List[Segment], words: List[str], prons_per_word: List[List[str]], inter_word_token: Optional[str]) -> List[Segment]:
    if len(words) != len(prons_per_word):
        raise ValueError(f'len(words)={len(words)} != len(prons_per_word)={len(prons_per_word)}')
    wi = 0
    word_segs: List[Segment] = []
    for word_index, (word, pron) in enumerate(zip(words, prons_per_word)):
        if word_index > 0 and inter_word_token is not None:
            if wi >= len(phone_token_segs):
                raise RuntimeError(f'Ran out of phone token segments before inter_word_token at word_index={word_index}')
            if phone_token_segs[wi].label != inter_word_token:
                raise RuntimeError(f'Expected inter_word_token={inter_word_token!r} at phone_token_index={wi}, got {phone_token_segs[wi].label!r}')
            wi += 1
        n = len(pron)
        if n <= 0:
            raise RuntimeError(f'Empty pronunciation for word_index={word_index}, word={word!r}')
        if wi + n > len(phone_token_segs):
            raise RuntimeError(f'Ran out of phone token segments for word_index={word_index}, word={word!r}: need={n}, remaining={len(phone_token_segs) - wi}')
        got_phones = [s.label for s in phone_token_segs[wi:wi + n]]
        if got_phones != pron:
            raise RuntimeError(f'Phone-token/pronunciation mismatch at word_index={word_index}, word={word!r}: got={got_phones!r}, expected={pron!r}')
        start_frame = phone_token_segs[wi].start_frame
        end_frame = phone_token_segs[wi + n - 1].end_frame
        if end_frame <= start_frame:
            raise RuntimeError(f'Invalid word segment frame span at word_index={word_index}, word={word!r}: start={start_frame}, end={end_frame}')
        word_segs.append(Segment(word, start_frame, end_frame))
        wi += n
    if wi != len(phone_token_segs):
        remaining = [s.label for s in phone_token_segs[wi:wi + 20]]
        raise RuntimeError(f'Unconsumed phone token segments after word reconstruction: consumed={wi}, total={len(phone_token_segs)}, remaining_sample={remaining!r}')
    return word_segs




def word_segments_with_confidence(word_segs: List[Segment], phone_segs_conf: List[SegmentWithConf]) -> List[SegmentWithConf]:
    out: List[SegmentWithConf] = []
    pi = 0
    for w in word_segs:
        confs: List[float] = []
        while pi < len(phone_segs_conf) and phone_segs_conf[pi].end_frame <= w.start_frame:
            pi += 1
        pj = pi
        while pj < len(phone_segs_conf) and phone_segs_conf[pj].start_frame < w.end_frame:
            confs.append(phone_segs_conf[pj].conf_log)
            pj += 1
        if not confs:
            raise RuntimeError(f'No overlapping phone confidence for word segment: {w}')
        out.append(SegmentWithConf(w.label, w.start_frame, w.end_frame, float(sum(confs) / len(confs))))
        pi = pj
    return out




def first_mismatch(expected: List[str], actual: List[str]) -> Tuple[int, Optional[str], Optional[str]]:
    n = min(len(expected), len(actual))
    for i in range(n):
        if expected[i] != actual[i]:
            return (i, expected[i], actual[i])
    if len(expected) != len(actual):
        return (n, expected[n] if n < len(expected) else None, actual[n] if n < len(actual) else None)
    return (-1, None, None)




def assert_tokens_equal(name: str, expected: List[str], actual: List[str]) -> None:
    if expected == actual:
        return
    pos, exp, got = first_mismatch(expected, actual)
    left = max(0, pos - 5)
    right = pos + 6
    raise RuntimeError(f'Token consistency check failed: {name}\nexpected_len={len(expected)}, actual_len={len(actual)}, mismatch_pos={pos}\nexpected_token={exp!r}, actual_token={got!r}\nexpected_context={expected[left:right]!r}\nactual_context={actual[left:right]!r}')




def validate_monotonic_word_spans(word_spans: List[WordSpan]) -> None:
    prev_end = -1.0
    prev_frame = -1
    for ws in word_spans:
        if ws.end_s <= ws.start_s:
            raise RuntimeError(f'Non-positive word duration: {ws}')
        if ws.end_frame <= ws.start_frame:
            raise RuntimeError(f'Non-positive word frame duration: {ws}')
        if ws.start_s < prev_end - 1e-06:
            raise RuntimeError(f'Word time spans overlap or go backwards: prev_end={prev_end}, current={ws}')
        if ws.start_frame < prev_frame:
            raise RuntimeError(f'Word frame spans go backwards: prev_frame={prev_frame}, current={ws}')
        prev_end = ws.end_s
        prev_frame = ws.start_frame




def emission_frames_by_token_index(points: List[Point], num_tokens: int) -> List[int]:
    if num_tokens <= 0:
        raise ValueError(f'num_tokens must be positive, got {num_tokens}')
    frames: List[Optional[int]] = [None] * num_tokens
    for p in points:
        if p.token_index < 0 or p.token_index >= num_tokens:
            raise RuntimeError(f'Point token_index out of range: token_index={p.token_index}, num_tokens={num_tokens}')
        if frames[p.token_index] is not None:
            raise RuntimeError(f'Duplicate emission point for token_index={p.token_index}')
        frames[p.token_index] = int(p.time_index)
    missing = [i for i, x in enumerate(frames) if x is None]
    if missing:
        raise RuntimeError(f'Missing emission frames for token indices: {missing[:20]}')
    return [int(x) for x in frames]




def word_phone_token_ranges(phone_token_segs: List[Segment], words: List[str], prons_per_word: List[List[str]], inter_word_token: Optional[str]) -> List[Tuple[int, int]]:
    if len(words) != len(prons_per_word):
        raise ValueError(f'len(words)={len(words)} != len(prons_per_word)={len(prons_per_word)}')
    token_i = 0
    ranges: List[Tuple[int, int]] = []
    for word_index, (word, pron) in enumerate(zip(words, prons_per_word)):
        if word_index > 0 and inter_word_token is not None:
            if token_i >= len(phone_token_segs):
                raise RuntimeError(f'Ran out of phone token segments before inter_word_token at word_index={word_index}')
            if phone_token_segs[token_i].label != inter_word_token:
                raise RuntimeError(f'Expected inter_word_token={inter_word_token!r} at phone_token_index={token_i}, got {phone_token_segs[token_i].label!r}')
            token_i += 1
        n = len(pron)
        if n <= 0:
            raise RuntimeError(f'Empty pronunciation for word_index={word_index}, word={word!r}')
        if token_i + n > len(phone_token_segs):
            raise RuntimeError(f'Ran out of phone token segments for word_index={word_index}, word={word!r}: need={n}, remaining={len(phone_token_segs) - token_i}')
        got_phones = [seg.label for seg in phone_token_segs[token_i:token_i + n]]
        if got_phones != pron:
            raise RuntimeError(f'Phone-token/pronunciation mismatch at word_index={word_index}, word={word!r}: got={got_phones!r}, expected={pron!r}')
        ranges.append((token_i, token_i + n))
        token_i += n
    if token_i != len(phone_token_segs):
        remaining = [seg.label for seg in phone_token_segs[token_i:token_i + 20]]
        raise RuntimeError(f'Unconsumed phone token segments after word-token range construction: consumed={token_i}, total={len(phone_token_segs)}, remaining_sample={remaining!r}')
    return ranges




def make_word_anchors_from_emissions(word_spans: List[WordSpan], token_ranges: List[Tuple[int, int]], token_emission_frames: List[int], *, spf: float, anchor_pad_s: float, audio_dur_s: float) -> List[WordAnchor]:
    if len(word_spans) != len(token_ranges):
        raise ValueError(f'len(word_spans)={len(word_spans)} != len(token_ranges)={len(token_ranges)}')
    if spf <= 0 or not math.isfinite(spf):
        raise ValueError(f'spf must be positive finite, got {spf}')
    if anchor_pad_s < 0 or not math.isfinite(anchor_pad_s):
        raise ValueError(f'anchor_pad_s must be non-negative finite, got {anchor_pad_s}')
    if audio_dur_s <= 0 or not math.isfinite(audio_dur_s):
        raise ValueError(f'audio_dur_s must be positive finite, got {audio_dur_s}')
    anchors: List[WordAnchor] = []
    for ws, (tok_start, tok_end) in zip(word_spans, token_ranges):
        if tok_end <= tok_start:
            raise RuntimeError(f'Empty token range for word_index={ws.word_index}, word={ws.word!r}')
        frames = token_emission_frames[tok_start:tok_end]
        if not frames:
            raise RuntimeError(f'No emission frames for word_index={ws.word_index}, word={ws.word!r}')
        emit_start_frame = min(frames)
        emit_end_frame = max(frames)
        emit_start_s = float(emit_start_frame) * spf
        emit_end_s = float(emit_end_frame) * spf
        anchor_start_s = max(0.0, emit_start_s - anchor_pad_s)
        anchor_end_s = min(float(audio_dur_s), emit_end_s + anchor_pad_s)
        if anchor_end_s <= anchor_start_s:
            raise RuntimeError(f'Invalid word anchor for word_index={ws.word_index}, word={ws.word!r}: emit=({emit_start_s}, {emit_end_s}), anchor=({anchor_start_s}, {anchor_end_s})')
        anchors.append(WordAnchor(word_index=ws.word_index, word=ws.word, emit_start_frame=emit_start_frame, emit_end_frame=emit_end_frame, emit_start_s=emit_start_s, emit_end_s=emit_end_s, anchor_start_s=anchor_start_s, anchor_end_s=anchor_end_s))
    return anchors




def merge_word_anchors_into_chunks(word_anchors: List[WordAnchor], *, anchor_merge_gap_s: float) -> List[Chunk]:
    if not word_anchors:
        raise ValueError('merge_word_anchors_into_chunks received empty word_anchors')
    if anchor_merge_gap_s < 0 or not math.isfinite(anchor_merge_gap_s):
        raise ValueError(f'anchor_merge_gap_s must be non-negative finite, got {anchor_merge_gap_s}')
    anchors = sorted(word_anchors, key=lambda a: (a.anchor_start_s, a.anchor_end_s, a.word_index))
    chunks: List[Chunk] = []
    cur_start = anchors[0].anchor_start_s
    cur_end = anchors[0].anchor_end_s
    cur_words = [anchors[0].word]
    cur_indices = [anchors[0].word_index]
    for a in anchors[1:]:
        gap = a.anchor_start_s - cur_end
        if gap < anchor_merge_gap_s:
            cur_end = max(cur_end, a.anchor_end_s)
            cur_words.append(a.word)
            cur_indices.append(a.word_index)
            continue
        chunks.append(Chunk(start=cur_start, end=cur_end, words=list(cur_words), word_indices=list(cur_indices)))
        cur_start = a.anchor_start_s
        cur_end = a.anchor_end_s
        cur_words = [a.word]
        cur_indices = [a.word_index]
    chunks.append(Chunk(start=cur_start, end=cur_end, words=list(cur_words), word_indices=list(cur_indices)))
    for i, c in enumerate(chunks):
        if c.end <= c.start:
            raise RuntimeError(f'Invalid merged anchor chunk at index={i}: {c}')
        if sorted(c.word_indices) != c.word_indices:
            raise RuntimeError(f'Merged anchor chunk word_indices are not monotonic at index={i}: {c}')
    return chunks




@dataclass(frozen=True)
class EmitEdge:
    u: int
    v: int
    phone: str
    phone_id: int
    word_index: Optional[int]
    word: Optional[str]




@dataclass
class PhoneState:
    edge: EmitEdge
    preds: Tuple[int, ...]
    succs: Tuple[int, ...]




@dataclass
class PhoneGraph:
    states: List[PhoneState]
    start_states: List[int]
    end_states: List[int]




def _eps_closure(num_nodes: int, eps_adj: List[List[int]]) -> List[Set[int]]:
    """closure[u] = nodes reachable from u via epsilon edges (including u)."""
    closure: List[Set[int]] = [set() for _ in range(num_nodes)]
    for u in range(num_nodes):
        seen = {u}
        stack = [u]
        while stack:
            x = stack.pop()
            for y in eps_adj[x]:
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
        closure[u] = seen
    return closure




def build_phone_graph_optional_sil_sph(words: List[str], prondict: PronouncingDictionary, phone_to_id: Dict[str, int], sil_phone: Optional[str]='SIL', optional_sil_between_words: bool=True, optional_sil_at_start: Optional[bool]=None, optional_sil_at_end: Optional[bool]=None, sil_cost: float=0.0, sph_phone: Optional[str]='sph', optional_sph_between_words: bool=False, optional_sph_at_start: Optional[bool]=None, optional_sph_at_end: Optional[bool]=None, sph_cost: float=-2.5, sph_word_label: str='[missing]') -> Tuple[PhoneGraph, np.ndarray]:
    """
    Pronunciation DAG with optional SIL and optional SPH.

    SIL is intended for silence/pauses.
    SPH is intended for generic speech corresponding to words that are present in
    the audio but missing from the transcript.

    Between transcript words, and also at utterance boundaries when enabled, this
    graph can allow:
      - epsilon, i.e., no gap material
      - SIL
      - SPH
      - SIL + SPH
      - SPH + SIL
      - SIL + SPH + SIL

    Important boundary behavior:
      At the beginning, optional SPH is reachable directly from START.
      At the end, optional SPH is an explicit end state that can terminate at END.

    Each emitting edge can last multiple frames through the Viterbi self-loop,
    so a single SPH edge can absorb a missing word or a short missing phrase.
    """
    if optional_sil_at_start is None:
        optional_sil_at_start = optional_sil_between_words
    if optional_sil_at_end is None:
        optional_sil_at_end = optional_sil_between_words
    if optional_sph_at_start is None:
        optional_sph_at_start = optional_sph_between_words
    if optional_sph_at_end is None:
        optional_sph_at_end = optional_sph_between_words
    next_node = 0

    def new_node() -> int:
        nonlocal next_node
        nid = next_node
        next_node += 1
        return nid
    START = new_node()
    emit_edges: List[EmitEdge] = []
    eps_edges: List[Tuple[int, int]] = []
    entry_bias: List[float] = []

    def add_emit(u: int, v: int, phone: str, widx: Optional[int], w: Optional[str], bias: float=0.0):
        if phone not in phone_to_id:
            raise KeyError(f"Phone '{phone}' not in model vocab.")
        emit_edges.append(EmitEdge(u=u, v=v, phone=phone, phone_id=phone_to_id[phone], word_index=widx, word=w))
        entry_bias.append(bias)

    def add_eps(u: int, v: int):
        eps_edges.append((u, v))

    def add_sil(u: int, v: int):
        if sil_phone is None:
            return
        add_emit(u, v, sil_phone, None, None, bias=sil_cost)

    def add_sph(u: int, v: int):
        if sph_phone is None:
            return
        add_emit(u, v, sph_phone, None, sph_word_label, bias=sph_cost)

    def add_optional_gap(u: int, v: int, allow_sil: bool, allow_sph: bool):
        """
        Add optional material between two anchors.

        This function is deliberately used for BOTH internal word gaps and
        utterance boundary gaps. This avoids a common failure mode where SPH is
        allowed between words but not actually reachable from START or allowed to
        terminate at END.
        """
        add_eps(u, v)
        if allow_sil and sil_phone is not None:
            add_sil(u, v)
        if allow_sph and sph_phone is not None:
            add_sph(u, v)
        if allow_sil and allow_sph and (sil_phone is not None) and (sph_phone is not None):
            m1 = new_node()
            add_sil(u, m1)
            add_sph(m1, v)
            m2 = new_node()
            add_sph(u, m2)
            add_sil(m2, v)
            m3 = new_node()
            m4 = new_node()
            add_sil(u, m3)
            add_sph(m3, m4)
            add_sil(m4, v)
    start_node = new_node()
    add_optional_gap(START, start_node, allow_sil=optional_sil_at_start, allow_sph=optional_sph_at_start)
    cur_node = start_node
    for wi, w in enumerate(words):
        end_of_word = new_node()
        prons = prondict.get_prons(w)
        for pron in prons:
            u = cur_node
            for pi, ph in enumerate(pron):
                v = end_of_word if pi == len(pron) - 1 else new_node()
                add_emit(u, v, ph, wi, w, bias=0.0)
                u = v
        cur_node = end_of_word
        if wi != len(words) - 1:
            nxt = new_node()
            add_optional_gap(cur_node, nxt, allow_sil=optional_sil_between_words, allow_sph=optional_sph_between_words)
            cur_node = nxt
    final_node = cur_node
    END = new_node()
    add_optional_gap(final_node, END, allow_sil=optional_sil_at_end, allow_sph=optional_sph_at_end)
    num_nodes = next_node
    eps_adj = [[] for _ in range(num_nodes)]
    eps_rev = [[] for _ in range(num_nodes)]
    for u, v in eps_edges:
        eps_adj[u].append(v)
        eps_rev[v].append(u)
    fwd_cl = _eps_closure(num_nodes, eps_adj)
    bwd_cl = _eps_closure(num_nodes, eps_rev)
    out_emit: Dict[int, List[int]] = {}
    in_emit: Dict[int, List[int]] = {}
    for ei, e in enumerate(emit_edges):
        out_emit.setdefault(e.u, []).append(ei)
        in_emit.setdefault(e.v, []).append(ei)
    states: List[PhoneState] = []
    for e in emit_edges:
        pred_idxs: List[int] = []
        for node in bwd_cl[e.u]:
            pred_idxs.extend(in_emit.get(node, []))
        succ_idxs: List[int] = []
        for node in fwd_cl[e.v]:
            succ_idxs.extend(out_emit.get(node, []))
        states.append(PhoneState(edge=e, preds=tuple(sorted(set(pred_idxs))), succs=tuple(sorted(set(succ_idxs)))))
    start_states: List[int] = []
    for node in fwd_cl[START]:
        start_states.extend(out_emit.get(node, []))
    start_states = sorted(set(start_states))
    if not start_states:
        raise RuntimeError('No start states. Check transcript/lexicon/SIL/SPH settings.')
    end_states: List[int] = []
    for si, st in enumerate(states):
        if END in fwd_cl[st.edge.v]:
            end_states.append(si)
    if not end_states:
        end_states = [i for i, st in enumerate(states) if len(st.succs) == 0]
    if optional_sph_at_start and sph_phone is not None:
        has_start_sph = any((states[s].edge.phone == sph_phone for s in start_states))
        if not has_start_sph:
            raise RuntimeError('optional_sph_at_start=True, but SPH is not reachable from START.')
    if optional_sph_at_end and sph_phone is not None:
        has_end_sph = any((states[s].edge.phone == sph_phone for s in end_states))
        if not has_end_sph:
            raise RuntimeError('optional_sph_at_end=True, but SPH cannot terminate at END.')
    return (PhoneGraph(states=states, start_states=start_states, end_states=end_states), np.asarray(entry_bias, dtype=np.float32))




@dataclass
class AlignmentResult:
    phone_segments_f: List[Tuple[str, int, int]]
    word_segments_f: List[Tuple[str, int, int]]
    state_path: np.ndarray
    aligned_phone_ids: np.ndarray




def align_beam_viterbi(logp: np.ndarray, graph: PhoneGraph, entry_bias: np.ndarray, p_stay: float=0.92, beam_size: int=300, word_sil_label: str='sil', boundary_lambda: float=0.0, boundary_context_s: float=0.015, frame_hop_s: float=0.01, sil_phone_id: int | None=None, min_sil_dur_ms: float=0.0, sil_enter_cost: float=0.0, sph_phone_id: int | None=None, sph_enter_cost: float=0.0) -> AlignmentResult:
    T, V = logp.shape
    S = len(graph.states)
    if entry_bias.shape[0] != S:
        raise ValueError('entry_bias length != number of states')
    if T == 0:
        raise ValueError('No frames produced by model.')
    lp_stay = math.log(p_stay)
    lp_move = math.log(1.0 - p_stay)
    ctx = max(1, int(round(boundary_context_s / frame_hop_s)))
    if boundary_lambda != 0.0:
        pref = np.zeros((T + 1, V), dtype=np.float32)
        pref[1:] = np.cumsum(logp, axis=0)

        def _mean(pid: int, s: int, e: int) -> float:
            if e <= s:
                return 0.0
            return float((pref[e, pid] - pref[s, pid]) / (e - s))

        def boundary_score(t: int, a: int, b: int) -> float:
            l0 = 0 if t - ctx < 0 else t - ctx
            l1 = t
            r0 = t
            r1 = T if t + ctx > T else t + ctx
            left = _mean(a, l0, l1) - _mean(b, l0, l1)
            right = _mean(b, r0, r1) - _mean(a, r0, r1)
            return left + right
    else:

        def boundary_score(t: int, a: int, b: int) -> float:
            return 0.0
    min_sil_frames = 0
    if min_sil_dur_ms is not None and min_sil_dur_ms > 0.0 and (sil_phone_id is not None):
        min_sil_frames = max(1, int(round(min_sil_dur_ms / 1000.0 / frame_hop_s)))

    def _is_sil_phone(pid: int) -> bool:
        return sil_phone_id is not None and pid == sil_phone_id

    def _is_sph_phone(pid: int) -> bool:
        return sph_phone_id is not None and pid == sph_phone_id
    bp: List[Dict[tuple[int, int], tuple[int, int]]] = []
    cur_scores: Dict[tuple[int, int], float] = {}
    cur_bp: Dict[tuple[int, int], tuple[int, int]] = {}
    for s in graph.start_states:
        phid = graph.states[s].edge.phone_id
        if _is_sil_phone(phid) and min_sil_frames > 0:
            lock = min_sil_frames - 1
        else:
            lock = 0
        key = (int(s), int(lock))
        cur_scores[key] = float(logp[0, phid]) + float(entry_bias[s])
        cur_bp[key] = key
    if len(cur_scores) > beam_size:
        top = sorted(cur_scores.items(), key=lambda kv: kv[1], reverse=True)[:beam_size]
        cur_scores = {k: v for k, v in top}
        cur_bp = {k: cur_bp[k] for k, _ in top}
    bp.append(cur_bp)
    for t in range(1, T):
        nxt_scores: Dict[tuple[int, int], float] = {}
        nxt_bp: Dict[tuple[int, int], tuple[int, int]] = {}
        for (s, lock_prev), sc in cur_scores.items():
            st = graph.states[s]
            phid_prev = st.edge.phone_id
            prev_is_sil = _is_sil_phone(phid_prev)
            prev_is_sph = _is_sph_phone(phid_prev)
            emit_s = float(logp[t, phid_prev]) + float(entry_bias[s])
            cand = sc + lp_stay + emit_s
            if prev_is_sil and lock_prev > 0:
                lock_stay = lock_prev - 1
            else:
                lock_stay = 0
            key_stay = (int(s), int(lock_stay if prev_is_sil else 0))
            if cand > nxt_scores.get(key_stay, NEG_INF):
                nxt_scores[key_stay] = cand
                nxt_bp[key_stay] = (int(s), int(lock_prev))
            base = sc + lp_move
            for ns in st.succs:
                nst = graph.states[ns]
                phid_next = nst.edge.phone_id
                next_is_sil = _is_sil_phone(phid_next)
                next_is_sph = _is_sph_phone(phid_next)
                if prev_is_sil and lock_prev > 0 and (not next_is_sil):
                    continue
                emit_ns = float(logp[t, phid_next]) + float(entry_bias[ns])
                if next_is_sil:
                    if prev_is_sil:
                        lock_next = lock_prev - 1 if lock_prev > 0 else 0
                    else:
                        lock_next = min_sil_frames - 1 if min_sil_frames > 0 else 0
                else:
                    lock_next = 0
                key_next = (int(ns), int(lock_next))
                enter_pen = 0.0
                if not prev_is_sil and next_is_sil:
                    enter_pen += float(sil_enter_cost)
                if not prev_is_sph and next_is_sph:
                    enter_pen += float(sph_enter_cost)
                cand2 = base + emit_ns + enter_pen + boundary_lambda * boundary_score(t, phid_prev, phid_next)
                if cand2 > nxt_scores.get(key_next, NEG_INF):
                    nxt_scores[key_next] = cand2
                    nxt_bp[key_next] = (int(s), int(lock_prev))
        if len(nxt_scores) > beam_size:
            top = sorted(nxt_scores.items(), key=lambda kv: kv[1], reverse=True)[:beam_size]
            nxt_scores = {k: v for k, v in top}
            nxt_bp = {k: nxt_bp[k] for k, _ in top}
        cur_scores = nxt_scores
        bp.append(nxt_bp)
    end_set = set(graph.end_states)
    best_state = None
    best_score = NEG_INF
    for (s, lock_prev), sc in cur_scores.items():
        term = sc + lp_move
        if s in end_set and term > best_score:
            best_score = term
            best_state = (int(s), int(lock_prev))
    if best_state is None:
        raise RuntimeError(f'Viterbi failed to reach any end state. T={T}, num_states={S}, beam_size={beam_size}, num_end_states={len(graph.end_states)}, active_states={len(cur_scores)}')
    path = np.empty((T,), dtype=np.int32)
    cur_key = best_state
    for t in range(T - 1, -1, -1):
        path[t] = int(cur_key[0])
        cur_key = bp[t].get(cur_key, cur_key)
    aligned_phone_ids = np.array([graph.states[int(s)].edge.phone_id for s in path], dtype=np.int32)
    phone_segments_f: List[Tuple[str, int, int]] = []
    cur_edge0 = graph.states[int(path[0])].edge
    cur_ph = cur_edge0.phone
    cur_wi = cur_edge0.word_index
    start = 0
    for t in range(1, T):
        e = graph.states[int(path[t])].edge
        ph = e.phone
        wi = e.word_index
        if ph != cur_ph or wi != cur_wi:
            phone_segments_f.append((cur_ph, start, t))
            cur_ph = ph
            cur_wi = wi
            start = t
    phone_segments_f.append((cur_ph, start, T))
    word_segments_f: List[Tuple[str, int, int]] = []
    edge0 = graph.states[int(path[0])].edge
    cur_w = edge0.word if edge0.word is not None else word_sil_label
    cur_wi = edge0.word_index
    start = 0
    for t in range(1, T):
        edge = graph.states[int(path[t])].edge
        lab = edge.word if edge.word is not None else word_sil_label
        wi = edge.word_index
        if lab != cur_w or wi != cur_wi:
            word_segments_f.append((cur_w, start, t))
            cur_w = lab
            cur_wi = wi
            start = t
    word_segments_f.append((cur_w, start, T))
    return AlignmentResult(phone_segments_f=phone_segments_f, word_segments_f=word_segments_f, state_path=path, aligned_phone_ids=aligned_phone_ids)




@dataclass(frozen=True)
class FixedStateSpec:
    phone: str
    phone_id: int
    word_index: Optional[int]
    word: Optional[str]
    bias: float = 0.0




def extract_state_segments_from_path(graph: PhoneGraph, entry_bias: np.ndarray, path: np.ndarray) -> List[Tuple[FixedStateSpec, int, int]]:
    """
    Collapse the first-pass frame-level state path into a phone-state sequence.

    The collapsed unit preserves word_index/word, so identical phone symbols across
    word boundaries remain distinct sequence states.
    """
    if path.ndim != 1 or path.size <= 0:
        raise RuntimeError(f'Invalid state path shape: {path.shape}')
    out: List[Tuple[FixedStateSpec, int, int]] = []
    cur_sid = int(path[0])
    cur_edge = graph.states[cur_sid].edge
    cur_bias = float(entry_bias[cur_sid])
    start = 0
    for t in range(1, int(path.size)):
        sid = int(path[t])
        edge = graph.states[sid].edge
        bias = float(entry_bias[sid])
        if edge.phone != cur_edge.phone or edge.phone_id != cur_edge.phone_id or edge.word_index != cur_edge.word_index or (edge.word != cur_edge.word):
            spec = FixedStateSpec(phone=cur_edge.phone, phone_id=cur_edge.phone_id, word_index=cur_edge.word_index, word=cur_edge.word, bias=cur_bias)
            out.append((spec, start, t))
            cur_sid = sid
            cur_edge = edge
            cur_bias = bias
            start = t
    spec = FixedStateSpec(phone=cur_edge.phone, phone_id=cur_edge.phone_id, word_index=cur_edge.word_index, word=cur_edge.word, bias=cur_bias)
    out.append((spec, start, int(path.size)))
    return out




def prune_short_internal_sil_sph_segments(state_segments: List[Tuple[FixedStateSpec, int, int]], *, sil_phone: Optional[str], sph_phone: Optional[str], min_sil_dur_ms: float, min_sph_dur_ms: float, frame_hop_s: float) -> Tuple[List[FixedStateSpec], Dict[str, int]]:
    """
    Remove short internal SIL/SPH states from the first-pass sequence.

    Boundary states are preserved because chunk start/end may genuinely contain
    short residual silence or untranscribed speech from Stage 1 chunking.
    """
    if not state_segments:
        raise RuntimeError('Cannot prune an empty first-pass state sequence.')
    if frame_hop_s <= 0.0 or not math.isfinite(frame_hop_s):
        raise ValueError(f'Invalid frame_hop_s={frame_hop_s}')
    if min_sil_dur_ms < 0.0 or min_sph_dur_ms < 0.0:
        raise ValueError(f'min_sil_dur_ms/min_sph_dur_ms must be non-negative, got {min_sil_dur_ms}, {min_sph_dur_ms}')
    sil_threshold_frames = int(math.ceil(min_sil_dur_ms / 1000.0 / frame_hop_s))
    sph_threshold_frames = int(math.ceil(min_sph_dur_ms / 1000.0 / frame_hop_s))
    kept: List[FixedStateSpec] = []
    dropped_sil = 0
    dropped_sph = 0
    for i, (spec, s, e) in enumerate(state_segments):
        dur_frames = int(e) - int(s)
        if dur_frames <= 0:
            raise RuntimeError(f'Non-positive first-pass state duration at segment {i}: phone={spec.phone!r}, start={s}, end={e}')
        is_boundary_state = i == 0 or i == len(state_segments) - 1
        if is_boundary_state:
            kept.append(spec)
            continue
        is_sil = sil_phone is not None and spec.phone == sil_phone
        is_sph = sph_phone is not None and spec.phone == sph_phone
        if is_sil and dur_frames < sil_threshold_frames:
            dropped_sil += 1
            continue
        if is_sph and dur_frames < sph_threshold_frames:
            dropped_sph += 1
            continue
        kept.append(spec)
    if not kept:
        raise RuntimeError('All first-pass states were removed during short SIL/SPH pruning. Check min_sil_dur_ms/min_sph_dur_ms.')
    return (kept, {'first_pass_states': len(state_segments), 'fixed_states': len(kept), 'dropped_short_sil': dropped_sil, 'dropped_short_sph': dropped_sph})




def build_fixed_sequence_graph(specs: List[FixedStateSpec]) -> Tuple[PhoneGraph, np.ndarray]:
    """
    Build a linear fixed-sequence phone graph for the second pass.

    The second pass keeps the token order fixed but still re-estimates state
    durations and boundaries using Viterbi self-loops and moves.
    """
    if not specs:
        raise RuntimeError('Cannot build a fixed-sequence graph from an empty sequence.')
    states: List[PhoneState] = []
    entry_bias: List[float] = []
    for i, spec in enumerate(specs):
        edge = EmitEdge(u=i, v=i + 1, phone=spec.phone, phone_id=spec.phone_id, word_index=spec.word_index, word=spec.word)
        preds = (i - 1,) if i > 0 else tuple()
        succs = (i + 1,) if i + 1 < len(specs) else tuple()
        states.append(PhoneState(edge=edge, preds=preds, succs=succs))
        entry_bias.append(float(spec.bias))
    return (PhoneGraph(states=states, start_states=[0], end_states=[len(states) - 1]), np.asarray(entry_bias, dtype=np.float32))




def redecode_with_pruned_fixed_sequence(*, first_pass_ali: AlignmentResult, first_pass_graph: PhoneGraph, first_pass_entry_bias: np.ndarray, logp: np.ndarray, sil_phone: Optional[str], sil_phone_id: Optional[int], sph_phone: Optional[str], sph_phone_id: Optional[int], args) -> Tuple[AlignmentResult, Dict[str, int]]:
    first_pass_segments = extract_state_segments_from_path(graph=first_pass_graph, entry_bias=first_pass_entry_bias, path=first_pass_ali.state_path)
    fixed_specs, stats = prune_short_internal_sil_sph_segments(first_pass_segments, sil_phone=sil_phone, sph_phone=sph_phone, min_sil_dur_ms=args.min_sil_dur_ms, min_sph_dur_ms=args.min_sph_dur_ms, frame_hop_s=args.frame_hop_s)
    fixed_graph, fixed_entry_bias = build_fixed_sequence_graph(fixed_specs)
    ali2 = align_beam_viterbi(logp=logp, graph=fixed_graph, entry_bias=fixed_entry_bias, p_stay=args.p_stay, beam_size=args.beam, word_sil_label=args.word_sil_label, boundary_lambda=args.boundary_lambda, boundary_context_s=args.boundary_context_s, frame_hop_s=args.frame_hop_s, sil_phone_id=sil_phone_id, min_sil_dur_ms=0.0, sil_enter_cost=0.0, sph_phone_id=sph_phone_id, sph_enter_cost=0.0)
    return (ali2, stats)




@dataclass(frozen=True)
class Interval:
    xmin: float
    xmax: float
    text: str




@dataclass(frozen=True)
class IntervalTier:
    name: str
    xmin: float
    xmax: float
    intervals: List[Interval]




@dataclass(frozen=True)
class TextGrid:
    xmin: float
    xmax: float
    tiers: List[IntervalTier]




def _strip_quotes(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == '"' and (s[-1] == '"'):
        return s[1:-1].replace('""', '"')
    return s.replace('""', '"')




def _escape_textgrid(s: str) -> str:
    return s.replace('"', '""')




def parse_textgrid_long(path: Path) -> TextGrid:
    if not path.is_file():
        raise FileNotFoundError(f'TextGrid not found: {path}')
    lines = path.read_text(encoding='utf-8', errors='strict').splitlines()

    def first_value(pattern: str) -> Optional[str]:
        for line in lines:
            m = re.match(pattern, line.strip())
            if m:
                return m.group(1)
        return None
    xmin_s = first_value('xmin\\s*=\\s*([0-9.eE+-]+)\\s*$')
    xmax_s = first_value('xmax\\s*=\\s*([0-9.eE+-]+)\\s*$')
    if xmin_s is None or xmax_s is None:
        raise ValueError(f'Failed to parse global xmin/xmax: {path}')
    tg_xmin = float(xmin_s)
    tg_xmax = float(xmax_s)
    tiers: List[IntervalTier] = []
    i = 0
    n = len(lines)
    while i < n:
        if not re.match('item\\s*\\[\\d+\\]\\s*:', lines[i].strip()):
            i += 1
            continue
        cls = None
        name = None
        txmin = None
        txmax = None
        intervals: List[Interval] = []
        i += 1
        while i < n and (not re.match('item\\s*\\[\\d+\\]\\s*:', lines[i].strip())):
            s = lines[i].strip()
            if s.startswith('class'):
                cls = _strip_quotes(s.split('=', 1)[1])
            elif s.startswith('name'):
                name = _strip_quotes(s.split('=', 1)[1])
            elif s.startswith('xmin') and txmin is None:
                txmin = float(s.split('=', 1)[1])
            elif s.startswith('xmax') and txmax is None:
                txmax = float(s.split('=', 1)[1])
            elif re.match('intervals\\s*\\[\\d+\\]\\s*:', s):
                ixmin = None
                ixmax = None
                text = ''
                i += 1
                while i < n:
                    ss = lines[i].strip()
                    if ss.startswith('xmin'):
                        ixmin = float(ss.split('=', 1)[1])
                    elif ss.startswith('xmax'):
                        ixmax = float(ss.split('=', 1)[1])
                    elif ss.startswith('text'):
                        text = _strip_quotes(ss.split('=', 1)[1])
                    elif re.match('(intervals|item)\\s*\\[', ss):
                        i -= 1
                        break
                    i += 1
                if ixmin is None or ixmax is None:
                    raise ValueError(f'Bad interval near line={i} in {path}')
                intervals.append(Interval(ixmin, ixmax, text))
            i += 1
        if cls == 'IntervalTier':
            if name is None or txmin is None or txmax is None:
                raise ValueError(f'Incomplete IntervalTier header: {path}')
            tiers.append(IntervalTier(name, txmin, txmax, intervals))
    if not tiers:
        raise ValueError(f'No IntervalTier parsed from {path}')
    return TextGrid(tg_xmin, tg_xmax, tiers)




def write_textgrid_long(tg: TextGrid, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        f.write('File type = "ooTextFile"\n')
        f.write('Object class = "TextGrid"\n\n')
        f.write(f'xmin = {tg.xmin:.6f}\n')
        f.write(f'xmax = {tg.xmax:.6f}\n')
        f.write('tiers? <exists>\n')
        f.write(f'size = {len(tg.tiers)}\n')
        f.write('item []:\n')
        for ti, tier in enumerate(tg.tiers, start=1):
            f.write(f'    item [{ti}]:\n')
            f.write('        class = "IntervalTier"\n')
            f.write(f'        name = "{_escape_textgrid(tier.name)}"\n')
            f.write(f'        xmin = {tier.xmin:.6f}\n')
            f.write(f'        xmax = {tier.xmax:.6f}\n')
            f.write(f'        intervals: size = {len(tier.intervals)}\n')
            for ii, interval in enumerate(tier.intervals, start=1):
                f.write(f'        intervals [{ii}]:\n')
                f.write(f'            xmin = {interval.xmin:.6f}\n')
                f.write(f'            xmax = {interval.xmax:.6f}\n')
                f.write(f'            text = "{_escape_textgrid(interval.text)}"\n')




def merge_adjacent(intervals: Iterable[Interval], eps: float=1e-05, merge_texts: Optional[set[str]]=None) -> List[Interval]:
    """
    Merge only labels explicitly allowed by merge_texts.

    For strict word preservation, do NOT merge lexical word intervals.
    Usually only NULL gaps should be merged.
    """
    out: List[Interval] = []
    for itv in sorted(intervals, key=lambda x: (x.xmin, x.xmax, x.text)):
        if itv.xmax - itv.xmin <= EPS:
            continue
        allow_merge = merge_texts is None or itv.text in merge_texts
        if allow_merge and out and (out[-1].text == itv.text) and (abs(out[-1].xmax - itv.xmin) <= eps):
            out[-1] = Interval(out[-1].xmin, max(out[-1].xmax, itv.xmax), out[-1].text)
        else:
            out.append(itv)
    return out




def clip_shift_interval(local: Interval, chunk_start: float, chunk_end: float, local_xmax: float) -> Optional[Interval]:
    """
    Local TextGrid interval -> global-time interval.

    Assumption:
    local TextGrid uses local time, i.e. [0, local_xmax].
    The output is shifted by chunk_start.

    This function must NOT compare local.xmin directly with chunk_start.
    """
    chunk_dur = chunk_end - chunk_start
    if chunk_dur <= EPS:
        raise RuntimeError(f'Invalid chunk duration: {chunk_start:.6f}..{chunk_end:.6f}')
    lxmin = float(local.xmin)
    lxmax = float(local.xmax)
    if lxmax - lxmin <= EPS:
        return None
    valid_local_end = min(float(local_xmax), float(chunk_dur))
    ls = max(0.0, lxmin)
    le = min(lxmax, valid_local_end)
    if le - ls <= EPS:
        return None
    return Interval(xmin=chunk_start + ls, xmax=chunk_start + le, text=local.text)




def labels_from_intervals(intervals: Iterable[Interval], ignore_labels: set[str]) -> List[str]:
    ignore = {x.strip().lower() for x in ignore_labels}
    out: List[str] = []
    for itv in intervals:
        lab = (itv.text or '').strip()
        if not lab:
            continue
        if lab.lower() in ignore:
            continue
        out.append(lab)
    return out




def validate_word_sequence(*, actual: List[str], expected: List[str], context: str) -> None:
    if actual == expected:
        return
    n = min(len(actual), len(expected))
    mismatch_pos = None
    for i in range(n):
        if actual[i] != expected[i]:
            mismatch_pos = i
            break
    if mismatch_pos is None:
        mismatch_pos = n
    got = actual[mismatch_pos] if mismatch_pos < len(actual) else None
    exp = expected[mismatch_pos] if mismatch_pos < len(expected) else None
    raise RuntimeError(f'Word sequence mismatch during merge: {context}\nactual_len={len(actual)}, expected_len={len(expected)}, mismatch_pos={mismatch_pos}, actual_token={got!r}, expected_token={exp!r}\nactual_window={actual[max(0, mismatch_pos - 5):mismatch_pos + 6]}\nexpected_window={expected[max(0, mismatch_pos - 5):mismatch_pos + 6]}')





# Pipeline-specific orchestration is defined below.


@dataclass(frozen=True)
class PronouncingDictionary:
    lex: Dict[str, List[List[str]]]

    def get_prons(self, word: str) -> List[List[str]]:
        if word not in self.lex:
            raise KeyError(f"Word not in lexicon: {word!r}")
        prons = self.lex[word]
        if not prons:
            raise RuntimeError(f"Word has no pronunciations: {word!r}")
        return prons


@dataclass(frozen=True)
class RuntimeChunk:
    chunk_id: str
    start_ms: int
    end_ms: int
    start_sample: int
    end_sample: int
    words: List[str]
    word_indices: List[int]

    @property
    def start_s(self) -> float:
        return self.start_ms / 1000.0

    @property
    def end_s(self) -> float:
        return self.end_ms / 1000.0

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass(frozen=True)
class AlignConfig:
    optional_sil: bool = True
    sil_phone: str = "sil"
    sil_cost: float = -0.5
    sil_enter_cost: float = -0.5
    min_sil_dur_ms: float = 65.0
    optional_sph: bool = True
    sph_phone: str = "sph"
    sph_cost: float = -2.0
    sph_enter_cost: float = -3.0
    sph_word_label: str = "[missing]"
    min_sph_dur_ms: float = 50.0
    beam: int = 400
    p_stay: float = 0.92
    boundary_lambda: float = 200.0
    boundary_context_s: float = 0.03
    frame_hop_s: float = 0.01
    word_sil_label: str = "sil"


@dataclass(frozen=True)
class LocalAlignment:
    textgrid: TextGrid
    redecode_stats: Dict[str, int]


def read_input_words(text: Optional[str], text_path: Optional[Path]) -> Tuple[str, List[str]]:
    if (text is None) == (text_path is None):
        raise ValueError("Exactly one of --text and --text_path must be provided.")

    if text_path is not None:
        if not text_path.is_file():
            raise FileNotFoundError(f"--text_path is not a file: {text_path}")
        raw_text = text_path.read_text(encoding="utf-8", errors="strict")
    else:
        raw_text = text

    if raw_text is None or not raw_text.strip():
        raise ValueError("Input transcript is empty.")

    raw_tokens = raw_text.strip().split()
    words: List[str] = []
    for index, token in enumerate(raw_tokens):
        word = normalize_word(token)
        if not word:
            raise ValueError(
                f"Transcript token became empty after normalization: "
                f"token_index={index}, raw_token={token!r}"
            )
        words.append(word)

    if not words:
        raise ValueError("Input transcript has no word tokens after normalization.")
    return raw_text, words


def load_raw_lexicon(path: Path) -> PronouncingDictionary:
    if not path.is_file():
        raise FileNotFoundError(f"Lexicon not found: {path}")

    lex: Dict[str, List[List[str]]] = {}
    with path.open("r", encoding="utf-8", errors="strict") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(
                    f"Invalid lexicon line {line_number} in {path}: {line!r}"
                )
            word = normalize_word(parts[0])
            if not word:
                raise ValueError(
                    f"Lexicon word became empty after normalization at "
                    f"line {line_number} in {path}: {parts[0]!r}"
                )
            pronunciation = parts[1:]
            if any(not phone for phone in pronunciation):
                raise ValueError(
                    f"Empty phone at lexicon line {line_number} in {path}: {line!r}"
                )
            lex.setdefault(word, []).append(pronunciation)

    if not lex:
        raise ValueError(f"No lexicon entries loaded: {path}")
    return PronouncingDictionary(lex=lex)


def build_chunk_lexicon(
    raw_lexicon: PronouncingDictionary,
) -> Dict[str, List[List[str]]]:
    out: Dict[str, List[List[str]]] = {}
    for word, pronunciations in raw_lexicon.lex.items():
        out[word] = [
            [strip_arpabet_stress(phone) for phone in pronunciation]
            for pronunciation in pronunciations
        ]
    return out


def validate_transcript_lexicon(
    words: List[str],
    raw_lexicon: PronouncingDictionary,
) -> None:
    for word_index, word in enumerate(words):
        if word not in raw_lexicon.lex:
            raise KeyError(
                f"OOV word not found in lexicon: word_index={word_index}, word={word!r}"
            )
        if not raw_lexicon.lex[word]:
            raise RuntimeError(
                f"Word has no pronunciations: word_index={word_index}, word={word!r}"
            )


def validate_align_phones(
    words: List[str],
    raw_lexicon: PronouncingDictionary,
    align_vocab: Dict[str, int],
    align_vocab_size: int,
) -> None:
    for word_index, word in enumerate(words):
        for pronunciation_index, pronunciation in enumerate(
            raw_lexicon.get_prons(word)
        ):
            if not pronunciation:
                raise RuntimeError(
                    f"Empty aligner pronunciation: word_index={word_index}, "
                    f"word={word!r}, pronunciation_index={pronunciation_index}"
                )
            for phone in pronunciation:
                if phone not in align_vocab:
                    raise KeyError(
                        f"Aligner phone not in model vocab: phone={phone!r}, "
                        f"word_index={word_index}, word={word!r}, "
                        f"pronunciation_index={pronunciation_index}, "
                        f"pronunciation={pronunciation!r}"
                    )
                phone_id = align_vocab[phone]
                if not isinstance(phone_id, int) or not 0 <= phone_id < align_vocab_size:
                    raise KeyError(
                        f"Aligner phone ID is outside model output range: "
                        f"phone={phone!r}, phone_id={phone_id!r}, "
                        f"model_vocab_size={align_vocab_size}, word={word!r}"
                    )


def load_pcm16_mono_wav(path: Path) -> Tuple[np.ndarray, int]:
    if not path.is_file():
        raise FileNotFoundError(f"--wav_path is not a file: {path}")

    try:
        with wave.open(str(path), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            compression = wav_file.getcomptype()
            num_frames = wav_file.getnframes()

            if channels != 1:
                raise ValueError(
                    f"Expected mono WAV, got channels={channels}: {path}"
                )
            if sample_width != 2:
                raise ValueError(
                    f"Expected PCM16 WAV, got sample_width={sample_width}: {path}"
                )
            if sample_rate != TARGET_SAMPLE_RATE:
                raise ValueError(
                    f"Expected {TARGET_SAMPLE_RATE} Hz WAV, got "
                    f"sample_rate={sample_rate}: {path}"
                )
            if compression != "NONE":
                raise ValueError(
                    f"Expected uncompressed PCM WAV, got compression={compression!r}: {path}"
                )
            if num_frames <= 0:
                raise ValueError(f"Empty WAV: {path}")

            raw = wav_file.readframes(num_frames)
    except wave.Error as exc:
        raise ValueError(f"Invalid PCM WAV file: {path}: {exc}") from exc

    samples_i16 = np.frombuffer(raw, dtype="<i2")
    if samples_i16.shape[0] != num_frames:
        raise RuntimeError(
            f"WAV frame count mismatch: header={num_frames}, "
            f"decoded={samples_i16.shape[0]}, path={path}"
        )
    audio = samples_i16.astype(np.float32) / 32768.0
    if audio.ndim != 1 or audio.size <= 0:
        raise RuntimeError(f"Invalid decoded waveform shape={audio.shape}: {path}")
    if not np.isfinite(audio).all():
        raise RuntimeError(f"Decoded waveform contains NaN/Inf: {path}")
    return np.ascontiguousarray(audio), sample_rate


def processor_sample_rate(processor, model_name: str) -> int:
    feature_extractor = getattr(processor, "feature_extractor", None)
    sample_rate = getattr(feature_extractor, "sampling_rate", None)
    if not isinstance(sample_rate, int):
        raise RuntimeError(
            f"{model_name} processor has no integer feature_extractor.sampling_rate: "
            f"{sample_rate!r}"
        )
    if sample_rate != TARGET_SAMPLE_RATE:
        raise RuntimeError(
            f"{model_name} processor expects {sample_rate} Hz, but this script "
            f"requires {TARGET_SAMPLE_RATE} Hz."
        )
    return sample_rate


def validate_model_vocab_size(
    model,
    vocab: Dict[str, int],
    model_name: str,
    *,
    require_equal: bool = True,
) -> int:
    model_vocab_size = getattr(getattr(model, "config", None), "vocab_size", None)
    if not isinstance(model_vocab_size, int):
        raise RuntimeError(f"{model_name} config.vocab_size is not an integer.")
    if require_equal and model_vocab_size != len(vocab):
        raise RuntimeError(
            f"{model_name} vocab size mismatch: model.config.vocab_size="
            f"{model_vocab_size}, loaded_vocab_size={len(vocab)}"
        )
    return model_vocab_size


def load_chunk_model(
    chunk_model_path: Path,
    chunk_vocab: Dict[str, int],
):
    if not chunk_model_path.is_dir():
        raise FileNotFoundError(f"--chunk_model is not a directory: {chunk_model_path}")

    chunk_processor = AutoProcessor.from_pretrained(
        str(chunk_model_path), local_files_only=True
    )
    chunk_model = AutoModelForCTC.from_pretrained(
        str(chunk_model_path), local_files_only=True
    ).to(torch.device("cpu"))
    chunk_model.eval()

    processor_sample_rate(chunk_processor, "chunk model")
    validate_model_vocab_size(chunk_model, chunk_vocab, "chunk model")
    return chunk_processor, chunk_model


def load_align_model(align_model_path: Path):
    if not align_model_path.is_dir():
        raise FileNotFoundError(f"--align_model is not a directory: {align_model_path}")

    align_processor = AutoProcessor.from_pretrained(
        str(align_model_path), local_files_only=True
    )
    align_model = AutoModelForCTC.from_pretrained(
        str(align_model_path), local_files_only=True
    ).to(torch.device("cpu"))
    align_model.eval()

    processor_sample_rate(align_processor, "align model")

    align_tokenizer = getattr(align_processor, "tokenizer", None)
    if align_tokenizer is None or not hasattr(align_tokenizer, "get_vocab"):
        raise RuntimeError("Align processor has no tokenizer.get_vocab().")
    align_vocab = align_tokenizer.get_vocab()
    if not isinstance(align_vocab, dict) or not align_vocab:
        raise RuntimeError("Align tokenizer returned an empty or invalid vocabulary.")
    validate_model_vocab_size(
        align_model,
        align_vocab,
        "align model",
        require_equal=False,
    )

    return align_processor, align_model, align_vocab


def run_chunking(
    *,
    audio: np.ndarray,
    sample_rate: int,
    words: List[str],
    chunk_lexicon: Dict[str, List[List[str]]],
    chunk_vocab: Dict[str, int],
    chunk_processor,
    chunk_model,
    blank_token: str,
    blank_id: int,
    word_spans_out: Optional[List[WordSpan]] = None,
) -> List[Chunk]:
    if chunk_vocab.get(blank_token) != blank_id:
        raise RuntimeError(
            f"Resolved blank token/id mismatch: token={blank_token!r}, "
            f"blank_id={blank_id}, vocab_id={chunk_vocab.get(blank_token)!r}"
        )
    audio_tensor = torch.from_numpy(audio).contiguous()
    log_probs, seconds_per_frame = compute_log_probs(
        chunk_model,
        chunk_processor,
        audio_tensor,
        sample_rate,
        torch.device("cpu"),
    )
    if log_probs.shape[1] != len(chunk_vocab):
        raise RuntimeError(
            f"Chunk log_probs vocab mismatch: log_probs.shape[1]={log_probs.shape[1]}, "
            f"chunk_vocab_size={len(chunk_vocab)}"
        )

    pronunciation = choose_greedy_pronunciations(
        words=words,
        lex=chunk_lexicon,
        phone_to_id=chunk_vocab,
        inter_word_token=None,
    )
    phone_ids = [chunk_vocab[phone] for phone in pronunciation.phones]
    trellis = build_trellis(log_probs, phone_ids, blank_id)
    points = backtrace(trellis, log_probs, phone_ids, blank_id)
    phone_token_segments = points_to_segments(points, pronunciation.phones)
    phone_segments_conf = attach_phone_confidence_from_points(
        phone_token_segs=phone_token_segments,
        points=points,
        target_labels=pronunciation.phones,
        phone_to_id=chunk_vocab,
        log_probs=log_probs,
        mode="emission",
    )

    token_ranges = word_phone_token_ranges(
        phone_token_segs=phone_token_segments,
        words=words,
        prons_per_word=pronunciation.chosen_prons,
        inter_word_token=None,
    )
    word_segments = phones_to_word_segments_by_offsets(
        phone_token_segs=phone_token_segments,
        words=words,
        prons_per_word=pronunciation.chosen_prons,
        inter_word_token=None,
    )
    word_segments_conf = word_segments_with_confidence(
        word_segments, phone_segments_conf
    )

    span_tokens = [segment.label for segment in word_segments_conf]
    assert_tokens_equal("input_transcript_vs_word_spans", words, span_tokens)

    word_spans: List[WordSpan] = []
    for word_index, (segment, pron) in enumerate(
        zip(word_segments_conf, pronunciation.chosen_prons)
    ):
        word_spans.append(
            WordSpan(
                word_index=word_index,
                word=segment.label,
                start_frame=segment.start_frame,
                end_frame=segment.end_frame,
                start_s=float(segment.start_frame * seconds_per_frame),
                end_s=float(segment.end_frame * seconds_per_frame),
                conf_log=float(segment.conf_log),
                pron=list(pron),
            )
        )
    validate_monotonic_word_spans(word_spans)
    if word_spans_out is not None:
        word_spans_out.extend(word_spans)

    emission_frames = emission_frames_by_token_index(
        points, len(pronunciation.phones)
    )
    anchors = make_word_anchors_from_emissions(
        word_spans=word_spans,
        token_ranges=token_ranges,
        token_emission_frames=emission_frames,
        spf=seconds_per_frame,
        anchor_pad_s=0.3,
        audio_dur_s=float(audio.shape[0]) / sample_rate,
    )
    chunks = merge_word_anchors_into_chunks(
        anchors,
        anchor_merge_gap_s=0.2,
    )

    concatenated_chunk_tokens = [word for chunk in chunks for word in chunk.words]
    assert_tokens_equal(
        "input_transcript_vs_concatenated_chunks",
        words,
        concatenated_chunk_tokens,
    )
    return chunks


def round_chunks_to_legacy_grid(
    *,
    raw_chunks: List[Chunk],
    utt_id: str,
    words: List[str],
    num_samples: int,
    sample_rate: int,
) -> List[RuntimeChunk]:
    if not raw_chunks:
        raise RuntimeError("Chunker returned no chunks.")

    audio_duration_s = num_samples / float(sample_rate)
    chunks: List[RuntimeChunk] = []
    for chunk_index, raw_chunk in enumerate(raw_chunks, start=1):
        start_ms = int(round(float(raw_chunk.start) * 1000.0))
        end_ms = int(round(float(raw_chunk.end) * 1000.0))
        if end_ms <= start_ms:
            raise RuntimeError(
                f"Invalid ms-rounded chunk span: chunk_index={chunk_index}, "
                f"raw_start_s={raw_chunk.start}, raw_end_s={raw_chunk.end}, "
                f"start_ms={start_ms}, end_ms={end_ms}"
            )

        raw_end_s = float(raw_chunk.end)
        if raw_end_s > audio_duration_s + 1e-6:
            raise RuntimeError(
                f"Raw chunk exceeds audio duration before ms rounding: "
                f"chunk_index={chunk_index}, raw_end_s={raw_end_s:.9f}, "
                f"audio_duration_s={audio_duration_s:.9f}"
            )

        end_s = end_ms / 1000.0
        if end_s > audio_duration_s:
            overflow_s = end_s - audio_duration_s
            if overflow_s > 0.001:
                raise RuntimeError(
                    f"Chunk exceeds audio duration after ms rounding: "
                    f"chunk_index={chunk_index}, end_s={end_s:.3f}, "
                    f"audio_duration_s={audio_duration_s:.9f}, "
                    f"overflow_s={overflow_s:.9f}"
                )
            max_end_ms = int(math.floor(audio_duration_s * 1000.0))
            if max_end_ms <= start_ms:
                raise RuntimeError(
                    f"Invalid tail clamp after ms rounding: chunk_index={chunk_index}, "
                    f"start_ms={start_ms}, max_end_ms={max_end_ms}"
                )
            end_ms = max_end_ms

        start_sample = int(round((start_ms / 1000.0) * sample_rate))
        end_sample = int(round((end_ms / 1000.0) * sample_rate))
        if start_sample < 0 or end_sample > num_samples or end_sample <= start_sample:
            raise RuntimeError(
                f"Invalid chunk sample span: chunk_index={chunk_index}, "
                f"start_sample={start_sample}, end_sample={end_sample}, "
                f"num_samples={num_samples}"
            )

        chunks.append(
            RuntimeChunk(
                chunk_id=f"{utt_id}.chunk{chunk_index:03d}",
                start_ms=start_ms,
                end_ms=end_ms,
                start_sample=start_sample,
                end_sample=end_sample,
                words=list(raw_chunk.words),
                word_indices=list(raw_chunk.word_indices),
            )
        )

    previous_end_sample = 0
    concatenated_words: List[str] = []
    concatenated_indices: List[int] = []
    for chunk_index, chunk in enumerate(chunks):
        if chunk.start_sample < previous_end_sample:
            raise RuntimeError(
                f"Overlapping chunks after legacy rounding: chunk_index={chunk_index}, "
                f"chunk_id={chunk.chunk_id}, start_sample={chunk.start_sample}, "
                f"previous_end_sample={previous_end_sample}"
            )
        if sorted(chunk.word_indices) != chunk.word_indices:
            raise RuntimeError(
                f"Non-monotonic word indices in chunk {chunk.chunk_id}: "
                f"{chunk.word_indices}"
            )
        previous_end_sample = chunk.end_sample
        concatenated_words.extend(chunk.words)
        concatenated_indices.extend(chunk.word_indices)

    assert_tokens_equal(
        "input_transcript_vs_rounded_chunks",
        words,
        concatenated_words,
    )
    expected_indices = list(range(len(words)))
    if concatenated_indices != expected_indices:
        raise RuntimeError(
            f"Chunk word-index coverage mismatch: expected={expected_indices!r}, "
            f"actual={concatenated_indices!r}"
        )
    return chunks


@torch.inference_mode()
def compute_align_log_probs(
    *,
    model,
    processor,
    audio: np.ndarray,
    sample_rate: int,
) -> np.ndarray:
    inputs = processor(audio, sampling_rate=sample_rate, return_tensors="pt")
    inputs = {name: value.to(torch.device("cpu")) for name, value in inputs.items()}
    logits = model(**inputs).logits
    if logits.ndim != 3 or logits.shape[0] != 1:
        raise RuntimeError(
            f"Expected align logits shape [1,T,V], got {tuple(logits.shape)}"
        )
    log_probs = torch.log_softmax(logits, dim=-1)[0].detach().cpu().numpy()
    if log_probs.ndim != 2 or log_probs.shape[0] <= 0 or log_probs.shape[1] <= 0:
        raise RuntimeError(f"Invalid align log_probs shape: {log_probs.shape}")
    if not np.isfinite(log_probs).all():
        raise RuntimeError("Align model produced NaN/Inf log_probs.")
    return log_probs


def filtered_word_labels(
    word_segments: List[Tuple[str, int, int]],
    config: AlignConfig,
) -> List[str]:
    ignore = {
        "",
        config.word_sil_label.strip().lower(),
        config.sph_word_label.strip().lower(),
        "null",
    }
    return [
        label
        for label, _start, _end in word_segments
        if label.strip().lower() not in ignore
    ]


def align_chunk(
    *,
    audio: np.ndarray,
    sample_rate: int,
    words: List[str],
    raw_lexicon: PronouncingDictionary,
    align_vocab: Dict[str, int],
    align_processor,
    align_model,
    config: AlignConfig,
    context: str,
) -> LocalAlignment:
    if audio.ndim != 1 or audio.size <= 0:
        raise RuntimeError(f"Invalid chunk waveform for {context}: shape={audio.shape}")

    log_probs = compute_align_log_probs(
        model=align_model,
        processor=align_processor,
        audio=audio,
        sample_rate=sample_rate,
    )
    align_model_vocab_size = int(align_model.config.vocab_size)
    if log_probs.shape[1] != align_model_vocab_size:
        raise RuntimeError(
            f"Align log_probs vocab mismatch for {context}: "
            f"log_probs.shape[1]={log_probs.shape[1]}, "
            f"model_vocab_size={align_model_vocab_size}"
        )

    sil_phone_id = align_vocab.get(config.sil_phone)
    sph_phone_id = align_vocab.get(config.sph_phone)
    graph, entry_bias = build_phone_graph_optional_sil_sph(
        words=words,
        prondict=raw_lexicon,
        phone_to_id=align_vocab,
        sil_phone=config.sil_phone,
        optional_sil_between_words=config.optional_sil,
        optional_sil_at_start=None,
        optional_sil_at_end=None,
        sil_cost=config.sil_cost,
        sph_phone=config.sph_phone,
        optional_sph_between_words=config.optional_sph,
        optional_sph_at_start=None,
        optional_sph_at_end=None,
        sph_cost=config.sph_cost,
        sph_word_label=config.sph_word_label,
    )

    first_pass = align_beam_viterbi(
        logp=log_probs,
        graph=graph,
        entry_bias=entry_bias,
        p_stay=config.p_stay,
        beam_size=config.beam,
        word_sil_label=config.word_sil_label,
        boundary_lambda=config.boundary_lambda,
        boundary_context_s=config.boundary_context_s,
        frame_hop_s=config.frame_hop_s,
        sil_phone_id=sil_phone_id,
        min_sil_dur_ms=0.0,
        sil_enter_cost=config.sil_enter_cost,
        sph_phone_id=sph_phone_id,
        sph_enter_cost=config.sph_enter_cost,
    )
    aligned, redecode_stats = redecode_with_pruned_fixed_sequence(
        first_pass_ali=first_pass,
        first_pass_graph=graph,
        first_pass_entry_bias=entry_bias,
        logp=log_probs,
        sil_phone=config.sil_phone,
        sil_phone_id=sil_phone_id,
        sph_phone=config.sph_phone,
        sph_phone_id=sph_phone_id,
        args=config,
    )

    actual_words = filtered_word_labels(aligned.word_segments_f, config)
    validate_word_sequence(actual=actual_words, expected=words, context=context)

    duration_s = float(log_probs.shape[0]) * config.frame_hop_s
    phone_intervals = [
        Interval(
            xmin=float(start_frame) * config.frame_hop_s,
            xmax=float(end_frame) * config.frame_hop_s,
            text=label,
        )
        for label, start_frame, end_frame in aligned.phone_segments_f
    ]
    word_intervals = [
        Interval(
            xmin=float(start_frame) * config.frame_hop_s,
            xmax=float(end_frame) * config.frame_hop_s,
            text=label,
        )
        for label, start_frame, end_frame in aligned.word_segments_f
    ]
    textgrid = TextGrid(
        xmin=0.0,
        xmax=duration_s,
        tiers=[
            IntervalTier("phones", 0.0, duration_s, phone_intervals),
            IntervalTier("words", 0.0, duration_s, word_intervals),
        ],
    )
    validate_textgrid_structure(textgrid, context=context)
    return LocalAlignment(textgrid=textgrid, redecode_stats=redecode_stats)


def validate_textgrid_structure(tg: TextGrid, *, context: str) -> None:
    if not math.isfinite(tg.xmin) or not math.isfinite(tg.xmax):
        raise RuntimeError(f"Non-finite TextGrid bounds for {context}: {tg}")
    if tg.xmax <= tg.xmin:
        raise RuntimeError(f"Invalid TextGrid bounds for {context}: {tg.xmin}..{tg.xmax}")
    if not tg.tiers:
        raise RuntimeError(f"TextGrid has no tiers for {context}")

    tier_names = [tier.name for tier in tg.tiers]
    if len(tier_names) != len(set(tier_names)):
        raise RuntimeError(f"Duplicate tier names for {context}: {tier_names}")
    if "phones" not in tier_names or "words" not in tier_names:
        raise RuntimeError(
            f"TextGrid must contain phones and words tiers for {context}; "
            f"found={tier_names}"
        )

    for tier in tg.tiers:
        if abs(tier.xmin - tg.xmin) > EPS or abs(tier.xmax - tg.xmax) > EPS:
            raise RuntimeError(
                f"Tier/global bounds mismatch for {context}, tier={tier.name}: "
                f"tier={tier.xmin}..{tier.xmax}, global={tg.xmin}..{tg.xmax}"
            )
        previous_end = tier.xmin
        for interval_index, interval in enumerate(tier.intervals):
            if not math.isfinite(interval.xmin) or not math.isfinite(interval.xmax):
                raise RuntimeError(
                    f"Non-finite interval for {context}, tier={tier.name}, "
                    f"interval_index={interval_index}: {interval}"
                )
            if interval.xmax <= interval.xmin:
                raise RuntimeError(
                    f"Non-positive interval for {context}, tier={tier.name}, "
                    f"interval_index={interval_index}: {interval}"
                )
            if interval.xmin < tier.xmin - EPS or interval.xmax > tier.xmax + EPS:
                raise RuntimeError(
                    f"Interval outside tier bounds for {context}, tier={tier.name}, "
                    f"interval_index={interval_index}: {interval}"
                )
            if interval.xmin < previous_end - EPS:
                raise RuntimeError(
                    f"Overlapping/backward intervals for {context}, tier={tier.name}, "
                    f"interval_index={interval_index}, previous_end={previous_end}, "
                    f"interval={interval}"
                )
            previous_end = interval.xmax


def merge_local_alignments(
    *,
    chunks: List[RuntimeChunk],
    local_alignments: List[LocalAlignment],
    full_duration_s: float,
    expected_words: List[str],
    config: AlignConfig,
) -> TextGrid:
    if len(chunks) != len(local_alignments):
        raise RuntimeError(
            f"Chunk/alignment count mismatch: chunks={len(chunks)}, "
            f"alignments={len(local_alignments)}"
        )
    if not chunks:
        raise RuntimeError("Cannot merge zero chunks.")
    if full_duration_s <= 0.0 or not math.isfinite(full_duration_s):
        raise RuntimeError(f"Invalid full audio duration: {full_duration_s}")

    tier_names = [tier.name for tier in local_alignments[0].textgrid.tiers]
    if tier_names != ["phones", "words"]:
        raise RuntimeError(f"Unexpected local tier order: {tier_names}")
    accumulated: Dict[str, List[Interval]] = {name: [] for name in tier_names}
    ignored_word_labels = {
        "",
        "NULL",
        config.word_sil_label,
        config.sph_word_label,
        "null",
    }

    def add_gap(start_s: float, end_s: float) -> None:
        if end_s - start_s <= EPS:
            return
        for tier_name in tier_names:
            accumulated[tier_name].append(Interval(start_s, end_s, "NULL"))

    previous_end = 0.0
    concatenated_local_words: List[str] = []
    for chunk, local_alignment in zip(chunks, local_alignments):
        if chunk.start_s > previous_end + EPS:
            add_gap(previous_end, chunk.start_s)
        elif chunk.start_s < previous_end - EPS:
            raise RuntimeError(
                f"Chunk overlap during merge: chunk_id={chunk.chunk_id}, "
                f"start={chunk.start_s}, previous_end={previous_end}"
            )

        local_tg = local_alignment.textgrid
        local_tier_map = {tier.name: tier for tier in local_tg.tiers}
        if list(local_tier_map) != tier_names:
            raise RuntimeError(
                f"Local tier mismatch for {chunk.chunk_id}: "
                f"actual={list(local_tier_map)}, expected={tier_names}"
            )

        shifted_words: List[Interval] = []
        for tier_name in tier_names:
            local_tier = local_tier_map[tier_name]
            for interval in local_tier.intervals:
                shifted = clip_shift_interval(
                    interval,
                    chunk_start=chunk.start_s,
                    chunk_end=chunk.end_s,
                    local_xmax=local_tier.xmax,
                )
                if shifted is not None:
                    accumulated[tier_name].append(shifted)
                    if tier_name == "words":
                        shifted_words.append(shifted)

        actual_chunk_words = labels_from_intervals(
            shifted_words,
            ignore_labels=ignored_word_labels,
        )
        validate_word_sequence(
            actual=actual_chunk_words,
            expected=chunk.words,
            context=f"shifted local chunk_id={chunk.chunk_id}",
        )
        concatenated_local_words.extend(actual_chunk_words)
        previous_end = chunk.end_s

    if full_duration_s > previous_end + EPS:
        add_gap(previous_end, full_duration_s)
    elif previous_end > full_duration_s + 0.1:
        raise RuntimeError(
            f"Last chunk end {previous_end:.6f}s exceeds full duration "
            f"{full_duration_s:.6f}s"
        )

    final_tiers = [
        IntervalTier(
            name=tier_name,
            xmin=0.0,
            xmax=full_duration_s,
            intervals=merge_adjacent(
                accumulated[tier_name], merge_texts={"NULL"}
            ),
        )
        for tier_name in tier_names
    ]
    merged = TextGrid(0.0, full_duration_s, final_tiers)
    validate_textgrid_structure(merged, context="merged TextGrid")

    word_tier = next(tier for tier in merged.tiers if tier.name == "words")
    merged_words = labels_from_intervals(
        word_tier.intervals,
        ignore_labels=ignored_word_labels,
    )
    validate_word_sequence(
        actual=merged_words,
        expected=expected_words,
        context="final merged TextGrid",
    )
    validate_word_sequence(
        actual=concatenated_local_words,
        expected=expected_words,
        context="concatenated shifted local word intervals",
    )
    return merged


def validate_written_textgrid(
    *,
    path: Path,
    expected_words: List[str],
    config: AlignConfig,
) -> TextGrid:
    parsed = parse_textgrid_long(path)
    validate_textgrid_structure(parsed, context=f"written TextGrid {path}")
    word_tier = next(tier for tier in parsed.tiers if tier.name == "words")
    actual_words = labels_from_intervals(
        word_tier.intervals,
        ignore_labels={
            "",
            "NULL",
            config.word_sil_label,
            config.sph_word_label,
            "null",
        },
    )
    validate_word_sequence(
        actual=actual_words,
        expected=expected_words,
        context=f"written TextGrid word tier: {path}",
    )
    if len(actual_words) != len(expected_words):
        raise RuntimeError(
            f"Final 1:1 word count mismatch: input={len(expected_words)}, "
            f"output={len(actual_words)}"
        )
    for word_index, (expected, actual) in enumerate(zip(expected_words, actual_words)):
        if expected != actual:
            raise RuntimeError(
                f"Final 1:1 word mismatch: word_index={word_index}, "
                f"input={expected!r}, output={actual!r}"
            )
    return parsed


def write_validated_textgrid(
    *,
    textgrid: TextGrid,
    output_path: Path,
    expected_words: List[str],
    config: AlignConfig,
) -> None:
    if output_path.exists():
        raise FileExistsError(f"--output_path already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    if temporary_path.exists():
        raise FileExistsError(f"Temporary output already exists: {temporary_path}")

    write_textgrid_long(textgrid, temporary_path)
    try:
        validate_written_textgrid(
            path=temporary_path,
            expected_words=expected_words,
            config=config,
        )
        temporary_path.replace(output_path)
    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        raise


def write_chunker_metadata(
    *,
    word_spans: List[WordSpan],
    output_path: Path,
) -> None:
    """Write optional product metadata without changing alignment behavior.

    Input format:
        The ordered ``WordSpan`` values already computed by Stage 1.
    Output format:
        UTF-8 JSON with one item per transcript word.
    Logic:
        Exposes the existing CTC emission confidence and the first pronunciation
        used by the Chunker so the Web layer can show low-confidence warnings.
    Usage example:
        ``--chunker_metadata_path result.chunker.json``.
    """
    if output_path.exists():
        raise FileExistsError(f"Chunker metadata already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    if temporary_path.exists():
        raise FileExistsError(
            f"Temporary chunker metadata already exists: {temporary_path}"
        )
    payload = {
        "schema_version": 1,
        "confidence_definition": (
            "geometric mean of Chunker CTC target-emission probabilities"
        ),
        "words": [
            {
                "word_index": span.word_index,
                "word": span.word,
                "start_seconds": span.start_s,
                "end_seconds": span.end_s,
                "confidence": span.conf_prob,
                "confidence_log": span.conf_log,
                "chunker_pronunciation": span.pron,
            }
            for span in word_spans
        ],
    }
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    try:
        temporary_path.replace(output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "CPU-only single-utterance FlexAligner. Input WAV must be "
            "16 kHz, mono, uncompressed PCM16."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--wav_path", type=Path, required=True,
        help="Input 16 kHz mono PCM16 WAV.",
    )
    text_group = parser.add_mutually_exclusive_group(required=True)
    text_group.add_argument("--text", type=str, help="Raw transcript string.")
    text_group.add_argument(
        "--text_path", type=Path, help="UTF-8 transcript text file."
    )
    parser.add_argument(
        "--lexicon", type=Path, required=True,
        help="Pronunciation dictionary used by both stages.",
    )
    parser.add_argument(
        "--chunk_model", type=Path, required=True,
        help="Local CTC chunk model directory containing vocab.json.",
    )
    parser.add_argument(
        "--align_model", type=Path, required=True,
        help="Local CE phone aligner model directory.",
    )
    parser.add_argument(
        "--output_path", type=Path, required=True,
        help="Final TextGrid path; an existing file is rejected.",
    )
    parser.add_argument(
        "--chunker_metadata_path",
        type=Path,
        default=None,
        help=(
            "Optional JSON path for per-word Chunker confidence and the "
            "pronunciation used during Stage 1."
        ),
    )
    parser.add_argument(
        "--num_threads", type=int, default=1,
        help="Positive PyTorch CPU thread count.",
    )
    args = parser.parse_args()
    if args.num_threads <= 0:
        parser.error(f"--num_threads must be positive, got {args.num_threads}")
    return args


def main() -> None:
    total_start = time.perf_counter()
    args = parse_args()
    if args.output_path.exists():
        raise FileExistsError(f"--output_path already exists: {args.output_path}")
    if (
        args.chunker_metadata_path is not None
        and args.chunker_metadata_path.exists()
    ):
        raise FileExistsError(
            f"--chunker_metadata_path already exists: "
            f"{args.chunker_metadata_path}"
        )
    temporary_output_path = args.output_path.with_name(args.output_path.name + ".tmp")
    if temporary_output_path.exists():
        raise FileExistsError(
            f"Temporary output already exists: {temporary_output_path}"
        )
    torch.set_num_threads(args.num_threads)
    device = torch.device("cpu")

    print(f"[ENV] python={sys.version.split()[0]}", flush=True)
    print(f"[ENV] torch={torch.__version__}", flush=True)
    import transformers
    print(f"[ENV] transformers={transformers.__version__}", flush=True)
    print(f"[ENV] numpy={np.__version__}", flush=True)
    print(f"[ENV] device={device}", flush=True)
    print(f"[ENV] torch_threads={torch.get_num_threads()}", flush=True)

    _raw_text, words = read_input_words(args.text, args.text_path)
    raw_lexicon = load_raw_lexicon(args.lexicon)
    validate_transcript_lexicon(words, raw_lexicon)
    chunk_lexicon = build_chunk_lexicon(raw_lexicon)

    if not args.chunk_model.is_dir():
        raise FileNotFoundError(f"--chunk_model is not a directory: {args.chunk_model}")
    if not args.align_model.is_dir():
        raise FileNotFoundError(f"--align_model is not a directory: {args.align_model}")

    audio_load_start = time.perf_counter()
    audio, sample_rate = load_pcm16_mono_wav(args.wav_path)
    utt_id = args.wav_path.stem
    print(
        f"[AUDIO_LOADED] duration_s={audio.shape[0] / sample_rate:.3f} "
        f"elapsed_s={time.perf_counter() - audio_load_start:.3f}",
        flush=True,
    )

    chunk_vocab_path = args.chunk_model / "vocab.json"
    chunk_vocab = read_phone_json(chunk_vocab_path)
    print(f"[MODEL_LOAD_START] stage=chunk path={args.chunk_model}", flush=True)
    chunk_model_load_start = time.perf_counter()
    chunk_processor, chunk_model = load_chunk_model(
        args.chunk_model,
        chunk_vocab,
    )
    print(
        f"[MODEL_LOAD_DONE] stage=chunk "
        f"elapsed_s={time.perf_counter() - chunk_model_load_start:.3f}",
        flush=True,
    )

    blank_token, blank_id = resolve_blank_token(
        "<pad>", chunk_processor, chunk_vocab
    )

    chunk_start = time.perf_counter()
    chunker_word_spans: List[WordSpan] = []
    raw_chunks = run_chunking(
        audio=audio,
        sample_rate=sample_rate,
        words=words,
        chunk_lexicon=chunk_lexicon,
        chunk_vocab=chunk_vocab,
        chunk_processor=chunk_processor,
        chunk_model=chunk_model,
        blank_token=blank_token,
        blank_id=blank_id,
        word_spans_out=chunker_word_spans,
    )
    chunks = round_chunks_to_legacy_grid(
        raw_chunks=raw_chunks,
        utt_id=utt_id,
        words=words,
        num_samples=int(audio.shape[0]),
        sample_rate=sample_rate,
    )
    chunk_elapsed_s = time.perf_counter() - chunk_start

    chunk_release_start = time.perf_counter()
    del raw_chunks
    del chunk_model
    del chunk_processor
    del chunk_lexicon
    del chunk_vocab
    del blank_token
    del blank_id
    gc.collect()
    print(
        f"[MODEL_RELEASED] stage=chunk "
        f"elapsed_s={time.perf_counter() - chunk_release_start:.3f}",
        flush=True,
    )

    print(
        f"[CHUNKING_DONE] elapsed_s={chunk_elapsed_s:.3f} "
        f"chunk_count={len(chunks)}",
        flush=True,
    )
    for chunk in chunks:
        print(
            f"[CHUNK] chunk_id={chunk.chunk_id} "
            f"chunk_time={chunk.start_s:.3f}-{chunk.end_s:.3f} "
            f"chunk_words={json.dumps(' '.join(chunk.words), ensure_ascii=False)}",
            flush=True,
        )

    print(f"[MODEL_LOAD_START] stage=align path={args.align_model}", flush=True)
    align_model_load_start = time.perf_counter()
    align_processor, align_model, align_vocab = load_align_model(args.align_model)
    align_model_vocab_size = int(align_model.config.vocab_size)
    validate_align_phones(
        words,
        raw_lexicon,
        align_vocab,
        align_vocab_size=align_model_vocab_size,
    )

    config = AlignConfig()
    for special_name, special_phone in (
        ("sil_phone", config.sil_phone),
        ("sph_phone", config.sph_phone),
    ):
        if special_phone not in align_vocab:
            raise KeyError(
                f"{special_name}={special_phone!r} is not in align model vocab."
            )
        special_id = align_vocab[special_phone]
        if (
            not isinstance(special_id, int)
            or not 0 <= special_id < align_model_vocab_size
        ):
            raise KeyError(
                f"{special_name}={special_phone!r} has invalid model output ID="
                f"{special_id!r}; model_vocab_size={align_model_vocab_size}"
            )
    print(
        f"[MODEL_LOAD_DONE] stage=align "
        f"elapsed_s={time.perf_counter() - align_model_load_start:.3f}",
        flush=True,
    )

    local_alignments: List[LocalAlignment] = []
    for chunk in chunks:
        chunk_words_text = json.dumps(" ".join(chunk.words), ensure_ascii=False)
        print(
            f"[ALIGN_START] chunk_id={chunk.chunk_id} "
            f"chunk_time={chunk.start_s:.3f}-{chunk.end_s:.3f} "
            f"chunk_words={chunk_words_text}",
            flush=True,
        )
        align_start = time.perf_counter()
        chunk_audio = np.ascontiguousarray(
            audio[chunk.start_sample:chunk.end_sample]
        )
        local_alignment = align_chunk(
            audio=chunk_audio,
            sample_rate=sample_rate,
            words=chunk.words,
            raw_lexicon=raw_lexicon,
            align_vocab=align_vocab,
            align_processor=align_processor,
            align_model=align_model,
            config=config,
            context=f"local chunk_id={chunk.chunk_id}",
        )
        local_alignments.append(local_alignment)
        align_elapsed_s = time.perf_counter() - align_start
        print(
            f"[ALIGN_DONE] chunk_id={chunk.chunk_id} "
            f"elapsed_s={align_elapsed_s:.3f}",
            flush=True,
        )

    align_release_start = time.perf_counter()
    del chunk_audio
    del align_model
    del align_processor
    del align_vocab
    gc.collect()
    print(
        f"[MODEL_RELEASED] stage=align "
        f"elapsed_s={time.perf_counter() - align_release_start:.3f}",
        flush=True,
    )

    merged = merge_local_alignments(
        chunks=chunks,
        local_alignments=local_alignments,
        full_duration_s=float(audio.shape[0]) / sample_rate,
        expected_words=words,
        config=config,
    )
    write_validated_textgrid(
        textgrid=merged,
        output_path=args.output_path,
        expected_words=words,
        config=config,
    )
    if args.chunker_metadata_path is not None:
        write_chunker_metadata(
            word_spans=chunker_word_spans,
            output_path=args.chunker_metadata_path,
        )

    validated = validate_written_textgrid(
        path=args.output_path,
        expected_words=words,
        config=config,
    )
    final_word_tier = next(tier for tier in validated.tiers if tier.name == "words")
    final_words = labels_from_intervals(
        final_word_tier.intervals,
        ignore_labels={
            "",
            "NULL",
            config.word_sil_label,
            config.sph_word_label,
            "null",
        },
    )
    print(
        f"[VALIDATION] input_word_count={len(words)} "
        f"output_word_count={len(final_words)} status=passed",
        flush=True,
    )
    print(f"[OUTPUT] textgrid={args.output_path.resolve()}", flush=True)
    total_elapsed_s = time.perf_counter() - total_start
    print(f"[TOTAL_DONE] elapsed_s={total_elapsed_s:.3f}", flush=True)


if __name__ == "__main__":
    main()
