#!/usr/bin/env python3
"""
CTC forced alignment using a Hugging Face Wav2Vec2ForCTC checkpoint.

Inputs:
  - audio file(s)
  - word transcript file(s) (plain text: words separated by whitespace)
  - lexicon (word -> phones), allowing multiple pronunciations per word
  - phone token list (or mapping) used by the CTC head

Outputs:
  - phone segments: start/end times + phone
  - (optional) word segments: start/end times + word

This implementation:
  - runs the model to get framewise log-probs
  - expands transcript words into phone sequences using lexicon
  - handles multiple pronunciations via beam search over pronunciation variants
  - aligns the selected phone sequence with CTC Viterbi trellis + backtrace

Dependencies:
  pip install torch torchaudio transformers soundfile

Notes:
  - Your model's CTC labels must correspond to phone tokens. This script assumes that.
  - You must provide a phone token list that matches the model's output vocabulary order.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
# 在文件最开头的 import 区域加入：
import soundfile as sf
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


# -----------------------------
# Utilities: lexicon + text
# -----------------------------

def normalize_word(w: str) -> str:
    # Adjust as you like: lowercase, strip punctuation, etc.
    w = w.strip()
    w = w.lower()
    # keep apostrophes inside words, remove leading/trailing punctuation
    w = re.sub(r"^[^\w']+|[^\w']+$", "", w)
    return w


def read_transcript(path: Path) -> List[str]:
    txt = path.read_text(encoding="utf-8", errors="ignore")
    words = [normalize_word(w) for w in txt.strip().split()]
    words = [w for w in words if w]
    return words


def read_lexicon(path: Path) -> Dict[str, List[List[str]]]:
    """
    Lexicon format (one pronunciation per line):
      WORD  PH1 PH2 PH3 ...
    Multiple lines for the same WORD => multiple pronunciations.

    Returns:
      lex[word] = [ [ph1, ph2, ...], [alt1, alt2, ...], ... ]
    """
    lex: Dict[str, List[List[str]]] = {}
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            word = normalize_word(parts[0])
            phones = parts[1:]
            lex.setdefault(word, []).append(phones)
    return lex


def read_phone_json(path: Path) -> Dict[str, int]:
    """
    JSON format:
      {
        "<pad>": 0,
        "AA": 1,
        "AE": 2,
        ...
      }
    """
    with path.open("r", encoding="utf-8") as f:
        phone_to_id = json.load(f)

    if not isinstance(phone_to_id, dict):
        raise ValueError("phones.json must be a dict {phone: token_id}")

    # sanity checks
    ids = list(phone_to_id.values())
    if not all(isinstance(v, int) for v in ids):
        raise ValueError("All token ids must be integers")

    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate token ids found in phones.json")

    return phone_to_id


# -----------------------------
# CTC forced alignment (Viterbi trellis + backtrace)
# -----------------------------

@dataclass
class Point:
    token_index: int  # index in target token sequence
    time_index: int   # frame index

def build_trellis(log_probs: torch.Tensor, targets: List[int], blank_id: int) -> torch.Tensor:
    """
    Vectorized Viterbi trellis for CTC forced alignment.

    log_probs: (T, V) log-probabilities
    targets: list of target token ids (length N)
    returns trellis: (T+1, N+1)
    """
    T, V = log_probs.shape
    N = len(targets)
    device = log_probs.device
    neg_inf = -float("inf")

    targets_t = torch.tensor(targets, device=device, dtype=torch.long)

    trellis = torch.full((T + 1, N + 1), neg_inf, device=device, dtype=log_probs.dtype)
    trellis[0, 0] = 0.0

    # First column: blanks only
    trellis[1:, 0] = torch.cumsum(log_probs[:, blank_id], dim=0)

    for t in range(1, T + 1):
        lp_t = log_probs[t - 1]
        blank = lp_t[blank_id]                       # scalar
        emit_scores = lp_t[targets_t]                # (N,)

        stay = trellis[t - 1, 1:] + blank            # (N,)
        emit = trellis[t - 1, :-1] + emit_scores     # (N,)

        trellis[t, 1:] = torch.maximum(stay, emit)

    return trellis

def backtrace(trellis: torch.Tensor, log_probs: torch.Tensor, targets: List[int], blank_id: int) -> List[Point]:
    """
    Backtrace best path from trellis.
    Returns list of Points with (token_index, time_index) when token was consumed.
    """
    _T = trellis.size(0) - 1
    N = trellis.size(1) - 1

    j = N
    t = int(torch.argmax(trellis[:, j]).item())  # best end time for consuming all tokens
    path: List[Point] = []

    while t > 0 and j > 0:
        lp_t = log_probs[t - 1]
        # Compare whether we stayed (blank) or emitted target[j-1]
        score_stay = trellis[t - 1, j] + lp_t[blank_id]
        score_emit = trellis[t - 1, j - 1] + lp_t[targets[j - 1]]

        if score_emit > score_stay:
            # emitted token j-1 at time t-1
            path.append(Point(token_index=j - 1, time_index=t - 1))
            j -= 1
            t -= 1
        else:
            # stayed with blank
            t -= 1

    path.reverse()
    return path


@dataclass
class Segment:
    label: str
    start_frame: int
    end_frame: int  # exclusive


def points_to_segments(points: List[Point], target_labels: List[str]) -> List[Segment]:
    """
    Convert token-consumption points into contiguous segments in frame indices.
    Each token occurs once in this simple trellis formulation.
    """
    if not points:
        return []

    segs: List[Segment] = []
    for i, p in enumerate(points):
        start = p.time_index
        end = points[i + 1].time_index if i + 1 < len(points) else (p.time_index + 1)
        segs.append(Segment(label=target_labels[p.token_index], start_frame=start, end_frame=end))
    return segs


def merge_repeated_labels(segs: List[Segment]) -> List[Segment]:
    if not segs:
        return []
    merged = [segs[0]]
    for s in segs[1:]:
        last = merged[-1]
        if s.label == last.label and s.start_frame <= last.end_frame:
            last.end_frame = max(last.end_frame, s.end_frame)
        else:
            merged.append(s)
    return merged
# -----------------------------
# Confidence + TextGrid helpers
# -----------------------------

@dataclass
class SegmentWithConf:
    label: str
    start_frame: int
    end_frame: int  # exclusive
    conf_log: float  # log-prob confidence (higher is better)


def compute_segment_confidence(
    seg: Segment,
    label: str,
    label_to_id: Dict[str, int],
    log_probs: torch.Tensor,
    emission_frame: Optional[int],
    mode: str = "emission",
) -> float:
    """Return log confidence for a segment.
    - emission: log p(label | emission_frame), with midpoint fallback
    - avg_frame: mean_t log p(label | frame=t) over the segment frames
    """
    pid = label_to_id[label]
    T = int(log_probs.size(0))

    if mode == "emission":
        t = emission_frame
        if t is None:
            t = (seg.start_frame + seg.end_frame) // 2
        t = max(0, min(int(t), T - 1))
        return float(log_probs[t, pid].item())

    # avg_frame
    s = max(0, int(seg.start_frame))
    e = min(int(seg.end_frame), T)
    if e <= s:
        t = max(0, min((seg.start_frame + seg.end_frame) // 2, T - 1))
        return float(log_probs[t, pid].item())
    return float(log_probs[s:e, pid].mean().item())


def attach_phone_confidence_from_points(
    merged_phone_segs: List[Segment],
    points: List[Point],
    target_labels: List[str],  # labels per token BEFORE merge
    phone_to_id: Dict[str, int],
    log_probs: torch.Tensor,
    mode: str,
) -> List[SegmentWithConf]:
    """Attach log-confidence to merged phone segments using emission frames from token-level points."""
    emission_frames: List[Optional[int]] = [None] * len(target_labels)
    for p in points:
        emission_frames[p.token_index] = p.time_index

    out: List[SegmentWithConf] = []
    token_i = 0
    for seg in merged_phone_segs:
        while token_i < len(target_labels) and target_labels[token_i] != seg.label:
            token_i += 1
        emit_t = emission_frames[token_i] if token_i < len(emission_frames) else None
        conf_log = compute_segment_confidence(seg, seg.label, phone_to_id, log_probs, emit_t, mode=mode)
        out.append(SegmentWithConf(seg.label, seg.start_frame, seg.end_frame, conf_log))
        token_i += 1
    return out


def word_segments_with_confidence(
    word_segs: List[Segment],
    phone_segs_conf: List[SegmentWithConf],
) -> List[SegmentWithConf]:
    """Word log-confidence = mean of overlapping phone log-confidences."""
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
        conf_log = float(sum(confs) / len(confs)) if confs else float("nan")
        out.append(SegmentWithConf(w.label, w.start_frame, w.end_frame, conf_log))
        pi = pj
    return out


def write_ctm_with_optional_conf(
    path: Path,
    utt_id: str,
    segs: List[SegmentWithConf],
    spf: float,
    channel: int = 1,
    include_conf: bool = False,
):
    """Write CTM. If include_conf=True, append a confidence column exp(conf_log)."""
    lines = []
    for s in segs:
        start = s.start_frame * spf
        dur = max(0.0, (s.end_frame - s.start_frame) * spf)
        if include_conf:
            conf = math.exp(s.conf_log) if (s.conf_log == s.conf_log) else -1.0  # NaN -> -1
            lines.append(f"{utt_id} {channel} {start:.3f} {dur:.3f} {s.label} {conf:.6f}")
        else:
            lines.append(f"{utt_id} {channel} {start:.3f} {dur:.3f} {s.label}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_textgrid(
    path: Path,
    xmin: float,
    xmax: float,
    tiers: Dict[str, List[Tuple[float, float, str]]],
):
    """Write Praat TextGrid (text format) with IntervalTier(s)."""
    def esc(s: str) -> str:
        return s.replace('"', '""')

    tier_names = list(tiers.keys())
    lines = []
    lines.append('File type = "ooTextFile"')
    lines.append('Object class = "TextGrid"')
    lines.append("")
    lines.append(f"xmin = {xmin:.6f}")
    lines.append(f"xmax = {xmax:.6f}")
    lines.append("tiers? <exists>")
    lines.append(f"size = {len(tier_names)}")
    lines.append("item []:")

    for i, name in enumerate(tier_names, 1):
        intervals = tiers[name]
        lines.append(f"    item [{i}]:")
        lines.append('        class = "IntervalTier"')
        lines.append(f'        name = "{esc(name)}"')
        lines.append(f"        xmin = {xmin:.6f}")
        lines.append(f"        xmax = {xmax:.6f}")
        lines.append(f"        intervals: size = {len(intervals)}")
        for j, (s, e, lab) in enumerate(intervals, 1):
            lines.append(f"        intervals [{j}]:")
            lines.append(f"            xmin = {s:.6f}")
            lines.append(f"            xmax = {e:.6f}")
            lines.append(f'            text = "{esc(lab)}"')

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# -----------------------------
# Chunking helpers (based on word CTM)
# -----------------------------

@dataclass
class WordSeg:
    start: float
    dur: float
    word: str

    @property
    def end(self) -> float:
        return self.start + self.dur


@dataclass
class Chunk:
    start: float
    end: float
    words: List[str]

    @property
    def dur(self) -> float:
        return self.end - self.start


def read_ctm_words(ctm_path: Path) -> List[WordSeg]:
    """Read word CTM: utt channel start duration word [optional...]"""
    segs: List[WordSeg] = []
    with ctm_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            _utt, _ch, start, dur, word = parts[:5]
            try:
                s = float(start)
                d = float(dur)
            except ValueError:
                continue
            if d <= 0:
                continue
            segs.append(WordSeg(start=s, dur=d, word=word))
    segs.sort(key=lambda x: x.start)
    return segs


def merge_words_into_chunks(
    words: List[WordSeg],
    max_gap_s: float,
    min_chunk_s: float,
    max_chunk_s: float,
    min_words: int,
) -> List[Chunk]:
    if not words:
        return []
    chunks: List[Chunk] = []
    cur_words = [words[0].word]
    cur_start = words[0].start
    cur_end = words[0].end

    for w in words[1:]:
        gap = w.start - cur_end
        proposed_end = w.end
        proposed_dur = proposed_end - cur_start

        if gap <= max_gap_s and proposed_dur <= max_chunk_s:
            cur_end = proposed_end
            cur_words.append(w.word)
        else:
            if (cur_end - cur_start) >= min_chunk_s and len(cur_words) >= min_words:
                chunks.append(Chunk(start=cur_start, end=cur_end, words=cur_words))
            cur_start = w.start
            cur_end = w.end
            cur_words = [w.word]

    if (cur_end - cur_start) >= min_chunk_s and len(cur_words) >= min_words:
        chunks.append(Chunk(start=cur_start, end=cur_end, words=cur_words))

    return chunks


def pad_chunks_without_cutting_words(
    chunks: List[Chunk],
    words: List[WordSeg],
    pad_s: float,
    max_pad_into_gap_s: float,
    audio_dur_s: float,
) -> List[Chunk]:
    if not chunks:
        return chunks

    out: List[Chunk] = []
    # map word boundaries for quick lookup
    for c in chunks:
        # find first/last word indices by time
        first_i = None
        last_i = None
        for i, w in enumerate(words):
            if first_i is None and abs(w.start - c.start) < 1e-6:
                first_i = i
            if abs(w.end - c.end) < 1e-6:
                last_i = i
        left_limit = words[first_i - 1].end if (first_i is not None and first_i > 0) else 0.0
        right_limit = words[last_i + 1].start if (last_i is not None and last_i + 1 < len(words)) else audio_dur_s

        left_gap = max(0.0, c.start - left_limit)
        right_gap = max(0.0, right_limit - c.end)

        left_pad = min(pad_s, max_pad_into_gap_s, left_gap)
        right_pad = min(pad_s, max_pad_into_gap_s, right_gap)

        ns = max(0.0, c.start - left_pad)
        ne = min(audio_dur_s, c.end + right_pad)

        out.append(Chunk(start=ns, end=ne, words=c.words))
    return out


# def save_chunk_wavs_and_manifests(
#     audio_path: Path,
#     out_dir: Path,
#     chunks: List[Chunk],
#     base: str,
# ):
#     out_dir.mkdir(parents=True, exist_ok=True)
#     wav, sr = torchaudio.load(str(audio_path))
#     audio_dur_s = wav.size(1) / sr

#     jsonl_path = out_dir / f"{base}.chunks.jsonl"
#     tsv_path = out_dir / f"{base}.chunks.tsv"
#     with jsonl_path.open("w", encoding="utf-8") as fj, tsv_path.open("w", encoding="utf-8") as ft:
#         ft.write("chunk_id\tstart_s\tend_s\tdur_s\twords\n")
#         for i, c in enumerate(chunks, 1):
#             s0 = int(round(c.start * sr))
#             s1 = int(round(c.end * sr))
#             s0 = max(0, min(s0, wav.size(1)))
#             s1 = max(0, min(s1, wav.size(1)))
#             if s1 <= s0:
#                 continue

#             chunk_id = f"{base}.chunk{i:03d}"
#             out_wav = out_dir / f"{chunk_id}_{c.start:.3f}-{c.end:.3f}.wav"
#             torchaudio.save(str(out_wav), wav[:, s0:s1], sr)

#             obj = {
#                 "chunk_id": chunk_id,
#                 "audio": str(out_wav),
#                 "start_s": round(c.start, 3),
#                 "end_s": round(c.end, 3),
#                 "dur_s": round(c.end - c.start, 3),
#                 "words": c.words,
#                 "text": " ".join(c.words),
#             }
#             fj.write(json.dumps(obj, ensure_ascii=False) + "\n")
#             ft.write(f"{chunk_id}\t{obj['start_s']}\t{obj['end_s']}\t{obj['dur_s']}\t{obj['text']}\n")

# 别忘了在文件最开头确保有这个导入：
# import soundfile as sf

def save_chunk_wavs_and_manifests(
    audio_path: Path,
    out_dir: Path,
    chunks: List[Chunk],
    base: str,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 🔴 [修改点 1] 替换 torchaudio.load -> sf.read
    # wav, sr = torchaudio.load(str(audio_path))
    wav_np, sr = sf.read(str(audio_path))
    wav = torch.from_numpy(wav_np).float()
    if wav.ndim == 1:
        wav = wav.unsqueeze(0) # (1, T)
    else:
        wav = wav.t()          # (T, C) -> (C, T)

    # audio_dur_s = wav.size(1) / sr

    jsonl_path = out_dir / f"{base}.chunks.jsonl"
    tsv_path = out_dir / f"{base}.chunks.tsv"
    with jsonl_path.open("w", encoding="utf-8") as fj, tsv_path.open("w", encoding="utf-8") as ft:
        ft.write("chunk_id\tstart_s\tend_s\tdur_s\twords\n")
        for i, c in enumerate(chunks, 1):
            s0 = int(round(c.start * sr))
            s1 = int(round(c.end * sr))
            s0 = max(0, min(s0, wav.size(1)))
            s1 = max(0, min(s1, wav.size(1)))
            if s1 <= s0:
                continue

            chunk_id = f"{base}.chunk{i:03d}"
            out_wav = out_dir / f"{chunk_id}_{c.start:.3f}-{c.end:.3f}.wav"
            
            # 🔴 [修改点 2] 替换 torchaudio.save -> sf.write
            # torchaudio.save(str(out_wav), wav[:, s0:s1], sr)
            segment_numpy = wav[:, s0:s1].squeeze(0).numpy()
            sf.write(str(out_wav), segment_numpy, sr)

            obj = {
                "chunk_id": chunk_id,
                "audio": str(out_wav),
                "start_s": round(c.start, 3),
                "end_s": round(c.end, 3),
                "dur_s": round(c.end - c.start, 3),
                "words": c.words,
                "text": " ".join(c.words),
            }
            fj.write(json.dumps(obj, ensure_ascii=False) + "\n")
            ft.write(f"{chunk_id}\t{obj['start_s']}\t{obj['end_s']}\t{obj['dur_s']}\t{obj['text']}\n")

# -----------------------------
# Pronunciation beam search
# -----------------------------

@dataclass
class PronCandidate:
    phones: List[str]
    pron_choice_idxs: List[int]  # pronunciation index per word
    score: float  # CTC best-path trellis score (higher is better)


def words_to_pronunciations(words: List[str], lex: Dict[str, List[List[str]]]) -> List[List[List[str]]]:
    """
    For each word, return list of pronunciation variants (each is list of phones).
    Raises on OOV.
    """
    out = []
    oov = []
    for w in words:
        if w not in lex:
            oov.append(w)
        else:
            out.append(lex[w])
    if oov:
        raise ValueError(f"OOV words not found in lexicon: {sorted(set(oov))}")
    return out


def score_phone_sequence_ctc(
    log_probs: torch.Tensor,
    phone_ids: List[int],
    blank_id: int,
) -> float:
    trellis = build_trellis(log_probs, phone_ids, blank_id)
    # best end score for consuming all tokens
    best = float(torch.max(trellis[:, len(phone_ids)]).item())
    return best


def beam_search_pronunciations(
    log_probs: torch.Tensor,
    words: List[str],
    prons_per_word: List[List[List[str]]],
    phone_to_id: Dict[str, int],
    blank_id: int,
    beam_size: int = 8,
    inter_word_token: Optional[str] = None,
) -> PronCandidate:
    """
    Beam over pronunciation variants by repeatedly extending partial sequences.
    To keep it simple and robust, we score full candidates with CTC DP each time.
    This is O(beam * variants * DP) but works fine for typical utterances.

    inter_word_token: optional token (e.g., "SIL" or "|") inserted between words if present in phone vocab.
    """
    beam: List[PronCandidate] = [PronCandidate(phones=[], pron_choice_idxs=[], score=-float("inf"))]

    for wi, word in enumerate(words):
        new_beam: List[PronCandidate] = []
        variants = prons_per_word[wi]  # list of phone lists

        for cand in beam:
            for pi, pron in enumerate(variants):
                phones = cand.phones.copy()
                if phones and inter_word_token is not None:
                    phones.append(inter_word_token)
                phones.extend(pron)

                # map to ids; if a phone is unknown to the model vocab, fail fast
                try:
                    ids = [phone_to_id[p] for p in phones]
                except KeyError as e:
                    raise ValueError(f"Phone '{e.args[0]}' not found in phone token list / model vocab.")

                score = score_phone_sequence_ctc(log_probs, ids, blank_id)
                new_beam.append(
                    PronCandidate(
                        phones=phones,
                        pron_choice_idxs=cand.pron_choice_idxs + [pi],
                        score=score,
                    )
                )

        # keep top beam_size by score
        new_beam.sort(key=lambda x: x.score, reverse=True)
        beam = new_beam[:beam_size]

    # best
    best = max(beam, key=lambda x: x.score)
    return best


# -----------------------------
# Model inference
# -----------------------------

# def load_audio(path: Path, target_sr: int) -> torch.Tensor:
#     wav, sr = torchaudio.load(str(path))
#     if wav.size(0) > 1:
#         wav = torch.mean(wav, dim=0, keepdim=True)  # mono
#     if sr != target_sr:
#         wav = torchaudio.functional.resample(wav, sr, target_sr)
#     return wav.squeeze(0)  # (num_samples,)

def load_audio(path: Path, target_sr: int) -> torch.Tensor:
    """
    Robust audio loading using soundfile directly.
    Bypasses torchaudio.load to avoid backend dependency hell.
    """
    # 1. 直接用 soundfile 读取 (返回 numpy array 和 sample_rate)
    # sf.read 返回的是 (Time, Channels) 或 (Time,)
    wav_numpy, sr = sf.read(str(path))
    
    # 2. 转为 Tensor
    wav = torch.from_numpy(wav_numpy).float()
    
    # 3. 处理维度: 确保变成 (Channels, Time)
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)  # (1, Time)
    else:
        wav = wav.t()           # (Time, Channels) -> (Channels, Time)
        
    # 4. 转单声道 (Mono)
    if wav.size(0) > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)
        
    # 5. 重采样 (Resample)
    if sr != target_sr:
        # 只使用 torchaudio 的函数库，不涉及 I/O，非常安全
        import torchaudio.functional as F
        wav = F.resample(wav, sr, target_sr)
        
    return wav.squeeze(0)  # (Num_Samples,)

@torch.inference_mode()
def compute_log_probs(
    model: Wav2Vec2ForCTC,
    processor: Wav2Vec2Processor,
    audio: torch.Tensor,
    sr: int,
    device: torch.device,
) -> Tuple[torch.Tensor, float]:
    """
    Returns:
      log_probs: (T, V) in log space
      seconds_per_frame: approximate seconds per frame
    """
    inputs = processor(audio.numpy(), sampling_rate=sr, return_tensors="pt")
    input_values = inputs.input_values.to(device)

    logits = model(input_values).logits  # (B, T, V)
    log_probs = torch.log_softmax(logits, dim=-1).squeeze(0)  # (T, V)

    # Approx frame rate: Wav2Vec2 has a feature encoder with stride (~20ms typical).
    # We estimate seconds_per_frame by dividing audio length by number of frames.
    seconds_per_frame = float(audio.numel() / sr) / float(log_probs.size(0))

    return log_probs, seconds_per_frame


# -----------------------------
# Output helpers
# -----------------------------

def write_ctm(path: Path, utt_id: str, segs: List[Segment], spf: float, channel: int = 1):
    """
    CTM format: utt channel start duration label
    """
    lines = []
    for s in segs:
        start = s.start_frame * spf
        dur = max(0.0, (s.end_frame - s.start_frame) * spf)
        lines.append(f"{utt_id} {channel} {start:.3f} {dur:.3f} {s.label}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def phones_to_word_segments(phone_segs: List[Segment], words: List[str], prons: List[List[str]], inter_word_token: Optional[str]) -> List[Segment]:
    """
    Build word segments by grouping phone segments according to the chosen pronunciations.
    Assumes phone_segs correspond exactly to the expanded phone target (including inter_word_token if used),
    after merging repeated labels.
    """
    # Flatten target with boundaries
    flat: List[str] = []
    for i, phs in enumerate(prons):
        if i > 0 and inter_word_token is not None:
            flat.append(inter_word_token)
        flat.extend(phs)

    # Create index mapping from flat phones to segments (best-effort)
    # We walk through phone_segs and flat together.
    wi = 0
    word_segs: List[Segment] = []
    flat_i = 0

    def consume_boundary():
        nonlocal flat_i
        if inter_word_token is not None and flat_i < len(flat) and flat[flat_i] == inter_word_token:
            flat_i += 1

    for word_i, w in enumerate(words):
        # boundary before this word except first
        if word_i > 0:
            consume_boundary()

        # consume phones for this word
        phs = prons[word_i]
        start_frame = None
        end_frame = None
        for _ in phs:
            if flat_i >= len(flat):
                break
            target_phone = flat[flat_i]
            # find next matching phone seg
            # (We assume alignment is consistent; if not, this may degrade.)
            # Advance through phone_segs until labels match.
            while wi < len(phone_segs) and phone_segs[wi].label != target_phone:
                wi += 1
            if wi >= len(phone_segs):
                break
            seg = phone_segs[wi]
            if start_frame is None:
                start_frame = seg.start_frame
            end_frame = seg.end_frame
            wi += 1
            flat_i += 1

        if start_frame is None:
            # couldn't form; skip
            continue
        word_segs.append(Segment(label=w, start_frame=start_frame, end_frame=end_frame if end_frame is not None else start_frame + 1))

    return word_segs


def phones_to_word_segments_by_offsets(
    phone_token_segs,
    words,
    prons_per_word,
    inter_word_token=None,
):
    """
    Build word segments by consuming the token-level phone segments in order.

    Args:
        phone_token_segs: list[Segment]
            Token-level segments aligned to the *target phone token sequence*.
            IMPORTANT: this must be the un-merged output of points_to_segments(points, best.phones),
            i.e., one segment per target token, in the same order as best.phones.
        words: list[str]
            Word sequence (same length as prons_per_word).
        prons_per_word: list[list[str]]
            The chosen pronunciation per word, as a list of phone symbols per word.
            (This should correspond to the same phone token stream used in alignment.)
        inter_word_token: Optional[str]
            Phone token inserted between words in the target stream (e.g., "|", "SIL", "SP").
            If provided, this function will consume it between words.

    Returns:
        word_segs: list[Segment]
            Each Segment has label=word, start_frame/end_frame covering only that word’s phones.
    """
    if len(words) != len(prons_per_word):
        raise ValueError(f"len(words)={len(words)} != len(prons_per_word)={len(prons_per_word)}")

    wi = 0  # index into phone_token_segs
    word_segs = []

    for k, (word, prons) in enumerate(zip(words, prons_per_word)):
        # consume inter-word token before every word except the first
        if k > 0 and inter_word_token is not None:
            if wi >= len(phone_token_segs):
                raise ValueError("Ran out of phone token segments while consuming inter-word token.")
            # Be tolerant: consume exactly one token; optionally sanity-check the label
            if phone_token_segs[wi].label != inter_word_token:
                # If your pipeline sometimes omits it, you can comment this out.
                # But failing fast is usually better for debugging alignment/tokenization mismatches.
                raise ValueError(
                    f"Expected inter_word_token '{inter_word_token}' at token idx {wi}, "
                    f"got '{phone_token_segs[wi].label}'"
                )
            wi += 1

        n = len(prons)
        if n == 0:
            # Empty pronunciation: create a zero-length segment anchored at current position.
            # (Rare; keep or remove depending on your lexicon.)
            if wi == 0:
                start = end = 0
            else:
                start = end = phone_token_segs[wi - 1].end_frame
            word_segs.append(Segment(word, start, end))
            continue

        if wi + n > len(phone_token_segs):
            raise ValueError(
                f"Ran out of phone token segments for word '{word}': need {n}, "
                f"have {len(phone_token_segs) - wi}"
            )

        start_frame = phone_token_segs[wi].start_frame
        end_frame = phone_token_segs[wi + n - 1].end_frame
        word_segs.append(Segment(word, start_frame, end_frame))
        wi += n

    return word_segs


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, required=True,
                    help="Path to HF checkpoint dir (contains config.json, pytorch_model.bin, etc.)")
    ap.add_argument("--processor_dir", type=str, default=None,
                    help="Path to HF processor dir. If omitted, uses model_dir.")
    ap.add_argument("--lexicon", type=str, required=True,
                    help="Pronouncing dictionary: WORD PH1 PH2 ... (multiple lines per word allowed)")
    ap.add_argument("--phone_json", type=str, required=True,
                help="JSON mapping phone -> token id")
    ap.add_argument("--blank_token", type=str, default=None,
                    help="Blank token string. If omitted, will try processor.tokenizer.pad_token or '<pad>'.")
    ap.add_argument("--inter_word_token", type=str, default=None,
                    help="Optional token inserted between words (e.g., 'SIL' or '|') if your model uses it.")
    ap.add_argument("--beam_size", type=int, default=8,
                    help="Beam size for pronunciation search.")
    ap.add_argument("--audio", type=str, required=True,
                help="Audio file (single utterance).")
    ap.add_argument("--transcript", type=str, required=True,
                help="Transcript file (single utterance, plain text words).")
    ap.add_argument("--out_dir", type=str, required=True,
                    help="Output directory.")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--write_word_ctm", action="store_true",
                    help="Also write word-level CTM.")
    ap.add_argument("--write_textgrid", action="store_true",
                        help="Also write Praat TextGrid (phones tier + optional words tier).")
    ap.add_argument("--confidence_mode", type=str, default="emission",
                        choices=["emission", "avg_frame"],
                        help="Confidence calculation: emission=logprob at emission frame; avg_frame=mean logprob over segment frames.")
    ap.add_argument("--write_conf_ctm", action="store_true",
                        help="If set, append confidence column to CTM outputs.")
    ap.add_argument("--chunk_audio", action="store_true",
                        help="If set, chunk the audio into clean portions based on the word alignment.")
    ap.add_argument("--chunks_out_dir", type=str, default=None,
                        help="Output directory for chunked wavs/manifests. Default: <out_dir>/chunks")
    ap.add_argument("--max_gap_s", type=float, default=0.35,
                        help="Max allowed pause between consecutive words inside a chunk.")
    ap.add_argument("--min_chunk_s", type=float, default=1.0,
                        help="Minimum chunk duration (seconds).")
    ap.add_argument("--max_chunk_s", type=float, default=12.0,
                        help="Maximum chunk duration (seconds).")
    ap.add_argument("--min_words", type=int, default=2,
                        help="Minimum number of words in a kept chunk.")
    ap.add_argument("--pad_s", type=float, default=0.15,
                        help="Try to pad chunk boundaries into surrounding silence (seconds), without crossing word boundaries.")
    ap.add_argument("--max_pad_into_gap_s", type=float, default=0.25,
                        help="Never pad more than this into a surrounding gap (seconds).")
    ap.add_argument("--no_pad", action="store_true",
                        help="Disable boundary padding when chunking.")
    args = ap.parse_args()

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        device = torch.device(args.device)
    print(f"[INFO] Using device: {device}")

    wav_path = Path(args.audio)
    tr_path = Path(args.transcript)

    if not wav_path.exists():
        raise FileNotFoundError(f"Audio not found: {wav_path}")
    if not tr_path.exists():
        raise FileNotFoundError(f"Transcript not found: {tr_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load lexicon + tokens
    lex = read_lexicon(Path(args.lexicon))
    phone_to_id = read_phone_json(Path(args.phone_json))

    # Load model + processor
    device = torch.device(args.device)
    model_dir = Path(args.model_dir)
    processor_dir = Path(args.processor_dir) if args.processor_dir else model_dir

    processor = Wav2Vec2Processor.from_pretrained(str(processor_dir))
    model = Wav2Vec2ForCTC.from_pretrained(str(model_dir)).to(device)
    model.eval()

    # Determine blank token id
    blank_token = args.blank_token
    if blank_token is None:
        # Try tokenizer.pad_token; otherwise common default
        blank_token = getattr(processor.tokenizer, "pad_token", None) or "<pad>"
    if blank_token not in phone_to_id:
        raise ValueError(
            f"Blank token '{blank_token}' not found in phone token list. "
            f"Please set --blank_token correctly."
        )
    blank_id = phone_to_id[blank_token]

    # Optional inter-word token check
    inter_word_token = args.inter_word_token
    if inter_word_token is not None and inter_word_token not in phone_to_id:
        raise ValueError(
            f"--inter_word_token '{inter_word_token}' not found in phone token list."
        )

    target_sr = getattr(processor.feature_extractor, "sampling_rate", 16000)

    utt_id = wav_path.stem
    words = read_transcript(tr_path)
    if not words:
        print(f"[WARN] Empty transcript for {utt_id}, skipping.")
        return

    audio = load_audio(wav_path, target_sr)
    log_probs, spf = compute_log_probs(model, processor, audio, target_sr, device)

    # Get pronunciations per word
    prons_per_word = words_to_pronunciations(words, lex)

    # Beam-search pronunciation choices (best phone string)
    best = beam_search_pronunciations(
        log_probs=log_probs,
        words=words,
        prons_per_word=prons_per_word,
        phone_to_id=phone_to_id,
        blank_id=blank_id,
        beam_size=args.beam_size,
        inter_word_token=inter_word_token,
    )

    # Align the chosen phone sequence
    phone_ids = [phone_to_id[p] for p in best.phones]
    trellis = build_trellis(log_probs, phone_ids, blank_id)
    points = backtrace(trellis, log_probs, phone_ids, blank_id)
    phone_segs = points_to_segments(points, best.phones)
    #phone_segs = merge_repeated_labels(phone_segs)
    # token-level phone segments (DO NOT MERGE)
    phone_token_segs = points_to_segments(points, best.phones)

    # Save phone CTM
    phone_ctm_path = out_dir / f"{utt_id}.phones.ctm"
    phone_segs_conf = attach_phone_confidence_from_points(
        merged_phone_segs=phone_segs,
        points=points,
        target_labels=best.phones,
        phone_to_id=phone_to_id,
        log_probs=log_probs,
        mode=args.confidence_mode,
    )
    # Preserve old CTM format unless --write_conf_ctm is set
    write_ctm_with_optional_conf(
        phone_ctm_path, utt_id, phone_segs_conf, spf, include_conf=args.write_conf_ctm
    )

    # Optionally build word CTM
    word_segs: Optional[List[Segment]] = None
    word_segs_conf: Optional[List[SegmentWithConf]] = None

    if args.write_word_ctm or args.write_textgrid or args.chunk_audio:
        # reconstruct per-word chosen pronunciations (indices best.pron_choice_idxs)
        chosen_prons: List[List[str]] = []
        for w_i, _w in enumerate(words):
            chosen_prons.append(prons_per_word[w_i][best.pron_choice_idxs[w_i]])

        #word_segs = phones_to_word_segments(phone_segs, words, chosen_prons, inter_word_token=inter_word_token)
        # build word segments from token offsets (NEW, robust)
        word_segs = phones_to_word_segments_by_offsets(
            phone_token_segs,
            words=words,
            prons_per_word=chosen_prons,
            inter_word_token=inter_word_token,
        )
        word_segs_conf = word_segments_with_confidence(word_segs, phone_segs_conf)

        if args.write_word_ctm:
            word_ctm_path = out_dir / f"{utt_id}.words.ctm"
            write_ctm_with_optional_conf(
                word_ctm_path, utt_id, word_segs_conf, spf, include_conf=args.write_conf_ctm
            )

    # TextGrid output (phones tier + optional words tier)
    if args.write_textgrid:
        total_dur = float(audio.numel()) / target_sr
        phone_intervals = [(s.start_frame * spf, s.end_frame * spf, s.label) for s in phone_segs]
        tiers: Dict[str, List[Tuple[float, float, str]]] = {"phones": phone_intervals}
        if word_segs is not None:
            word_intervals = [(s.start_frame * spf, s.end_frame * spf, s.label) for s in word_segs]
            tiers["words"] = word_intervals
        tg_path = out_dir / f"{utt_id}.TextGrid"
        write_textgrid(tg_path, xmin=0.0, xmax=total_dur, tiers=tiers)

    # Chunk audio based on word alignment (requires word segments)
    if args.chunk_audio:
        if word_segs is None:
            raise RuntimeError("chunk_audio requested but word segments were not generated.")
        audio_dur_s = float(audio.numel()) / target_sr
        word_ctm_like: List[WordSeg] = [
            WordSeg(start=s.start_frame * spf, dur=(s.end_frame - s.start_frame) * spf, word=s.label)
            for s in word_segs
        ]
        chunks = merge_words_into_chunks(
            word_ctm_like,
            max_gap_s=args.max_gap_s,
            min_chunk_s=args.min_chunk_s,
            max_chunk_s=args.max_chunk_s,
            min_words=args.min_words,
        )
        if (not args.no_pad) and chunks:
            chunks = pad_chunks_without_cutting_words(
                chunks,
                word_ctm_like,
                pad_s=args.pad_s,
                max_pad_into_gap_s=args.max_pad_into_gap_s,
                audio_dur_s=audio_dur_s,
            )
        chunks_out_dir = Path(args.chunks_out_dir) if args.chunks_out_dir else (out_dir / "chunks")
        save_chunk_wavs_and_manifests(wav_path, chunks_out_dir, chunks, base=utt_id)

    # Metadata
    meta = {
        "utt_id": utt_id,
        "audio": str(wav_path),
        "transcript": str(tr_path),
        "words": words,
        "beam_size": args.beam_size,
        "best_pron_choice_idxs": best.pron_choice_idxs,
        "best_score": best.score,
        "seconds_per_frame": spf,
        "blank_token": blank_token,
        "inter_word_token": inter_word_token,
    }
    (out_dir / f"{utt_id}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] {utt_id}: wrote {phone_ctm_path.name}" + (f" and {utt_id}.words.ctm" if args.write_word_ctm else ""))


if __name__ == "__main__":
    main()


