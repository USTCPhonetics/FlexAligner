#!/usr/bin/env python3
"""
Forced alignment for a framewise (cross-entropy) wav2vec2 phone classifier.

Features
- Beam-search Viterbi over a lexicon/pronunciation graph
- Optional SIL insertion:
  * between words
  * before first word and after last word
- Word tier silences labeled "sil"
- Outputs:
  * Praat TextGrid (phones + words)
  * JSON alignment with confidence per segment (phones + words)

IMPORTANT NOTE ABOUT CONFIDENCE
- Phone confidence is computed from posterior of that phone over frames in the phone segment.
- Word confidence is computed from posterior of the *aligned phone at each frame* within that word segment
  (since words are not in the model vocab).

Usage example:
python align_ce_w2v2_json_full.py \
  --ckpt ../outputs_fairseq/ce \
  --wav /path/to/SP01_001.wav \
  --lexicon lexicon.txt \
  --transcript transcript.txt \
  --out_textgrid SP01_001.TextGrid \
  --out_json SP01_001.align.json \
  --optional_sil --sil_at_ends --sil_phone SIL --sil_cost -0.5 \
  --beam 400 --p_stay 0.92 --frame_hop_s 0.01
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
import soundfile as sf
import torch
from transformers import AutoProcessor, AutoModelForCTC

NEG_INF = -1e30


# -------------------------
# Lexicon
# -------------------------
class PronouncingDictionary:
    """
    word -> list of pronunciations; each pronunciation is a list of phones (strings).
    Format per line:
      WORD PH1 PH2 ...
    Multiple lines with same WORD => multiple pronunciations.
    """
    def __init__(self):
        self.lex: Dict[str, List[List[str]]] = {}

    def add(self, word: str, pron: List[str]) -> None:
        self.lex.setdefault(word, []).append(list(pron))

    def get_prons(self, word: str) -> List[List[str]]:
        if word not in self.lex:
            raise KeyError(f"Word not in lexicon: {word}")
        prons = self.lex[word]
        if not prons:
            raise KeyError(f"Word has no pronunciations: {word}")
        return prons

    @staticmethod
    def from_path(path: str, lowercase: bool = False) -> "PronouncingDictionary":
        pd = PronouncingDictionary()
        with open(path, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln or ln.startswith("#"):
                    continue
                parts = ln.split()
                if len(parts) < 2:
                    continue
                w = parts[0].lower() if lowercase else parts[0]
                pd.add(w, parts[1:])
        return pd


def read_transcript_words(path: str, lowercase: bool = False) -> List[str]:
    txt = open(path, "r", encoding="utf-8").read().strip()
    words = [w for w in txt.split() if w]
    if lowercase:
        words = [w.lower() for w in words]
    return words


def utt_id_from_wav(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


# -------------------------
# Graph structures
# -------------------------
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


def build_phone_graph_optional_sil(
    words: List[str],
    prondict: PronouncingDictionary,
    phone_to_id: Dict[str, int],
    sil_phone: Optional[str] = "SIL",
    optional_sil_between_words: bool = True,
    optional_sil_at_start: bool = True,
    optional_sil_at_end: bool = True,
    sil_cost: float = 0.0,  # additive per-frame bias for SIL states (negative discourages SIL)
) -> Tuple[PhoneGraph, np.ndarray]:
    """
    Pronunciation DAG with optional SIL:
      - between words (epsilon OR SIL)
      - before first word (epsilon OR SIL)
      - after last word (epsilon OR SIL)

    We remove epsilons via epsilon-closure to decode purely over emitting states.

    Returns:
      graph, entry_bias_by_state (np.ndarray shape [S]) applied each frame to that state.
    """
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

    def add_emit(u: int, v: int, phone: str, widx: Optional[int], w: Optional[str], bias: float = 0.0):
        if phone not in phone_to_id:
            raise KeyError(f"Phone '{phone}' not in model vocab.")
        emit_edges.append(EmitEdge(u=u, v=v, phone=phone, phone_id=phone_to_id[phone], word_index=widx, word=w))
        entry_bias.append(bias)

    def add_eps(u: int, v: int):
        eps_edges.append((u, v))

    # Optional SIL at start
    start_node = new_node()
    if sil_phone is not None and optional_sil_at_start:
        add_eps(START, start_node)
        add_emit(START, start_node, sil_phone, None, None, bias=sil_cost)
    else:
        add_eps(START, start_node)

    cur_node = start_node

    # Words with optional SIL between
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

        if optional_sil_between_words and wi != len(words) - 1 and sil_phone is not None:
            nxt = new_node()
            add_eps(cur_node, nxt)
            add_emit(cur_node, nxt, sil_phone, None, None, bias=sil_cost)
            cur_node = nxt

    final_node = cur_node

    # END marker
    END = new_node()

    # Optional SIL at end
    if sil_phone is not None and optional_sil_at_end:
        add_eps(final_node, END)
        tail = new_node()
        add_emit(final_node, tail, sil_phone, None, None, bias=sil_cost)
        add_eps(tail, END)
    else:
        add_eps(final_node, END)

    num_nodes = next_node

    # epsilon adjacency
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

    # start states: any emitting edge starting in eps-closure(START)
    start_states: List[int] = []
    for node in fwd_cl[START]:
        start_states.extend(out_emit.get(node, []))
    start_states = sorted(set(start_states))
    if not start_states:
        raise RuntimeError("No start states. Check transcript/lexicon/SIL settings.")

    # end states: those whose end node can reach END by eps
    end_states: List[int] = []
    for si, st in enumerate(states):
        if END in fwd_cl[st.edge.v]:
            end_states.append(si)
    if not end_states:
        end_states = [i for i, st in enumerate(states) if len(st.succs) == 0]

    return PhoneGraph(states=states, start_states=start_states, end_states=end_states), np.asarray(entry_bias, dtype=np.float32)


# -------------------------
# Beam Viterbi decoding
# -------------------------
@dataclass
class AlignmentResult:
    # segments are (label, start_frame, end_frame_excl)
    phone_segments_f: List[Tuple[str, int, int]]
    word_segments_f: List[Tuple[str, int, int]]
    state_path: np.ndarray  # (T,)
    aligned_phone_ids: np.ndarray  # (T,) phone id for aligned state at each frame


def align_beam_viterbi(
    logp: np.ndarray,          # (T, V) log-probabilities
    graph: PhoneGraph,
    entry_bias: np.ndarray,    # (S,)
    p_stay: float = 0.92,
    beam_size: int = 300,
    word_sil_label: str = "sil",
) -> AlignmentResult:
    T, V = logp.shape
    S = len(graph.states)
    if entry_bias.shape[0] != S:
        raise ValueError("entry_bias length != number of states")
    if T == 0:
        raise ValueError("No frames produced by model.")

    lp_stay = math.log(p_stay)
    lp_move = math.log(1.0 - p_stay)

    bp: List[Dict[int, int]] = []

    # init
    cur_scores: Dict[int, float] = {}
    cur_bp: Dict[int, int] = {}
    for s in graph.start_states:
        phid = graph.states[s].edge.phone_id
        cur_scores[s] = float(logp[0, phid]) + float(entry_bias[s])
        cur_bp[s] = s

    if len(cur_scores) > beam_size:
        top = sorted(cur_scores.items(), key=lambda kv: kv[1], reverse=True)[:beam_size]
        cur_scores = {k: v for k, v in top}
        cur_bp = {k: cur_bp[k] for k, _ in top}
    bp.append(cur_bp)

    # forward
    for t in range(1, T):
        nxt_scores: Dict[int, float] = {}
        nxt_bp: Dict[int, int] = {}

        for s, sc in cur_scores.items():
            st = graph.states[s]
            emit_s = float(logp[t, st.edge.phone_id]) + float(entry_bias[s])

            # stay
            cand = sc + lp_stay + emit_s
            if cand > nxt_scores.get(s, NEG_INF):
                nxt_scores[s] = cand
                nxt_bp[s] = s

            # move
            base = sc + lp_move
            for ns in st.succs:
                nst = graph.states[ns]
                emit_ns = float(logp[t, nst.edge.phone_id]) + float(entry_bias[ns])
                cand2 = base + emit_ns
                if cand2 > nxt_scores.get(ns, NEG_INF):
                    nxt_scores[ns] = cand2
                    nxt_bp[ns] = s

        if len(nxt_scores) > beam_size:
            top = sorted(nxt_scores.items(), key=lambda kv: kv[1], reverse=True)[:beam_size]
            nxt_scores = {k: v for k, v in top}
            nxt_bp = {k: nxt_bp[k] for k, _ in top}

        cur_scores = nxt_scores
        bp.append(nxt_bp)

    # termination
    end_set = set(graph.end_states)
    best_state = None
    best_score = NEG_INF
    for s, sc in cur_scores.items():
        term = sc + lp_move
        if s in end_set and term > best_score:
            best_score = term
            best_state = s
    if best_state is None:
        best_state = max(cur_scores.items(), key=lambda kv: kv[1])[0]

    # backtrace
    path = np.empty((T,), dtype=np.int32)
    cur = int(best_state)
    for t in range(T - 1, -1, -1):
        path[t] = cur
        cur = int(bp[t].get(cur, cur))

    # aligned phone ids per frame
    aligned_phone_ids = np.array([graph.states[int(s)].edge.phone_id for s in path], dtype=np.int32)

    # phone segments in frames
    phone_segments_f: List[Tuple[str, int, int]] = []
    cur_ph = graph.states[int(path[0])].edge.phone
    start = 0
    for t in range(1, T):
        ph = graph.states[int(path[t])].edge.phone
        if ph != cur_ph:
            phone_segments_f.append((cur_ph, start, t))
            cur_ph = ph
            start = t
    phone_segments_f.append((cur_ph, start, T))

    # word segments in frames (use edge.word; TextGrid correctness implies this is fine in your setup)
    word_segments_f: List[Tuple[str, int, int]] = []
    w0 = graph.states[int(path[0])].edge.word
    cur_w = w0 if w0 is not None else word_sil_label
    start = 0
    for t in range(1, T):
        w = graph.states[int(path[t])].edge.word
        lab = w if w is not None else word_sil_label
        if lab != cur_w:
            word_segments_f.append((cur_w, start, t))
            cur_w = lab
            start = t
    word_segments_f.append((cur_w, start, T))

    return AlignmentResult(
        phone_segments_f=phone_segments_f,
        word_segments_f=word_segments_f,
        state_path=path,
        aligned_phone_ids=aligned_phone_ids,
    )


# -------------------------
# Confidence scoring
# -------------------------
def phone_segment_confidence(
    logp: np.ndarray,          # (T,V) log posteriors
    tok2id: Dict[str, int],
    phone_label: str,
    start_f: int,
    end_f: int,
) -> Dict[str, float]:
    """Confidence for a phone segment: posterior of that phone over its frames."""
    if phone_label not in tok2id or end_f <= start_f:
        return {"avg_logp": None, "avg_prob": None, "min_prob": None, "p10_prob": None}
    tid = tok2id[phone_label]
    lp = logp[start_f:end_f, tid]
    p = np.exp(lp)
    return {
        "avg_logp": float(lp.mean()),
        "avg_prob": float(p.mean()),
        "min_prob": float(p.min()),
        "p10_prob": float(np.percentile(p, 10)),
    }


def confidence_from_aligned_phone_path(
    logp: np.ndarray,              # (T,V) log posteriors
    aligned_phone_ids: np.ndarray, # (T,) aligned phone id each frame
    start_f: int,
    end_f: int,
) -> Dict[str, float]:
    """
    Confidence for a word segment (or any segment): posterior of the aligned phone at each frame.
    This avoids NaN for word labels that are not in the phone vocab.
    """
    if end_f <= start_f:
        return {"avg_logp": None, "avg_prob": None, "min_prob": None, "p10_prob": None}
    idx = np.arange(start_f, end_f, dtype=np.int32)
    lp = logp[idx, aligned_phone_ids[idx]]
    p = np.exp(lp)
    return {
        "avg_logp": float(lp.mean()),
        "avg_prob": float(p.mean()),
        "min_prob": float(p.min()),
        "p10_prob": float(np.percentile(p, 10)),
    }


# -------------------------
# Outputs: TextGrid + JSON
# -------------------------
def write_textgrid(
    out_path: str,
    duration_s: float,
    phone_segments: List[Tuple[str, float, float]],
    word_segments: List[Tuple[str, float, float]],
    phone_tier_name: str = "phones",
    word_tier_name: str = "words",
) -> None:
    def esc(s: str) -> str:
        return s.replace('"', '""')

    def tier(name: str, segs: List[Tuple[str, float, float]]) -> str:
        out = []
        out.append('        class = "IntervalTier"')
        out.append(f'        name = "{esc(name)}"')
        out.append("        xmin = 0")
        out.append(f"        xmax = {duration_s:.6f}")
        out.append(f"        intervals: size = {len(segs)}")
        for i, (lab, x1, x2) in enumerate(segs, start=1):
            out.append(f"        intervals [{i}]:")
            out.append(f"            xmin = {x1:.6f}")
            out.append(f"            xmax = {x2:.6f}")
            out.append(f'            text = "{esc(lab)}"')
        return "\n".join(out)

    tg = []
    tg.append('File type = "ooTextFile"')
    tg.append('Object class = "TextGrid"')
    tg.append("")
    tg.append("xmin = 0")
    tg.append(f"xmax = {duration_s:.6f}")
    tg.append("tiers? <exists>")
    tg.append("size = 2")
    tg.append("item []:")
    tg.append("    item [1]:")
    tg.append(tier(phone_tier_name, phone_segments))
    tg.append("    item [2]:")
    tg.append(tier(word_tier_name, word_segments))

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(tg) + "\n")


def write_alignment_json(
    out_path: str,
    utt_id: str,
    wav_path: str,
    sr: int,
    frame_hop_s: float,
    logp: np.ndarray,                        # (T,V)
    tok2id: Dict[str, int],
    aligned_phone_ids: np.ndarray,           # (T,)
    phone_segments_f: List[Tuple[str, int, int]],
    word_segments_f: List[Tuple[str, int, int]],
) -> None:
    T = int(logp.shape[0])
    duration_s = T * frame_hop_s

    phones = []
    for lab, s, e in phone_segments_f:
        obj = {
            "label": lab,
            "start_frame": int(s),
            "end_frame": int(e),
            "start_time": float(s * frame_hop_s),
            "end_time": float(e * frame_hop_s),
            "duration": float((e - s) * frame_hop_s),
        }
        obj.update(phone_segment_confidence(logp, tok2id, lab, s, e))
        phones.append(obj)

    words = []
    for lab, s, e in word_segments_f:
        obj = {
            "label": lab,
            "start_frame": int(s),
            "end_frame": int(e),
            "start_time": float(s * frame_hop_s),
            "end_time": float(e * frame_hop_s),
            "duration": float((e - s) * frame_hop_s),
        }
        # Word confidence from aligned phone path (NOT tok2id[word])
        obj.update(confidence_from_aligned_phone_path(logp, aligned_phone_ids, s, e))
        words.append(obj)

    out = {
        "utt_id": utt_id,
        "wav_path": wav_path,
        "sampling_rate": int(sr),
        "frame_hop_s": float(frame_hop_s),
        "num_frames": int(T),
        "duration_s": float(duration_s),
        "phones": phones,
        "words": words,
    }

    # strict JSON: disallow NaN/Infinity. We used None for missing values.
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, allow_nan=False)


# -------------------------
# Model forward (CE framewise)
# -------------------------
def compute_logp_frames(model, processor, audio: np.ndarray, sr: int, device: str) -> np.ndarray:
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model = model.to(device)
    with torch.no_grad():
        logits = model(**inputs).logits  # (1, T, V)
        logp = torch.log_softmax(logits, dim=-1)[0]
    return logp.detach().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--wav", type=str, required=True)
    ap.add_argument("--lexicon", type=str, required=True)
    ap.add_argument("--transcript", type=str, required=True)

    ap.add_argument("--out_textgrid", type=str, required=True)
    ap.add_argument("--out_json", type=str, required=True)

    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--lowercase", action="store_true")

    ap.add_argument("--optional_sil", action="store_true", help="optional SIL between words")
    ap.add_argument("--sil_at_ends", action="store_true", help="optional SIL at utterance start+end")
    ap.add_argument("--sil_phone", type=str, default="SIL", help="silence phone token (empty disables)")
    ap.add_argument("--sil_cost", type=float, default=-0.5, help="negative discourages SIL (per SIL frame)")

    ap.add_argument("--beam", type=int, default=400)
    ap.add_argument("--p_stay", type=float, default=0.92)
    ap.add_argument("--frame_hop_s", type=float, default=0.01)

    ap.add_argument("--force_sr", type=int, default=16000)

    # silence label in word tier (visible)
    ap.add_argument("--word_sil_label", type=str, default="sil")

    args = ap.parse_args()

    utt_id = utt_id_from_wav(args.wav)

    # load model/processor
    processor = AutoProcessor.from_pretrained(args.ckpt)
    model = AutoModelForCTC.from_pretrained(args.ckpt).eval()

    # token->id (must match lexicon phones)
    tok2id: Dict[str, int] = processor.tokenizer.get_vocab()

    # audio
    audio, sr = sf.read(args.wav, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != args.force_sr:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=args.force_sr)
        sr = args.force_sr

    import time
    t0 = time.time()

    # emissions
    logp = compute_logp_frames(model, processor, audio, sr, device=args.device)  # (T,V)
    T = logp.shape[0]
    duration_s = T * args.frame_hop_s
    print(f"[TIME] model forward: {time.time() - t0:.1f}s  logp.shape={logp.shape}")

    # lexicon + transcript
    lex = PronouncingDictionary.from_path(args.lexicon, lowercase=args.lowercase)
    words = read_transcript_words(args.transcript, lowercase=args.lowercase)

    sil_phone = args.sil_phone.strip() or None

    # build graph
    graph, entry_bias = build_phone_graph_optional_sil(
        words=words,
        prondict=lex,
        phone_to_id=tok2id,
        sil_phone=sil_phone,
        optional_sil_between_words=args.optional_sil,
        optional_sil_at_start=args.sil_at_ends,
        optional_sil_at_end=args.sil_at_ends,
        sil_cost=args.sil_cost,
    )

    succ_lens = [len(st.succs) for st in graph.states]
    print(
    "DEBUG graph:",
    "num_states =", len(graph.states),
    "succ avg =", sum(succ_lens) / len(succ_lens),
    "succ max =", max(succ_lens),
    )

    pred_lens = [len(st.preds) for st in graph.states]
    print(
    "pred avg =", sum(pred_lens)/len(pred_lens),
    "pred max =", max(pred_lens),
    )

    # decode (segments in frames)
    ali = align_beam_viterbi(
        logp=logp,
        graph=graph,
        entry_bias=entry_bias,
        p_stay=args.p_stay,
        beam_size=args.beam,
        word_sil_label=args.word_sil_label,
    )

    # TextGrid segments in seconds
    phone_segments_s = [(lab, s * args.frame_hop_s, e * args.frame_hop_s) for (lab, s, e) in ali.phone_segments_f]
    word_segments_s = [(lab, s * args.frame_hop_s, e * args.frame_hop_s) for (lab, s, e) in ali.word_segments_f]

    write_textgrid(
        out_path=args.out_textgrid,
        duration_s=duration_s,
        phone_segments=phone_segments_s,
        word_segments=word_segments_s,
    )

    write_alignment_json(
        out_path=args.out_json,
        utt_id=utt_id,
        wav_path=args.wav,
        sr=sr,
        frame_hop_s=args.frame_hop_s,
        logp=logp,
        tok2id=tok2id,
        aligned_phone_ids=ali.aligned_phone_ids,
        phone_segments_f=ali.phone_segments_f,
        word_segments_f=ali.word_segments_f,
    )

    print(f"Utterance ID: {utt_id}")
    print(f"Wrote TextGrid: {args.out_textgrid}")
    print(f"Wrote JSON:     {args.out_json}")
    print(f"Frames={T} duration≈{duration_s:.3f}s hop={args.frame_hop_s:.3f}s")
    print(f"Optional SIL between words: {args.optional_sil} | at ends: {args.sil_at_ends} | SIL token: {sil_phone!r}")


if __name__ == "__main__":
    main()

