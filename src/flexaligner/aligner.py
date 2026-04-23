import torch
import numpy as np
import math
import os
import json
import csv
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Set
from pathlib import Path
from transformers import AutoModelForCTC, AutoProcessor
import soundfile as sf
# ==========================================
#  1. Reference Data Structures (完全复刻)
# ==========================================

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

@dataclass
class AlignmentResult:
    phone_segments_f: List[Tuple[str, int, int]]
    word_segments_f: List[Tuple[str, int, int]]
    state_path: np.ndarray 
    aligned_phone_ids: np.ndarray 

# 对外输出的通用结构 (适配 Pipeline)
@dataclass
class AlignmentSegment:
    label: str
    start: float
    end: float
    score: float = 0.0
    
@dataclass
class AcousticEvidence:
    chunk_id: str
    log_probs: np.ndarray        # (T, V)
    num_frames: int
    vocab_size: int
    frame_hop_s: float
    duration_s: float    
# @dataclass
# class ChunkRecord:
#     chunk_id: str
#     audio_path: str
#     text: str
#     start_time: float
#     end_time: float
#     dur_s: float    

# @dataclass
# class ChunkEvidence:
#     chunk_id: str
#     audio_path: str
#     text: str
#     start_time: float
#     end_time: float
#     dur_s: float
#     log_probs_path: str
#     num_frames: int
#     frame_hop_s: float
#     vocab_size: int
#     model_tag: str    
#     phone_vocab_path: Optional[str] = None
#     sample_rate: int = 16000
@dataclass
class DecodeResult:
    chunk_id: str
    phones: list
    words: list
    decode_tag: str    
# ==========================================
#  2. Reference Algorithms (逻辑 1:1 移植)
# ==========================================

NEG_INF = -1e30

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
    prondict, # Expecting PronouncingDictionary instance
    phone_to_id: Dict[str, int],
    sil_phone: Optional[str] = "SIL",
    optional_sil_between_words: bool = True,
    optional_sil_at_start: bool = True,
    optional_sil_at_end: bool = True,
    sil_cost: float = 0.0,
) -> Tuple[PhoneGraph, np.ndarray]:
    """
    [Reference Logic] 构建包含可选静音的发音图
    """
    # print(f"Building phone graph for words: {words}")
    # input("Press Enter to continue...")  # Debug pause to inspect input words
    
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
        # [Robustness] 增加 OOV 防御，防止 Key Error 导致崩溃，改为跳过
        if phone not in phone_to_id:
            assert phone.lower() == sil_phone.lower(), f"Phone '{phone}' not in vocab and is not the designated silence phone '{sil_phone}'."
            # 尝试去重音 (AO1 -> AO)
            pure = ''.join(filter(str.isalpha, phone))
            if pure in phone_to_id:
                phone_id = phone_to_id[pure]
            else:
                # 实在没有，打印警告并跳过这条边（可能导致图断裂，但在 Library 中比 crash 好）
                print(f"Warning: Phone '{phone}' not in vocab.")
                return 
        else:
            phone_id = phone_to_id[phone]
            
        emit_edges.append(EmitEdge(u=u, v=v, phone=phone, phone_id=phone_id, word_index=widx, word=w))
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

    # Words loop
    for wi, w in enumerate(words):
        end_of_word = new_node()
        try:
            prons = prondict.get_prons(w)
        except KeyError:
            # OOV words: skip in graph via epsilon
            add_eps(cur_node, end_of_word)
            cur_node = end_of_word
            continue

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

    end_states: List[int] = []
    for si, st in enumerate(states):
        if END in fwd_cl[st.edge.v]:
            end_states.append(si)
            
    # Fallback if no end states found (rare)
    if not end_states:
        end_states = [i for i, st in enumerate(states) if len(st.succs) == 0]

    return PhoneGraph(states=states, start_states=start_states, end_states=end_states), np.asarray(entry_bias, dtype=np.float32)

# def align_beam_viterbi(
#     logp: np.ndarray,      # (T, V)
#     graph: PhoneGraph,
#     entry_bias: np.ndarray,# (S,)
#     p_stay: float = 0.92,
#     beam_size: int = 400,
#     word_sil_label: str = "sil",
# ) -> AlignmentResult:
#     """
#     [Reference Logic] 标准 Beam Viterbi 解码
#     """
#     T, V = logp.shape
    
#     lp_stay = math.log(p_stay)
#     lp_move = math.log(1.0 - p_stay)

#     bp: List[Dict[int, int]] = []
#     cur_scores: Dict[int, float] = {}
#     cur_bp: Dict[int, int] = {}

#     # Init
#     for s in graph.start_states:
#         phid = graph.states[s].edge.phone_id
#         cur_scores[s] = float(logp[0, phid]) + float(entry_bias[s])
#         cur_bp[s] = s

#     # Init Pruning
#     if len(cur_scores) > beam_size:
#         top = sorted(cur_scores.items(), key=lambda kv: kv[1], reverse=True)[:beam_size]
#         cur_scores = {k: v for k, v in top}
#         cur_bp = {k: cur_bp[k] for k, _ in top}
#     bp.append(cur_bp)

#     # Forward
#     for t in range(1, T):
#         nxt_scores: Dict[int, float] = {}
#         nxt_bp: Dict[int, int] = {}

#         for s, sc in cur_scores.items():
#             st = graph.states[s]
#             emit_s = float(logp[t, st.edge.phone_id]) + float(entry_bias[s])

#             # 1. Stay
#             cand = sc + lp_stay + emit_s
#             if cand > nxt_scores.get(s, NEG_INF):
#                 nxt_scores[s] = cand
#                 nxt_bp[s] = s

#             # 2. Move
#             base = sc + lp_move
#             for ns in st.succs:
#                 nst = graph.states[ns]
#                 emit_ns = float(logp[t, nst.edge.phone_id]) + float(entry_bias[ns])
#                 cand2 = base + emit_ns
#                 if cand2 > nxt_scores.get(ns, NEG_INF):
#                     nxt_scores[ns] = cand2
#                     nxt_bp[ns] = s

#         # Pruning
#         if len(nxt_scores) > beam_size:
#             top = sorted(nxt_scores.items(), key=lambda kv: kv[1], reverse=True)[:beam_size]
#             nxt_scores = {k: v for k, v in top}
#             nxt_bp = {k: nxt_bp[k] for k, _ in top}

#         cur_scores = nxt_scores
#         bp.append(nxt_bp)

#     # Termination
#     end_set = set(graph.end_states)
#     best_state = None
#     best_score = NEG_INF
#     for s, sc in cur_scores.items():
#         term = sc + lp_move
#         if s in end_set and term > best_score:
#             best_score = term
#             best_state = s
            
#     if best_state is None and len(cur_scores) > 0:
#         best_state = max(cur_scores.items(), key=lambda kv: kv[1])[0]

#     # Backtrace
#     path = np.empty((T,), dtype=np.int32)
#     if best_state is not None:
#         cur = int(best_state)
#         for t in range(T - 1, -1, -1):
#             path[t] = cur
#             cur = int(bp[t].get(cur, cur))
#     else:
#         path.fill(0)

#     aligned_phone_ids = np.array([graph.states[int(s)].edge.phone_id for s in path], dtype=np.int32)

#     # Extract Phones
#     phone_segments_f = []
#     if T > 0:
#         cur_ph = graph.states[int(path[0])].edge.phone
#         start = 0
#         for t in range(1, T):
#             ph = graph.states[int(path[t])].edge.phone
#             if ph != cur_ph:
#                 phone_segments_f.append((cur_ph, start, t))
#                 cur_ph = ph
#                 start = t
#         phone_segments_f.append((cur_ph, start, T))

#     # Extract Words
#     word_segments_f = []
#     if T > 0:
#         w0 = graph.states[int(path[0])].edge.word
#         cur_w = w0 if w0 is not None else word_sil_label
#         start = 0
#         for t in range(1, T):
#             w = graph.states[int(path[t])].edge.word
#             lab = w if w is not None else word_sil_label
#             if lab != cur_w:
#                 word_segments_f.append((cur_w, start, t))
#                 cur_w = lab
#                 start = t
#         word_segments_f.append((cur_w, start, T))

#     return AlignmentResult(phone_segments_f, word_segments_f, path, aligned_phone_ids)

def align_beam_viterbi(
    logp: np.ndarray,          # (T, V) log-probabilities
    graph: PhoneGraph,
    entry_bias: np.ndarray,    # (S,)
    p_stay: float = 0.92,
    beam_size: int = 300,
    word_sil_label: str = "sil",
    # --- [Trick Parameters] ---
    boundary_lambda: float = 0.0,
    boundary_context_s: float = 0.015,
    frame_hop_s: float = 0.01,
    sil_phone_id: Optional[int] = None,
    min_sil_dur_ms: float = 0.0,
    sil_enter_cost: float = 0.0,
) -> AlignmentResult:
    """
    [Advanced Logic] 带物理惯性约束与边界平滑的 Beam Viterbi 解码
    """
    T, V = logp.shape
    S = len(graph.states)
    if entry_bias.shape[0] != S:
        raise ValueError("entry_bias length != number of states")
    if T == 0:
        raise ValueError("No frames produced by model.")

    lp_stay = math.log(p_stay)
    lp_move = math.log(1.0 - p_stay)

    # -------------------------
    # Trick 1: 局部概率悬崖探测器 (Boundary Score)
    # -------------------------
    ctx = max(1, int(round(boundary_context_s / frame_hop_s)))
    if boundary_lambda != 0.0:
        pref = np.zeros((T + 1, V), dtype=np.float32)
        pref[1:] = np.cumsum(logp, axis=0)

        def _mean(pid: int, s: int, e: int) -> float:
            if e <= s: return 0.0
            return float((pref[e, pid] - pref[s, pid]) / (e - s))

        def boundary_score(t: int, a: int, b: int) -> float:
            l0 = 0 if t - ctx < 0 else (t - ctx)
            l1 = t
            r0 = t
            r1 = T if t + ctx > T else (t + ctx)
            left = _mean(a, l0, l1) - _mean(b, l0, l1)
            right = _mean(b, r0, r1) - _mean(a, r0, r1)
            return left + right
    else:
        def boundary_score(t: int, a: int, b: int) -> float:
            return 0.0

    # -------------------------
    # Trick 2 & 3: 静音物理惯性锁与过路费 (Silence Lock & Toll)
    # -------------------------
    min_sil_frames = 0
    if (min_sil_dur_ms is not None) and (min_sil_dur_ms > 0.0) and (sil_phone_id is not None):
        min_sil_frames = max(1, int(round((min_sil_dur_ms / 1000.0) / frame_hop_s)))

    def _is_sil_phone(pid: int) -> bool:
        return (sil_phone_id is not None) and (pid == sil_phone_id)

    bp: List[Dict[tuple[int, int], tuple[int, int]]] = []
    cur_scores: Dict[tuple[int, int], float] = {}
    cur_bp: Dict[tuple[int, int], tuple[int, int]] = {}

    # Init
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

    # Forward
    for t in range(1, T):
        nxt_scores: Dict[tuple[int, int], float] = {}
        nxt_bp: Dict[tuple[int, int], tuple[int, int]] = {}

        for (s, lock_prev), sc in cur_scores.items():
            st = graph.states[s]
            phid_prev = st.edge.phone_id
            prev_is_sil = _is_sil_phone(phid_prev)
            emit_s = float(logp[t, phid_prev]) + float(entry_bias[s])

            # 1. Stay
            cand = sc + lp_stay + emit_s
            lock_stay = (lock_prev - 1) if (prev_is_sil and lock_prev > 0) else 0
            key_stay = (int(s), int(lock_stay if prev_is_sil else 0))
            
            if cand > nxt_scores.get(key_stay, NEG_INF):
                nxt_scores[key_stay] = cand
                nxt_bp[key_stay] = (int(s), int(lock_prev))

            # 2. Move
            base = sc + lp_move
            for ns in st.succs:
                nst = graph.states[ns]
                phid_next = nst.edge.phone_id
                next_is_sil = _is_sil_phone(phid_next)

                # [物理约束] 静音锁死逻辑：锁没归零，不准跳出
                if prev_is_sil and lock_prev > 0 and not next_is_sil:
                    continue

                emit_ns = float(logp[t, phid_next]) + float(entry_bias[ns])

                # 计算下一帧的静音锁
                if next_is_sil:
                    if prev_is_sil:
                        lock_next = (lock_prev - 1) if lock_prev > 0 else 0
                    else:
                        lock_next = (min_sil_frames - 1) if (min_sil_frames > 0) else 0
                else:
                    lock_next = 0

                key_next = (int(ns), int(lock_next))
                # [过路费扣除]
                enter_pen = float(sil_enter_cost) if ((not prev_is_sil) and next_is_sil) else 0.0
                
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

    # Termination
    end_set = set(graph.end_states)
    best_state = None
    best_score = NEG_INF
    for (s, lock_prev), sc in cur_scores.items():
        term = sc + lp_move
        if s in end_set and term > best_score:
            best_score = term
            best_state = (int(s), int(lock_prev))
            
    if best_state is None and len(cur_scores) > 0:
        best_state = max(cur_scores.items(), key=lambda kv: kv[1])[0]

    # Backtrace
    path = np.empty((T,), dtype=np.int32)
    if best_state is not None:
        cur_key = best_state 
        for t in range(T - 1, -1, -1):
            path[t] = int(cur_key[0])
            cur_key = bp[t].get(cur_key, cur_key)
    else:
        path.fill(0)

    aligned_phone_ids = np.array([graph.states[int(s)].edge.phone_id for s in path], dtype=np.int32)

    # Extract Phones
    phone_segments_f = []
    if T > 0:
        cur_edge0 = graph.states[int(path[0])].edge
        cur_ph = cur_edge0.phone
        cur_wi = cur_edge0.word_index 
        start = 0
        for t in range(1, T):
            e = graph.states[int(path[t])].edge
            ph = e.phone
            wi = e.word_index
            # [关键修复] 防止跨词但同音素时被合并 (比如 "A A")
            if (ph != cur_ph) or (wi != cur_wi):
                phone_segments_f.append((cur_ph, start, t))
                cur_ph = ph
                cur_wi = wi
                start = t
        phone_segments_f.append((cur_ph, start, T))

    # Extract Words
    word_segments_f = []
    if T > 0:
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

    return AlignmentResult(phone_segments_f, word_segments_f, path, aligned_phone_ids)


# ==========================================
#  3. Helper Class: Lexicon
# ==========================================

class PronouncingDictionary:
    def __init__(self):
        self.lex: Dict[str, List[List[str]]] = {}

    def add(self, word: str, pron: List[str]) -> None:
        self.lex.setdefault(word, []).append(list(pron))

    def get_prons(self, word: str) -> List[List[str]]:
        if word not in self.lex:
            # Case-insensitive fallback
            if word.upper() in self.lex: return self.lex[word.upper()]
            if word.lower() in self.lex: return self.lex[word.lower()]
            raise KeyError(f"Word not in lexicon: {word}")
        return self.lex[word]

    @staticmethod
    def from_path(path: str) -> "PronouncingDictionary":
        pd = PronouncingDictionary()
        if not os.path.exists(path):
            return pd
        with open(path, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln or ln.startswith("#"): continue
                parts = ln.split()
                if len(parts) < 2: continue
                w = parts[0].lower() # Force lowercase
                pd.add(w, parts[1:])
        return pd

# ==========================================
#  4. LocalAligner Wrapper
# ==========================================

class LocalAligner:
    def __init__(self, config: dict, phone_to_id: Optional[Dict[str, int]] = None, decode_only: bool = False):
        self.config = config or {}
        self.device = torch.device(self.config.get("device", "cpu"))
        
        # Verbose & Output
        self.verbose = self.config.get("verbose", False)
        self.align_out_dir = self.config.get("align_out_dir", None)

        if self.verbose:
            self._log_header("LocalAligner Initializing")
            print(f"  - Device:      {self.device}")

        # Params matching reference script
        self.beam_size = self.config.get("align_beam_size", 400)
        self.p_stay = self.config.get("p_stay", 0.92)
        self.sil_phone = self.config.get("sil_phone", "sil")
        self.sil_cost = self.config.get("sil_cost", -0.5)
        
        # Flags
        self.optional_sil = self.config.get("optional_sil", True)
        self.sil_at_ends = self.config.get("sil_at_ends", True) # New param from ref
        self.word_sil_label = self.config.get("word_sil_label", "sil")

        # Physics
        self.frame_hop = self.config.get("frame_hop_s", 0.01)
        self.offset_s = self.config.get("offset_s", 0.0) # Reference script doesn't use explicit offset, usually 0 or implicit in hop/2
        # Physics & Advanced Constraints (Trick Parameters)
        self.frame_hop = self.config.get("frame_hop_s", 0.01)
        self.offset_s = self.config.get("offset_s", 0.0)
        self.sil_enter_cost = self.config.get("sil_enter_cost", -0.5)
        self.min_sil_dur_ms = self.config.get("min_sil_dur_ms", 0.0)
        self.boundary_lambda = self.config.get("boundary_lambda", 200.0)
        self.boundary_context_s = self.config.get("boundary_context_s", 0.05)
        self.decode_only = decode_only
        # Resources
        self.model = None
        self.processor = None
        self.lexicon = None
        self.phone_to_id = {}
        
        # 1. Load Resources
        self._load_resources(decode_only=self.decode_only)

        # 2. Vocab Injection
        if phone_to_id is not None:
            self.phone_to_id = phone_to_id
        elif not self.phone_to_id:
            # Fallback to json if processor failed to provide vocab
            json_path = self.config.get("phone_json_path")
            if json_path and os.path.exists(json_path):
                with open(json_path, 'r', encoding='utf-8') as f:
                    self.phone_to_id = json.load(f)
        

    def _log(self, msg: str):
        if self.verbose: print(f"[LocalAligner] {msg}")

    def _log_header(self, title: str):
        if self.verbose: print(f"\n=== {title} ===")
    def save_log_probs(self, evidence: AcousticEvidence, save_path: str):
        """
        保存 Stage 2 产出的声学证据。
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "chunk_id": evidence.chunk_id,
            "log_probs": torch.from_numpy(evidence.log_probs),
            "num_frames": evidence.num_frames,
            "vocab_size": evidence.vocab_size,
            "frame_hop_s": evidence.frame_hop_s,
            "duration_s": evidence.duration_s,
        }
        torch.save(payload, save_path)

    def build_decode_graph(self, text: str):
        """
        Stage 3: 只根据 text 构建发音图。
        """
        if self.lexicon is None:
            raise RuntimeError("Lexicon not loaded.")

        words = text.split()

        graph, entry_bias = build_phone_graph_optional_sil(
            words=words,
            prondict=self.lexicon,
            phone_to_id=self.phone_to_id,
            sil_phone=self.sil_phone,
            optional_sil_between_words=self.optional_sil,
            optional_sil_at_start=self.sil_at_ends,
            optional_sil_at_end=self.sil_at_ends,
            sil_cost=self.sil_cost,
        )
        return graph, entry_bias
    def decode_log_probs(
        self,
        log_probs: np.ndarray,
        text: str,
        file_id: str = "segment",
        dump_tsv: bool = False
    ) -> Dict[str, List[AlignmentSegment]]:
        """
        Stage 3: 只基于固定的 log_probs + text 进行解码。
        """
        if self.lexicon is None:
            return {"phones": [], "words": []}

        T = log_probs.shape[0]

        if self.verbose:
            self._log_header(f"Decode: {file_id}")
            print(f"  - Frames: {T}")
            print(f"  - Text: {text}")

        # 1. Build Graph
        try:
            graph, entry_bias = self.build_decode_graph(text)
        except Exception as e:
            print(f"❌ Graph construction failed: {e}")
            return {"phones": [], "words": []}

        # 2. Decode
        sil_phone_id = self.phone_to_id.get(self.sil_phone) if self.sil_phone else None

        try:
            ali = align_beam_viterbi(
                logp=log_probs,
                graph=graph,
                entry_bias=entry_bias,
                p_stay=self.p_stay,
                beam_size=self.beam_size,
                word_sil_label=self.word_sil_label,
                boundary_lambda=self.boundary_lambda,
                boundary_context_s=self.boundary_context_s,
                frame_hop_s=self.frame_hop,
                sil_phone_id=sil_phone_id,
                min_sil_dur_ms=self.min_sil_dur_ms,
                sil_enter_cost=self.sil_enter_cost,
            )
        except Exception as e:
            print(f"❌ Viterbi failed: {e}")
            return {"phones": [], "words": []}

        # 3. Format Output
        result = self._format_alignment_result(ali)

        if dump_tsv and self.align_out_dir:
            self._save_tsv(file_id, result)

        return result
    def load_log_probs(self, load_path: str) -> AcousticEvidence:
        """
        读取 Stage 2 保存的声学证据。
        """
        payload = torch.load(load_path, map_location="cpu")

        log_probs = payload["log_probs"]
        if isinstance(log_probs, torch.Tensor):
            log_probs = log_probs.numpy()

        return AcousticEvidence(
            chunk_id=str(payload["chunk_id"]),
            log_probs=log_probs,
            num_frames=int(payload["num_frames"]),
            vocab_size=int(payload["vocab_size"]),
            frame_hop_s=float(payload["frame_hop_s"]),
            duration_s=float(payload["duration_s"]),
        )
    @torch.inference_mode()
    def forward_chunk(self, chunk_tensor: torch.Tensor, file_id: str = "segment") -> AcousticEvidence:
        """
        Stage 2: 只做声学前向，返回可复用的 log_probs 证据。
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("Aligner model/processor not loaded.")

        if chunk_tensor.ndim > 1:
            chunk_tensor = chunk_tensor.mean(dim=1)

        inputs = self.processor(
            chunk_tensor.numpy(),
            sampling_rate=16000,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        logits = self.model(**inputs).logits
        logp = torch.log_softmax(logits, dim=-1)[0].detach().cpu().numpy()  # (T, V)

        T, V = logp.shape
        actual_dur = chunk_tensor.size(0) / 16000.0

        if self.verbose:
            self._log_header(f"Forward: {file_id}")
            print(f"  - Frames: {T}, Duration: {actual_dur:.3f}s, Vocab: {V}")

        return AcousticEvidence(
            chunk_id=file_id,
            log_probs=logp,
            num_frames=T,
            vocab_size=V,
            frame_hop_s=self.frame_hop,
            duration_s=actual_dur,
        )
    def _format_alignment_result(self, ali: AlignmentResult) -> Dict[str, List[AlignmentSegment]]:
        phones_out = []
        for lab, s, e in ali.phone_segments_f:
            start_t = self.offset_s + (s * self.frame_hop)
            end_t = self.offset_s + (e * self.frame_hop)
            phones_out.append(AlignmentSegment(lab, start_t, end_t))

        words_out = []
        for lab, s, e in ali.word_segments_f:
            start_t = self.offset_s + (s * self.frame_hop)
            end_t = self.offset_s + (e * self.frame_hop)
            words_out.append(AlignmentSegment(lab, start_t, end_t))

        return {"phones": phones_out, "words": words_out}
    

    def _load_resources(self, decode_only: bool = False):
        # =====================================================================
        # 1. 加载词典（Stage 3 需要）
        # =====================================================================
        lex_path = self.config.get("lexicon_path")
        if lex_path:
            self.lexicon = PronouncingDictionary.from_path(lex_path)
        else:
            self.lexicon = PronouncingDictionary()

        # 英语 G2P 增广，保留
        if self.config.get("lang", "zh") == "en":
            try:
                from g2p_en.g2p import G2p
                g2p_inst = G2p()
                cmu_dict = g2p_inst.cmu

                added_words = 0
                local_words = set()
                if isinstance(self.lexicon, dict):
                    local_words = set(self.lexicon.keys())
                elif hasattr(self.lexicon, '_dict'):
                    local_words = set(self.lexicon._dict.keys())
                elif hasattr(self.lexicon, 'words'):
                    local_words = set(self.lexicon.words)

                for w, prons in cmu_dict.items():
                    w_upper = w.upper()
                    if local_words:
                        is_known = w_upper in local_words
                    else:
                        try:
                            is_known = w_upper in self.lexicon
                        except TypeError:
                            is_known = False

                    if not is_known:
                        clean_pron = [p for p in prons[0] if p.isalnum()]
                        try:
                            if hasattr(self.lexicon, 'add_word'):
                                self.lexicon.add_word(w_upper, clean_pron)
                            elif isinstance(self.lexicon, dict) or hasattr(self.lexicon, '__setitem__'):
                                self.lexicon[w_upper] = [clean_pron]
                            elif hasattr(self.lexicon, '_dict'):
                                self.lexicon._dict[w_upper] = [clean_pron]
                            added_words += 1
                        except Exception:
                            pass
            except Exception as e:
                print(f"[warning] Aligner G2P augmentation failed: {e}")

        # =====================================================================
        # 2. 优先加载 phone vocab（Stage 3 需要）
        # =====================================================================
        # self.phone_to_id = {}

        # json_path = self.config.get("phone_json_path")
        # if json_path and os.path.exists(json_path):
        #     with open(json_path, "r", encoding="utf-8") as f:
        #         self.phone_to_id = json.load(f)

        # =====================================================================
        # 3. 如果是 decode-only，到这里就结束，不加载模型
        # =====================================================================
        # if decode_only:
        #     if self.verbose:
        #         self._log("Decode-only mode: skip loading aligner model/processor.")
        #     return

        # =====================================================================
        # 4. 加载模型与 processor（只有 Stage 2/完整流程需要）
        # =====================================================================
        model_path = self.config.get("align_model_path")
        if not model_path:
            return

        self._log(f"Loading model from {model_path}...")

        is_local = os.path.isdir(model_path)
        load_kwargs = {}

        if not is_local:
            load_kwargs["subfolder"] = f"{self.config.get('lang', 'zh')}/aligner"

        try:
            self.processor = AutoProcessor.from_pretrained(model_path, **load_kwargs)
            self.model = AutoModelForCTC.from_pretrained(model_path, **load_kwargs).to(self.device)
        except Exception:
            self.processor = AutoProcessor.from_pretrained(model_path)
            self.model = AutoModelForCTC.from_pretrained(model_path).to(self.device)

        self.model.eval()

        # 如果之前没从 phone_json_path 拿到 vocab，则从 processor 取
        if not self.phone_to_id and self.processor:
            self.phone_to_id = self.processor.tokenizer.get_vocab()
    
    @torch.inference_mode()
    def align_locally(
        self,
        chunk_tensor: torch.Tensor,
        text: str,
        file_id: str = "segment"
    ) -> Dict[str, List[AlignmentSegment]]:
        """
        兼容旧接口：内部改为 Stage 2 forward + Stage 3 decode。
        """
        print(f"Aligning chunk '{file_id}' with text: {text}")

        if self.model is None or self.lexicon is None:
            return {"phones": [], "words": []}

        try:
            evidence = self.forward_chunk(chunk_tensor, file_id=file_id)
        except Exception as e:
            print(f"❌ Forward failed: {e}")
            return {"phones": [], "words": []}

        return self.decode_log_probs(
            log_probs=evidence.log_probs,
            text=text,
            file_id=file_id,
            dump_tsv=True,
        )
    def _coerce_batch_item(self, item, idx: int = 0):
        """
        将 batch 输入统一规整成:
        {
            "chunk_id": str,
            "audio_np": np.ndarray,   # mono float32, 16k expected by caller/pipeline
            "num_samples": int,
        }

        支持的 item 形式:
        1) torch.Tensor
        2) np.ndarray
        3) dict，包含:
        - {"chunk_id": ..., "audio": np.ndarray/tensor}
        - {"chunk_id": ..., "audio_path": "..."}
        4) 任意对象，具有属性:
        - .chunk_id
        - .tensor 或 .audio_path
        """
        chunk_id = f"segment_{idx:06d}"
        audio_np = None

        # -------- case 1: raw tensor --------
        if isinstance(item, torch.Tensor):
            x = item.detach().cpu()
            if x.ndim > 1:
                x = x.mean(dim=1)
            audio_np = x.numpy().astype(np.float32, copy=False)

        # -------- case 2: raw numpy --------
        elif isinstance(item, np.ndarray):
            audio_np = item
            if audio_np.ndim > 1:
                audio_np = audio_np.mean(axis=1)
            audio_np = audio_np.astype(np.float32, copy=False)

        # -------- case 3: dict --------
        elif isinstance(item, dict):
            chunk_id = str(item.get("chunk_id", chunk_id))

            if "audio" in item:
                x = item["audio"]
                if isinstance(x, torch.Tensor):
                    x = x.detach().cpu()
                    if x.ndim > 1:
                        x = x.mean(dim=1)
                    audio_np = x.numpy().astype(np.float32, copy=False)
                elif isinstance(x, np.ndarray):
                    if x.ndim > 1:
                        x = x.mean(axis=1)
                    audio_np = x.astype(np.float32, copy=False)
                else:
                    raise TypeError(f"Unsupported dict['audio'] type: {type(x)}")

            elif "audio_path" in item:
                wav, sr = sf.read(str(item["audio_path"]))
                if sr != 16000:
                    raise ValueError(
                        f"forward_batch currently expects 16k audio chunks, got sr={sr} "
                        f"for {item['audio_path']}"
                    )
                if wav.ndim > 1:
                    wav = wav.mean(axis=1)
                audio_np = wav.astype(np.float32, copy=False)

            else:
                raise ValueError("Dict item must contain 'audio' or 'audio_path'.")

        # -------- case 4: object with attributes --------
        else:
            if hasattr(item, "chunk_id"):
                chunk_id = str(item.chunk_id)

            if hasattr(item, "tensor") and item.tensor is not None:
                x = item.tensor.detach().cpu()
                if x.ndim > 1:
                    x = x.mean(dim=1)
                audio_np = x.numpy().astype(np.float32, copy=False)

            elif hasattr(item, "audio_path") and item.audio_path is not None:
                wav, sr = sf.read(str(item.audio_path))
                if sr != 16000:
                    raise ValueError(
                        f"forward_batch currently expects 16k audio chunks, got sr={sr} "
                        f"for {item.audio_path}"
                    )
                if wav.ndim > 1:
                    wav = wav.mean(axis=1)
                audio_np = wav.astype(np.float32, copy=False)

            else:
                raise TypeError(
                    f"Unsupported batch item type: {type(item)}. "
                    f"Expected tensor / ndarray / dict / object with tensor or audio_path."
                )

        if audio_np is None or audio_np.size == 0:
            raise ValueError(f"Empty audio for batch item: {chunk_id}")

        return {
            "chunk_id": chunk_id,
            "audio_np": audio_np,
            "num_samples": int(audio_np.shape[0]),
        }    
    @torch.inference_mode()
    def forward_batch(
        self,
        chunk_records_or_arrays,
        max_batch_items: int,
        max_batch_frames: int,
        sort_by_duration: bool = True
    ):
        """
        Stage 2: 批量声学前向。
        
        参数
        ----
        chunk_records_or_arrays:
            一个列表，元素可为:
            - torch.Tensor
            - np.ndarray
            - {"chunk_id": ..., "audio": ...} / {"chunk_id": ..., "audio_path": ...}
            - 具有 .chunk_id + .tensor / .audio_path 的对象
        max_batch_items:
            单个 micro-batch 最多包含多少条 chunk
        max_batch_frames:
            单个 micro-batch 允许的“估计总帧数”上限。
            这里使用 len(audio)/16000/frame_hop 估计，而不是模型真实输出帧。
        sort_by_duration:
            是否按时长排序后再组 batch，以减少 padding 浪费。
        
        返回
        ----
        List[AcousticEvidence]
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("Aligner model/processor not loaded.")

        if not chunk_records_or_arrays:
            return []

        if max_batch_items <= 0:
            raise ValueError(f"max_batch_items must be > 0, got {max_batch_items}")
        if max_batch_frames <= 0:
            raise ValueError(f"max_batch_frames must be > 0, got {max_batch_frames}")

        # ---------------------------------------------------------
        # 1. 规整输入
        # ---------------------------------------------------------
        items = []
        for i, raw_item in enumerate(chunk_records_or_arrays):
            item = self._coerce_batch_item(raw_item, idx=i)
            est_frames = max(1, int(round((item["num_samples"] / 16000.0) / self.frame_hop)))
            item["est_frames"] = est_frames
            item["orig_idx"] = i
            items.append(item)

        if sort_by_duration:
            items.sort(key=lambda x: x["num_samples"])

        # ---------------------------------------------------------
        # 2. 组 micro-batch
        # ---------------------------------------------------------
        micro_batches = []
        cur_batch = []
        cur_frames = 0

        for item in items:
            need_new_batch = (
                len(cur_batch) >= max_batch_items or
                (len(cur_batch) > 0 and cur_frames + item["est_frames"] > max_batch_frames)
            )
            if need_new_batch:
                micro_batches.append(cur_batch)
                cur_batch = []
                cur_frames = 0

            cur_batch.append(item)
            cur_frames += item["est_frames"]

        if cur_batch:
            micro_batches.append(cur_batch)

        if self.verbose:
            self._log_header("Forward Batch")
            print(f"  - Total chunks: {len(items)}")
            print(f"  - Micro-batches: {len(micro_batches)}")
            print(f"  - max_batch_items={max_batch_items}, max_batch_frames={max_batch_frames}")

        # ---------------------------------------------------------
        # 3. 逐 micro-batch 前向
        # ---------------------------------------------------------
        outputs = [None] * len(items)

        for mb_idx, batch in enumerate(micro_batches):
            audio_list = [x["audio_np"] for x in batch]
            chunk_ids = [x["chunk_id"] for x in batch]
            raw_input_lengths = torch.tensor(
                [x["num_samples"] for x in batch],
                dtype=torch.long
            )

            inputs = self.processor(
                audio_list,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            logits = self.model(**inputs).logits              # (B, T_pad, V)
            log_probs = torch.log_softmax(logits, dim=-1).detach().cpu()

            B, T_pad, V = log_probs.shape

            # 尽量使用模型自己的长度映射函数，得到每条样本的真实输出帧数
            out_lengths = None
            if hasattr(self.model, "_get_feat_extract_output_lengths"):
                try:
                    out_lengths = self.model._get_feat_extract_output_lengths(raw_input_lengths)
                    out_lengths = out_lengths.detach().cpu().tolist()
                except Exception:
                    out_lengths = None

            # fallback: 用输入长度按 padded 最大长度线性缩放
            if out_lengths is None:
                max_in_len = int(raw_input_lengths.max().item())
                out_lengths = [
                    max(1, int(round(T_pad * (int(n.item()) / max_in_len))))
                    for n in raw_input_lengths
                ]

            for j in range(B):
                T_real = int(out_lengths[j])
                T_real = max(1, min(T_real, T_pad))

                lp = log_probs[j, :T_real].numpy()
                dur_s = batch[j]["num_samples"] / 16000.0

                ev = AcousticEvidence(
                    chunk_id=chunk_ids[j],
                    log_probs=lp,
                    num_frames=lp.shape[0],
                    vocab_size=lp.shape[1],
                    frame_hop_s=self.frame_hop,
                    duration_s=dur_s,
                )

                outputs[batch[j]["orig_idx"]] = ev

            if self.verbose:
                total_est_frames = sum(x["est_frames"] for x in batch)
                print(
                    f"  - micro-batch {mb_idx + 1}/{len(micro_batches)}: "
                    f"B={B}, T_pad={T_pad}, V={V}, est_frames={total_est_frames}"
                )

        # 按原始输入顺序返回
        return outputs