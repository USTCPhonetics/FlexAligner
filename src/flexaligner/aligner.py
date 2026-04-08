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
    def __init__(self, config: dict, phone_to_id: Optional[Dict[str, int]] = None):
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
        # Resources
        self.model = None
        self.processor = None
        self.lexicon = None
        self.phone_to_id = {}

        # 1. Load Resources
        self._load_resources()

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

    # def _load_resources(self):
    #     # Lexicon
    #     lex_path = self.config.get("lexicon_path")
    #     if lex_path:
    #         self.lexicon = PronouncingDictionary.from_path(lex_path)
    #     else:
    #         self.lexicon = PronouncingDictionary()

    #     # Model
    #     model_path = self.config.get("align_model_path")
    #     if not model_path: return

    #     self._log(f"Loading model from {model_path}...")
        
    #     # Load Logic (Local vs Cloud)
    #     is_local = os.path.isdir(model_path)
    #     load_kwargs = {}
    #     print(f"[info]: model path {model_path}")
    #     if not is_local:
            
            
    #         load_kwargs["subfolder"] = f"{self.config.get('lang', 'zh')}/aligner"
            
    #         print(f"\[info\]: loading from cloud path---{ load_kwargs['subfolder'] }")

    #     try:
    #         self.processor = AutoProcessor.from_pretrained(model_path, **load_kwargs)
    #         self.model = AutoModelForCTC.from_pretrained(model_path, **load_kwargs).to(self.device)
    #     except Exception:
    #         # Fallback
    #         self.processor = AutoProcessor.from_pretrained(model_path)
    #         self.model = AutoModelForCTC.from_pretrained(model_path).to(self.device)
    
    #     self.model.eval()
    #     if self.processor:
    #         self.phone_to_id = self.processor.tokenizer.get_vocab()

    def _load_resources(self):
        # =====================================================================
        # 1. 基础物理词表加载
        # =====================================================================
        lex_path = self.config.get("lexicon_path")
        if lex_path:
            self.lexicon = PronouncingDictionary.from_path(lex_path)
            # print(f"[info] loaded local lexicon from {lex_path}")
        else:
            self.lexicon = PronouncingDictionary()
            # print("[info] initialized empty lexicon")

        # =====================================================================
        # 🛡️ 2. 增量装甲：全量吸收 G2P 巨型字典，打通终极闭环
        # =====================================================================
        # if self.config.get("lang", "zh") == "en":
        #     try:
        #         from g2p_en.g2p import G2p
        #         g2p_inst = G2p()
        #         cmu_dict = g2p_inst.cmu
                
        #         added_words = 0
        #         for w, prons in cmu_dict.items():
        #             w_upper = w.upper()
                    
        #             # 坚守原则：本地手工打磨的字典优先级最高，绝对不覆盖
        #             if w_upper not in self.lexicon:
        #                 # 剔除潜在的标点符号，提取最纯净的声学 ID
        #                 clean_pron = [p for p in prons[0] if p.isalnum()]
                        
        #                 # [全地形兼容探针] 强行绕过类的封装，注入数据
        #                 try:
        #                     if hasattr(self.lexicon, 'add_word'):
        #                         self.lexicon.add_word(w_upper, clean_pron)
        #                     elif isinstance(self.lexicon, dict) or hasattr(self.lexicon, '__setitem__'):
        #                         # 大多数发音字典映射格式为: Word -> List[List[Phone]]
        #                         self.lexicon[w_upper] = [clean_pron]
        #                     elif hasattr(self.lexicon, '_dict'):
        #                         self.lexicon._dict[w_upper] = [clean_pron]
        #                     added_words += 1
        #                 except Exception as inner_e:
        #                     pass # 忽略极少数无法兼容的奇葩单词
                            
        #         # print(f"[info] 🚀 Aligner G2P Augmentation: {added_words} OOV words injected into RAM.")
        #     except Exception as e:
        #         print(f"[warning] Aligner G2P augmentation failed: {e}")
        if self.config.get("lang", "zh") == "en":
            try:
                from g2p_en.g2p import G2p
                g2p_inst = G2p()
                cmu_dict = g2p_inst.cmu
                
                added_words = 0
                
                # ---------------------------------------------------------
                # [战术修正] 提前解包 PronouncingDictionary 提取底层词表
                # 避免在 13 万次循环中反复触发 TypeError，实现 O(1) 极速哈希查询
                # ---------------------------------------------------------
                local_words = set()
                if isinstance(self.lexicon, dict):
                    local_words = set(self.lexicon.keys())
                elif hasattr(self.lexicon, '_dict'):
                    local_words = set(self.lexicon._dict.keys())
                elif hasattr(self.lexicon, 'words'):
                    local_words = set(self.lexicon.words)
                
                for w, prons in cmu_dict.items():
                    w_upper = w.upper()
                    
                    # 坚守原则：本地手工打磨的字典优先级最高，绝对不覆盖
                    # 使用提取出的纯净 Set 进行判定，或使用 Try-Except 兜底
                    if local_words:
                        is_known = w_upper in local_words
                    else:
                        try:
                            is_known = w_upper in self.lexicon
                        except TypeError:
                            is_known = False

                    if not is_known:
                        # 剔除潜在的标点符号，提取最纯净的声学 ID
                        clean_pron = [p for p in prons[0] if p.isalnum()]
                        
                        # [全地形兼容探针] 强行绕过类的封装，注入数据
                        try:
                            if hasattr(self.lexicon, 'add_word'):
                                self.lexicon.add_word(w_upper, clean_pron)
                            elif isinstance(self.lexicon, dict) or hasattr(self.lexicon, '__setitem__'):
                                self.lexicon[w_upper] = [clean_pron]
                            elif hasattr(self.lexicon, '_dict'):
                                self.lexicon._dict[w_upper] = [clean_pron]
                            added_words += 1
                        except Exception:
                            pass # 忽略极少数无法兼容的奇葩单词
                            
                # 建议把这行打印打开一次，看看这针“强心剂”到底打进去了多少词
                print(f"[info] 🚀 Aligner G2P Augmentation: {added_words} OOV words injected into RAM.")
            except Exception as e:
                print(f"[warning] Aligner G2P augmentation failed: {e}")

        # =====================================================================
        # 3. 核心计算图与模型装载 (保留原有逻辑)
        # =====================================================================
        model_path = self.config.get("align_model_path")
        if not model_path: return

        self._log(f"Loading model from {model_path}...")
        
        is_local = os.path.isdir(model_path)
        load_kwargs = {}
        print(f"[info]: model path {model_path}")
        
        if not is_local:
            load_kwargs["subfolder"] = f"{self.config.get('lang', 'zh')}/aligner"
            # print(f"[info]: loading from cloud path---{ load_kwargs['subfolder'] }")

        try:
            self.processor = AutoProcessor.from_pretrained(model_path, **load_kwargs)
            self.model = AutoModelForCTC.from_pretrained(model_path, **load_kwargs).to(self.device)
        except Exception:
            # Fallback
            self.processor = AutoProcessor.from_pretrained(model_path)
            self.model = AutoModelForCTC.from_pretrained(model_path).to(self.device)
    
        self.model.eval()
        if self.processor:
            self.phone_to_id = self.processor.tokenizer.get_vocab()
    
    @torch.inference_mode()
    def align_locally(self, chunk_tensor: torch.Tensor, text: str, file_id: str = "segment") -> Dict[str, List[AlignmentSegment]]:
        """
        Executes reference alignment logic and adapts output to Pipeline format.
        """
        print(f"Aligning chunk '{file_id}' with text: {text}")
        # print("align_locally called with chunk_tensor shape:", chunk_tensor.shape)
        # input("Press Enter to continue...")  # Debug pause to inspect inputs
        if self.model is None or self.lexicon is None:
             return {"phones": [], "words": []}

        # 1. Forward (Match compute_logp_frames)
        inputs = self.processor(chunk_tensor.numpy(), sampling_rate=16000, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        logits = self.model(**inputs).logits 
        logp = torch.log_softmax(logits, dim=-1)[0].cpu().numpy() # (T, V)

        T = logp.shape[0]
        actual_dur = chunk_tensor.size(0) / 16000.0

        if self.verbose:
            self._log_header(f"Aligning: {file_id}")
            print(f"  - Frames: {T}, Duration: {actual_dur:.3f}s")
            print(f"  - Text: {text}")

        # 2. Build Graph (Reference Algorithm)
        words = text.split()
        try:
            graph, entry_bias = build_phone_graph_optional_sil(
                words=words,
                prondict=self.lexicon,
                phone_to_id=self.phone_to_id,
                sil_phone=self.sil_phone,
                optional_sil_between_words=self.optional_sil,
                optional_sil_at_start=self.sil_at_ends,
                optional_sil_at_end=self.sil_at_ends,
                sil_cost=self.sil_cost
            )
        except Exception as e:
            print(f"❌ Graph construction failed: {e}")
            return {"phones": [], "words": []}

        # 3. Decode (Reference Algorithm)
        # try:
        #     ali = align_beam_viterbi(
        #         logp=logp,
        #         graph=graph,
        #         entry_bias=entry_bias,
        #         p_stay=self.p_stay,
        #         beam_size=self.beam_size,
        #         word_sil_label=self.word_sil_label
        #     )
        # except Exception as e:
        #     print(f"❌ Viterbi failed: {e}")
        #     return {"phones": [], "words": []}
        
        sil_phone_id = self.phone_to_id.get(self.sil_phone) if self.sil_phone else None
        
        try:
            ali = align_beam_viterbi(
                logp=logp,
                graph=graph,
                entry_bias=entry_bias,
                p_stay=self.p_stay,
                beam_size=self.beam_size,
                word_sil_label=self.word_sil_label,
                # --- 核心物理挂载 ---
                boundary_lambda=self.boundary_lambda,
                boundary_context_s=self.boundary_context_s,
                frame_hop_s=self.frame_hop,
                sil_phone_id=sil_phone_id,
                min_sil_dur_ms=self.min_sil_dur_ms,
                sil_enter_cost=self.sil_enter_cost
            )
        except Exception as e:
            print(f"❌ Viterbi failed: {e}")
            return {"phones": [], "words": []}

        # 4. Format Output (Frames -> Seconds)
        # Using self.frame_hop from config to strictly match reference logic
        
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

        result = {"phones": phones_out, "words": words_out}

        # 5. Optional TSV Dump (Internal Debug)
        if self.align_out_dir:
            self._save_tsv(file_id, result)

        return result

    def _save_tsv(self, file_id, res):
        out = Path(self.align_out_dir)
        out.mkdir(parents=True, exist_ok=True)
        for kind in ["phones", "words"]:
            with open(out / f"{file_id}.{kind}.tsv", "w", newline="") as f:
                writer = csv.writer(f, delimiter="\t")
                writer.writerow(["start", "end", "label"])
                for s in res[kind]:
                    writer.writerow([f"{s.start:.4f}", f"{s.end:.4f}", s.label])