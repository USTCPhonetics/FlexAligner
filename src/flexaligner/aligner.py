import torch
import numpy as np
import math
import os
import json
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Set
from transformers import AutoModelForCTC, AutoProcessor

# ==========================================
#  1. Helper Structures (数据结构区)
# ==========================================

@dataclass
class AlignmentSegment:
    """对外输出的标准格式"""
    label: str
    start: float
    end: float
    score: float = 0.0

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

# ==========================================
#  2. Helper Class: Lexicon (词典区)
# ==========================================

class PronouncingDictionary:
    def __init__(self):
        self.lex: Dict[str, List[List[str]]] = {}

    def add(self, word: str, pron: List[str]) -> None:
        self.lex.setdefault(word, []).append(list(pron))

    def get_prons(self, word: str) -> List[List[str]]:
        # 简单归一化
        if word not in self.lex:
            # 尝试大写
            if word.upper() in self.lex:
                return self.lex[word.upper()]
            # 尝试小写
            if word.lower() in self.lex:
                return self.lex[word.lower()]
            raise KeyError(f"Word not in lexicon: {word}")
        return self.lex[word]

    @staticmethod
    def from_path(path: str) -> "PronouncingDictionary":
        pd = PronouncingDictionary()
        if not os.path.exists(path):
            print(f"[Warn] Lexicon not found at {path}")
            return pd
            
        with open(path, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln or ln.startswith("#"): 
                    continue
                parts = ln.split()
                if len(parts) < 2: 
                    continue
                
                # [核心修正] 强制转小写，与 Frontend/Chunker 保持符号一致性
                w = parts[0].lower()
                pd.add(w, parts[1:])
        return pd

# ==========================================
#  3. Core Class: LocalAligner (核心逻辑)
# ==========================================

class LocalAligner:
    # [核心修正] 增加 phone_to_id 接口，支持云端动态注入
    def __init__(self, config: dict, phone_to_id: Optional[Dict[str, int]] = None):
        self.config = config or {}
        # 统一从配置读取设备
        self.device = torch.device(self.config.get("device", "cpu"))
        
        # 资源占位
        self.model = None
        self.processor = None
        self.lexicon = None
        self.phone_to_id = {}
        
        # 参数加载 (带默认值兜底)
        self.beam_size = self.config.get("align_beam_size", 400)
        self.p_stay = self.config.get("p_stay", 0.92)
        self.sil_phone = self.config.get("sil_phone", "sil")
        self.sil_cost = self.config.get("sil_cost", -0.5)
        self.frame_hop = self.config.get("frame_hop_s", 0.01)
        self.optional_sil = self.config.get("optional_sil", True)
        self.offset_s = self.config.get("offset_s", 0.0125)
        
        # 1. 加载模型和词典
        self._load_resources()

        # 2. [核心修正] 音素表加载优先级：注入 > 模型自带 > 本地JSON
        if phone_to_id is not None:
            # 优先使用 Pipeline 传进来的 (云端英语模式)
            print(f"[LocalAligner] Using injected phone_to_id map ({len(phone_to_id)} tokens)")
            self.phone_to_id = phone_to_id
        elif not self.phone_to_id:
            # 如果模型没带 vocab (非 processor 加载)，尝试读本地 json
            json_path = self.config.get("phone_json_path")
            if json_path and os.path.exists(json_path):
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        self.phone_to_id = json.load(f)
                except Exception as e:
                    print(f"[LocalAligner] Failed to load phones.json: {e}")
        
        # 调试用反查表
        self.id_to_phone = {v: k for k, v in self.phone_to_id.items()}
        
        # 3. 确保 SIL ID 存在
        self.sil_id = self.phone_to_id.get(self.sil_phone, 0)

    def _load_resources(self):
        """加载模型和词典"""
        # 1. Lexicon
        lex_path = self.config.get("lexicon_path")
        if lex_path:
            self.lexicon = PronouncingDictionary.from_path(lex_path)
        else:
            self.lexicon = PronouncingDictionary() # 空词典兜底

        # [核心修正] Stage 2 OOV 热修复 (同步 Chunker 的逻辑)
        if self.config.get("lang") == "en":
            oov_patch = {
                "montreal": ["M", "AA1", "N", "T", "R", "IY0", "AA1", "L"],
                "forced": ["F", "AO1", "R", "S", "T"],
                "aligner": ["AH0", "L", "AY1", "N", "ER0"],
            }
            patched = 0
            for w, p in oov_patch.items():
                try:
                    self.lexicon.get_prons(w)
                except KeyError:
                    self.lexicon.add(w, p)
                    patched += 1
            if patched > 0:
                print(f"[LocalAligner] Applied hotfix for {patched} OOV words.")

        # 2. Model
        model_path = self.config.get("align_model_path")
        if not model_path:
            return 

        print(f"[LocalAligner] Loading model from {model_path}...")
        
        # [核心修正] 智能路由逻辑
        is_local_dir = os.path.isdir(model_path)
        load_kwargs = {}
        
        if not is_local_dir:
            # 动态决定 subfolder
            lang = self.config.get("lang", "zh")
            load_kwargs["subfolder"] = f"{lang}/aligner"
            print(f"[LocalAligner] Cloud mode detected. Target subfolder: {load_kwargs['subfolder']}")

        try:
            # 策略 A: 尝试带 subfolder 加载 (云端) 或 直接加载 (本地)
            self.processor = AutoProcessor.from_pretrained(model_path, **load_kwargs)
            self.model = AutoModelForCTC.from_pretrained(model_path, **load_kwargs).to(self.device)
        except (OSError, ValueError) as e:
            # 策略 B: 降级重试 (防止用户手动改了本地文件夹结构)
            print(f"[LocalAligner] Routing failed ({e}), falling back to root/default...")
            try:
                self.processor = AutoProcessor.from_pretrained(model_path)
                self.model = AutoModelForCTC.from_pretrained(model_path).to(self.device)
            except Exception as final_e:
                print(f"[LocalAligner] Critical: Failed to load model: {final_e}")
                return
            
        self.model.eval()
        
        # 如果加载了 Processor，优先提取它的 Vocab
        if self.processor:
            self.phone_to_id = self.processor.tokenizer.get_vocab()

    @torch.inference_mode()
    def align_locally(self, chunk_tensor: torch.Tensor, text: str) -> Dict[str, List[AlignmentSegment]]:
        """
        Stage 2 推理入口
        """
        if self.model is None or self.lexicon is None:
             return {"phones": [], "words": []}

        # 1. Forward Pass
        inputs = self.processor(chunk_tensor.numpy(), sampling_rate=16000, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        logits = self.model(**inputs).logits 
        log_probs = torch.log_softmax(logits, dim=-1).squeeze(0).cpu().numpy() # (T, V)

        # ==========================================
        # 🧪 [物理探针] 动态步长检测与诊断
        # ==========================================
        T_frames = log_probs.shape[0]
        actual_samples = chunk_tensor.size(0)
        actual_duration = actual_samples / 16000.0
        
        # 计算物理步长 (Seconds Per Frame)
        if T_frames > 1:
            calculated_hop = actual_duration / T_frames
        else:
            calculated_hop = self.frame_hop # Fallback
            
        print("\n" + "="*40)
        print(f"🧪 [Physics Probe] Audio Analysis:")
        print(f"   Samples:  {actual_samples}")
        print(f"   Duration: {actual_duration:.6f} sec")
        print(f"   Frames:   {T_frames}")
        print(f"   SPF(Calc): {calculated_hop*1000:.3f} ms")
        print(f"   SPF(Conf): {self.frame_hop*1000:.3f} ms")
        
        # 诊断判定
        diff = abs(calculated_hop - self.frame_hop) * 1000
        if diff > 1.0: # 误差超过 1ms
             print(f"⚠️  [WARNING] STRIDE MISMATCH DETECTED!")
        else:
             print(f"✅  [OK] Physics matches Config.")
        current_hop = self.frame_hop
             
        print("="*40 + "\n")
        # ==========================================

        # 2. Build Graph
        words = text.split()
        try:
            graph, entry_bias = build_phone_graph_optional_sil(
                words=words,
                prondict=self.lexicon,
                phone_to_id=self.phone_to_id,
                sil_phone=self.sil_phone,
                optional_sil_between_words=self.optional_sil,
                optional_sil_at_start=True,
                optional_sil_at_end=True,
                sil_cost=self.sil_cost
            )
        except Exception as e:
            print(f"[LocalAligner] Graph build failed for '{text}': {e}")
            return {"phones": [], "words": []}

        # 3. Viterbi Decode
        try:
            ali_result = align_beam_viterbi(
                log_probs, graph, entry_bias,
                p_stay=self.p_stay,
                beam_size=self.beam_size
            )
        except Exception as e:
            print(f"[LocalAligner] Viterbi failed for '{text}': {e}")
            return {"phones": [], "words": []}

        # 4. Convert Frames to Seconds
        # 使用上面诊断出的 current_hop
        
        phones_out = []
        for lab, s, e in ali_result.phone_segments_f:
            # 物理越界防御：min(t, duration)
            ####################################################
            start_t = min(self.offset_s + (s * current_hop), actual_duration)
            ####################################################
            end_t = min(self.offset_s + (e * current_hop), actual_duration)
            phones_out.append(AlignmentSegment(lab, start_t, end_t))
            
        words_out = []
        for lab, s, e in ali_result.word_segments_f:
            start_t = min(self.offset_s + (s * current_hop), actual_duration)
            end_t = min(self.offset_s + (e * current_hop), actual_duration)
            words_out.append(AlignmentSegment(lab, start_t, end_t))

        return {"phones": phones_out, "words": words_out}

# ==========================================
#  4. Algorithms (图算法区)
# ==========================================

NEG_INF = -1e30

def _eps_closure(num_nodes: int, eps_adj: List[List[int]]) -> List[Set[int]]:
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
    sil_cost: float = 0.0,
) -> Tuple[PhoneGraph, np.ndarray]:
    
    next_node = 0
    def new_node():
        nonlocal next_node
        nid = next_node
        next_node += 1
        return nid

    START = new_node()
    emit_edges = []
    eps_edges = []
    entry_bias = []

    def add_emit(u, v, phone, widx, w, bias=0.0):
        # [核心战术] 音素模糊匹配 (Fuzzy Matching)
        # 目标：解决 AO vs AO1, AH vs AH0 的不匹配问题
        
        target_id = None
        
        # 1. 尝试精确匹配
        if phone in phone_to_id:
            target_id = phone_to_id[phone]
        else:
            # 2. 尝试模糊匹配 (去重音 / 加重音)
            phone_pure = ''.join(filter(str.isalpha, phone)) # AO1 -> AO
            
            # 2a. 尝试纯音素 (AO)
            if phone_pure in phone_to_id:
                target_id = phone_to_id[phone_pure]
            else:
                # 2b. 尝试加重音变体 (AO -> AO1, AO0, AO2)
                # 优先尝试 1 (Primary Stress)
                for suffix in ["1", "0", "2"]:
                    variant = phone_pure + suffix
                    if variant in phone_to_id:
                        target_id = phone_to_id[variant]
                        break
        
        if target_id is None:
            # 确实没救了，跳过这条边（可能导致图断裂，但比崩溃好）
            # print(f"[Warn] OOV Phone: {phone} (and variants) not in vocab")
            return

        emit_edges.append(EmitEdge(u=u, v=v, phone=phone, phone_id=target_id, word_index=widx, word=w))
        entry_bias.append(bias)

    def add_eps(u, v):
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
            print(f"[Warn] OOV in Graph: {w}, skipping word in graph")
            # 遇到 OOV 无法构建发音路径，直接由 EPS 边跳过，避免图断裂
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

    out_emit = {}
    in_emit = {}
    for ei, e in enumerate(emit_edges):
        out_emit.setdefault(e.u, []).append(ei)
        in_emit.setdefault(e.v, []).append(ei)

    states = []
    for e in emit_edges:
        pred_idxs = []
        for node in bwd_cl[e.u]:
            pred_idxs.extend(in_emit.get(node, []))
        succ_idxs = []
        for node in fwd_cl[e.v]:
            succ_idxs.extend(out_emit.get(node, []))
        states.append(PhoneState(edge=e, preds=tuple(sorted(set(pred_idxs))), succs=tuple(sorted(set(succ_idxs)))))

    start_states = []
    for node in fwd_cl[START]:
        start_states.extend(out_emit.get(node, []))
    start_states = sorted(set(start_states))

    end_states = []
    for si, st in enumerate(states):
        if END in fwd_cl[st.edge.v]:
            end_states.append(si)
    
    return PhoneGraph(states=states, start_states=start_states, end_states=end_states), np.asarray(entry_bias, dtype=np.float32)


def align_beam_viterbi(
    logp: np.ndarray,
    graph: PhoneGraph,
    entry_bias: np.ndarray,
    p_stay: float = 0.92,
    beam_size: int = 400,
    word_sil_label: str = "sil",
) -> AlignmentResult:
    T, V = logp.shape
    
    lp_stay = math.log(p_stay)
    lp_move = math.log(1.0 - p_stay)

    bp: List[Dict[int, int]] = []
    cur_scores: Dict[int, float] = {}
    cur_bp: Dict[int, int] = {}

    for s in graph.start_states:
        phid = graph.states[s].edge.phone_id
        cur_scores[s] = float(logp[0, phid]) + float(entry_bias[s])
        cur_bp[s] = s

    # Initial Pruning
    if len(cur_scores) > beam_size:
        top = sorted(cur_scores.items(), key=lambda kv: kv[1], reverse=True)[:beam_size]
        cur_scores = {k: v for k, v in top}
        cur_bp = {k: cur_bp[k] for k, _ in top}
    bp.append(cur_bp)

    # Forward
    for t in range(1, T):
        nxt_scores = {}
        nxt_bp = {}
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
        
        # Beam Pruning
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
    for s, sc in cur_scores.items():
        term = sc + lp_move
        if s in end_set and term > best_score:
            best_score = term
            best_state = s
            
    if best_state is None and len(cur_scores) > 0:
        best_state = max(cur_scores.items(), key=lambda kv: kv[1])[0]

    # Backtrace
    path = np.empty((T,), dtype=np.int32)
    if best_state is not None:
        cur = int(best_state)
        for t in range(T - 1, -1, -1):
            path[t] = cur
            cur = int(bp[t].get(cur, cur))
    else:
        path.fill(0) 

    aligned_phone_ids = np.array([graph.states[int(s)].edge.phone_id for s in path], dtype=np.int32)

    # Extract Segments
    phone_segments_f = []
    if T > 0:
        cur_ph = graph.states[int(path[0])].edge.phone
        start = 0
        for t in range(1, T):
            ph = graph.states[int(path[t])].edge.phone
            if ph != cur_ph:
                phone_segments_f.append((cur_ph, start, t))
                cur_ph = ph
                start = t
        phone_segments_f.append((cur_ph, start, T))

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