import torch
import os
from typing import List, Optional
from dataclasses import dataclass
from transformers import Wav2Vec2ForCTC, AutoProcessor

# 引入我们在 io.py 定义的数据结构
try:
    from .io import AudioChunk
except ImportError:
    @dataclass
    class AudioChunk:
        tensor: torch.Tensor
        start_time: float
        end_time: float
        text: str
        chunk_id: str

# ==========================================
#  Helper Data Structures
# ==========================================

@dataclass
class Point:
    token_index: int 
    time_index: int 

@dataclass
class Segment:
    label: str
    start_frame: int
    end_frame: int

    @property
    def duration_frames(self):
        return self.end_frame - self.start_frame

@dataclass
class PronCandidate:
    phones: List[str]
    pron_choice_idxs: List[int]
    score: float

@dataclass
class WordSeg:
    start: float
    dur: float
    word: str
    @property
    def end(self) -> float:
        return self.start + self.dur

@dataclass
class InternalChunk:
    start: float
    end: float
    words: List[str]

# ==========================================
#  CTCChunker Class (核心逻辑)
# ==========================================

class CTCChunker:
    def __init__(self, config: dict):
        self.config = config or {}
        self.device = torch.device(self.config.get("device", "cpu"))
        self.lang = self.config.get("lang", "zh")
        
        # 资源
        self.model = None
        self.processor = None
        self.lexicon = {}
        self.phone_to_id = {} 
        self.blank_id = 0
        
        # 参数
        self.beam_size = self.config.get("beam_size", 10) # 默认较小，为了速度
        self.min_chunk_s = self.config.get("min_chunk_s", 1.0)
        self.max_chunk_s = self.config.get("max_chunk_s", 12.0)
        self.max_gap_s = self.config.get("max_gap_s", 0.35)
        self.min_words = self.config.get("min_words", 2)
        self.pad_s = self.config.get("pad_s", 0.15)
        self.blank_token = self.config.get("blank_token", "<pad>")
        
        # [物理常数]
        self.config_hop = self.config.get("frame_hop_s", 0.01)

        self._load_resources()


    def _load_resources(self):
        """加载资源 (含云端/本地逻辑 + 英语兼容性补丁)"""
        lex_path = self.config.get("lexicon_path")
        self.lexicon = self._read_lexicon(lex_path)
        
        model_path = self.config.get("chunk_model_path")
        if not model_path: return

        print(f"[CTCChunker] Requesting model: {model_path}")
        load_kwargs = {}
        is_local_dir = os.path.isdir(model_path)
        
        if not is_local_dir:
            # 云端模式：自动定位 subfolder
            load_kwargs["subfolder"] = f"{self.lang}/chunker"
            print(f"[CTCChunker] Mode: Cloud Repo ({load_kwargs['subfolder']})")
        else:
            print("[CTCChunker] Mode: Local Override")

        try:
            self.processor = AutoProcessor.from_pretrained(model_path, **load_kwargs)
            self.model = Wav2Vec2ForCTC.from_pretrained(model_path, **load_kwargs).to(self.device)
            self.model.eval()
            print(f"[CTCChunker] Successfully loaded model.")
        except Exception as e:
            print(f"[CTCChunker] Routing failed ({e}), falling back to root...")
            try:
                self.processor = AutoProcessor.from_pretrained(model_path)
                self.model = Wav2Vec2ForCTC.from_pretrained(model_path).to(self.device)
            except Exception as final_e:
                raise RuntimeError(f"Model load failed: {final_e}")

        if hasattr(self.processor, "tokenizer"):
            self.phone_to_id = self.processor.tokenizer.get_vocab()
            if self.blank_token in self.phone_to_id:
                self.blank_id = self.phone_to_id[self.blank_token]
            elif self.processor.tokenizer.pad_token_id is not None:
                self.blank_id = self.processor.tokenizer.pad_token_id
            else:
                self.blank_id = 0
            
            # ==========================================
            # 🛠️ [英语专用补丁] 建立模糊匹配索引
            # ==========================================
            if self.lang == "en":
                # 1. 识别垃圾桶音素 (Garbage/Null)，通常是 "O"
                # 如果词典里有 "O" 但模型没有，记录下来以免报错
                self.garbage_tokens = {"O", "[UNK]", "<unk>"}
                
                # 2. 预计算 "纯净版" 音素映射
                # 这样在 Beam Search 时，不用每次都 split 字符串，速度更快
                self.fuzzy_map = {}
                for token, pid in self.phone_to_id.items():
                    # 存入原始 key (如 "AA")
                    self.fuzzy_map[token] = pid
                    
                    # 存入去重音 key (如 "AA" -> pid)
                    # 这样当词典查 "AA1" 时，我们去数字变成 "AA"，查这个表就能拿到 ID
                    pure_token = ''.join(filter(str.isalpha, token))
                    if pure_token and pure_token not in self.fuzzy_map:
                        self.fuzzy_map[pure_token] = pid
                
                print(f"[CTCChunker] 🇬🇧 English Hotfix Applied: Fuzzy map built with {len(self.fuzzy_map)} entries.")
            # ==========================================
            
        else:
            self.blank_id = 0
    @torch.inference_mode()
    def find_chunks(self, audio_tensor: torch.Tensor, text_list: List[str]) -> List[AudioChunk]:
        """[主入口] 执行 Stage 1 完整流程 (含自适应重试 + Token对齐修复)"""
        if self.model is None:
            raise RuntimeError("CTC 模型未加载")

        # 1. Forward Pass
        input_values = self.processor(
            audio_tensor.numpy(), 
            sampling_rate=16000, 
            return_tensors="pt"
        ).input_values.to(self.device)
        
        logits = self.model(input_values).logits 
        log_probs = torch.log_softmax(logits, dim=-1).squeeze(0)
        
        # --- 物理探针 ---
        actual_samples = audio_tensor.size(0)
        actual_duration = actual_samples / 16000.0
        T_frames = log_probs.size(0)
        calculated_spf = actual_duration / T_frames if T_frames > 0 else 0
        
        # 自动纠正步长
        if abs(calculated_spf - self.config_hop) > 0.001:
             print(f"[CTCChunker] ⚠️ Physics override: hop={calculated_spf*1000:.2f}ms")
             spf = calculated_spf
        else:
             spf = self.config_hop
        # ----------------

        # 2. 发音变体 Beam Search (带自适应重试)
        prons_per_word = self._words_to_pronunciations(text_list)
        
        best_candidate = None
        for attempt_beam in [self.beam_size, 50, 200, 1000]:
            best_candidate = self._beam_search(log_probs, text_list, prons_per_word, beam_width=attempt_beam)
            if best_candidate:
                if attempt_beam > self.beam_size:
                    print(f"[CTCChunker] 💡 Expanded beam to {attempt_beam} to find path.")
                break
        
        if not best_candidate:
            print("[CTCChunker] ❌ Beam search failed even with beam=1000. Fallback to full audio.")
            return [AudioChunk(
                tensor=audio_tensor,
                start_time=0.0,
                end_time=audio_tensor.size(0)/16000.0,
                text=" ".join(text_list),
                chunk_id="chunk_fallback"
            )]

        # 3. Viterbi 强制对齐
        # target_ids 只包含模型认识的 token ID (过滤掉了 OOV 和 Garbage)
        target_ids = [self.phone_to_id[p] for p in best_candidate.phones if p in self.phone_to_id]
        if not target_ids: return []

        trellis = build_trellis(log_probs, target_ids, self.blank_id)
        points = backtrace(trellis, log_probs, target_ids, self.blank_id)
        
        # 4. 转回物理时间戳
        # [核心修复点 1] points_to_segments 需要的 labels 必须与 target_ids 对应
        # 因此我们不能传 best_candidate.phones (含 O)，而要传过滤后的列表
        filtered_phones = [p for p in best_candidate.phones if p in self.phone_to_id]
        token_segs = points_to_segments(points, filtered_phones)
        
        # [核心修复点 2] 构建清洗过的 prons 列表，供 word segment 还原使用
        # 确保 word 切分逻辑看到的音素数量与 Viterbi 输出的一致
        clean_prons = []
        for i, idx in enumerate(best_candidate.pron_choice_idxs):
            raw_pron = prons_per_word[i][idx]
            # 同样只保留模型认识的音素
            if len(raw_pron) > 0 and isinstance(raw_pron[0], list): raw_pron = raw_pron[0] # 防御性解包
            filtered_pron = [p for p in raw_pron if p in self.phone_to_id]
            clean_prons.append(filtered_pron)

        word_segs = self._phones_to_word_segments_robust(
            token_segs, text_list, 
            clean_prons 
        )

        # 5. 切分逻辑
        word_objects = [
            WordSeg(s.start_frame * spf, (s.end_frame - s.start_frame) * spf, s.label)
            for s in word_segs
        ]
        
        internal_chunks = self._merge_words_into_chunks(word_objects)
        audio_dur_s = audio_tensor.size(0) / 16000
        internal_chunks = self._pad_chunks(internal_chunks, word_objects, audio_dur_s)

        # 6. 生成对象
        final_chunks = []
        sr = 16000
        for i, c in enumerate(internal_chunks):
            s_samp = int(c.start * sr)
            e_samp = int(c.end * sr)
            s_samp = max(0, s_samp)
            e_samp = min(audio_tensor.size(0), e_samp)
            
            if e_samp <= s_samp: continue

            chunk_obj = AudioChunk(
                tensor=audio_tensor[s_samp:e_samp].clone(),
                start_time=c.start,
                end_time=c.end,
                text=" ".join(c.words),
                chunk_id=f"chunk_{i:03d}"
            )
            final_chunks.append(chunk_obj)

        print(f"[CTCChunker] Found {len(final_chunks)} chunks.")
        return final_chunks

    # --- 内部核心 ---

    def _read_lexicon(self, path: Optional[str]):
        lexicon = {}
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        word = parts[0].lower() 
                        phones = parts[1:] 
                        lexicon[word] = phones
        
        if self.lang == "en":
            oov_patch = {
                "montreal": ["M", "AA1", "N", "T", "R", "IY0", "AA1", "L"],
                "forced": ["F", "AO1", "R", "S", "T"],
                "aligner": ["AH0", "L", "AY1", "N", "ER0"],
            }
            for word, phones in oov_patch.items():
                if word not in lexicon:
                    lexicon[word] = phones
        return lexicon

    def _words_to_pronunciations(self, words: List[str]) -> List[List[List[str]]]:
        out = []
        for i, w in enumerate(words):
            w_norm = w.strip().lower()
            if not w_norm: continue
            if w_norm not in self.lexicon:
                raise ValueError(f"[CTCChunker] OOV Word: {w_norm}")
            out.append([self.lexicon[w_norm]])
        return out

    def _beam_search(self, log_probs, words, prons_per_word, beam_width=10) -> Optional[PronCandidate]:
        """
        支持动态调整 Beam Width + [核心升级] 音素模糊匹配
        """
        beam = [PronCandidate(phones=[], pron_choice_idxs=[], score=0.0)]
        
        for i, word in enumerate(words):
            new_beam = []
            variants = prons_per_word[i]
            
            for cand in beam:
                for p_idx, pron in enumerate(variants):
                    # 防御性扁平化处理
                    if len(pron) > 0 and isinstance(pron[0], list): pron = pron[0] 
                    
                    new_phones = cand.phones + pron
                    new_ids = []
                    valid_pron = True
                    
                    # --- [核心修复] 音素模糊匹配逻辑 ---
                    for p in new_phones:
                        if not isinstance(p, str): continue
                        
                        # 1. 尝试精确匹配 (例如: "AA", "sil")
                        if p in self.phone_to_id:
                            new_ids.append(self.phone_to_id[p])
                        else:
                            # 2. 尝试去重音匹配 (例如: "AA1" -> "AA")
                            # Chunker 词表通常不带数字，但词典带数字
                            p_pure = ''.join(filter(str.isalpha, p))
                            if p_pure in self.phone_to_id:
                                new_ids.append(self.phone_to_id[p_pure])
                            elif p == "O": 
                                # 3. 特殊处理 "O" (Garbage/Null)
                                # 如果模型词表里没有 "O"，我们选择跳过它，
                                # 而不是判定路径断裂。CTC 会自动处理中间的空白。
                                continue
                            else:
                                # 4. 确实是未知的 OOV，标记路径无效
                                # print(f"DEBUG: Real OOV found: {p} (pure: {p_pure})")
                                valid_pron = False
                                break
                    # -----------------------------------
                    
                    if not valid_pron:
                        continue 
                    
                    # 计算 Viterbi 得分
                    try:
                        trellis = build_trellis(log_probs, new_ids, self.blank_id)
                        # Trellis shape: (T+1, S+1), 取最后时刻最后状态的得分
                        score = float(torch.max(trellis[-1, -1]).item())
                    except Exception:
                        score = -float("inf")
                    
                    # 只有分数有效的路径才保留
                    if score > -1e8: 
                        new_beam.append(PronCandidate(
                            phones=new_phones,
                            pron_choice_idxs=cand.pron_choice_idxs + [p_idx],
                            score=score
                        ))
            
            if not new_beam: return None # Dead end
            new_beam.sort(key=lambda x: x.score, reverse=True)
            beam = new_beam[:beam_width] # 使用传入的 beam_width
            
        return beam[0] if beam else None

    def _phones_to_word_segments_robust(self, token_segs, words, prons):
        word_segs = []
        wi = 0 
        for word, pron in zip(words, prons):
            n_phones = len(pron)
            if n_phones == 0: continue
            if wi + n_phones > len(token_segs): break
            start_frame = token_segs[wi].start_frame
            end_frame = token_segs[wi + n_phones - 1].end_frame
            word_segs.append(Segment(word, start_frame, end_frame))
            wi += n_phones
        return word_segs

    def _merge_words_into_chunks(self, words: List[WordSeg]) -> List[InternalChunk]:
        if not words: return []
        chunks = []
        cur_words = [words[0].word]
        cur_start = words[0].start
        cur_end = words[0].end
        
        for w in words[1:]:
            gap = w.start - cur_end
            proposed_dur = w.end - cur_start
            if gap <= self.max_gap_s and proposed_dur <= self.max_chunk_s:
                cur_end = w.end
                cur_words.append(w.word)
            else:
                if (cur_end - cur_start) >= self.min_chunk_s and len(cur_words) >= self.min_words:
                    chunks.append(InternalChunk(cur_start, cur_end, cur_words))
                cur_start = w.start
                cur_end = w.end
                cur_words = [w.word]
        
        if len(cur_words) > 0:
             chunks.append(InternalChunk(cur_start, cur_end, cur_words))
        return chunks

    def _pad_chunks(self, chunks, words, audio_dur):
        if not chunks: return []
        out = []
        for c in chunks:
            new_start = max(0.0, c.start - self.pad_s)
            new_end = min(audio_dur, c.end + self.pad_s)
            out.append(InternalChunk(new_start, new_end, c.words))
        return out

# ==========================================
#  Static Functions
# ==========================================

def build_trellis(log_probs: torch.Tensor, targets: List[int], blank_id: int) -> torch.Tensor:
    T, V = log_probs.shape
    N = len(targets)
    device = log_probs.device
    neg_inf = -1e9 

    targets_t = torch.tensor(targets, device=device, dtype=torch.long)
    trellis = torch.full((T + 1, N + 1), neg_inf, device=device, dtype=log_probs.dtype)
    trellis[0, 0] = 0.0
    trellis[1:, 0] = torch.cumsum(log_probs[:, blank_id], dim=0)

    for t in range(1, T + 1):
        lp_t = log_probs[t - 1]
        blank = lp_t[blank_id]
        emit_scores = lp_t[targets_t]
        
        stay = trellis[t - 1, 1:] + blank
        emit = trellis[t - 1, :-1] + emit_scores
        trellis[t, 1:] = torch.maximum(stay, emit)

    return trellis

def backtrace(trellis: torch.Tensor, log_probs: torch.Tensor, targets: List[int], blank_id: int) -> List[Point]:
    _T = trellis.size(0) - 1
    N = trellis.size(1) - 1
    j = N
    t = _T 
    path = []

    while t > 0 and j > 0:
        lp_t = log_probs[t - 1]
        score_current = trellis[t, j]
        score_stay = trellis[t - 1, j] + lp_t[blank_id]
        score_emit = trellis[t - 1, j - 1] + lp_t[targets[j - 1]]
        
        if abs(score_current - score_emit) < 1e-4:
            path.append(Point(token_index=j - 1, time_index=t - 1))
            j -= 1
            t -= 1
        elif abs(score_current - score_stay) < 1e-4:
            t -= 1
        else:
            t -= 1
            
    path.reverse()
    return path

def points_to_segments(points: List[Point], labels: List[str]) -> List[Segment]:
    if not points: return []
    segs = []
    for i, p in enumerate(points):
        start = p.time_index
        end = points[i + 1].time_index if i + 1 < len(points) else start + 1
        segs.append(Segment(labels[p.token_index], start, end))
    return segs