import torch
import json
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass
from transformers import Wav2Vec2ForCTC, AutoProcessor
import os
# 引入我们在 io.py 定义的数据结构
from .io import AudioChunk

# ==========================================
#  Helper Data Structures (内部使用)
# ==========================================

@dataclass
class Point:
    token_index: int  # index in target token sequence
    time_index: int   # frame index

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
    """中间状态的 Chunk，只存时间信息，还没切 Tensor"""
    start: float
    end: float
    words: List[str]

# ==========================================
#  CTCChunker Class (核心逻辑)
# ==========================================

class CTCChunker:
    def __init__(self, config: dict):
        self.config = config or {}
        # 统一从配置读取设备
        self.device = torch.device(self.config.get("device", "cpu"))
        
        # 资源占位符
        self.model = None
        self.processor = None
        self.lexicon = {}
        self.phone_to_id = {}
        self.blank_id = 0
        
        # 加载超参
        self.beam_size = self.config.get("beam_size", 10)
        self.min_chunk_s = self.config.get("min_chunk_s", 1.0)
        self.max_chunk_s = self.config.get("max_chunk_s", 12.0)
        self.max_gap_s = self.config.get("max_gap_s", 0.35)
        self.min_words = self.config.get("min_words", 2)
        self.pad_s = self.config.get("pad_s", 0.15)
        
        # 启动加载
        self._load_resources()


    def _load_resources(self):
        """加载词典和模型配置（通用化重构版）"""
        # 1. 加载 Lexicon
        lex_path = self.config.get("lexicon_path")
        if lex_path:
            self.lexicon = self._read_lexicon(Path(lex_path))
        
        # 2. 加载 Phones JSON
        phone_path = self.config.get("phone_json_path")
        if phone_path:
            with open(phone_path, "r", encoding="utf-8") as f:
                self.phone_to_id = json.load(f)
        
        # 3. 加载模型 (Global Loading Logic)
        model_path = self.config.get("chunk_model_path")
        if not model_path: return

        print(f"[CTCChunker] Requesting model: {model_path}")

        # --- 配置加载参数 ---
        load_kwargs = {}
        
        # 策略 A: 官方特供版 (包含子目录)
        # 只有我们的官方库需要这个 subfolder 参数，其他通用模型不需要
        if "USTCPhonetics/FlexAligner" in model_path:
            print("[CTCChunker] Mode: Official Repo (with subfolder)")
            load_kwargs["subfolder"] = "hf_phs"
            
        # 策略 B: 用户指定的本地路径
        elif os.path.isdir(model_path):
            print("[CTCChunker] Mode: Local Override")
            # 本地路径不需要额外参数，transformers 会自动识别
            
        # 策略 C: 通用 Hugging Face 模型 (如 English 模型)
        # e.g., "facebook/wav2vec2-base-960h"
        else:
            print("[CTCChunker] Mode: Generic HF Hub (Cache -> Download)")
            # 不需要 subfolder，直接加载

        # --- 统一执行加载 ---
        # transformers 的 from_pretrained 默认逻辑就是：
        # 1. 检查 model_path 是否为本地文件夹 -> 是则加载
        # 2. 检查 ~/.cache/huggingface 下是否有缓存 -> 是则加载
        # 3. 联网下载 -> 下载并缓存
        try:
            from transformers import AutoProcessor, Wav2Vec2ForCTC
            
            self.processor = AutoProcessor.from_pretrained(model_path, **load_kwargs)
            self.model = Wav2Vec2ForCTC.from_pretrained(model_path, **load_kwargs).to(self.device)
            self.model.eval()
            print(f"[CTCChunker] Successfully loaded model from {model_path}")
            
        except Exception as e:
            raise RuntimeError(
                f"Model load failed.\n"
                f"Path: {model_path}\n"
                f"Args: {load_kwargs}\n"
                f"Error: {e}\n"
                "Check network connection or model name."
            )

        # 4. 确定 blank_id (保持不变)
        blank_token = self.config.get("blank_token", "<pad>")
        if blank_token in self.phone_to_id:
            self.blank_id = self.phone_to_id[blank_token]
        elif hasattr(self.processor, "tokenizer") and self.processor.tokenizer.pad_token_id is not None:
            self.blank_id = self.processor.tokenizer.pad_token_id
        else:
            self.blank_id = 0
    @torch.inference_mode()
    def find_chunks(self, audio_tensor: torch.Tensor, text_list: List[str]) -> List[AudioChunk]:
        """
        [主入口] 执行 Stage 1 完整流程：
        Audio -> LogProbs -> BeamSearch -> Viterbi -> WordSegs -> Merge -> AudioChunks
        """
        if self.model is None:
            raise RuntimeError("CTC 模型未加载，请检查 config 中的 chunk_model_path")

        # 1. 计算声学概率 (Forward Pass)
        # audio_tensor: (Time,) -> (1, Time) -> Model
        input_values = self.processor(
            audio_tensor.numpy(), 
            sampling_rate=16000, 
            return_tensors="pt"
        ).input_values.to(self.device)
        
        logits = self.model(input_values).logits  # (1, T, V)
        log_probs = torch.log_softmax(logits, dim=-1).squeeze(0) # (T, V)
        
        # 计算每帧秒数 (SPF)
        # num_frames = log_probs.size(0)
        # audio_dur = audio_tensor.size(0) / 16000
        spf = (audio_tensor.size(0) / 16000) / log_probs.size(0)

        # 2. 发音变体 Beam Search
        prons_per_word = self._words_to_pronunciations(text_list)
        best_candidate = self._beam_search(log_probs, text_list, prons_per_word)

        # 3. Viterbi 强制对齐 (得到 token 级别的对齐点)
        target_ids = [self.phone_to_id[p] for p in best_candidate.phones]
        trellis = build_trellis(log_probs, target_ids, self.blank_id)
        points = backtrace(trellis, log_probs, target_ids, self.blank_id)
        
        # 4. 转换回 Word Segments (物理时间戳)
        # 先转为 token 级别的 segment
        token_segs = points_to_segments(points, best_candidate.phones)
        
        # 再组合成 Word
        word_segs = self._phones_to_word_segments_robust(
            token_segs, text_list, 
            [prons_per_word[i][idx] for i, idx in enumerate(best_candidate.pron_choice_idxs)]
        )

        # 5. 智能合并成 Chunk (宏观切分)
        # 将 Segment 转换为带真实时间的 WordSeg 对象
        word_objects = [
            WordSeg(s.start_frame * spf, (s.end_frame - s.start_frame) * spf, s.label)
            for s in word_segs
        ]
        
        internal_chunks = self._merge_words_into_chunks(word_objects)
        
        # 6. Padding (向两侧静音延展)
        audio_dur_s = audio_tensor.size(0) / 16000
        internal_chunks = self._pad_chunks(internal_chunks, word_objects, audio_dur_s)

        # 7. 生成最终 AudioChunk 对象 (In-Memory Slicing!)
        final_chunks = []
        sr = 16000
        for i, c in enumerate(internal_chunks):
            # 计算 sample 索引
            s_samp = int(c.start * sr)
            e_samp = int(c.end * sr)
            
            # 边界保护
            s_samp = max(0, s_samp)
            e_samp = min(audio_tensor.size(0), e_samp)
            
            if e_samp <= s_samp:
                continue

            # 切片！(Copy to avoid memory leaks if large tensor is kept)
            chunk_tensor = audio_tensor[s_samp:e_samp].clone()
            
            chunk_obj = AudioChunk(
                tensor=chunk_tensor,
                start_time=c.start,
                end_time=c.end,
                text=" ".join(c.words),
                chunk_id=f"chunk_{i:03d}"
            )
            final_chunks.append(chunk_obj)

        return final_chunks

    # --- 内部核心算法 ---

    def _read_lexicon(self, path: Path):
        lexicon = {}
        if not path.exists():
            return lexicon
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    word = parts[0]
                    # 关键：直接存 List[str]，不要套娃
                    phones = parts[1:] 
                    lexicon[word] = phones
        return lexicon


    def _words_to_pronunciations(self, words: List[str]) -> List[List[List[str]]]:
        out = []
        print(f"\n[DEBUG] --- Entering _words_to_pronunciations (Total words: {len(words)}) ---")
        
        for i, w in enumerate(words):
            w_norm = w.strip().lower()
            if not w_norm:
                continue
                
            if w_norm not in self.lexicon:
                raise ValueError(f"[CTCChunker] OOV Word: {w_norm}")
            
            # 1. 原始查表结果
            flat_phones = self.lexicon[w_norm]
            
            # 2. 构造当前的词候选 (包裹一层)
            word_candidates = [flat_phones]
            
            # 🔴 关键调试打印：只打印前 3 个词，防止日志爆炸
            if i < 3:
                print(f"[DEBUG] Word[{i}]: '{w_norm}'")
                print(f"        Lexicon says: {flat_phones} (Type: {type(flat_phones)})")
                print(f"        Wrapped into: {word_candidates} (Depth: 2)")
            
            out.append(word_candidates)
        
        # 3. 最终整体结构验证
        if out:
            print(f"[DEBUG] Final structure sample (out[0]): {out[0]}")
            # 物理验证：如果是正确的，out[0][0] 应该是一个 list (音素列表)，而不是 string
            if len(out[0]) > 0:
                print(f"        Verification: out[0][0][0] is '{out[0][0][0]}' (Should be first phone char)")
        
        print(f"[DEBUG] --- End of _words_to_pronunciations ---\n")
        return out
            


    def _beam_search(self, log_probs, words, prons_per_word) -> PronCandidate:
        """简单的 Beam Search，寻找最佳发音组合"""
        # 1. 确保初始化时 phones 是一个纯净的空列表
        beam = [PronCandidate(phones=[], pron_choice_idxs=[], score=0.0)]
        
        for i, word in enumerate(words):
            new_beam = []
            variants = prons_per_word[i] # 结构: [['t', 'a', ...]]
            
            for cand in beam:
                for p_idx, pron in enumerate(variants):
                    # --- 维度防御检查 ---
                    # 确保 pron 是 ['t', 'a'] 而不是 [['t', 'a']]
                    if len(pron) > 0 and isinstance(pron[0], list):
                        print(f"[DEBUG] Dimension Error detected at word '{word}', flattening...")
                        pron = pron[0] 

                    # 2. 拼接音素序列
                    new_phones = cand.phones + pron
                    
                    # --- 再次防御：确保 new_phones 里的每一个元素都是字符串 ---
                    try:
                        # 核心查表
                        new_ids = [self.phone_to_id[p] for p in new_phones if isinstance(p, str)]
                    except KeyError as e:
                        # 某些音素不在词表中，打印出来看看是哪个
                        # print(f"[CTCChunker] Missing Phone in Vocabulary: {e}")
                        continue
                    except TypeError as e:
                        # 如果走到这里，说明 new_phones 里混进了 list
                        # 我们打印出 new_phones 的前几个元素来抓现行
                        print(f"[CRITICAL] new_phones sample: {new_phones[:3]}")
                        raise e
                    
                    # 3. 计算 Viterbi 得分 (build_trellis)
                    # 这里的 score 计算逻辑保持你的不变
                    try:
                        trellis = build_trellis(log_probs, new_ids, self.blank_id)
                        # 注意：trellis 维度通常是 (T, S+1)
                        score = float(torch.max(trellis[:, len(new_ids)]).item())
                    except Exception:
                        score = -float("inf")
                    
                    new_beam.append(PronCandidate(
                        phones=new_phones,
                        pron_choice_idxs=cand.pron_choice_idxs + [p_idx],
                        score=score
                    ))
            
            # 4. 剪枝
            new_beam.sort(key=lambda x: x.score, reverse=True)
            beam = new_beam[:self.beam_size]
            
        return beam[0] if beam else None

    def _phones_to_word_segments_robust(self, token_segs, words, prons):
        """根据音素长度反推单词边界"""
        word_segs = []
        wi = 0 # token index
        
        for word, pron in zip(words, prons):
            n_phones = len(pron)
            if n_phones == 0:
                continue
            
            if wi + n_phones > len(token_segs):
                break
                
            start_frame = token_segs[wi].start_frame
            end_frame = token_segs[wi + n_phones - 1].end_frame
            
            word_segs.append(Segment(word, start_frame, end_frame))
            wi += n_phones
            
        return word_segs

    def _merge_words_into_chunks(self, words: List[WordSeg]) -> List[InternalChunk]:
        """[核心] 你的孤岛切分逻辑"""
        if not words:
            return []
        
        chunks = []
        cur_words = [words[0].word]
        cur_start = words[0].start
        cur_end = words[0].end
        
        for w in words[1:]:
            gap = w.start - cur_end
            proposed_dur = w.end - cur_start
            
            # 判断是否断开
            if gap <= self.max_gap_s and proposed_dur <= self.max_chunk_s:
                # 连起来
                cur_end = w.end
                cur_words.append(w.word)
            else:
                # 断开，保存前一个
                if (cur_end - cur_start) >= self.min_chunk_s and len(cur_words) >= self.min_words:
                    chunks.append(InternalChunk(cur_start, cur_end, cur_words))
                
                # 开启新 Chunk
                cur_start = w.start
                cur_end = w.end
                cur_words = [w.word]
                
        # 扫尾
        if (cur_end - cur_start) >= self.min_chunk_s and len(cur_words) >= self.min_words:
            chunks.append(InternalChunk(cur_start, cur_end, cur_words))
            
        return chunks

    def _pad_chunks(self, chunks, words, audio_dur):
        """向两侧安全地填充静音"""
        if not chunks: 
            return []
        out = []
        
        # 为了快速查找，建立 word map
        # 这里简化处理：假设 chunks 和 words 都是有序的
        # 实际逻辑：在 words 列表中找到 chunk 的第一个词和最后一个词的前后邻居
        
        # 简单版：只做基础 padding，不处理复杂的 word 邻居避让（为了代码清晰）
        # 如果需要原来的严格逻辑，可以把 chunks2.py 里的 pad_chunks_without_cutting_words 完整搬过来
        # 这里演示基础逻辑：
        for c in chunks:
            new_start = max(0.0, c.start - self.pad_s)
            new_end = min(audio_dur, c.end + self.pad_s)
            out.append(InternalChunk(new_start, new_end, c.words))
            
        return out


# ==========================================
#  Static Pure Functions (数学运算部分)
# ==========================================

def build_trellis(log_probs: torch.Tensor, targets: List[int], blank_id: int) -> torch.Tensor:
    """Viterbi Trellis 构建 (Vectorized)"""
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
        blank = lp_t[blank_id]
        emit_scores = lp_t[targets_t]

        stay = trellis[t - 1, 1:] + blank
        emit = trellis[t - 1, :-1] + emit_scores

        trellis[t, 1:] = torch.maximum(stay, emit)

    return trellis

def backtrace(trellis: torch.Tensor, log_probs: torch.Tensor, targets: List[int], blank_id: int) -> List[Point]:
    """回溯寻找最佳路径点"""
    _T = trellis.size(0) - 1
    N = trellis.size(1) - 1
    j = N
    t = int(torch.argmax(trellis[:, j]).item())
    path = []

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
    return path

def points_to_segments(points: List[Point], labels: List[str]) -> List[Segment]:
    """将点转换为区间"""
    if not points: 
        return []
    segs = []
    for i, p in enumerate(points):
        start = p.time_index
        end = points[i + 1].time_index if i + 1 < len(points) else (p.time_index + 1)
        segs.append(Segment(labels[p.token_index], start, end))
    return segs