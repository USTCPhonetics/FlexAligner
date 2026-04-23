# import torch
# import os
# import json
# import soundfile as sf
# from pathlib import Path
# from typing import List, Optional, Dict
# from dataclasses import dataclass, asdict
# from transformers import Wav2Vec2ForCTC, AutoProcessor

# # 引入 io.py 数据结构
# try:
#     from .io import AudioChunk
# except ImportError:
#     @dataclass
#     class AudioChunk:
#         tensor: torch.Tensor
#         start_time: float
#         end_time: float
#         text: str
#         chunk_id: str

# # ==========================================
# #  Helper Data Structures
# # ==========================================
# @dataclass
# class Point:
#     token_index: int 
#     time_index: int 

# @dataclass
# class Segment:
#     label: str
#     start_frame: int
#     end_frame: int
#     @property
#     def duration_frames(self):
#         return self.end_frame - self.start_frame

# @dataclass
# class PronCandidate:
#     phones: List[str]
#     pron_choice_idxs: List[int]
#     score: float

# @dataclass
# class WordSeg:
#     start: float
#     dur: float
#     word: str
#     @property
#     def end(self) -> float:
#         return self.start + self.dur

# @dataclass
# class InternalChunk:
#     start: float
#     end: float
#     words: List[str]

# # ==========================================
# #  CTCChunker Class (Strictly Equivalent)
# # ==========================================

# class CTCChunker:
#     def __init__(self, config: dict):
#         self.config = config or {}
#         self.device = torch.device(self.config.get("device", "cpu"))
#         self.lang = self.config.get("lang", "zh")
        
#         # [Verbose]
#         self.verbose = self.config.get("verbose", False)
#         self.chunks_out_dir = self.config.get("chunks_out_dir", None)
        
#         if self.verbose:
#             self._log_header("CTCChunker Initializing")
#             print(f"  - Device: {self.device}")
#             print(f"  - Lang: {self.lang}")
#             print(f"  - Intermediate Output: {self.chunks_out_dir if self.chunks_out_dir else 'Disabled'}")

#         # Resources
#         self.model = None
#         self.processor = None
#         self.lexicon = {}
#         self.phone_to_id = {} 
#         self.blank_id = 0
        
#         # Hyperparameters (Stage 1)
#         self.beam_size = self.config.get("beam_size", 10)
#         self.min_chunk_s = self.config.get("min_chunk_s", 1.0)
#         self.max_chunk_s = self.config.get("max_chunk_s", 12.0)
#         self.max_gap_s = self.config.get("max_gap_s", 0.35)
#         self.min_words = self.config.get("min_words", 2)
#         self.pad_s = self.config.get("pad_s", 0.15)
#         self.blank_token = self.config.get("blank_token", "<pad>")
#         # print("chunker config:")
#         # print(self.config)
#         # input("check:")
#         # [Physics Fix] 不再使用固定 hop，而在运行时计算
#         # self.config_hop = 0.02 

#         self._load_resources()

#     def _log(self, msg: str):
#         if self.verbose: print(f"[CTCChunker] {msg}")

#     def _log_header(self, title: str):
#         if self.verbose: print(f"\n=== {title} ===")

#     # def _load_resources(self):
#     #     lex_path = self.config.get("lexicon_path")
#     #     self.lexicon = self._read_lexicon(lex_path)
#     #     print(f"[info] loaded lexicon with {len(self.lexicon)} entries from {lex_path}")
#     #     # input("Check lexicon load, press Enter to continue...")  # Debug pause to inspect lexicon load
#     #     if self.verbose: print(f"  - Lexicon loaded: {len(self.lexicon)} words")
        
#     #     model_path = self.config.get("chunk_model_path")
#     #     if not model_path: return

#     #     self._log(f"Requesting model: {model_path}")
#     #     load_kwargs = {}
#     #     print(f"[info] loading model from {model_path}")
#     #     if not os.path.isdir(model_path):
#     #         load_kwargs["subfolder"] = f"{self.lang}/chunker"
#     #         self._log(f"Mode: Cloud Repo ({load_kwargs['subfolder']})")
#     #     else:
#     #         self._log("Mode: Local Override")

#     #     try:
#     #         self.processor = AutoProcessor.from_pretrained(model_path, **load_kwargs)
#     #         self.model = Wav2Vec2ForCTC.from_pretrained(model_path, **load_kwargs).to(self.device)
#     #         self.model.eval()
#     #         self._log("Successfully loaded model.")
#     #     except Exception as e:
#     #         self._log(f"Routing failed ({e}), falling back to root...")
#     #         try:
#     #             self.processor = AutoProcessor.from_pretrained(model_path)
#     #             self.model = Wav2Vec2ForCTC.from_pretrained(model_path).to(self.device)
#     #         except Exception as final_e:
#     #             raise RuntimeError(f"Model load failed: {final_e}")

#     #     if hasattr(self.processor, "tokenizer"):
#     #         self.phone_to_id = self.processor.tokenizer.get_vocab()
#     #         # Resolve blank token
#     #         if self.blank_token in self.phone_to_id:
#     #             self.blank_id = self.phone_to_id[self.blank_token]
#     #         elif self.processor.tokenizer.pad_token_id is not None:
#     #             self.blank_id = self.processor.tokenizer.pad_token_id
#     #         else:
#     #             self.blank_id = 0
            
#     #         # English Hotfix
#     #         if self.lang == "en":
#     #             self.garbage_tokens = {"O", "[UNK]", "<unk>"}
#     #             self.fuzzy_map = {}
#     #             for token, pid in self.phone_to_id.items():
#     #                 self.fuzzy_map[token] = pid
#     #                 pure_token = ''.join(filter(str.isalpha, token))
#     #                 if pure_token and pure_token not in self.fuzzy_map:
#     #                     self.fuzzy_map[pure_token] = pid
#     #             self._log(f"🇬🇧 English Hotfix Applied: Fuzzy map built with {len(self.fuzzy_map)} entries.")
#     #     else:
#     #         self.blank_id = 0
#     def _load_resources(self):
#         # =====================================================================
#         # 1. 加载本地基础词典
#         # =====================================================================
#         lex_path = self.config.get("lexicon_path")
#         self.lexicon = self._read_lexicon(lex_path) if lex_path else {}
#         # print(f"[info] loaded local lexicon with {len(self.lexicon)} entries from {lex_path}")
        
#         # =====================================================================
#         # 🛡️ 2. 核心增量：吸收 G2P 巨型字典，一次性消除 OOV
#         # =====================================================================
#         if self.lang == "en":
#             try:
#                 from g2p_en.g2p import G2p
#                 g2p_inst = G2p()
#                 cmu_dict = g2p_inst.cmu  # 提取内置的 CMU 字典 { 'hello': [['HH', 'AH0', 'L', 'OW1']], ... }
                
#                 added_words = 0
#                 for w, prons in cmu_dict.items():
#                     w_upper = w.upper()
#                     # 坚守原则：本地词典优先级最高，不覆盖已有定义
#                     if w_upper not in self.lexicon and w not in self.lexicon:
#                         # 默认取第一种发音变体，并剔除附带的潜在标点
#                         clean_pron = [p for p in prons[0] if p.isalnum()]
#                         self.lexicon[w_upper] = clean_pron
#                         added_words += 1
                        
#                 # print(f"[info] 🚀 G2P augmentation complete: added {added_words} OOV words into RAM.")
#             except Exception as e:
#                 print(f"[warning] G2P dictionary augmentation failed: {e}")

#         if self.verbose: print(f"  - Total Lexicon size: {len(self.lexicon)} words")

#         # =====================================================================
#         # 3. 加载声学模型与分词器
#         # =====================================================================
#         model_path = self.config.get("chunk_model_path")
#         if not model_path: return

#         self._log(f"Requesting model: {model_path}")
#         load_kwargs = {}
#         # print(f"[info] loading model from {model_path}")
#         if not os.path.isdir(model_path):
#             load_kwargs["subfolder"] = f"{self.lang}/chunker"
#             self._log(f"Mode: Cloud Repo ({load_kwargs['subfolder']})")
#         else:
#             self._log("Mode: Local Override")

#         try:
#             self.processor = AutoProcessor.from_pretrained(model_path, **load_kwargs)
#             self.model = Wav2Vec2ForCTC.from_pretrained(model_path, **load_kwargs).to(self.device)
#             self.model.eval()
#             self._log("Successfully loaded model.")
#         except Exception as e:
#             self._log(f"Routing failed ({e}), falling back to root...")
#             try:
#                 self.processor = AutoProcessor.from_pretrained(model_path)
#                 self.model = Wav2Vec2ForCTC.from_pretrained(model_path).to(self.device)
#             except Exception as final_e:
#                 raise RuntimeError(f"Model load failed: {final_e}")

#         # =====================================================================
#         # 4. 音素映射与降维池化 (Fuzzy Map)
#         # =====================================================================
#         if hasattr(self.processor, "tokenizer"):
#             self.phone_to_id = self.processor.tokenizer.get_vocab()
            
#             # Resolve blank token
#             if self.blank_token in self.phone_to_id:
#                 self.blank_id = self.phone_to_id[self.blank_token]
#             elif self.processor.tokenizer.pad_token_id is not None:
#                 self.blank_id = self.processor.tokenizer.pad_token_id
#             else:
#                 self.blank_id = 0
            
#             # English Hotfix 终极加固
#             if self.lang == "en":
#                 self.garbage_tokens = {"O", "[UNK]", "<unk>"}
#                 self.fuzzy_map = {}
                
#                 for token, pid in self.phone_to_id.items():
#                     self.fuzzy_map[token] = pid
#                     self.fuzzy_map[token.upper()] = pid
#                     self.fuzzy_map[token.lower()] = pid
                    
#                     # 降维池化：将带有数字的补充音素映射回基础 ID
#                     # 例如 'OW2' 会在这里找到对应 'OW' 的 ID
#                     pure_token = ''.join(filter(str.isalpha, token))
#                     if pure_token:
#                         if pure_token.upper() not in self.fuzzy_map:
#                             self.fuzzy_map[pure_token.upper()] = pid
#                         if pure_token.lower() not in self.fuzzy_map:
#                             self.fuzzy_map[pure_token.lower()] = pid
                            
#                 self._log(f"🇬🇧 English Hotfix Applied: Omni-Fuzzy map built with {len(self.fuzzy_map)} entries.")
#         else:
#             self.blank_id = 0

#     @torch.inference_mode()
#     def find_chunks(self, audio_tensor: torch.Tensor, text_list: List[str], file_id: str = "unknown") -> List[AudioChunk]:
#         # print(f"find_chunks called with file_id: {file_id}, text_list: {text_list}")
#         # input("Check find_chunks inputs, press Enter to continue...")  # Debug pause to inspect inputs
#         if self.model is None: raise RuntimeError("CTC 模型未加载")

#         if self.verbose:
#             self._log_header(f"Processing: {file_id}")

#         # 1. Forward Pass
#         # 注意: chunks2.py 默认 target_sr = 16000
#         input_values = self.processor(
#             audio_tensor.numpy(), 
#             sampling_rate=16000, 
#             return_tensors="pt"
#         ).input_values.to(self.device)
        
#         logits = self.model(input_values).logits 
#         log_probs = torch.log_softmax(logits, dim=-1).squeeze(0)
        
#         # [Fix 1: Dynamic SPF Calculation]
#         # 完全复刻 chunks2.py: seconds_per_frame = float(audio.numel() / sr) / float(log_probs.size(0))
#         spf = float(audio_tensor.numel() / 16000) / float(log_probs.size(0))
#         # spf = 0.02
#         if self.verbose:
#             print(f"  - Audio Duration: {audio_tensor.size(0)/16000:.3f}s")
#             print(f"  - Logits Shape: {log_probs.shape}")
#             print(f"  - Calculated SPF: {spf:.6f} (was fixed 0.02)")

#         # 2. Beam Search
#         # print(f"Converting words to pronunciations for text: {text_list}")
#         prons_per_word = self._words_to_pronunciations(text_list)
#         # print(f"Pronunciations per word: {prons_per_word}")
#         best_candidate = None
#         for attempt_beam in [self.beam_size, 50, 200, 1000]:
#             best_candidate = self._beam_search(log_probs, text_list, prons_per_word, beam_width=attempt_beam)
#             if best_candidate:
#                 if attempt_beam > self.beam_size: self._log(f"💡 Expanded beam to {attempt_beam}.")
#                 break
        
#         if not best_candidate:
#             self._log("❌ Beam search failed. Fallback to full audio.")
#             return [AudioChunk(
#                 tensor=audio_tensor, start_time=0.0, end_time=audio_tensor.size(0)/16000.0,
#                 text=" ".join(text_list), chunk_id=f"{file_id}.chunk_fallback"
#             )]

#         # 3. Viterbi Alignment
#         target_ids = [self.phone_to_id[p] for p in best_candidate.phones if p in self.phone_to_id]
#         if not target_ids: return []

#         trellis = build_trellis(log_probs, target_ids, self.blank_id)
#         points = backtrace(trellis, log_probs, target_ids, self.blank_id)
        
#         # 4. Convert to Segments
#         filtered_phones = [p for p in best_candidate.phones if p in self.phone_to_id]
#         token_segs = points_to_segments(points, filtered_phones)
        
#         clean_prons = []
#         for i, idx in enumerate(best_candidate.pron_choice_idxs):
#             raw_pron = prons_per_word[i][idx]
#             if len(raw_pron) > 0 and isinstance(raw_pron[0], list): raw_pron = raw_pron[0]
#             filtered_pron = [p for p in raw_pron if p in self.phone_to_id]
#             clean_prons.append(filtered_pron)

#         word_segs = self._phones_to_word_segments_robust(token_segs, text_list, clean_prons)

#         # 5. Logical Chunking (using dynamic SPF)
#         word_objects = [
#             WordSeg(s.start_frame * spf, (s.end_frame - s.start_frame) * spf, s.label)
#             for s in word_segs
#         ]
        
#         internal_chunks = self._merge_words_into_chunks(word_objects)
#         audio_dur_s = float(audio_tensor.size(0)) / 16000
#         internal_chunks = self._pad_chunks(internal_chunks, word_objects, audio_dur_s)

#         # 6. Physical Extraction & Object Creation
#         # final_chunks = []
#         # sr = 16000
#         # for i, c in enumerate(internal_chunks):
#         #     # chunks2.py: s0 = int(round(c.start * sr))
#         #     s_samp = int(round(c.start * sr))
#         #     e_samp = int(round(c.end * sr))
#         #     s_samp = max(0, min(s_samp, audio_tensor.size(0)))
#         #     e_samp = max(0, min(e_samp, audio_tensor.size(0)))
            
#         #     if e_samp <= s_samp: continue

#         #     chunk_id_str = f"{file_id}.chunk{i+1:03d}"
            
#         #     chunk_obj = AudioChunk(
#         #         tensor=audio_tensor[s_samp:e_samp].clone(),
#         #         start_time=c.start,
#         #         end_time=c.end,
#         #         text=" ".join(c.words),
#         #         chunk_id=chunk_id_str
#         #     )
#         #     final_chunks.append(chunk_obj)

#         # self._log(f"Found {len(final_chunks)} chunks.")

#         # if self.chunks_out_dir:
#         #     self._save_intermediate_results(final_chunks, file_id)

#         # 
        
#         # ==========================================================
#         # 6. Physical Extraction & Object Creation (全局单调对齐装甲)
#         # ==========================================================
#         final_chunks = []
#         sr = 16000
        
#         # 🛡️ 预计算阶段：扫描全局序列，为每个 Chunk 锁定在纯净 text_list 中的绝对起始坐标
#         chunk_starts = [0] * len(internal_chunks)
#         orig_cursor = 0
        
#         for i, c in enumerate(internal_chunks):
#             # 异常防御：如果当前 chunk 被 CTC 切空了，起点直接继承当前游标
#             if not c.words:
#                 chunk_starts[i] = orig_cursor
#                 continue
                
#             # 1. 抓取当前 Chunk 的第一个词作为“寻星镜”锚点 (去数字/小写)
#             anchor_ctc = ''.join(filter(str.isalpha, c.words[0].lower()))
            
#             # 2. 游标在原始序列中推进，直到命中锚点
#             while orig_cursor < len(text_list):
#                 anchor_orig = ''.join(filter(str.isalpha, text_list[orig_cursor].lower()))
#                 if anchor_orig == anchor_ctc:
#                     break
#                 orig_cursor += 1
                
#             chunk_starts[i] = min(orig_cursor, len(text_list))
            
#             # 3. 匹配完起点后，用内部循环把当前 Chunk 剩下的词全“消化”掉
#             # 保证下一次外层循环寻星时，游标已经跨过了当前 Chunk 的领空
#             for cw in c.words[1:]:
#                 cw_clean = ''.join(filter(str.isalpha, cw.lower()))
#                 while orig_cursor < len(text_list) - 1:
#                     orig_cursor += 1
#                     tw_clean = ''.join(filter(str.isalpha, text_list[orig_cursor].lower()))
#                     if tw_clean == cw_clean:
#                         break
                        
#             # 消化完当前 chunk 的最后一个词后，游标必须往前再走一格，准备迎接下个 chunk
#             orig_cursor += 1 

#         # 🛡️ 压入终点坐标，确保最后一个 Chunk 像黑洞一样吸收所有剩余残骸
#         chunk_starts.append(len(text_list))
        
#         # ==========================================================
#         # 🗡️ 切割阶段：根据绝对坐标无损切割
#         # ==========================================================
#         for i, c in enumerate(internal_chunks):
#             s_samp = int(round(c.start * sr))
#             e_samp = int(round(c.end * sr))
#             s_samp = max(0, min(s_samp, audio_tensor.size(0)))
#             e_samp = max(0, min(e_samp, audio_tensor.size(0)))
            
#             if e_samp <= s_samp: continue

#             chunk_id_str = f"{file_id}.chunk{i+1:03d}"
            
#             # 🎯 物理级切割：左闭右开，严丝合缝
#             # 哪怕 chunk_starts[i] 和 [i+1] 之间隔了 10 个被 CTC 丢弃的 OOV
#             # 这行切片也能把它们完美打包进当前 Chunk 的尾部！
#             real_words = text_list[chunk_starts[i] : chunk_starts[i+1]]
            
#             chunk_obj = AudioChunk(
#                 tensor=audio_tensor[s_samp:e_samp].clone(),
#                 start_time=c.start,
#                 end_time=c.end,
#                 text=" ".join(real_words), 
#                 chunk_id=chunk_id_str
#             )
#             final_chunks.append(chunk_obj)
#             print(f"  - Created chunk: {chunk_obj.chunk_id} ({chunk_obj.start_time:.3f}s - {chunk_obj.end_time:.3f}s, dur={(chunk_obj.end_time-chunk_obj.start_time):.3f}s, text='{chunk_obj.text}')")
#         self._log(f"Found {len(final_chunks)} chunks.")
#         if self.chunks_out_dir:
#             self._save_intermediate_results(final_chunks, file_id)
#         return final_chunks

#     def _save_intermediate_results(self, chunks, file_id, source_audio=None) -> Path:
#         """
#         [Fix 2: Path Logic] 确保文件生成在指定的 chunks_out_dir 内，而不是 parent。
#         """
#         out_root = Path(self.chunks_out_dir)
#         out_root.mkdir(parents=True, exist_ok=True)
        
#         # 修正：直接在 out_root 下生成 jsonl/tsv
#         jsonl_path = out_root / f"{file_id}.chunks.jsonl"
#         tsv_path = out_root / f"{file_id}.chunks.tsv"

#         if self.verbose:
#             print(f"  - Saving artifacts to: {out_root}")

#         with open(jsonl_path, "w", encoding="utf-8") as fj, \
#              open(tsv_path, "w", encoding="utf-8") as ft:
            
#             ft.write("chunk_id\tstart_s\tend_s\tdur_s\twords\n")

#             for c in chunks:
#                 wav_name = f"{c.chunk_id}_{c.start_time:.3f}-{c.end_time:.3f}.wav"
#                 wav_path = out_root / wav_name
                
#                 # sf.write(str(wav_path), c.tensor.squeeze().numpy(), 16000)
#                 sf.write(
#                     str(wav_path),
#                     c.tensor.squeeze().cpu().numpy().astype("float32"),
#                     16000,
#                     subtype="FLOAT"
#                 )
#                 dur = c.end_time - c.start_time
#                 words_list = c.text.split()
                
#                 # 关键：保留 round(x, 3) 以匹配 chunks2.py 的输出格式
#                 obj = {
#                     "chunk_id": c.chunk_id,
#                     "audio": str(wav_path),
#                     "start_s": round(c.start_time, 3),
#                     "end_s": round(c.end_time, 3),
#                     "dur_s": round(dur, 3),
#                     "words": words_list,
#                     "text": c.text
#                 }
#                 print(f"  - Saved chunk: {obj['chunk_id']} ({obj['start_s']:.3f}s - {obj['end_s']:.3f}s, dur={obj['dur_s']:.3f}s, words={len(words_list)})")
#                 print(f"text: '{c.text}'")
#                 # input("Check saved chunk info, press Enter to continue...")  # Debug pause to inspect saved chunk info
#                 fj.write(json.dumps(obj, ensure_ascii=False) + "\n")
#                 # TSV 也是用的 rounded values
#                 ft.write(f"{c.chunk_id}\t{obj['start_s']}\t{obj['end_s']}\t{obj['dur_s']}\t{obj['text']}\n")
#         return jsonl_path
#     # --- 内部方法 (保持不变) ---

#     def _read_lexicon(self, path: Optional[str]):
#         lexicon = {}
#         if path and os.path.exists(path):
#             with open(path, "r", encoding="utf-8") as f:
#                 for line in f:
#                     parts = line.strip().split()
#                     if len(parts) >= 2:
#                         lexicon[parts[0].lower()] = parts[1:]
#         # English patch omitted for brevity (same as before)
#         return lexicon

#     # def _words_to_pronunciations(self, words: List[str]):
#     #     out = []
#     #     for w in words:
#     #         w_norm = w.strip().lower()
#     #         if not w_norm: continue
#     #         if w_norm not in self.lexicon:
                
#     #             if self.verbose: print(f"⚠️  [CTCChunker] OOV Word: '{w_norm}'")
#     #             raise ValueError(f"[CTCChunker] OOV Word: {w_norm}")
#     #         out.append([self.lexicon[w_norm]])
#     #     return out
    
#     # def _words_to_pronunciations(self, words: List[str]):
#     #     # print(f"words to convert: {words}")
#     #     # input("check words:")
#     #     out = []
#     #     for w in words:
#     #         w_clean = w.strip()
#     #         if not w_clean: continue
            
#     #         # 构建多维探测形态
#     #         w_upper = w_clean.upper()
#     #         w_lower = w_clean.lower()
#     #         # 🛡️ [核心降维装甲] 剥离数字 (例如 IY1 -> IY, AE2 -> AE)
#     #         w_no_digit = ''.join([c for c in w_upper if not c.isdigit()])
            
#     #         # 1. 第一级：绝对精确匹配 (如果 Chunker 词表升级了，优先用精确的)
#     #         if w_upper in self.lexicon:
#     #             out.append([self.lexicon[w_upper]])
                
#     #         # 2. 第二级：降维打击 (Chunker 词表只有 IY，没有 IY1，在这里被完美接住)
#     #         elif w_no_digit in self.lexicon:
#     #             # if self.verbose: print(f"  [Downgrade] {w_clean} -> {w_no_digit}")
#     #             out.append([self.lexicon[w_no_digit]])
                
#     #         # 3. 第三级：小写兜底
#     #         elif w_lower in self.lexicon:
#     #             out.append([self.lexicon[w_lower]])
                
#     #         # 4. 第四级：原词兜底 (如 sil)
#     #         elif w_clean in self.lexicon:
#     #             out.append([self.lexicon[w_clean]])
                
#     #         else:
#     #             # 破防了，真正的 OOV
#     #             if self.verbose: print(f"⚠️  [CTCChunker] OOV Word: '{w_clean}' (Tried: {w_no_digit})")
#     #             raise ValueError(f"[CTCChunker] OOV Word: {w_clean}")
#     #     # print(f"Pronunciations: {out}")
#     #     # input("check pronunciations:")
#     #     return out
    
#     # def _words_to_pronunciations(self, words: List[str]):
#     #     out = []
#     #     for w in words:
#     #         w_clean = w.strip()
#     #         if not w_clean: continue
            
#     #         w_upper = w_clean.upper()
#     #         w_no_digit = ''.join([c for c in w_upper if not c.isdigit()])
            
#     #         # ==========================================================
#     #         # 🛡️ 维度 A：它本来就是个纯音素 (如 'sil', 'IY1', 'HH')
#     #         # ==========================================================
#     #         if w_upper in self.lexicon:
#     #             out.append([self.lexicon[w_upper]])
#     #             continue
#     #         elif w_no_digit in self.lexicon:
#     #             out.append([self.lexicon[w_no_digit]])
#     #             continue
#     #         elif w_clean in self.lexicon:
#     #             out.append([self.lexicon[w_clean]])
#     #             continue
                
#     #         # ==========================================================
#     #         # 🗡️ 维度 B：如果走到这里，说明它是单词 (如 'we', 'huh', 'and')
#     #         # 启动内置 G2P 进行物理粉碎
#     #         # ==========================================================
#     #         # 延迟加载，不污染全局环境
#     #         if not hasattr(self, '_internal_g2p'):
#     #             from g2p_en import G2p
#     #             self._internal_g2p = G2p()
                
#     #         phonemes = self._internal_g2p(w_clean) # 例如 "and" -> ['AE1', 'N', 'D']
            
#     #         word_ids = []
#     #         for p in phonemes:
#     #             p_clean = ''.join([c for c in p.upper() if not c.isdigit()]).strip()
#     #             # 过滤掉 G2P 可能产生的标点符号
#     #             if not p_clean or p_clean in ["'", ".", ",", "?", "!"]: 
#     #                 continue
                    
#     #             if p_clean in self.lexicon:
#     #                 word_ids.append(self.lexicon[p_clean])
#     #             else:
#     #                 if self.verbose: print(f"⚠️ [CTCChunker] G2P 解析出的音素不在词表: '{p_clean}'")
            
#     #         if word_ids:
#     #             # 🎯 核心：把单词粉碎后的一组音素 ID，作为一个整体 Append 进去！
#     #             out.append(word_ids)
#     #         else:
#     #             # 破防了，这玩意儿连 G2P 都救不回来
#     #             if self.verbose: print(f"❌ [CTCChunker] 彻底无法解析的 OOV: '{w_clean}'")
#     #             raise ValueError(f"[CTCChunker] OOV Word: {w_clean}")

#     #     return out
#     def _words_to_pronunciations(self, words: List[str]):
#         out = []
#         # print(f"Converting words to pronunciations: {words}")
#         # print(f"lexicon keys sample: {list(self.lexicon.keys())[:30]}")
#         for w in words:
#             w_clean = w.strip()
#             if not w_clean: continue
            
#             w_upper = w_clean.upper()
#             w_lower = w_clean.lower()
#             # 🛡️ 核心降维逻辑：剥离数字 (例如 IY1 -> IY, AH0 -> AH)
#             w_no_digit = ''.join([c for c in w_upper if not c.isdigit()])
            
#             # 1. 第一级：完全精确匹配 (如 'SH', 'sil')
#             if w_upper in self.lexicon:
#                 out.append([self.lexicon[w_upper]])
                
#             # 2. 第二级：降维打击 (完美接住前端传来的 IY1)
#             elif w_no_digit in self.lexicon:
#                 out.append([self.lexicon[w_no_digit]])
                
#             # 3. 兜底匹配 (如原始的小写 sil)
#             elif w_lower in self.lexicon:
#                 out.append([self.lexicon[w_lower]])
                
#             elif w_clean in self.lexicon:
#                 out.append([self.lexicon[w_clean]])
                
#             else:
#                 # 只有真正连降维都不认识的乱码才会走到这里
#                 if self.verbose: 
#                     print(f"⚠️ [CTCChunker] OOV Phoneme: '{w_clean}' (Tried: '{w_no_digit}')")
#                 # raise ValueError(f"[CTCChunker] 无法识别的音素符号: {w_clean}")
#                 print(f"⚠️ [CTCChunker] OOV Phoneme: '{w_clean}' (Tried: '{w_no_digit}') - Skipping this word.")
#                 # 安排一个 sil记号，让它在后续的 beam search 中被当作无声处理掉
#                 out.append([["sil"]])
                
#         return out
#     # def _beam_search(self, log_probs, words, prons_per_word, beam_width=10):
#     #     beam = [PronCandidate(phones=[], pron_choice_idxs=[], score=0.0)]
#     #     for i, word in enumerate(words):
#     #         new_beam = []
#     #         variants = prons_per_word[i]
#     #         for cand in beam:
#     #             for p_idx, pron in enumerate(variants):
#     #                 if len(pron) > 0 and isinstance(pron[0], list): pron = pron[0]
                    
#     #                 # Inter-word token logic (Strictly matching chunks2.py: if phones...)
#     #                 current_phones = list(cand.phones)
#     #                 inter_token = self.config.get("inter_word_token", None)
#     #                 if current_phones and inter_token: 
#     #                     current_phones.append(inter_token)
                    
#     #                 new_phones = current_phones + pron
#     #                 new_ids = []
#     #                 valid_pron = True
#     #                 for p in new_phones:
#     #                     if not isinstance(p, str): continue
#     #                     if p in self.phone_to_id: new_ids.append(self.phone_to_id[p])
#     #                     elif p == "O": continue
#     #                     else: 
#     #                         # Try fuzzy
#     #                         p_pure = ''.join(filter(str.isalpha, p))
#     #                         if p_pure in self.phone_to_id: new_ids.append(self.phone_to_id[p_pure])
#     #                         else: 
#     #                             valid_pron = False; break
                    
#     #                 if not valid_pron: continue
#     #                 try:
#     #                     trellis = build_trellis(log_probs, new_ids, self.blank_id)
#     #                     score = float(torch.max(trellis[-1, -1]).item())
#     #                 except: score = -float("inf")
                    
#     #                 if score > -1e8:
#     #                     new_beam.append(PronCandidate(new_phones, cand.pron_choice_idxs + [p_idx], score))
#     #         if not new_beam: return None
#     #         new_beam.sort(key=lambda x: x.score, reverse=True)
#     #         beam = new_beam[:beam_width]
#     #     return beam[0] if beam else None

#     def _beam_search(self, log_probs, words, prons_per_word, beam_width=10):
#         """
#         // Modified: 彻底移除了导致时间复杂度爆炸的局部 DP 评估 (build_trellis)。
#         // 采用 O(N) 的贪心构建逻辑，毫秒级返回最可能的基础音素序列。
#         // 真正的对齐运算应当延后至整个序列构建完毕后进行单次计算。
#         """
#         current_phones = []
#         pron_choice_idxs = []
        
#         for i, word in enumerate(words):
#             variants = prons_per_word[i]
#             valid_pron_found = False
            
#             # 贪心策略：遍历该词的所有发音变体，采纳第一个合法的发音
#             for p_idx, pron in enumerate(variants):
#                 if len(pron) > 0 and isinstance(pron[0], list): 
#                     pron = pron[0]
                
#                 # 词间符号逻辑 (Inter-word token)
#                 temp_phones = list(current_phones)
#                 inter_token = self.config.get("inter_word_token", None)
#                 if temp_phones and inter_token: 
#                     temp_phones.append(inter_token)
                
#                 temp_phones.extend(pron)
                
#                 # 验证当前发音变体的合法性 (避免 OOV 导致整个序列崩溃)
#                 is_valid = True
#                 for p in pron:
#                     if not isinstance(p, str): continue
#                     if p in self.phone_to_id or p == "O": continue
                    
#                     # 尝试模糊匹配 (Fuzzy Match)
#                     p_pure = ''.join(filter(str.isalpha, p))
#                     if p_pure not in self.phone_to_id:
#                         is_valid = False
#                         break
                
#                 if is_valid:
#                     current_phones = temp_phones
#                     pron_choice_idxs.append(p_idx)
#                     valid_pron_found = True
#                     break # 找到首个合法发音，立刻跳出变体循环
            
#             if not valid_pron_found:
#                 # 如果所有变体都不合法，交由上层异常处理逻辑 (记录 OOV)
#                 return None 
        
#         # 返回伪造了满分 (score=0.0) 的候选对象，满足上层 API 接口需求
#         return PronCandidate(phones=current_phones, pron_choice_idxs=pron_choice_idxs, score=0.0)
    
#     def _phones_to_word_segments_robust(self, token_segs, words, prons):
#         word_segs = []
#         wi = 0 
#         for i, (word, pron) in enumerate(zip(words, prons)):
#             n_phones = len(pron)
#             if n_phones == 0: continue
#             if wi + n_phones > len(token_segs): break 
#             word_segs.append(Segment(word, token_segs[wi].start_frame, token_segs[wi + n_phones - 1].end_frame))
#             wi += n_phones
#         return word_segs

#     def _merge_words_into_chunks(self, words: List[WordSeg]):
#         if not words: return []
#         chunks = []
#         cur_words = [words[0].word]
#         cur_start = words[0].start
#         cur_end = words[0].end
#         for w in words[1:]:
#             gap = w.start - cur_end
#             proposed_dur = w.end - cur_start
#             if gap <= self.max_gap_s and proposed_dur <= self.max_chunk_s:
#                 cur_end = w.end
#                 cur_words.append(w.word)
#             else:
#                 if (cur_end - cur_start) >= self.min_chunk_s and len(cur_words) >= self.min_words:
#                     chunks.append(InternalChunk(cur_start, cur_end, cur_words))
#                 cur_start = w.start
#                 cur_end = w.end
#                 cur_words = [w.word]
#         if len(cur_words) > 0 and (cur_end - cur_start) >= self.min_chunk_s and len(cur_words) >= self.min_words:
#              chunks.append(InternalChunk(cur_start, cur_end, cur_words))
#         return chunks

#     def _pad_chunks(self, chunks, words, audio_dur):
#         if not chunks: return []
#         out = []
#         for c in chunks:
#             # Replicated logic from previous turn (Correct)
#             first_word_idx = -1; last_word_idx = -1
#             for i_w, w in enumerate(words):
#                 if abs(w.start - c.start) < 1e-4: first_word_idx = i_w
#                 if abs(w.end - c.end) < 1e-4: last_word_idx = i_w
            
#             left_limit = 0.0
#             if first_word_idx > 0: left_limit = words[first_word_idx - 1].end
#             right_limit = audio_dur
#             if last_word_idx != -1 and last_word_idx + 1 < len(words): right_limit = words[last_word_idx + 1].start
            
#             left_gap = max(0.0, c.start - left_limit)
#             right_gap = max(0.0, right_limit - c.end)
#             max_pad = self.config.get("max_pad_into_gap_s", 0.25)
            
#             new_start = max(0.0, c.start - min(self.pad_s, max_pad, left_gap))
#             new_end = min(audio_dur, c.end + min(self.pad_s, max_pad, right_gap))
#             out.append(InternalChunk(new_start, new_end, c.words))
#         return out

# # Static Functions (Unchanged)
# def build_trellis(log_probs, targets, blank_id):
#     T, V = log_probs.shape
#     N = len(targets)
#     device = log_probs.device
#     neg_inf = -1e9
#     targets_t = torch.tensor(targets, device=device, dtype=torch.long)
#     trellis = torch.full((T + 1, N + 1), neg_inf, device=device, dtype=log_probs.dtype)
#     trellis[0, 0] = 0.0
#     trellis[1:, 0] = torch.cumsum(log_probs[:, blank_id], dim=0)
#     for t in range(1, T + 1):
#         lp_t = log_probs[t - 1]
#         stay = trellis[t - 1, 1:] + lp_t[blank_id]
#         emit = trellis[t - 1, :-1] + lp_t[targets_t]
#         trellis[t, 1:] = torch.maximum(stay, emit)
#     return trellis



# def backtrace(trellis, log_probs, targets, blank_id):
#     _T = trellis.size(0) - 1
#     N = trellis.size(1) - 1
#     j = N; t = _T; path = []
#     while t > 0 and j > 0:
#         lp_t = log_probs[t - 1]
#         score_current = trellis[t, j]
#         score_stay = trellis[t - 1, j] + lp_t[blank_id]
#         score_emit = trellis[t - 1, j - 1] + lp_t[targets[j - 1]]
#         if abs(score_current - score_emit) < 1e-4:
#             path.append(Point(j - 1, t - 1)); j -= 1; t -= 1
#         else: t -= 1
#     path.reverse()
#     return path

# def points_to_segments(points, labels):
#     if not points: return []
#     segs = []
#     for i, p in enumerate(points):
#         start = p.time_index
#         end = points[i + 1].time_index if i + 1 < len(points) else start + 1
#         segs.append(Segment(labels[p.token_index], start, end))
#     return segs


import torch
import os
import json
import soundfile as sf
from pathlib import Path
from typing import List, Optional, Dict
from dataclasses import dataclass
from transformers import Wav2Vec2ForCTC, AutoProcessor

# 引入 io.py 数据结构
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
#  CTCChunker Class
#  - 逻辑尽量贴近“原始结果基于的版本”
#  - 兼容新的 3-stage artifact 输出
# ==========================================
class CTCChunker:
    def __init__(self, config: dict):
        self.config = config or {}
        self.device = torch.device(self.config.get("device", "cpu"))
        self.lang = self.config.get("lang", "zh")

        # Verbose / output
        self.verbose = self.config.get("verbose", False)
        self.chunks_out_dir = self.config.get("chunks_out_dir", None)

        if self.verbose:
            self._log_header("CTCChunker Initializing")
            print(f"  - Device: {self.device}")
            print(f"  - Lang: {self.lang}")
            print(f"  - Intermediate Output: {self.chunks_out_dir if self.chunks_out_dir else 'Disabled'}")

        # Resources
        self.model = None
        self.processor = None
        self.lexicon = {}
        self.phone_to_id = {}
        self.blank_id = 0

        # Hyperparameters (Stage 1)
        self.beam_size = self.config.get("beam_size", 10)
        self.min_chunk_s = self.config.get("min_chunk_s", 1.0)
        self.max_chunk_s = self.config.get("max_chunk_s", 12.0)
        self.max_gap_s = self.config.get("max_gap_s", 0.35)
        self.min_words = self.config.get("min_words", 2)
        self.pad_s = self.config.get("pad_s", 0.15)
        self.blank_token = self.config.get("blank_token", "<pad>")

        self._load_resources()

    def _log(self, msg: str):
        if self.verbose:
            print(f"[CTCChunker] {msg}")

    def _log_header(self, title: str):
        if self.verbose:
            print(f"\n=== {title} ===")

    # ==========================================================
    # Resource Loading
    # - 保持接近“原始结果那版”
    # - 不做 G2P 大规模增广
    # ==========================================================
    def _load_resources(self):
        lex_path = self.config.get("lexicon_path")
        self.lexicon = self._read_lexicon(lex_path)
        if self.verbose:
            print(f"  - Lexicon loaded: {len(self.lexicon)} words")

        model_path = self.config.get("chunk_model_path")
        if not model_path:
            return

        self._log(f"Requesting model: {model_path}")
        load_kwargs = {}

        if not os.path.isdir(model_path):
            load_kwargs["subfolder"] = f"{self.lang}/chunker"
            self._log(f"Mode: Cloud Repo ({load_kwargs['subfolder']})")
        else:
            self._log("Mode: Local Override")

        try:
            self.processor = AutoProcessor.from_pretrained(model_path, **load_kwargs)
            self.model = Wav2Vec2ForCTC.from_pretrained(model_path, **load_kwargs).to(self.device)
            self.model.eval()
            self._log("Successfully loaded model.")
        except Exception as e:
            self._log(f"Routing failed ({e}), falling back to root...")
            try:
                self.processor = AutoProcessor.from_pretrained(model_path)
                self.model = Wav2Vec2ForCTC.from_pretrained(model_path).to(self.device)
                self.model.eval()
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

            # English fuzzy map：保留你原始版本里的 hotfix
            if self.lang == "en":
                self.garbage_tokens = {"O", "[UNK]", "<unk>"}
                self.fuzzy_map = {}
                for token, pid in self.phone_to_id.items():
                    self.fuzzy_map[token] = pid
                    pure_token = "".join(filter(str.isalpha, token))
                    if pure_token and pure_token not in self.fuzzy_map:
                        self.fuzzy_map[pure_token] = pid
                self._log(f"🇬🇧 English Hotfix Applied: Fuzzy map built with {len(self.fuzzy_map)} entries.")
        else:
            self.blank_id = 0

    # ==========================================================
    # Main Entry
    # - 只生成一套最终 chunk
    # - 保存和返回严格一致
    # ==========================================================
    @torch.inference_mode()
    def find_chunks(self, audio_tensor: torch.Tensor, text_list: List[str], file_id: str = "unknown") -> List[AudioChunk]:
        if self.model is None:
            raise RuntimeError("CTC 模型未加载")

        if self.verbose:
            self._log_header(f"Processing: {file_id}")

        # 1. Forward Pass
        input_values = self.processor(
            audio_tensor.numpy(),
            sampling_rate=16000,
            return_tensors="pt"
        ).input_values.to(self.device)

        logits = self.model(input_values).logits
        log_probs = torch.log_softmax(logits, dim=-1).squeeze(0)

        # 与你原始版本一致：动态 SPF
        spf = float(audio_tensor.numel() / 16000.0) / float(log_probs.size(0))

        if self.verbose:
            print(f"  - Audio Duration: {audio_tensor.size(0)/16000:.3f}s")
            print(f"  - Logits Shape: {log_probs.shape}")
            print(f"  - Calculated SPF: {spf:.6f}")

        # 2. Beam Search
        prons_per_word = self._words_to_pronunciations(text_list)

        best_candidate = None
        for attempt_beam in [self.beam_size, 50, 200, 1000]:
            best_candidate = self._beam_search(
                log_probs,
                text_list,
                prons_per_word,
                beam_width=attempt_beam
            )
            if best_candidate:
                if attempt_beam > self.beam_size:
                    self._log(f"💡 Expanded beam to {attempt_beam}.")
                break

        if not best_candidate:
            self._log("❌ Beam search failed. Fallback to full audio.")
            fallback = [AudioChunk(
                tensor=audio_tensor,
                start_time=0.0,
                end_time=audio_tensor.size(0) / 16000.0,
                text=" ".join(text_list),
                chunk_id=f"{file_id}.chunk_fallback"
            )]
            if self.chunks_out_dir:
                self._save_intermediate_results(fallback, file_id)
            return fallback

        # 3. Viterbi Alignment
        target_ids = [self.phone_to_id[p] for p in best_candidate.phones if p in self.phone_to_id]
        if not target_ids:
            return []

        trellis = build_trellis(log_probs, target_ids, self.blank_id)
        points = backtrace(trellis, log_probs, target_ids, self.blank_id)

        # 4. Convert to token / word segments
        filtered_phones = [p for p in best_candidate.phones if p in self.phone_to_id]
        token_segs = points_to_segments(points, filtered_phones)

        clean_prons = []
        for i, idx in enumerate(best_candidate.pron_choice_idxs):
            raw_pron = prons_per_word[i][idx]
            if len(raw_pron) > 0 and isinstance(raw_pron[0], list):
                raw_pron = raw_pron[0]
            filtered_pron = [p for p in raw_pron if p in self.phone_to_id]
            clean_prons.append(filtered_pron)

        word_segs = self._phones_to_word_segments_robust(token_segs, text_list, clean_prons)

        # 5. Logical Chunking
        word_objects = [
            WordSeg(s.start_frame * spf, (s.end_frame - s.start_frame) * spf, s.label)
            for s in word_segs
        ]

        internal_chunks = self._merge_words_into_chunks(word_objects)
        audio_dur_s = float(audio_tensor.size(0)) / 16000.0
        internal_chunks = self._pad_chunks(internal_chunks, word_objects, audio_dur_s)

        # 6. Final Physical Chunks
        # 保留你“原始结果基于版本”的文本逻辑：
        # 使用 chunk_starts 把 text_list 单调切回 chunk 文本
        final_chunks = []
        sr = 16000

        chunk_starts = [0] * len(internal_chunks)
        orig_cursor = 0

        for i, c in enumerate(internal_chunks):
            if not c.words:
                chunk_starts[i] = orig_cursor
                continue

            anchor_ctc = "".join(filter(str.isalpha, c.words[0].lower()))

            while orig_cursor < len(text_list):
                anchor_orig = "".join(filter(str.isalpha, text_list[orig_cursor].lower()))
                if anchor_orig == anchor_ctc:
                    break
                orig_cursor += 1

            chunk_starts[i] = min(orig_cursor, len(text_list))

            for cw in c.words[1:]:
                cw_clean = "".join(filter(str.isalpha, cw.lower()))
                while orig_cursor < len(text_list) - 1:
                    orig_cursor += 1
                    tw_clean = "".join(filter(str.isalpha, text_list[orig_cursor].lower()))
                    if tw_clean == cw_clean:
                        break

            orig_cursor += 1

        chunk_starts.append(len(text_list))

        for i, c in enumerate(internal_chunks):
            s_samp = int(round(c.start * sr))
            e_samp = int(round(c.end * sr))

            s_samp = max(0, min(s_samp, audio_tensor.size(0)))
            e_samp = max(0, min(e_samp, audio_tensor.size(0)))

            if e_samp <= s_samp:
                continue

            chunk_id_str = f"{file_id}.chunk{i+1:03d}"
            real_words = text_list[chunk_starts[i]:chunk_starts[i+1]]

            chunk_obj = AudioChunk(
                tensor=audio_tensor[s_samp:e_samp].clone(),
                start_time=c.start,
                end_time=c.end,
                text=" ".join(real_words),
                chunk_id=chunk_id_str
            )
            final_chunks.append(chunk_obj)

        self._log(f"Found {len(final_chunks)} chunks.")

        # 兼容 3-stage：保存的就是最终返回的这一版
        if self.chunks_out_dir:
            self._save_intermediate_results(final_chunks, file_id)

        return final_chunks

    # ==========================================================
    # Artifact Saving
    # - 保存最终版 chunk
    # - 用 FLOAT 避免量化误差
    # ==========================================================
    def _save_intermediate_results(self, chunks: List[AudioChunk], file_id: str):
        out_root = Path(self.chunks_out_dir)
        out_root.mkdir(parents=True, exist_ok=True)

        jsonl_path = out_root / f"{file_id}.chunks.jsonl"
        tsv_path = out_root / f"{file_id}.chunks.tsv"

        if self.verbose:
            print(f"  - Saving artifacts to: {out_root}")

        with open(jsonl_path, "w", encoding="utf-8") as fj, \
             open(tsv_path, "w", encoding="utf-8") as ft:

            ft.write("chunk_id\tstart_s\tend_s\tdur_s\twords\n")

            for c in chunks:
                wav_name = f"{c.chunk_id}_{c.start_time:.3f}-{c.end_time:.3f}.wav"
                wav_path = out_root / wav_name

                sf.write(
                    str(wav_path),
                    c.tensor.squeeze().cpu().numpy().astype("float32"),
                    16000,
                    subtype="FLOAT"
                )

                dur = c.end_time - c.start_time
                words_list = c.text.split()

                obj = {
                    "chunk_id": c.chunk_id,
                    "audio": str(wav_path),
                    "start_s": round(c.start_time, 3),
                    "end_s": round(c.end_time, 3),
                    "dur_s": round(dur, 3),
                    "words": words_list,
                    "text": c.text
                }

                fj.write(json.dumps(obj, ensure_ascii=False) + "\n")
                ft.write(
                    f"{c.chunk_id}\t{obj['start_s']}\t{obj['end_s']}\t{obj['dur_s']}\t{obj['text']}\n"
                )

    # ==========================================================
    # Internal Helpers
    # ==========================================================
    def _read_lexicon(self, path: Optional[str]):
        lexicon = {}
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        lexicon[parts[0].lower()] = parts[1:]
        return lexicon

    def _words_to_pronunciations(self, words: List[str]):
        out = []
        for w in words:
            w_clean = w.strip()
            if not w_clean:
                continue

            w_upper = w_clean.upper()
            w_lower = w_clean.lower()
            w_no_digit = "".join([c for c in w_upper if not c.isdigit()])

            if w_upper in self.lexicon:
                out.append([self.lexicon[w_upper]])
            elif w_no_digit in self.lexicon:
                out.append([self.lexicon[w_no_digit]])
            elif w_lower in self.lexicon:
                out.append([self.lexicon[w_lower]])
            elif w_clean in self.lexicon:
                out.append([self.lexicon[w_clean]])
            else:
                if self.verbose:
                    print(f"⚠️  [CTCChunker] OOV Word: '{w_clean}' (Tried: {w_no_digit})")
                raise ValueError(f"[CTCChunker] OOV Word: {w_clean}")
        return out
    def _beam_search(self, log_probs, words, prons_per_word, beam_width=10):
        current_phones = []
        pron_choice_idxs = []

        for i, word in enumerate(words):
            variants = prons_per_word[i]
            valid_pron_found = False

            for p_idx, pron in enumerate(variants):
                if len(pron) > 0 and isinstance(pron[0], list):
                    pron = pron[0]

                temp_phones = list(current_phones)
                inter_token = self.config.get("inter_word_token", None)
                if temp_phones and inter_token:
                    temp_phones.append(inter_token)

                temp_phones.extend(pron)

                is_valid = True
                for p in pron:
                    if not isinstance(p, str):
                        continue
                    if p in self.phone_to_id or p == "O":
                        continue

                    p_pure = "".join(filter(str.isalpha, p))
                    if p_pure not in self.phone_to_id and not (hasattr(self, "fuzzy_map") and p_pure in self.fuzzy_map):
                        is_valid = False
                        break

                if is_valid:
                    current_phones = temp_phones
                    pron_choice_idxs.append(p_idx)
                    valid_pron_found = True
                    break

            if not valid_pron_found:
                return None

        return PronCandidate(
            phones=current_phones,
            pron_choice_idxs=pron_choice_idxs,
            score=0.0
        )
    # def _beam_search(self, log_probs, words, prons_per_word, beam_width=10):
    #     beam = [PronCandidate(phones=[], pron_choice_idxs=[], score=0.0)]

    #     for i, word in enumerate(words):
    #         new_beam = []
    #         variants = prons_per_word[i]

    #         for cand in beam:
    #             for p_idx, pron in enumerate(variants):
    #                 if len(pron) > 0 and isinstance(pron[0], list):
    #                     pron = pron[0]

    #                 current_phones = list(cand.phones)
    #                 inter_token = self.config.get("inter_word_token", None)
    #                 if current_phones and inter_token:
    #                     current_phones.append(inter_token)

    #                 new_phones = current_phones + pron
    #                 new_ids = []
    #                 valid_pron = True

    #                 for p in new_phones:
    #                     if not isinstance(p, str):
    #                         continue
    #                     if p in self.phone_to_id:
    #                         new_ids.append(self.phone_to_id[p])
    #                     elif p == "O":
    #                         continue
    #                     else:
    #                         p_pure = "".join(filter(str.isalpha, p))
    #                         if hasattr(self, "fuzzy_map") and p_pure in self.fuzzy_map:
    #                             new_ids.append(self.fuzzy_map[p_pure])
    #                         elif p_pure in self.phone_to_id:
    #                             new_ids.append(self.phone_to_id[p_pure])
    #                         else:
    #                             valid_pron = False
    #                             break

    #                 if not valid_pron or not new_ids:
    #                     continue

    #                 try:
    #                     trellis = build_trellis(log_probs, new_ids, self.blank_id)
    #                     score = float(torch.max(trellis[-1, -1]).item())
    #                 except Exception:
    #                     score = -float("inf")

    #                 if score > -1e8:
    #                     new_beam.append(
    #                         PronCandidate(
    #                             new_phones,
    #                             cand.pron_choice_idxs + [p_idx],
    #                             score
    #                         )
    #                     )

    #         if not new_beam:
    #             return None

    #         new_beam.sort(key=lambda x: x.score, reverse=True)
    #         beam = new_beam[:beam_width]

    #     return beam[0] if beam else None

    def _phones_to_word_segments_robust(self, token_segs, words, prons):
        word_segs = []
        wi = 0
        for i, (word, pron) in enumerate(zip(words, prons)):
            n_phones = len(pron)
            if n_phones == 0:
                continue
            if wi + n_phones > len(token_segs):
                break
            word_segs.append(
                Segment(word, token_segs[wi].start_frame, token_segs[wi + n_phones - 1].end_frame)
            )
            wi += n_phones
        return word_segs

    def _merge_words_into_chunks(self, words: List[WordSeg]):
        if not words:
            return []
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

        if len(cur_words) > 0 and (cur_end - cur_start) >= self.min_chunk_s and len(cur_words) >= self.min_words:
            chunks.append(InternalChunk(cur_start, cur_end, cur_words))

        return chunks

    def _pad_chunks(self, chunks, words, audio_dur):
        if not chunks:
            return []
        out = []
        for c in chunks:
            first_word_idx = -1
            last_word_idx = -1
            for i_w, w in enumerate(words):
                if abs(w.start - c.start) < 1e-4:
                    first_word_idx = i_w
                if abs(w.end - c.end) < 1e-4:
                    last_word_idx = i_w

            left_limit = 0.0
            if first_word_idx > 0:
                left_limit = words[first_word_idx - 1].end

            right_limit = audio_dur
            if last_word_idx != -1 and last_word_idx + 1 < len(words):
                right_limit = words[last_word_idx + 1].start

            left_gap = max(0.0, c.start - left_limit)
            right_gap = max(0.0, right_limit - c.end)
            max_pad = self.config.get("max_pad_into_gap_s", 0.25)

            new_start = max(0.0, c.start - min(self.pad_s, max_pad, left_gap))
            new_end = min(audio_dur, c.end + min(self.pad_s, max_pad, right_gap))
            out.append(InternalChunk(new_start, new_end, c.words))
        return out


# ==========================================================
# Static Functions
# ==========================================================
def build_trellis(log_probs, targets, blank_id):
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
        stay = trellis[t - 1, 1:] + lp_t[blank_id]
        emit = trellis[t - 1, :-1] + lp_t[targets_t]
        trellis[t, 1:] = torch.maximum(stay, emit)

    return trellis


def backtrace(trellis, log_probs, targets, blank_id):
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
            path.append(Point(j - 1, t - 1))
            j -= 1
            t -= 1
        else:
            t -= 1

    path.reverse()
    return path


def points_to_segments(points, labels):
    if not points:
        return []
    segs = []
    for i, p in enumerate(points):
        start = p.time_index
        end = points[i + 1].time_index if i + 1 < len(points) else start + 1
        segs.append(Segment(labels[p.token_index], start, end))
    return segs