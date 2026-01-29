import torch
import numpy as np
import time
from pathlib import Path
from dataclasses import asdict
from typing import Dict, Union, List, Tuple, Optional

# 尝试导入进度条，如果没有则使用哑巴包装器
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs): return iterable

# [核心依赖]
from .config import AlignmentConfig
from .frontend import TextFrontend
# 延迟导入防止循环依赖
from .chunker import CTCChunker
from .aligner import LocalAligner

class FlexAligner:
    def __init__(self, config: Union[Dict, AlignmentConfig, None] = None):
        """
        FlexAligner 核心控制器
        """
        # 1. 配置对象化
        if isinstance(config, dict):
            self.config = AlignmentConfig(**config)
        elif isinstance(config, AlignmentConfig):
            self.config = config
        else:
            self.config = AlignmentConfig()
        
        config_dict = asdict(self.config)
        
        # 2. 初始化前端
        mode = getattr(self.config, "validation_mode", "FAST")
        self.frontend = TextFrontend(config=config_dict, mode=mode)
        
        # 3. 初始化 Chunker (Stage 1)
        self.chunker = CTCChunker(config=config_dict)
        
        # 4. 初始化 Aligner (Stage 2)
        # 共享词表优化：如果 Aligner 没指定词表，直接复用 Chunker 的
        if self.config.phone_json_path is None and hasattr(self.chunker, 'phone_to_id'):
            self.aligner = LocalAligner(config=config_dict, phone_to_id=self.chunker.phone_to_id)
        else:
            self.aligner = LocalAligner(config=config_dict)

    def align(self, audio_path: str, text_path: str, output_path: str, verbose: bool = True):
        """
        [单文件入口] 全闭环对齐流程
        """
        return self._align_core(audio_path, text_path, output_path, verbose=verbose)

    def align_batch(self, tasks: List[Tuple[str, str, str]]):
        """
        [批处理入口] 高效处理文件列表
        Args:
            tasks: List of (audio_path, text_path, output_path)
        """
        if not tasks:
            print("[FlexAligner] Warning: Empty task list.")
            return

        print("\n" + "="*80)
        print(f"🚀 [Batch Mode] Starting alignment for {len(tasks)} files...")
        print(f"   Device: {self.config.device} | Base Lang: {self.config.lang if self.config.lang else 'Auto-Detect'}")
        print("="*80)

        # 1. 锁定语言 (Language Lock)
        # 如果 Config 指定了语言，则以此为准；否则以第一条数据探测结果为准
        target_lang = self.config.lang
        
        success_count = 0
        fail_count = 0
        start_time = time.time()

        # 使用 tqdm 显示进度条
        pbar = tqdm(tasks, unit="file", desc="Aligning")
        
        for i, (audio_p, text_p, out_p) in enumerate(pbar):
            try:
                # 预加载文本以进行语言一致性检查
                # 注意：为了效率，我们尽量不重复读取，但在 batch 模式下安全第一
                raw_text = self.frontend.load_text(text_p)
                current_lang = self.frontend.detect_language(raw_text)

                # [核心逻辑] 语言一致性熔断
                if i == 0 and not target_lang:
                    target_lang = current_lang
                    # 动态反写回 config，确保后续组件感知
                    self.config.lang = target_lang
                    tqdm.write(f"🔒 [Language Lock] Batch language set to: {target_lang.upper()}")
                
                if target_lang and current_lang != target_lang:
                    raise AssertionError(
                        f"Language Mismatch! Expected {target_lang.upper()}, "
                        f"but file '{Path(text_p).name}' is detected as {current_lang.upper()}."
                    )

                # 调用核心逻辑 (关闭 verbose 以提高速度和减少刷屏)
                # 我们复用 _align_core，传入预读取的文本以减少 IO（需要稍微修改 core 接口支持）
                # 这里为了代码简洁，直接传路径，TextFrontend 内部有 LRU Cache，不会太慢
                self._align_core(audio_p, text_p, out_p, verbose=False, pre_lang=target_lang)
                success_count += 1
                
            except Exception as e:
                fail_count += 1
                tqdm.write(f"❌ Error processing {Path(audio_p).name}: {str(e)}")
                # 在 Batch 模式下，个别失败不应中断整个进程，除非是严重的语言错误
                if isinstance(e, AssertionError):
                    raise e # 语言不对直接炸

        total_time = time.time() - start_time
        print("\n" + "-"*80)
        print(f"🏁 Batch Completed in {total_time:.2f}s")
        print(f"   ✅ Success: {success_count}")
        print(f"   ❌ Failed:  {fail_count}")
        print(f"   ⚡ Speed:   {len(tasks)/total_time:.2f} files/sec")
        print("="*80 + "\n")

    def _align_core(self, audio_path: str, text_path: str, output_path: str, verbose: bool = True, pre_lang: str = None):
        """
        [内部核心] 执行单次对齐，支持静默模式
        """
        # --- 1. IO & Preprocessing ---
        audio_np = self.frontend.load_audio(audio_path)
        raw_text = self.frontend.load_text(text_path)
        
        # 语言决策优先级：pre_lang (Batch传入) > config.lang > 自动检测
        if pre_lang:
            lang = pre_lang
        elif hasattr(self.config, 'lang') and self.config.lang:
            lang = self.config.lang
        else:
            lang = self.frontend.detect_language(raw_text)
        
        raw_tokens = self.frontend.get_phonemes(raw_text, lang)
        text_list = [t.strip() for t in raw_tokens if t.strip()]
        
        audio_tensor = torch.from_numpy(audio_np).float()
        audio_duration = audio_tensor.size(0) / 16000.0

        # --- 仪表盘 (仅 Verbose 模式) ---
        if verbose:
            print("\n" + "="*80)
            print(f"🎛️  [FlexAligner Dashboard] Processing: {Path(audio_path).name}")
            print(f"   Audio Dur: {audio_duration:.3f}s | Lang: {lang.upper()} | Words: {len(text_list)}")
            print("="*80)
            print(f"🛰️  [Stage 1] Executing Macro-Segmentation (Beam Size={self.chunker.beam_size})...")

        # --- 2. Stage 1: Chunker ---
        chunks = self.chunker.find_chunks(audio_tensor, text_list)

        if verbose:
            n_chunks = len(chunks)
            avg_w_p_c = len(text_list) / n_chunks if n_chunks > 0 else 0
            print("-" * 80)
            print(f"📊 [Stage 1 Report] Found {n_chunks} chunks (Avg {avg_w_p_c:.1f} words/chunk)")
            print(f"{'ID':<6} | {'START':<8} | {'END':<8} | {'DUR':<6} | {'TEXT PREVIEW':<35}")
            print("-" * 80)
        
        global_phones = []
        global_words = []
        prev_end_time = 0.0

        # --- 3. Stage 2: Local Alignment Loop ---
        for i, chunk in enumerate(chunks):
            if verbose:
                chunk_dur = chunk.end_time - chunk.start_time
                txt_preview = (chunk.text[:32] + "..") if len(chunk.text) > 32 else chunk.text
                print(f"{i:<6} | {chunk.start_time:8.3f} | {chunk.end_time:8.3f} | {chunk_dur:6.3f} | {txt_preview:<35}")

            # 填充 Gap
            gap_dur = chunk.start_time - prev_end_time
            if gap_dur > 1e-6:
                null_seg = ("NULL", prev_end_time, chunk.start_time)
                global_phones.append(null_seg)
                global_words.append(null_seg)

            # Local Alignment
            local_result = self.aligner.align_locally(chunk.tensor, chunk.text)
            offset = chunk.start_time
            
            # --- Result Merging & Physics Fuse ---
            for seg in local_result["phones"]:
                g_start = offset + seg.start
                g_end = offset + seg.end
                if g_end > g_start + 1e-6:
                    global_phones.append((seg.label, g_start, g_end))
                
            for seg in local_result["words"]:
                g_start = offset + seg.start
                g_end = offset + seg.end
                if g_end > g_start + 1e-6:
                    global_words.append((seg.label, g_start, g_end))
                elif verbose:
                    print(f"      ⚠️  [Physics Fuse] Dropped zero-len word: {seg.label} at {g_start:.3f}s")
            
            prev_end_time = chunk.end_time

        # --- 4. Finalization ---
        if prev_end_time < audio_duration - 1e-6:
            last_null = ("NULL", prev_end_time, audio_duration)
            global_phones.append(last_null)
            global_words.append(last_null)

        if verbose:
            print("-" * 80)
            print(f"🏁 [Pipeline] Finished. Total Aligned Words: {len(global_words) - global_words.count('NULL')}")
            print("=" * 80 + "\n")

        # --- 5. Export ---
        self._export_textgrid(
            output_path, 
            audio_duration, 
            {"phones": global_phones, "words": global_words}
        )
        
        return chunks
    
    def _export_textgrid(self, path: str, duration: float, tiers_data: dict):
        """
        [工业级导出] 自研 TextGrid 生成器
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        
        def fmt(val): return f"{val:.6f}"

        def format_tier(name, segments):
            lines = []
            lines.append('        class = "IntervalTier"')
            lines.append(f'        name = "{name}"')
            lines.append('        xmin = 0') 
            lines.append(f'        xmax = {fmt(duration)}') 
            lines.append(f'        intervals: size = {len(segments)}')
            
            for i, (label, start, end) in enumerate(segments):
                lines.append(f'        intervals [{i+1}]:')
                lines.append(f'            xmin = {fmt(start)}')
                lines.append(f'            xmax = {fmt(end)}')
                safe_label = str(label).replace('"', '""')
                lines.append(f'            text = "{safe_label}"')
            return lines

        lines = [
            'File type = "ooTextFile"',
            'Object class = "TextGrid"',
            '',
            'xmin = 0',
            f'xmax = {fmt(duration)}',
            'tiers? <exists>',
            f'size = {len(tiers_data)}',
            'item []:'
        ]
        
        tier_idx = 1
        for name in ["words", "phones"]:
            if name in tiers_data:
                lines.append(f'    item [{tier_idx}]:')
                lines.extend(format_tier(name, tiers_data[name]))
                tier_idx += 1
                
        content = "\n".join(lines) + "\n"
        p.write_text(content, encoding="utf-8")