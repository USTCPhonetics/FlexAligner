import torch
import numpy as np
import time
import pandas as pd
import soundfile as sf
import shutil
import gc
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Union, List, Tuple, Optional

# 进度条适配
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs): return iterable

# [核心依赖]
from .config import AlignmentConfig
from .frontend import TextFrontend
from .chunker import CTCChunker
from .aligner import LocalAligner

# 定义一个轻量级的数据结构，用于在 Stage 1 和 Stage 2 之间传递数据
@dataclass
class AlignmentTask:
    chunk_id: str
    text: str
    start_time: float
    end_time: float
    tensor: torch.Tensor

class FlexAligner:
    def __init__(self, config: Union[Dict, AlignmentConfig, None] = None):
        """
        FlexAligner 控制器：严格分步执行 (Segmentation -> Alignment)
        """
        # 1. 配置加载
        if isinstance(config, dict):
            self.config = AlignmentConfig(**config)
        elif isinstance(config, AlignmentConfig):
            self.config = config
        else:
            self.config = AlignmentConfig()
        
        self.config_dict = asdict(self.config)
        
        # 2. 初始化前端 (轻量级，常驻)
        mode = getattr(self.config, "validation_mode", "FAST")
        self.frontend = TextFrontend(config=self.config_dict, mode=mode)
        
        # 3. 模型组件 (懒加载 / 按需加载，初始为空)
        self.chunker: Optional[CTCChunker] = None
        self.aligner: Optional[LocalAligner] = None

    # =========================================================================
    # [入口 1] 单文件处理 (Strict Mode)
    # =========================================================================
    def align(self, audio_path: str, text_path: str, output_path: str, verbose: bool = True):
        """
        单文件全流程：针对单文件任务，如果出错，应当直接抛出异常。
        """
        tasks = [(audio_path, text_path, output_path)]
        self.align_batch(tasks, raise_on_error=True)

    # =========================================================================
    # [入口 2] 批量处理 (Robust Mode - Two Stage)
    # =========================================================================
    def align_batch(self, tasks: List[Tuple[str, str, str]], raise_on_error: bool = False):
        """
        严格分步执行批量任务：
        Phase 1: 全部切分 -> 内存暂存 -> 卸载 Chunker
        Phase 2: 加载 Aligner -> 读取暂存 -> 对齐拼接
        """
        if not tasks: return

        print("\n" + "="*80)
        print(f"🚀 [FlexAligner] Batch Processing: {len(tasks)} files")
        print(f"   Strategy: Two-Stage Sequential (Chunk -> Align)")
        print("="*80)

        # --- Phase 1: Segmentation ---
        print(f"\n[Phase 1] Segmentation & Text Preprocessing...")
        
        if self.chunker is None:
            print(f"   -> Loading Chunker model...")
            self.chunker = CTCChunker(config=self.config_dict)
        
        # 暂存所有文件的 Chunk 信息
        # 结构: [ {"output_path": str, "full_duration": float, "chunks": List[AlignmentTask]}, ... ]
        batch_data = []
        
        pbar_seg = tqdm(tasks, desc="Seg", unit="file")
        for audio_p, text_p, out_p in pbar_seg:
            try:
                # IO
                audio_np = self.frontend.load_audio(audio_p)
                raw_text = self.frontend.load_text(text_p)
                
                # Preprocess
                lang = self.config.lang if self.config.lang else self.frontend.detect_language(raw_text)
                tokens = self.frontend.get_phonemes(raw_text, lang)
                text_list = [t.strip() for t in tokens if t.strip()]
                
                audio_tensor = torch.from_numpy(audio_np).float()
                full_dur = audio_tensor.size(0) / 16000.0
                
                # Chunking
                file_id = Path(audio_p).stem
                raw_chunks = self.chunker.find_chunks(audio_tensor, text_list, file_id=file_id)
                
                # 转换为标准 Task 对象
                task_chunks = []
                for rc in raw_chunks:
                    task_chunks.append(AlignmentTask(
                        chunk_id=rc.chunk_id,
                        text=rc.text,
                        start_time=rc.start_time,
                        end_time=rc.end_time,
                        tensor=rc.tensor
                    ))

                batch_data.append({
                    "output_path": out_p,
                    "full_duration": full_dur,
                    "chunks": task_chunks,
                    "src_name": Path(audio_p).name
                })
                
            except Exception as e:
                if raise_on_error: raise e
                tqdm.write(f"❌ Segmentation Failed {Path(audio_p).name}: {e}")
        
        # 显存清理
        print(f"   -> Unloading Chunker to free VRAM...")
        del self.chunker
        self.chunker = None
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        # --- Phase 2: Alignment ---
        print(f"\n[Phase 2] Alignment & Stitching...")
        if not batch_data: return

        if self.aligner is None:
            print(f"   -> Loading Aligner model...")
            self.aligner = LocalAligner(config=self.config_dict)

        pbar_ali = tqdm(batch_data, desc="Align", unit="file")
        for item in pbar_ali:
            try:
                # 调用统一的缝合逻辑
                self._stitch_and_export(
                    chunks=item['chunks'],
                    full_duration=item['full_duration'],
                    output_path=item['output_path']
                )
            except Exception as e:
                if raise_on_error: raise e
                tqdm.write(f"❌ Alignment Failed {item['src_name']}: {e}")

        print("\n" + "="*80)
        print(f"🏁 Batch Processing Completed.")
        print("="*80 + "\n")

    # =========================================================================
    # [入口 3] 从 Manifest 恢复 (Stage 2 Only)
    # =========================================================================
    def align_from_manifest(
        self, 
        manifest_path: str, 
        audio_dir: str, 
        output_path: str, 
        full_audio_path: Optional[str] = None,
        verbose: bool = True
    ):
        """
        Stage 2 独立模式：读取 TSV -> 寻找音频 -> 转换为 Task 对象 -> 统一缝合
        """
        tsv_path = Path(manifest_path)
        wav_dir_path = Path(audio_dir)
        
        if not tsv_path.exists(): raise FileNotFoundError(f"Manifest not found: {tsv_path}")
        if not wav_dir_path.exists(): raise FileNotFoundError(f"Chunk audio dir: {wav_dir_path}")

        if verbose:
            print(f"🧩 [Resume] Processing {tsv_path.name}")

        if self.aligner is None:
            self.aligner = LocalAligner(config=self.config_dict)

        # 1. 确定总时长
        target_duration = 0.0
        if full_audio_path and Path(full_audio_path).exists():
            target_duration = sf.info(full_audio_path).duration
        
        # 2. 读取 TSV
        try:
            df = pd.read_csv(tsv_path, sep='\t')
        except Exception as e:
            raise RuntimeError(f"Failed to parse TSV: {e}")

        if target_duration == 0.0 and not df.empty:
            # 降级：估算
            try: target_duration = float(df.iloc[-1]['end_s'])
            except: target_duration = 0.0

        # 3. 构建 Task 列表 (模拟 Phase 1 的输出)
        task_chunks = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading Audio", disable=not verbose):
            chunk_id = row['chunk_id']
            text = str(row.get('text', row.get('words', ''))).strip()
            start = float(row['start_s'])
            end = float(row['end_s'])

            # 寻找音频 (兼容 Legacy 命名 {id}_*.wav)
            candidates = list(wav_dir_path.glob(f"{chunk_id}_*.wav"))
            if not candidates:
                candidates = list(wav_dir_path.glob(f"{chunk_id}.wav"))
            
            if not candidates:
                if verbose: print(f"❌ Chunk audio missing: {chunk_id}")
                continue
            
            # 读取音频转 Tensor
            try:
                wav, sr = sf.read(str(candidates[0]))
                if sr != 16000 and verbose: print(f"⚠️ Resampling required for {chunk_id}")
                chunk_tensor = torch.from_numpy(wav).float()
                if chunk_tensor.ndim > 1: chunk_tensor = chunk_tensor.mean(dim=1)
                
                task_chunks.append(AlignmentTask(
                    chunk_id=chunk_id,
                    text=text,
                    start_time=start,
                    end_time=end,
                    tensor=chunk_tensor
                ))
            except Exception as e:
                print(f"❌ Error loading {chunk_id}: {e}")

        # 4. 调用统一缝合逻辑
        self._stitch_and_export(task_chunks, target_duration, output_path)
        if verbose: print(f"✅ Saved to {output_path}")

    # =========================================================================
    # [核心私有方法] 缝合与导出 (The Core Stitcher)
    # =========================================================================
    def _stitch_and_export(self, chunks: List[AlignmentTask], full_duration: float, output_path: str):
        """
        核心逻辑：接收标准化 Chunk 列表，执行对齐，处理 Gap/Padding，生成 TextGrid
        """
        global_phones = []
        global_words = []
        prev_global_end = 0.0  # 物理锚点归零

        for chunk in chunks:
            chunk_start = chunk.start_time
            chunk_end = chunk.end_time
            
            # A. 核心对齐推理
            result = self.aligner.align_locally(chunk.tensor, chunk.text, file_id=chunk.chunk_id)
            
            if not result['phones']: 
                # 如果对齐失败，为了保持时间轴连续，可能需要填补？
                # 目前逻辑是跳过，这会导致大 Gap
                continue

            # B. 头部缝合 (Stitch Gap)
            gap = chunk_start - prev_global_end
            if gap > 0.001:
                gap_seg = ("NULL", prev_global_end, chunk_start)
                global_phones.append(gap_seg)
                global_words.append(gap_seg)
            
            # C. 添加对齐结果 (Offset Shift)
            for seg in result['phones']:
                global_phones.append((seg.label, chunk_start + seg.start, chunk_start + seg.end))
            for seg in result['words']:
                global_words.append((seg.label, chunk_start + seg.start, chunk_start + seg.end))
            
            prev_global_end = chunk_end

        # D. 尾部补齐 (Final Padding)
        # 获取最后一个有效对齐点的结束时间
        final_valid_end = max(prev_global_end, global_phones[-1][2] if global_phones else 0.0)
        
        if full_duration > final_valid_end + 0.001:
            pad_seg = ("NULL", final_valid_end, full_duration)
            global_phones.append(pad_seg)
            global_words.append(pad_seg)
            final_valid_end = full_duration # 更新为真实时长

        # E. 写入文件
        self._export_textgrid_file(output_path, final_valid_end, {"phones": global_phones, "words": global_words})

    def _export_textgrid_file(self, path: str, duration: float, tiers_data: dict):
        """底层 I/O：TextGrid 格式化写入"""
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