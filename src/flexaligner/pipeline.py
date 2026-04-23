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
import json
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
# @dataclass
# class AlignmentTask:
#     chunk_id: str
#     text: str
#     start_time: float
#     end_time: float
#     tensor: torch.Tensor

@dataclass
class AlignmentTask:
    chunk_id: str
    text: str
    start_time: float
    end_time: float
    audio_path: Optional[str] = None
    tensor: Optional[torch.Tensor] = None

@dataclass
class ChunkRecord:
    chunk_id: str
    audio_path: str
    text: str
    start_time: float
    end_time: float
    dur_s: float    


@dataclass
class ChunkEvidenceRecord:
    chunk_id: str
    audio_path: str
    text: str
    start_time: float
    end_time: float
    dur_s: float
    log_probs_path: str
    num_frames: int
    frame_hop_s: float
    vocab_size: int
@dataclass
class DecodeResult:
    chunk_id: str
    phones: list
    words: list
    decode_tag: str    


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
        # print("flexaligner config:")
        # print(self.config)
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
        单文件全流程：三阶段编排壳。
        """
        tasks = [(audio_path, text_path, output_path)]
        self.align_batch(tasks, raise_on_error=True)

    # =========================================================================
    # [入口 2] 批量处理 (Robust Mode - Two Stage)
    # =========================================================================
    def align_batch(self, tasks: List[Tuple[str, str, str]], raise_on_error: bool = False):
        """
        三阶段批量处理：
        Phase 1: Chunk -> chunks manifest
        Phase 2: Forward -> evidence manifest
        Phase 3: Decode -> TextGrid
        """
        if not tasks:
            return

        print("\n" + "=" * 80)
        print(f"🚀 [FlexAligner] Batch Processing: {len(tasks)} files")
        print(f"   Strategy: Three-Stage Orchestration (Chunk -> Forward -> Decode)")
        print("=" * 80)

        # -----------------------------------------------------------------
        # Phase 1: Chunk
        # -----------------------------------------------------------------
        print(f"\n[Phase 1] Chunking & Manifest Export...")

        if self.chunker is None:
            print(f"   -> Loading Chunker model...")
            self.chunker = CTCChunker(config=self.config_dict)

        if not self.config.chunks_out_dir:
            raise RuntimeError("chunks_out_dir must be set for three-stage pipeline.")

        batch_data = []

        pbar_chunk = tqdm(tasks, desc="Chunk", unit="file")
        for audio_p, text_p, out_p in pbar_chunk:
            try:
                audio_np = self.frontend.load_audio(audio_p)
                raw_text = self.frontend.load_text(text_p)

                lang = self.config.lang if self.config.lang else self.frontend.detect_language(raw_text)
                tokens = self.frontend.get_phonemes(raw_text, lang)
                text_list = [t.strip() for t in tokens if t.strip()]

                audio_tensor = torch.from_numpy(audio_np).float()
                full_dur = audio_tensor.size(0) / 16000.0

                file_id = Path(audio_p).stem

                # Stage 1: chunker 内部负责写出 chunks manifest / wav
                _ = self.chunker.find_chunks(audio_tensor, text_list, file_id=file_id)

                manifest_path = Path(self.config.chunks_out_dir) / f"{file_id}.chunks.jsonl"
                if not manifest_path.exists():
                    raise RuntimeError(f"Expected manifest not found: {manifest_path}")

                batch_data.append({
                    "audio_path": str(audio_p),
                    "text_path": str(text_p),
                    "manifest_path": str(manifest_path),
                    "output_path": str(out_p),
                    "full_duration": full_dur,
                    "src_name": Path(audio_p).name,
                })

            except Exception as e:
                if raise_on_error:
                    raise e
                tqdm.write(f"❌ Chunk Failed {Path(audio_p).name}: {e}")

        print(f"   -> Unloading Chunker to free VRAM...")
        del self.chunker
        self.chunker = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if not batch_data:
            print("⚠️ No valid files passed Phase 1.")
            return

        # -----------------------------------------------------------------
        # Phase 2: Forward
        # -----------------------------------------------------------------
        print(f"\n[Phase 2] Acoustic Forward & Evidence Export...")

        if self.aligner is None:
            print(f"   -> Loading Aligner model...")
            self.aligner = LocalAligner(config=self.config_dict)

        pbar_forward = tqdm(batch_data, desc="Forward", unit="file")
        for item in pbar_forward:
            try:
                manifest_path = item["manifest_path"]

                # 默认把 evidence 放到 chunks_out_dir 同级的 evidence/ 下
                chunks_dir = Path(manifest_path).parent
                if chunks_dir.name == "chunks":
                    evidence_dir = chunks_dir.parent / "evidence"
                else:
                    evidence_dir = chunks_dir / "evidence"

                evidence_manifest_path = self.forward_from_manifest(
                    manifest_path=manifest_path,
                    evidence_dir=str(evidence_dir),
                    verbose=False,
                    max_batch_items=self.config.stage2_max_batch_items,
                    max_batch_frames=self.config.stage2_max_batch_frames,
                    sort_by_duration=self.config.stage2_sort_by_duration,
                )

                item["evidence_manifest_path"] = evidence_manifest_path

            except Exception as e:
                if raise_on_error:
                    raise e
                tqdm.write(f"❌ Forward Failed {item['src_name']}: {e}")

        if not any("evidence_manifest_path" in item for item in batch_data):
            print("⚠️ No valid files passed Phase 2.")
            return

        # -----------------------------------------------------------------
        # Phase 3: Decode
        # -----------------------------------------------------------------
        print(f"\n[Phase 3] Decode & TextGrid Export...")

        pbar_decode = tqdm(batch_data, desc="Decode", unit="file")
        for item in pbar_decode:
            if "evidence_manifest_path" not in item:
                continue

            try:
                self.decode_from_evidence(
                    evidence_manifest_path=item["evidence_manifest_path"],
                    output_path=item["output_path"],
                    full_duration=item["full_duration"],
                    verbose=False
                )
            except Exception as e:
                if raise_on_error:
                    raise e
                tqdm.write(f"❌ Decode Failed {item['src_name']}: {e}")

        print("\n" + "=" * 80)
        print("🏁 Batch Processing Completed.")
        print("=" * 80 + "\n")

    
    
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
            # print(f"\nProcessing chunk '{chunk.chunk_id}': {chunk_start:.3f}s - {chunk_end:.3f}s, text: '{chunk.text}'")
            # input("Press Enter to align this chunk...")  # Debug pause before alignment
            # A. 核心对齐推理
            result = self.aligner.align_locally(chunk.tensor, chunk.text, file_id=chunk.chunk_id)
            
            if not result['phones']: 
                # 如果对齐失败，为了保持时间轴连续，可能需要填补？
                # 目前逻辑是跳过，这会导致大 Gap
                continue

            # B. 头部缝合 (Stitch Gap)
            # gap = chunk_start - prev_global_end
            gap = round(chunk_start - prev_global_end, 3)
            if gap >= 0.001:
                gap_seg = ("NULL", prev_global_end, chunk_start)
                global_phones.append(gap_seg)
                global_words.append(gap_seg)
            
            # C. 添加对齐结果 (Offset Shift)
            # for seg in result['phones']:
            #     global_phones.append((seg.label, chunk_start + seg.start, chunk_start + seg.end))
            # for seg in result['words']:
            #     global_words.append((seg.label, chunk_start + seg.start, chunk_start + seg.end))
            for seg in result['phones']:
                abs_start = round(chunk_start + seg.start, 3)
                abs_end = round(chunk_start + seg.end, 3)
                global_phones.append((seg.label, abs_start, abs_end))
                
            for seg in result['words']:
                abs_start = round(chunk_start + seg.start, 3)
                abs_end = round(chunk_start + seg.end, 3)
                global_words.append((seg.label, abs_start, abs_end))
            # prev_global_end = chunk_end
            prev_global_end = global_phones[-1][2]

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

    def _load_chunk_records(self, manifest_path: str) -> List[ChunkRecord]:
        """
        读取 JSONL manifest，并做最小字段校验。
        """
        manifest_path = Path(manifest_path)
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        records: List[ChunkRecord] = []
        required_fields = {"chunk_id", "audio", "start_s", "end_s"}

        with open(manifest_path, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    raise RuntimeError(
                        f"Failed to parse JSONL at line {lineno} in {manifest_path}: {e}"
                    )

                missing = required_fields - set(obj.keys())
                if missing:
                    raise RuntimeError(
                        f"Manifest line {lineno} missing required fields {sorted(missing)} "
                        f"in {manifest_path}"
                    )

                text = str(obj.get("text", "")).strip()
                if not text:
                    words = obj.get("words", [])
                    if isinstance(words, list):
                        text = " ".join(map(str, words)).strip()
                    else:
                        text = str(words).strip()

                start_time = float(obj["start_s"])
                end_time = float(obj["end_s"])
                dur_s = float(obj.get("dur_s", end_time - start_time))

                records.append(ChunkRecord(
                    chunk_id=str(obj["chunk_id"]),
                    audio_path=str(obj["audio"]),
                    text=text,
                    start_time=start_time,
                    end_time=end_time,
                    dur_s=dur_s,
                ))

        if not records:
            raise RuntimeError(f"Manifest is empty: {manifest_path}")

        return records


    def _records_to_tasks(
        self,
        records: List[ChunkRecord],
        verbose: bool = True
    ) -> List[AlignmentTask]:
        """
        从 ChunkRecord 列表加载音频并构造 AlignmentTask。
        """
        task_chunks: List[AlignmentTask] = []

        for rec in tqdm(records, total=len(records), desc="Loading Audio", disable=not verbose):
            audio_path = Path(rec.audio_path)

            if not audio_path.exists():
                if verbose:
                    print(f"❌ Chunk audio missing: {audio_path}")
                continue

            try:
                wav, sr = sf.read(str(audio_path))
                if sr != 16000 and verbose:
                    print(f"⚠️ Resampling required for {rec.chunk_id} (sr={sr})")

                chunk_tensor = torch.from_numpy(wav).float()
                if chunk_tensor.ndim > 1:
                    chunk_tensor = chunk_tensor.mean(dim=1)

                task_chunks.append(AlignmentTask(
                    chunk_id=rec.chunk_id,
                    text=rec.text,
                    start_time=rec.start_time,
                    end_time=rec.end_time,
                    tensor=chunk_tensor
                ))
            except Exception as e:
                if verbose:
                    print(f"❌ Error loading {rec.chunk_id} from {audio_path}: {e}")

        return task_chunks
    def _save_evidence_records(
        self,
        records: List[ChunkEvidenceRecord],
        evidence_manifest_path: str
    ):
        """
        将 Stage 2 的证据记录保存为 JSONL。
        """
        evidence_manifest_path = Path(evidence_manifest_path)
        evidence_manifest_path.parent.mkdir(parents=True, exist_ok=True)

        with open(evidence_manifest_path, "w", encoding="utf-8") as f:
            for rec in records:
                obj = {
                    "chunk_id": rec.chunk_id,
                    "audio": rec.audio_path,
                    "text": rec.text,
                    "start_s": round(rec.start_time, 3),
                    "end_s": round(rec.end_time, 3),
                    "dur_s": round(rec.dur_s, 3),
                    "log_probs": rec.log_probs_path,
                    "num_frames": rec.num_frames,
                    "frame_hop_s": rec.frame_hop_s,
                    "vocab_size": rec.vocab_size,
                }
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")


    def _load_evidence_records(self, evidence_manifest_path: str) -> List[ChunkEvidenceRecord]:
            """
            读取 Stage 2 生成的 evidence JSONL。
            """
            evidence_manifest_path = Path(evidence_manifest_path)
            if not evidence_manifest_path.exists():
                raise FileNotFoundError(f"Evidence manifest not found: {evidence_manifest_path}")

            required_fields = {
                "chunk_id", "audio", "text", "start_s", "end_s", "log_probs",
                "num_frames", "frame_hop_s", "vocab_size"
            }

            records: List[ChunkEvidenceRecord] = []

            with open(evidence_manifest_path, "r", encoding="utf-8") as f:
                for lineno, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError as e:
                        raise RuntimeError(
                            f"Failed to parse evidence JSONL at line {lineno} in "
                            f"{evidence_manifest_path}: {e}"
                        )

                    missing = required_fields - set(obj.keys())
                    if missing:
                        raise RuntimeError(
                            f"Evidence line {lineno} missing required fields {sorted(missing)} "
                            f"in {evidence_manifest_path}"
                        )

                    start_time = float(obj["start_s"])
                    end_time = float(obj["end_s"])

                    records.append(ChunkEvidenceRecord(
                        chunk_id=str(obj["chunk_id"]),
                        audio_path=str(obj["audio"]),
                        text=str(obj["text"]).strip(),
                        start_time=start_time,
                        end_time=end_time,
                        dur_s=float(obj.get("dur_s", end_time - start_time)),
                        log_probs_path=str(obj["log_probs"]),
                        num_frames=int(obj["num_frames"]),
                        frame_hop_s=float(obj["frame_hop_s"]),
                        vocab_size=int(obj["vocab_size"]),
                    ))

            if not records:
                raise RuntimeError(f"Evidence manifest is empty: {evidence_manifest_path}")

            return records
    def _stitch_decoded_results_and_export(
            self,
            decoded_chunks: List[dict],
            full_duration: float,
            output_path: str
        ):
            """
            接收已经 decode 完成的 chunk 结果，做全局缝合并导出 TextGrid。
            decoded_chunks 的每个元素格式：
            {
                "chunk_id": str,
                "start_time": float,
                "end_time": float,
                "result": {"phones": [...], "words": [...]}
            }
            """
            global_phones = []
            global_words = []
            prev_global_end = 0.0

            for item in decoded_chunks:
                chunk_start = item["start_time"]
                chunk_end = item["end_time"]
                result = item["result"]

                if not result["phones"]:
                    continue

                gap = round(chunk_start - prev_global_end, 3)
                if gap >= 0.001:
                    gap_seg = ("NULL", prev_global_end, chunk_start)
                    global_phones.append(gap_seg)
                    global_words.append(gap_seg)

                for seg in result["phones"]:
                    abs_start = round(chunk_start + seg.start, 3)
                    abs_end = round(chunk_start + seg.end, 3)
                    global_phones.append((seg.label, abs_start, abs_end))

                for seg in result["words"]:
                    abs_start = round(chunk_start + seg.start, 3)
                    abs_end = round(chunk_start + seg.end, 3)
                    global_words.append((seg.label, abs_start, abs_end))

                prev_global_end = global_phones[-1][2]

            final_valid_end = max(prev_global_end, global_phones[-1][2] if global_phones else 0.0)

            if full_duration > final_valid_end + 0.001:
                pad_seg = ("NULL", final_valid_end, full_duration)
                global_phones.append(pad_seg)
                global_words.append(pad_seg)
                final_valid_end = full_duration

            self._export_textgrid_file(
                output_path,
                final_valid_end,
                {"phones": global_phones, "words": global_words}
            )
    def forward_from_manifest(
        self,
        manifest_path: str,
        evidence_dir: Optional[str] = None,
        verbose: bool = True,
        max_batch_items: int = 32,
        max_batch_frames: int = 120000,
        sort_by_duration: bool = True
    ) -> str:
        """
        Stage 2: 从 chunks manifest 读取 chunk，批量运行 aligner 的声学前向，
        保存 log_probs，并生成 evidence manifest。

        Returns:
            evidence_manifest_path (str)
        """
        manifest_path = Path(manifest_path)
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        if evidence_dir is None:
            if manifest_path.parent.name == "chunks":
                evidence_root = manifest_path.parent.parent / "evidence"
            else:
                evidence_root = manifest_path.parent / "evidence"
        else:
            evidence_root = Path(evidence_dir)

        evidence_root.mkdir(parents=True, exist_ok=True)

        base_name = manifest_path.name.replace(".chunks.jsonl", "")
        evidence_manifest_path = evidence_root / f"{base_name}.evidence.jsonl"

        if verbose:
            print(f"⚙️ [Stage 2] Forward from manifest: {manifest_path.name}")
            print(f"   -> Evidence dir: {evidence_root}")
            print(
                f"   -> Batch policy: max_batch_items={max_batch_items}, "
                f"max_batch_frames={max_batch_frames}, "
                f"sort_by_duration={sort_by_duration}"
            )

        if self.aligner is None:
            self.aligner = LocalAligner(config=self.config_dict)

        records = self._load_chunk_records(str(manifest_path))

        # ---------------------------------------------------------
        # 1. 过滤掉音频不存在的 chunk
        # ---------------------------------------------------------
        valid_records = []
        for rec in records:
            audio_path = Path(rec.audio_path)
            if not audio_path.exists():
                if verbose:
                    print(f"❌ Chunk audio missing: {audio_path}")
                continue
            valid_records.append(rec)

        if not valid_records:
            raise RuntimeError(f"No valid chunk audio found for manifest: {manifest_path}")

        # ---------------------------------------------------------
        # 2. 构建 forward_batch 输入
        # ---------------------------------------------------------
        batch_inputs = [
            {
                "chunk_id": rec.chunk_id,
                "audio_path": rec.audio_path,
            }
            for rec in valid_records
        ]

        # ---------------------------------------------------------
        # 3. 批量前向
        # ---------------------------------------------------------
        try:
            evidences = self.aligner.forward_batch(
                batch_inputs,
                max_batch_items=max_batch_items,
                max_batch_frames=max_batch_frames,
                sort_by_duration=sort_by_duration
            )
        except Exception as e:
            raise RuntimeError(f"forward_batch failed for manifest {manifest_path}: {e}")

        if len(evidences) != len(valid_records):
            raise RuntimeError(
                f"forward_batch returned mismatched size: "
                f"{len(evidences)} vs expected {len(valid_records)}"
            )

        # ---------------------------------------------------------
        # 4. 保存证据文件并构建 evidence manifest
        # ---------------------------------------------------------
        evidence_records: List[ChunkEvidenceRecord] = []

        for rec, evidence in zip(valid_records, evidences):
            if evidence is None:
                if verbose:
                    print(f"❌ Empty evidence returned for {rec.chunk_id}")
                continue

            try:
                log_probs_path = evidence_root / f"{rec.chunk_id}.log_probs.pt"
                self.aligner.save_log_probs(evidence, str(log_probs_path))

                evidence_records.append(ChunkEvidenceRecord(
                    chunk_id=rec.chunk_id,
                    audio_path=rec.audio_path,
                    text=rec.text,
                    start_time=rec.start_time,
                    end_time=rec.end_time,
                    dur_s=rec.dur_s,
                    log_probs_path=str(log_probs_path),
                    num_frames=evidence.num_frames,
                    frame_hop_s=evidence.frame_hop_s,
                    vocab_size=evidence.vocab_size,
                ))
            except Exception as e:
                if verbose:
                    print(f"❌ Failed to save evidence for {rec.chunk_id}: {e}")

        if not evidence_records:
            raise RuntimeError(f"No evidence generated from manifest: {manifest_path}")

        self._save_evidence_records(evidence_records, str(evidence_manifest_path))

        if verbose:
            print(f"✅ Evidence manifest saved -> {evidence_manifest_path}")

        return str(evidence_manifest_path)
    
    def decode_from_evidence(
        self,
        evidence_manifest_path: str,
        output_path: str,
        full_audio_path: Optional[str] = None,
        full_duration: Optional[float] = None,
        verbose: bool = True
    ):
        """
        Stage 3: 从 evidence manifest 读取 log_probs + text，
        进行解码并缝合导出 TextGrid。
        """
        evidence_manifest_path = Path(evidence_manifest_path)
        if not evidence_manifest_path.exists():
            raise FileNotFoundError(f"Evidence manifest not found: {evidence_manifest_path}")

        if verbose:
            print(f"🧠 [Stage 3] Decode from evidence: {evidence_manifest_path.name}")

        if self.aligner is None:
            self.aligner = LocalAligner(config=self.config_dict, decode_only=True)

        target_duration = 0.0
        if full_duration is not None:
            target_duration = float(full_duration)
        elif full_audio_path and Path(full_audio_path).exists():
            target_duration = sf.info(full_audio_path).duration

        records = self._load_evidence_records(str(evidence_manifest_path))

        if target_duration == 0.0:
            try:
                target_duration = float(records[-1].end_time)
            except Exception:
                target_duration = 0.0

        decoded_chunks = []

        for rec in tqdm(records, total=len(records), desc="Decode", disable=not verbose):
            log_probs_path = Path(rec.log_probs_path)
            if not log_probs_path.exists():
                if verbose:
                    print(f"❌ Missing log_probs file: {log_probs_path}")
                continue

            try:
                evidence = self.aligner.load_log_probs(str(log_probs_path))
                result = self.aligner.decode_log_probs(
                    log_probs=evidence.log_probs,
                    text=rec.text,
                    file_id=rec.chunk_id,
                    dump_tsv=False
                )

                decoded_chunks.append({
                    "chunk_id": rec.chunk_id,
                    "start_time": rec.start_time,
                    "end_time": rec.end_time,
                    "result": result
                })
            except Exception as e:
                if verbose:
                    print(f"❌ Decode failed for {rec.chunk_id}: {e}")

        if not decoded_chunks:
            raise RuntimeError(f"No decoded chunks produced from evidence: {evidence_manifest_path}")

        self._stitch_decoded_results_and_export(
            decoded_chunks=decoded_chunks,
            full_duration=target_duration,
            output_path=output_path
        )

        if verbose:
            print(f"✅ Saved TextGrid -> {output_path}")
        
    def align_from_manifest(
        self,
        manifest_path: str,
        output_path: str,
        full_audio_path: Optional[str] = None,
        full_duration: Optional[float] = None,
        evidence_dir: Optional[str] = None,
        verbose: bool = True
    ):
        """
        从 Stage 1 的 chunks manifest 恢复：
        Stage 2: forward_from_manifest(...)
        Stage 3: decode_from_evidence(...)
        """
        manifest_path = Path(manifest_path)
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        if verbose:
            print(f"🧩 [Resume] Processing {manifest_path.name}")

        evidence_manifest_path = self.forward_from_manifest(
            manifest_path=str(manifest_path),
            evidence_dir=evidence_dir,
            verbose=verbose
        )

        self.decode_from_evidence(
            evidence_manifest_path=evidence_manifest_path,
            output_path=output_path,
            full_audio_path=full_audio_path,
            full_duration=full_duration,
            verbose=verbose
        )