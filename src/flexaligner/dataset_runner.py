from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional
import json

@dataclass
class DatasetItem:
    utt_id: str
    audio_path: str
    text_path: str

@dataclass
class ShardSpec:
    shard_id: int
    manifest_path: str
    num_items: int
def load_dataset_manifest(path: str) -> List[DatasetItem]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset manifest not found: {path}")

    items = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            if "utt_id" not in obj or "audio_path" not in obj or "text_path" not in obj:
                raise RuntimeError(
                    f"Dataset manifest line {lineno} missing one of "
                    f"['utt_id', 'audio_path', 'text_path']"
                )

            items.append(DatasetItem(
                utt_id=str(obj["utt_id"]),
                audio_path=str(obj["audio_path"]),
                text_path=str(obj["text_path"]),
            ))
    return items


def save_dataset_manifest(items: List[DatasetItem], path: str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")
    
def build_shards(
    dataset_manifest_path: str,
    shard_size: int,
    shard_dir: str
) -> List[ShardSpec]:
    items = load_dataset_manifest(dataset_manifest_path)
    shard_dir = Path(shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)

    if shard_size <= 0:
        raise ValueError(f"shard_size must be > 0, got {shard_size}")

    shards = []
    for shard_id, start in enumerate(range(0, len(items), shard_size)):
        chunk = items[start:start + shard_size]
        manifest_path = shard_dir / f"shard_{shard_id:05d}.jsonl"
        save_dataset_manifest(chunk, str(manifest_path))
        shards.append(ShardSpec(
            shard_id=shard_id,
            manifest_path=str(manifest_path),
            num_items=len(chunk),
        ))
    return shards
import gc
import torch
from pathlib import Path
from typing import List, Optional

from .config import AlignmentConfig
from .pipeline import FlexAligner

class DatasetRunner:
    def __init__(self, config: AlignmentConfig):
        self.config = config
        self.run_root = Path(config.run_root or "runs/default")
        self.dataset_dir = self.run_root / "dataset"
        self.shard_dir = self.dataset_dir / "shards"
        self.stage1_root = self.run_root / "stage1"
        self.stage2_root = self.run_root / "stage2"
        self.stage3_root = self.run_root / "stage3" / config.decode_tag
        self.status_root = self.run_root / "status"

        for p in [
            self.dataset_dir, self.shard_dir,
            self.stage1_root, self.stage2_root,
            self.stage3_root, self.status_root
        ]:
            p.mkdir(parents=True, exist_ok=True)

    def _cleanup(self, *objs):
        for obj in objs:
            try:
                del obj
            except Exception:
                pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def chunk_shard(self, shard_manifest_path: str, overwrite: bool = False):
        shard_manifest_path = Path(shard_manifest_path)
        shard_name = shard_manifest_path.stem
        done_flag = self.status_root / f"stage1_{shard_name}.done"

        if done_flag.exists() and not overwrite:
            print(f"[stage1] skip {shard_name} (done)")
            return

        items = load_dataset_manifest(str(shard_manifest_path))
        out_dir = self.stage1_root / shard_name / "chunks"
        out_dir.mkdir(parents=True, exist_ok=True)

        cfg = AlignmentConfig(**{**self.config.__dict__, "chunks_out_dir": str(out_dir)})
        fa = FlexAligner(cfg)

        tasks = []
        for item in items:
            # Stage 1 这里只需要让 pipeline 产出 chunks manifest
            # output_path 先给占位，后面 stage3 再真正输出 TextGrid
            dummy_out = str((self.stage3_root / shard_name / f"{item.utt_id}.TextGrid"))
            tasks.append((item.audio_path, item.text_path, dummy_out))

        # 这里仍然会走 1->2->3，所以这一步我们后面要再拆
        # 当前先不要直接调用 align_batch
        # 改为逐文件仅触发 chunk 阶段，见下一个最小实现
        for item in items:
            audio_path = item.audio_path
            text_path = item.text_path

            audio_np = fa.frontend.load_audio(audio_path)
            raw_text = fa.frontend.load_text(text_path)
            lang = cfg.lang if cfg.lang else fa.frontend.detect_language(raw_text)
            tokens = fa.frontend.get_phonemes(raw_text, lang)
            text_list = [t.strip() for t in tokens if t.strip()]
            audio_tensor = torch.from_numpy(audio_np).float()

            if fa.chunker is None:
                fa.chunker = fa.chunker or __import__("flexaligner.chunker", fromlist=["CTCChunker"]).CTCChunker(config=fa.config_dict)

            _ = fa.chunker.find_chunks(audio_tensor, text_list, file_id=item.utt_id)

        done_flag.write_text("done", encoding="utf-8")
        self._cleanup(fa)

    def chunk_shard(self, shard_manifest_path: str, overwrite: bool = False):
        shard_manifest_path = Path(shard_manifest_path)
        shard_name = shard_manifest_path.stem
        done_flag = self.status_root / f"stage1_{shard_name}.done"

        if done_flag.exists() and not overwrite:
            print(f"[stage1] skip {shard_name} (done)")
            return

        items = load_dataset_manifest(str(shard_manifest_path))
        out_dir = self.stage1_root / shard_name / "chunks"
        out_dir.mkdir(parents=True, exist_ok=True)

        cfg = AlignmentConfig(**{**self.config.__dict__, "chunks_out_dir": str(out_dir)})
        fa = FlexAligner(cfg)

        tasks = []
        for item in items:
            # Stage 1 这里只需要让 pipeline 产出 chunks manifest
            # output_path 先给占位，后面 stage3 再真正输出 TextGrid
            dummy_out = str((self.stage3_root / shard_name / f"{item.utt_id}.TextGrid"))
            tasks.append((item.audio_path, item.text_path, dummy_out))

        # 这里仍然会走 1->2->3，所以这一步我们后面要再拆
        # 当前先不要直接调用 align_batch
        # 改为逐文件仅触发 chunk 阶段，见下一个最小实现
        for item in items:
            audio_path = item.audio_path
            text_path = item.text_path

            audio_np = fa.frontend.load_audio(audio_path)
            raw_text = fa.frontend.load_text(text_path)
            lang = cfg.lang if cfg.lang else fa.frontend.detect_language(raw_text)
            tokens = fa.frontend.get_phonemes(raw_text, lang)
            text_list = [t.strip() for t in tokens if t.strip()]
            audio_tensor = torch.from_numpy(audio_np).float()

            if fa.chunker is None:
                fa.chunker = fa.chunker or __import__("flexaligner.chunker", fromlist=["CTCChunker"]).CTCChunker(config=fa.config_dict)

            _ = fa.chunker.find_chunks(audio_tensor, text_list, file_id=item.utt_id)

        done_flag.write_text("done", encoding="utf-8")
        self._cleanup(fa)
    def forward_shard(self, shard_manifest_path: str, overwrite: bool = False):
        shard_manifest_path = Path(shard_manifest_path)
        shard_name = shard_manifest_path.stem
        done_flag = self.status_root / f"stage2_{shard_name}.done"

        if done_flag.exists() and not overwrite:
            print(f"[stage2] skip {shard_name} (done)")
            return

        items = load_dataset_manifest(str(shard_manifest_path))

        chunks_dir = self.stage1_root / shard_name / "chunks"
        evidence_dir = self.stage2_root / shard_name / "evidence"
        evidence_dir.mkdir(parents=True, exist_ok=True)

        cfg = AlignmentConfig(**self.config.__dict__)
        fa = FlexAligner(cfg)

        for item in items:
            manifest_path = chunks_dir / f"{item.utt_id}.chunks.jsonl"
            if not manifest_path.exists():
                print(f"[stage2] missing chunks manifest: {manifest_path}")
                continue

            fa.forward_from_manifest(
                manifest_path=str(manifest_path),
                evidence_dir=str(evidence_dir),
                verbose=False,
                max_batch_items=cfg.stage2_max_batch_items,
                max_batch_frames=cfg.stage2_max_batch_frames,
                sort_by_duration=cfg.stage2_sort_by_duration,
            )

        done_flag.write_text("done", encoding="utf-8")
        self._cleanup(fa)
    def decode_shard(self, shard_manifest_path: str, overwrite: bool = False):
        shard_manifest_path = Path(shard_manifest_path)
        shard_name = shard_manifest_path.stem
        done_flag = self.status_root / f"stage3_{self.config.decode_tag}_{shard_name}.done"

        if done_flag.exists() and not overwrite:
            print(f"[stage3] skip {shard_name} (done)")
            return

        items = load_dataset_manifest(str(shard_manifest_path))

        evidence_dir = self.stage2_root / shard_name / "evidence"
        out_dir = self.stage3_root / shard_name
        out_dir.mkdir(parents=True, exist_ok=True)

        cfg = AlignmentConfig(**self.config.__dict__)
        fa = FlexAligner(cfg)

        for item in items:
            evidence_manifest_path = evidence_dir / f"{item.utt_id}.evidence.jsonl"
            if not evidence_manifest_path.exists():
                print(f"[stage3] missing evidence manifest: {evidence_manifest_path}")
                continue

            out_tg = out_dir / f"{item.utt_id}.TextGrid"
            fa.decode_from_evidence(
                evidence_manifest_path=str(evidence_manifest_path),
                output_path=str(out_tg),
                full_audio_path=item.audio_path,
                verbose=False,
            )

        done_flag.write_text("done", encoding="utf-8")
        self._cleanup(fa)
    def prepare_dataset(self, dataset_manifest_path: str):
        dataset_manifest_path = Path(dataset_manifest_path)
        target = self.dataset_dir / "all.jsonl"
        target.write_text(dataset_manifest_path.read_text(encoding="utf-8"), encoding="utf-8")

        shards = build_shards(
            dataset_manifest_path=str(target),
            shard_size=self.config.shard_size,
            shard_dir=str(self.shard_dir),
        )
        return shards

    def run_stage1_all(self):
        for shard_manifest in sorted(self.shard_dir.glob("shard_*.jsonl")):
            self.chunk_shard(str(shard_manifest))

    def run_stage2_all(self):
        for shard_manifest in sorted(self.shard_dir.glob("shard_*.jsonl")):
            self.forward_shard(str(shard_manifest))

    def run_stage3_all(self):
        for shard_manifest in sorted(self.shard_dir.glob("shard_*.jsonl")):
            self.decode_shard(str(shard_manifest))

    def run_all(self):
        self.run_stage1_all()
        self.run_stage2_all()
        self.run_stage3_all()