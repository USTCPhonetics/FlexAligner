import sys
import os
import json
import gc
import shutil
from pathlib import Path

import torch
import soundfile as sf

# =====================================================================
# 1. 环境设置
# =====================================================================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"   # 按需修改

PROJECT_ROOT = Path("/home/wangyiming/projects/FlexAligner")
sys.path.append(str(PROJECT_ROOT / "src"))

from flexaligner.pipeline import FlexAligner
from flexaligner.config import AlignmentConfig
from flexaligner.chunker import CTCChunker
from flexaligner.frontend import TextFrontend
from flexaligner.aligner import LocalAligner

# =====================================================================
# 2. 固定测试样本
# =====================================================================
TARGET_TXT = Path("/mnt/hd/data_wangyiming/Buckeye_Evaluation/mfa_input_combined/s0101a.txt")
TARGET_WAV = Path("/mnt/hd/data_wangyiming/Buckeye_Evaluation/mfa_input_combined/s0101a.wav")

SEG_MODEL = PROJECT_ROOT / "models_hidden/en/hf_phs"
ALIGN_MODEL = PROJECT_ROOT / "models_hidden/en/ce17.6000"
DICT_PATH = PROJECT_ROOT / "assets/dictionaries/en.dict"

TEST_ROOT = PROJECT_ROOT / "tmp_full_pipeline_stage123_test"

# 约定目录
RUN_SPLIT = TEST_ROOT / "run_split_123"
RUN_RESUME23 = TEST_ROOT / "run_resume_23"
RUN_RESUME3 = TEST_ROOT / "run_resume_3"
RUN_FULL = TEST_ROOT / "run_full_123"

# =====================================================================
# 3. 基础工具
# =====================================================================
def assert_true(cond, msg):
    if not cond:
        raise AssertionError(msg)

def load_jsonl(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise RuntimeError(f"JSONL 解析失败: line={lineno}, file={path}, err={e}")
    return rows

def prepare_clean_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)

def cleanup_runtime(*objs):
    for obj in objs:
        try:
            del obj
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def assert_new_api_available():
    missing = []

    # pipeline 新接口
    for name in [
        "forward_from_manifest",
        "decode_from_evidence",
        "align_from_manifest",
    ]:
        if not hasattr(FlexAligner, name):
            missing.append(f"FlexAligner.{name}")

    # aligner 新接口
    for name in [
        "forward_chunk",
        "save_log_probs",
        "load_log_probs",
        "decode_log_probs",
    ]:
        if not hasattr(LocalAligner, name):
            missing.append(f"LocalAligner.{name}")

    if missing:
        raise RuntimeError(
            "检测到你本地代码还没有完成我们讨论的新接口改造，缺少：\n"
            + "\n".join(f"  - {x}" for x in missing)
        )

def validate_textgrid(path: Path):
    assert_true(path.exists(), f"TextGrid 不存在: {path}")
    assert_true(path.stat().st_size > 0, f"TextGrid 为空: {path}")
    content = path.read_text(encoding="utf-8", errors="ignore")
    assert_true('Object class = "TextGrid"' in content, f"不是有效 TextGrid: {path}")
    assert_true('IntervalTier' in content, f"TextGrid 缺少 tier: {path}")

def validate_chunks_manifest(manifest_path: Path):
    assert_true(manifest_path.exists(), f"chunks manifest 不存在: {manifest_path}")
    rows = load_jsonl(manifest_path)
    assert_true(len(rows) > 0, f"chunks manifest 为空: {manifest_path}")

    required = {"chunk_id", "audio", "start_s", "end_s", "dur_s", "words", "text"}
    for i, row in enumerate(rows):
        missing = required - set(row.keys())
        assert_true(not missing, f"chunks manifest 第 {i} 行缺字段: {sorted(missing)}")
        wav_path = Path(row["audio"])
        assert_true(wav_path.exists(), f"chunk wav 不存在: {wav_path}")
        wav, sr = sf.read(str(wav_path))
        assert_true(len(wav) > 0, f"chunk wav 为空: {wav_path}")
        assert_true(sr == 16000, f"chunk wav 采样率异常: {wav_path}, sr={sr}")
    return rows

def validate_evidence_manifest(evidence_manifest_path: Path):
    assert_true(evidence_manifest_path.exists(), f"evidence manifest 不存在: {evidence_manifest_path}")
    rows = load_jsonl(evidence_manifest_path)
    assert_true(len(rows) > 0, f"evidence manifest 为空: {evidence_manifest_path}")

    required = {
        "chunk_id", "audio", "text", "start_s", "end_s", "dur_s",
        "log_probs", "num_frames", "frame_hop_s", "vocab_size"
    }

    for i, row in enumerate(rows):
        missing = required - set(row.keys())
        assert_true(not missing, f"evidence manifest 第 {i} 行缺字段: {sorted(missing)}")

        lp_path = Path(row["log_probs"])
        assert_true(lp_path.exists(), f"log_probs 文件不存在: {lp_path}")

        payload = torch.load(lp_path, map_location="cpu")
        assert_true("log_probs" in payload, f"log_probs payload 缺少 log_probs: {lp_path}")

        lp = payload["log_probs"]
        if isinstance(lp, torch.Tensor):
            assert_true(lp.ndim == 2, f"log_probs 维度异常: {lp_path}, ndim={lp.ndim}")
            T, V = lp.shape
        else:
            raise AssertionError(f"log_probs payload 不是 tensor: {lp_path}")

        assert_true(T > 0 and V > 0, f"log_probs shape 异常: {lp_path}, shape={(T, V)}")

    return rows

def build_config(chunks_out_dir: Path) -> AlignmentConfig:
    return AlignmentConfig(
        lang="en",
        device="cuda:0",
        chunk_model_path=str(SEG_MODEL),
        align_model_path=str(ALIGN_MODEL),
        lexicon_path=str(DICT_PATH),

        # chunk 阶段参数
        beam_size=400,

        # decode 阶段参数
        align_beam_size=400,
        sil_cost=-8,
        sil_enter_cost=-0.5,
        min_sil_dur_ms=50.0,
        sil_at_ends=True,
        optional_sil=True,
        sil_phone="sil",
        p_stay=0.92,
        frame_hop_s=0.01,
        boundary_lambda=200.0,

        chunks_out_dir=str(chunks_out_dir),
        verbose=False,
    )

def preprocess_for_stage1(config: AlignmentConfig):
    config_dict = config.__dict__.copy()
    frontend = TextFrontend(config=config_dict, mode="FAST")

    audio_np = frontend.load_audio(str(TARGET_WAV))
    raw_text = frontend.load_text(str(TARGET_TXT))
    lang = config.lang if config.lang else frontend.detect_language(raw_text)
    tokens = frontend.get_phonemes(raw_text, lang)
    text_list = [t.strip() for t in tokens if t.strip()]
    audio_tensor = torch.from_numpy(audio_np).float()
    full_dur = audio_tensor.size(0) / 16000.0
    return audio_tensor, text_list, full_dur

# =====================================================================
# 4. 阶段执行函数
# =====================================================================
def run_stage1_chunk_only(run_root: Path):
    print("\n" + "-" * 80)
    print("[CASE A] Stage 1 only: chunk -> manifest")
    print("-" * 80)

    chunks_dir = run_root / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    config = build_config(chunks_dir)
    audio_tensor, text_list, full_dur = preprocess_for_stage1(config)

    file_id = TARGET_WAV.stem
    manifest_path = chunks_dir / f"{file_id}.chunks.jsonl"

    chunker = CTCChunker(config=config.__dict__.copy())
    with torch.inference_mode():
        final_chunks = chunker.find_chunks(audio_tensor, text_list, file_id=file_id)

    assert_true(len(final_chunks) > 0, "Stage 1 未返回任何 chunk")
    rows = validate_chunks_manifest(manifest_path)
    assert_true(len(rows) == len(final_chunks),
                f"manifest 行数与 final_chunks 数量不一致: {len(rows)} vs {len(final_chunks)}")

    print(f"✅ Stage 1 成功: {manifest_path}")
    cleanup_runtime(chunker)
    return manifest_path, full_dur

def run_stage2_forward_only(run_root: Path, manifest_path: Path):
    print("\n" + "-" * 80)
    print("[CASE B] Stage 2 only: manifest -> evidence")
    print("-" * 80)

    chunks_dir = run_root / "chunks"
    evidence_dir = run_root / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)

    config = build_config(chunks_dir)
    aligner = FlexAligner(config)
    aligner.config_dict["verbose"] = False

    with torch.inference_mode():
        evidence_manifest_path = aligner.forward_from_manifest(
            manifest_path=str(manifest_path),
            evidence_dir=str(evidence_dir),
            verbose=False,
        )

    evidence_rows = validate_evidence_manifest(Path(evidence_manifest_path))
    print(f"✅ Stage 2 成功: {evidence_manifest_path} ({len(evidence_rows)} chunks)")
    cleanup_runtime(aligner)
    return Path(evidence_manifest_path)

def run_stage3_decode_only(run_root: Path, evidence_manifest_path: Path, full_dur: float, tag: str):
    print("\n" + "-" * 80)
    print(f"[CASE C] Stage 3 only: evidence -> TextGrid ({tag})")
    print("-" * 80)

    chunks_dir = run_root / "chunks"
    decode_dir = run_root / "decode"
    decode_dir.mkdir(parents=True, exist_ok=True)

    config = build_config(chunks_dir)
    aligner = FlexAligner(config)
    aligner.config_dict["verbose"] = False

    out_tg = decode_dir / f"{TARGET_WAV.stem}.{tag}.TextGrid"

    with torch.inference_mode():
        aligner.decode_from_evidence(
            evidence_manifest_path=str(evidence_manifest_path),
            output_path=str(out_tg),
            full_duration=full_dur,
            verbose=False,
        )

    validate_textgrid(out_tg)
    print(f"✅ Stage 3 成功: {out_tg}")
    cleanup_runtime(aligner)
    return out_tg

def run_resume_23(run_root: Path, manifest_path: Path, full_dur: float):
    print("\n" + "=" * 80)
    print("[TEST 2] 恢复路径：2 -> 3（从 chunks manifest 恢复）")
    print("=" * 80)

    chunks_dir = run_root / "chunks"
    evidence_dir = run_root / "evidence"
    decode_dir = run_root / "decode"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    decode_dir.mkdir(parents=True, exist_ok=True)

    # 模拟进程重启
    cleanup_runtime()

    config = build_config(chunks_dir)
    aligner = FlexAligner(config)
    aligner.config_dict["verbose"] = False

    out_tg = decode_dir / f"{TARGET_WAV.stem}.resume23.TextGrid"

    with torch.inference_mode():
        aligner.align_from_manifest(
            manifest_path=str(manifest_path),
            output_path=str(out_tg),
            full_duration=full_dur,
            evidence_dir=str(evidence_dir),
            verbose=False,
        )

    evidence_manifest_path = evidence_dir / f"{TARGET_WAV.stem}.evidence.jsonl"
    validate_chunks_manifest(manifest_path)
    validate_evidence_manifest(evidence_manifest_path)
    validate_textgrid(out_tg)

    print(f"✅ 2->3 恢复成功: {out_tg}")
    cleanup_runtime(aligner)
    return evidence_manifest_path, out_tg

def run_resume_3(run_root: Path, evidence_manifest_path: Path, full_dur: float):
    print("\n" + "=" * 80)
    print("[TEST 3] 恢复路径：3（从 evidence 直接恢复）")
    print("=" * 80)

    chunks_dir = run_root / "chunks"
    decode_dir = run_root / "decode"
    decode_dir.mkdir(parents=True, exist_ok=True)

    # 模拟进程重启
    cleanup_runtime()

    config = build_config(chunks_dir)
    aligner = FlexAligner(config)
    aligner.config_dict["verbose"] = False

    out_tg = decode_dir / f"{TARGET_WAV.stem}.resume3.TextGrid"

    with torch.inference_mode():
        aligner.decode_from_evidence(
            evidence_manifest_path=str(evidence_manifest_path),
            output_path=str(out_tg),
            full_duration=full_dur,
            verbose=False,
        )

    validate_evidence_manifest(evidence_manifest_path)
    validate_textgrid(out_tg)

    print(f"✅ 3 恢复成功: {out_tg}")
    cleanup_runtime(aligner)
    return out_tg

def run_full_123_via_align_batch(run_root: Path):
    print("\n" + "=" * 80)
    print("[TEST 4] 整体路径：1 -> 2 -> 3（align_batch 一把跑完）")
    print("=" * 80)

    chunks_dir = run_root / "chunks"
    decode_dir = run_root / "decode"
    evidence_dir = run_root / "evidence"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    decode_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)

    config = build_config(chunks_dir)
    aligner = FlexAligner(config)
    aligner.config_dict["verbose"] = False

    out_tg = decode_dir / f"{TARGET_WAV.stem}.full.TextGrid"

    tasks = [(str(TARGET_WAV), str(TARGET_TXT), str(out_tg))]
    with torch.inference_mode():
        aligner.align_batch(tasks, raise_on_error=True)

    # 按我们前面设计的三阶段目录约定检查
    manifest_path = chunks_dir / f"{TARGET_WAV.stem}.chunks.jsonl"
    evidence_manifest_path = evidence_dir / f"{TARGET_WAV.stem}.evidence.jsonl"

    validate_chunks_manifest(manifest_path)
    validate_evidence_manifest(evidence_manifest_path)
    validate_textgrid(out_tg)

    print(f"✅ 整体 1->2->3 成功: {out_tg}")
    cleanup_runtime(aligner)
    return manifest_path, evidence_manifest_path, out_tg

# =====================================================================
# 5. 主测试逻辑
# =====================================================================
def main():
    print("=" * 80)
    print("🧪 Full Pipeline Test: chunk / evidence / decode / resume")
    print("=" * 80)

    assert_true(TARGET_WAV.exists(), f"测试音频不存在: {TARGET_WAV}")
    assert_true(TARGET_TXT.exists(), f"测试文本不存在: {TARGET_TXT}")

    assert_new_api_available()

    prepare_clean_dir(TEST_ROOT)
    RUN_SPLIT.mkdir(parents=True, exist_ok=True)
    RUN_RESUME23.mkdir(parents=True, exist_ok=True)
    RUN_RESUME3.mkdir(parents=True, exist_ok=True)
    RUN_FULL.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------
    # TEST 1: 显式分阶段 1 -> 2 -> 3
    # -----------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[TEST 1] 分阶段路径：1 -> 2 -> 3")
    print("=" * 80)

    manifest_path, full_dur = run_stage1_chunk_only(RUN_SPLIT)
    # 模拟清空旧状态，只保留落盘结果
    cleanup_runtime()

    evidence_manifest_path = run_stage2_forward_only(RUN_SPLIT, manifest_path)
    # 再次清空旧状态，只保留落盘结果
    cleanup_runtime()

    tg_split = run_stage3_decode_only(RUN_SPLIT, evidence_manifest_path, full_dur, tag="split123")

    # -----------------------------------------------------------------
    # TEST 2: 恢复路径 2 -> 3
    # -----------------------------------------------------------------
    evidence_manifest_23, tg_23 = run_resume_23(RUN_RESUME23, manifest_path, full_dur)

    # -----------------------------------------------------------------
    # TEST 3: 恢复路径 3
    # -----------------------------------------------------------------
    tg_3 = run_resume_3(RUN_RESUME3, evidence_manifest_path, full_dur)

    # -----------------------------------------------------------------
    # TEST 4: 整体一把跑完 1 -> 2 -> 3
    # -----------------------------------------------------------------
    manifest_full, evidence_full, tg_full = run_full_123_via_align_batch(RUN_FULL)

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------
    print("\n" + "=" * 80)
    print("✅ 全部测试通过")
    print("=" * 80)
    print("结果摘要：")
    print(f"  [1->2->3 split] chunks manifest : {manifest_path}")
    print(f"  [1->2->3 split] evidence       : {evidence_manifest_path}")
    print(f"  [1->2->3 split] textgrid       : {tg_split}")
    print(f"  [2->3 resume] evidence         : {evidence_manifest_23}")
    print(f"  [2->3 resume] textgrid         : {tg_23}")
    print(f"  [3 resume] textgrid            : {tg_3}")
    print(f"  [full 1->2->3] manifest        : {manifest_full}")
    print(f"  [full 1->2->3] evidence        : {evidence_full}")
    print(f"  [full 1->2->3] textgrid        : {tg_full}")
    print(f"\n📂 测试根目录: {TEST_ROOT}")

    cleanup_runtime()

if __name__ == "__main__":
    main()