import sys
import os
import json
import gc
import shutil
from pathlib import Path

import torch
import soundfile as sf

# =====================================================================
# 1. 核心定址与环境防御
# =====================================================================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"  # 锁定单卡测试

PROJECT_ROOT = Path("/home/wangyiming/projects/FlexAligner")
sys.path.append(str(PROJECT_ROOT / "src"))

from flexaligner.pipeline import FlexAligner
from flexaligner.config import AlignmentConfig
from flexaligner.chunker import CTCChunker
from flexaligner.frontend import TextFrontend


# =====================================================================
# 2. 固定测试样本
# =====================================================================
TARGET_TXT = Path("/mnt/hd/data_wangyiming/Buckeye_Evaluation/mfa_input_combined/s0101a.txt")
TARGET_WAV = Path("/mnt/hd/data_wangyiming/Buckeye_Evaluation/mfa_input_combined/s0101a.wav")

# 输出目录
TEST_ROOT = PROJECT_ROOT / "tmp_stage1_contract_test"
CHUNKS_OUT_DIR = TEST_ROOT / "chunks_out"
ALIGN_OUT_DIR = TEST_ROOT / "align_out"
ALIGN_OUT_DIR.mkdir(parents=True, exist_ok=True)

# 模型与资产路径
SEG_MODEL = PROJECT_ROOT / "models_hidden/en/hf_phs"
ALIGN_MODEL = PROJECT_ROOT / "models_hidden/en/ce17.6000"
DICT_PATH = PROJECT_ROOT / "assets/dictionaries/en.dict"


# =====================================================================
# 3. 配置：沿用你示例中的参数风格
# =====================================================================
def build_config():
    return AlignmentConfig(
        lang="en",
        device="cuda:0",
        chunk_model_path=str(SEG_MODEL),
        align_model_path=str(ALIGN_MODEL),
        lexicon_path=str(DICT_PATH),

        # --- 你示例中的参数 ---
        sil_cost=-8,
        sil_enter_cost=-0.5,
        min_sil_dur_ms=50.0,
        sil_at_ends=True,
        optional_sil=True,
        sil_phone="sil",
        beam_size=400,
        p_stay=0.92,
        frame_hop_s=0.01,
        boundary_lambda=200.0,

        # --- Stage 1 契约测试关键 ---
        chunks_out_dir=str(CHUNKS_OUT_DIR),
        verbose=False,
    )


# =====================================================================
# 4. 基础工具
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


# =====================================================================
# 5. 测试主逻辑
# =====================================================================
def main():
    print("=" * 80)
    print("🧪 Stage 1 Manifest Contract Test")
    print("=" * 80)

    assert_true(TARGET_WAV.exists(), f"测试音频不存在: {TARGET_WAV}")
    assert_true(TARGET_TXT.exists(), f"测试文本不存在: {TARGET_TXT}")

    prepare_clean_dir(TEST_ROOT)
    CHUNKS_OUT_DIR.mkdir(parents=True, exist_ok=True)
    ALIGN_OUT_DIR.mkdir(parents=True, exist_ok=True)

    config = build_config()
    config_dict = config.__dict__.copy()

    # ---------------------------------------------------------
    # A. 预处理：得到 Stage 1 输入
    # ---------------------------------------------------------
    print("\n[A] 前端预处理 ...")
    frontend = TextFrontend(config=config_dict, mode="FAST")

    audio_np = frontend.load_audio(str(TARGET_WAV))
    raw_text = frontend.load_text(str(TARGET_TXT))
    lang = config.lang if config.lang else frontend.detect_language(raw_text)
    tokens = frontend.get_phonemes(raw_text, lang)
    text_list = [t.strip() for t in tokens if t.strip()]
    audio_tensor = torch.from_numpy(audio_np).float()

    file_id = TARGET_WAV.stem
    manifest_path = CHUNKS_OUT_DIR / f"{file_id}.chunks.jsonl"
    tsv_path = CHUNKS_OUT_DIR / f"{file_id}.chunks.tsv"

    print(f"  - file_id = {file_id}")
    print(f"  - token_count = {len(text_list)}")
    print(f"  - expected_manifest = {manifest_path}")

    # ---------------------------------------------------------
    # B. Stage 1：直接跑 chunker.find_chunks()
    # ---------------------------------------------------------
    print("\n[B] 执行 Stage 1 切分与保存 ...")
    chunker = CTCChunker(config=config_dict)

    with torch.inference_mode():
        final_chunks = chunker.find_chunks(audio_tensor, text_list, file_id=file_id)

    assert_true(len(final_chunks) > 0, "Stage 1 未返回任何 chunk")
    assert_true(manifest_path.exists(), f"JSONL manifest 未生成: {manifest_path}")
    assert_true(tsv_path.exists(), f"TSV 未生成: {tsv_path}")

    print(f"  - final_chunks = {len(final_chunks)}")
    print(f"  - manifest exists = True")

    # ---------------------------------------------------------
    # C. 校验 JSONL 是否满足新契约
    # ---------------------------------------------------------
    print("\n[C] 校验 JSONL manifest 契约 ...")
    rows = load_jsonl(manifest_path)
    assert_true(len(rows) == len(final_chunks),
                f"manifest 行数与最终 chunks 数不一致: {len(rows)} vs {len(final_chunks)}")

    required_fields = {"chunk_id", "audio", "start_s", "end_s", "dur_s", "words", "text"}
    for i, row in enumerate(rows):
        missing = required_fields - set(row.keys())
        assert_true(not missing, f"第 {i} 行缺少字段: {sorted(missing)}")

    # ---------------------------------------------------------
    # D. 校验 manifest 与内存返回的 final_chunks 一致
    # ---------------------------------------------------------
    print("\n[D] 校验 manifest 与 find_chunks() 返回值一致 ...")
    for i, (row, chunk) in enumerate(zip(rows, final_chunks)):
        row_audio = Path(row["audio"])

        assert_true(row["chunk_id"] == chunk.chunk_id,
                    f"[{i}] chunk_id 不一致: {row['chunk_id']} vs {chunk.chunk_id}")

        assert_true(abs(float(row["start_s"]) - round(float(chunk.start_time), 3)) < 1e-6,
                    f"[{i}] start_s 不一致: {row['start_s']} vs {round(float(chunk.start_time), 3)}")

        assert_true(abs(float(row["end_s"]) - round(float(chunk.end_time), 3)) < 1e-6,
                    f"[{i}] end_s 不一致: {row['end_s']} vs {round(float(chunk.end_time), 3)}")

        assert_true(abs(float(row["dur_s"]) - round(float(chunk.end_time - chunk.start_time), 3)) < 1e-6,
                    f"[{i}] dur_s 不一致")

        assert_true(str(row["text"]).strip() == str(chunk.text).strip(),
                    f"[{i}] text 不一致:\nmanifest={row['text']}\nchunk={chunk.text}")

        assert_true(row_audio.exists(), f"[{i}] audio 文件不存在: {row_audio}")

        wav, sr = sf.read(str(row_audio))
        assert_true(sr == 16000, f"[{i}] chunk wav 采样率异常: {sr}")
        assert_true(len(wav) > 0, f"[{i}] chunk wav 为空: {row_audio}")

    print("  - manifest 与最终 returned chunks 一致 ✅")

    # ---------------------------------------------------------
    # E. 通过 FlexAligner 新接口验证恢复逻辑
    # ---------------------------------------------------------
    print("\n[E] 验证 _load_chunk_records() / _records_to_tasks() ...")
    aligner = FlexAligner(config)
    aligner.config_dict["verbose"] = False

    records = aligner._load_chunk_records(str(manifest_path))
    assert_true(len(records) == len(rows),
                f"_load_chunk_records() 数量异常: {len(records)} vs {len(rows)}")

    task_chunks = aligner._records_to_tasks(records, verbose=False)
    assert_true(len(task_chunks) == len(rows),
                f"_records_to_tasks() 数量异常: {len(task_chunks)} vs {len(rows)}")

    for i, (rec, task) in enumerate(zip(records, task_chunks)):
        assert_true(rec.chunk_id == task.chunk_id, f"[{i}] record/task chunk_id 不一致")
        assert_true(abs(float(rec.start_time) - float(task.start_time)) < 1e-6,
                    f"[{i}] record/task start_time 不一致")
        assert_true(abs(float(rec.end_time) - float(task.end_time)) < 1e-6,
                    f"[{i}] record/task end_time 不一致")
        assert_true(str(rec.text).strip() == str(task.text).strip(),
                    f"[{i}] record/task text 不一致")
        assert_true(task.tensor is not None, f"[{i}] task.tensor 为空")
        assert_true(task.tensor.numel() > 0, f"[{i}] task.tensor 长度为 0")

    print("  - manifest -> records -> tasks 恢复链路闭合 ✅")

    # ---------------------------------------------------------
    # F. 可选：继续验证 align_from_manifest() 是否能跑通
    # ---------------------------------------------------------
    print("\n[F] 可选验证 align_from_manifest() 恢复入口 ...")
    textgrid_out = ALIGN_OUT_DIR / f"{file_id}.from_manifest.TextGrid"

    with torch.inference_mode():
        aligner.align_from_manifest(
            manifest_path=str(manifest_path),
            output_path=str(textgrid_out),
            full_audio_path=str(TARGET_WAV),
            verbose=False,
        )

    assert_true(textgrid_out.exists(), f"align_from_manifest 未生成输出: {textgrid_out}")
    assert_true(textgrid_out.stat().st_size > 0, f"输出 TextGrid 为空: {textgrid_out}")

    print(f"  - align_from_manifest 输出存在 ✅ -> {textgrid_out}")

    # ---------------------------------------------------------
    # G. 汇总结论
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("✅ 测试通过：当前 Stage 1 新契约符合设计逻辑")
    print("=" * 80)
    print("验证结论：")
    print("1. Stage 1 最终返回的 final_chunks 已成功落盘为 JSONL manifest")
    print("2. manifest 是可恢复的单一事实来源（包含 audio/text/time）")
    print("3. _load_chunk_records() / _records_to_tasks() 可以正确恢复 Stage 2 输入")
    print("4. align_from_manifest() 可以基于 manifest 独立完成恢复执行")
    print(f"\n📂 测试目录: {TEST_ROOT}")

    # 清理显存
    del chunker
    del aligner
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()