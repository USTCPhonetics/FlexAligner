import pytest
import torch
import os
from pathlib import Path
from praatio import textgrid
from flexaligner.pipeline import FlexAligner
from flexaligner.config import AlignmentConfig
from dataclasses import asdict

# --- 环境配置 ---
EXAMPLES_DIR = Path(__file__).parent.parent / "assets" / "examples"
WAV_PATH = EXAMPLES_DIR / "en.flac" if (EXAMPLES_DIR / "en.flac").exists() else EXAMPLES_DIR / "en.wav"
TXT_PATH = EXAMPLES_DIR / "en.txt"
REF_TG_PATH = EXAMPLES_DIR / "en.TextGrid"

@pytest.fixture(scope="module")
def aligner():
    """初始化英语对齐器单例，节省模型加载时间"""
    config = AlignmentConfig(
        lang="en",
        frame_hop_s=0.01, # 适配 Stride 1 修改版
        device="cpu"      # 测试环境建议 CPU
    )
    return FlexAligner(config=asdict(config))

def get_tier_entries(tg_path, tier_name="words"):
    """兼容不同版本 praatio 的层级提取"""
    tg = textgrid.openTextgrid(str(tg_path), includeEmptyIntervals=False)
    # 属性探测适配
    t_names = tg.tierNames if hasattr(tg, 'tierNames') else list(tg.tierDict.keys())
    target = next((n for n in t_names if n.lower() == tier_name.lower()), t_names[0])
    return tg.getTier(target).entries

# ==========================================
#  TEST CASE 1: 物理步长与边界完整性
# ==========================================
def test_physical_integrity(aligner, tmp_path):
    """验证 10ms 模型输出的时间戳是否越界，以及是否覆盖全长"""
    out_tg = tmp_path / "integrity.TextGrid"
    aligner.align(str(WAV_PATH), str(TXT_PATH), str(out_tg))
    
    # 获取音频物理长度
    import soundfile as sf
    info = sf.info(WAV_PATH)
    duration = info.duration
    
    tg = textgrid.openTextgrid(str(out_tg), includeEmptyIntervals=False)
    assert abs(tg.maxTimestamp - duration) < 0.02, f"TextGrid 总时长与音频不符: {tg.maxTimestamp} vs {duration}"

# ==========================================
#  TEST CASE 2: OOV 与符号清洗鲁棒性
# ==========================================
def test_symbol_robustness(aligner, tmp_path):
    """测试特殊符号（如 en.txt 里的 %）是否会被前端正确处理而不干扰对齐"""
    out_tg = tmp_path / "robustness.TextGrid"
    
    # 验证 en.txt 内容
    with open(TXT_PATH, 'r') as f:
        content = f.read()
    
    # 即使有 %，对齐也不应崩溃
    aligner.align(str(WAV_PATH), str(TXT_PATH), str(out_tg))
    
    entries = get_tier_entries(out_tg, "words")
    labels = [e.label for e in entries if e.label != "NULL"]
    
    # 期望结果中不应包含原始文本里的干扰符号
    for lab in labels:
        assert "%" not in lab, f"符号清洗失败，结果中仍含有 %: {lab}"
    assert "montreal" in labels, "核心词汇 montreal 丢失"


#  TEST CASE 4: 模型路由验证
# ==========================================
def test_model_routing_correctness(aligner):
    """确保真的加载了英语模型而非默认的中文模型"""
    # 检查 chunker 内部的 vocab 映射
    # 英语模型通常包含 'A', 'B' 等字符，而中文模型（拼音版）全是 a, o, e
    vocab = aligner.chunker.phone_to_id
    # 这是一个典型的英文 Arpabet 音素
    assert "AA1" in vocab or "M" in vocab, "模型 Vocab 中未发现英文音素，可能路由到了错误模型"