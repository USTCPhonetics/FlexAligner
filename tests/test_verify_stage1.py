import pytest
import torch
import soundfile as sf
from pathlib import Path
from dataclasses import asdict

from flexaligner.chunker import CTCChunker
from flexaligner.config import AlignmentConfig
from flexaligner.pipeline import FlexAligner

# ==========================================
# 0. 配置 (指向你本地已有的模型和数据)
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# MODEL_PATH = PROJECT_ROOT / "Toolkit/models/hf_phs"
TEST_WAV = PROJECT_ROOT / "Toolkit/test/SP01_001.wav"
TEST_TXT = PROJECT_ROOT / "Toolkit/test/SP01_001.txt"

@pytest.fixture
def base_config():
    return AlignmentConfig(
        lang="zh",
        device="cpu",
        # chunk_model_path=str(MODEL_PATH),
        max_gap_s=0.35,
        pad_s=0.15
    )

# ==========================================
# 1. 核心逻辑测试
# ==========================================

def test_chunker_object_consistency(base_config):
    """
    验证 Chunker 产生的对象是否符合 Pipeline 预期的 Schema
    """
    chunker = CTCChunker(asdict(base_config))
    
    # 准备输入
    with open(TEST_TXT, "r", encoding="utf-8") as f:
        text_list = f.read().strip().split()
    wav, _ = sf.read(str(TEST_WAV))
    audio_tensor = torch.from_numpy(wav).float()

    chunks = chunker.find_chunks(audio_tensor, text_list, file_id="test_id")

    assert len(chunks) > 0
    for chunk in chunks:
        # 验证必需字段，这些字段直接决定了 Pipeline Phase 2 能否运行
        assert hasattr(chunk, 'tensor'), "Chunk 必须携带音频 Tensor 以便 Phase 2 免 IO"
        assert hasattr(chunk, 'start_time')
        assert isinstance(chunk.text, str)
        assert chunk.tensor.ndim == 1, "Chunk tensor 应该是单声道平铺"

def test_pipeline_integration_flow(base_config, tmp_path):
    """
    黑盒测试：验证单文件 align 是否能跑通全流程并生成 TextGrid
    """
    output_tg = tmp_path / "test_output.TextGrid"
    aligner = FlexAligner(config=asdict(base_config))
    
    # 执行全流程 (Phase 1 -> Phase 2)
    # 如果内部 raise_on_error 逻辑正确，这里报错会直接 fail
    aligner.align(
        audio_path=str(TEST_WAV),
        text_path=str(TEST_TXT),
        output_path=str(output_tg)
    )

    assert output_tg.exists(), "Pipeline 未能生成最终的对齐结果文件"
    content = output_tg.read_text()
    assert 'class = "IntervalTier"' in content
    assert 'name = "phones"' in content

def test_pipeline_batch_isolation(base_config, tmp_path):
    """
    验证批量模式下的内存隔离逻辑 (Phase 1 结束后 chunker 必须释放)
    """
    aligner = FlexAligner(config=asdict(base_config))
    tasks = [(str(TEST_WAV), str(TEST_TXT), str(tmp_path / "out.TextGrid"))]
    
    # 模拟批量运行
    aligner.align_batch(tasks)

    # 验证显存/内存管理逻辑
    assert aligner.chunker is None, "Phase 1 结束后 Chunker 应该被置为 None 以释放显存"
    # 注意：aligner.aligner 此时可能还持有模型，这是符合预期的（除非也显式卸载）