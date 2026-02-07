import pytest
import torch
import pandas as pd
from pathlib import Path
from flexaligner.pipeline import FlexAligner
from flexaligner.config import AlignmentConfig

# ==========================================
# 0. 路径配置
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# MODEL_DIR = PROJECT_ROOT / "Toolkit/models/ce2"
# 依赖之前的输出
STAGE1_DATA_DIR = PROJECT_ROOT / "verification_out_pytest/new"
MANIFEST_TSV = STAGE1_DATA_DIR / "SP01_001.chunks.tsv"

@pytest.fixture
def flex_aligner():
    """初始化 FlexAligner 控制器"""
    config = AlignmentConfig(
        lang="zh",
        device="cuda" if torch.cuda.is_available() else "cpu",
        # align_model_path=str(MODEL_DIR),
        # 允许不传 lexicon/phone_json 以使用模型默认值
    )
    return FlexAligner(config=config)

# ==========================================
# 1. 核心管线测试
# ==========================================

def test_align_from_manifest_flow(flex_aligner, tmp_path):
    """
    测试从 Manifest 恢复并完成对齐拼接的全流程 (Stage 2 核心逻辑)
    """
    if not MANIFEST_TSV.exists():
        pytest.skip(f"跳过测试：未找到 Stage 1 产物 {MANIFEST_TSV}")

    output_tg = tmp_path / "final_stitched.TextGrid"
    
    # 直接调用 Pipeline 的 Resume 模式
    flex_aligner.align_from_manifest(
        manifest_path=str(MANIFEST_TSV),
        audio_dir=str(STAGE1_DATA_DIR),
        output_path=str(output_tg),
        verbose=True
    )

    # 1. 物理存在验证
    assert output_tg.exists(), "未能生成 TextGrid 文件"
    
    # 2. 逻辑正确性验证 (读取生成的 TG 内容)
    content = output_tg.read_text(encoding="utf-8")
    assert 'name = "phones"' in content
    assert 'name = "words"' in content
    
    # 3. 缝合质量验证：检查是否包含 NULL 填充
    # 这是验证你 pipeline 中 `gap > 0.001` 逻辑是否生效的关键
    assert '"NULL"' in content, "TextGrid 中缺失 NULL 填充，检查缝合逻辑是否正确处理了 Gap"

# def test_stitch_logic_monotonicity(flex_aligner, tmp_path):
#     """
#     专门验证 _stitch_and_export 内部的时间轴单调性
#     """
#     from flexaligner.pipeline import AlignmentTask
    
#     # 模拟两个有间隙的 Chunk
#     mock_chunks = [
#         AlignmentTask(
#             chunk_id="c1", text="测试", start_time=0.0, end_time=1.0, 
#             tensor=torch.randn(16000) # 1s 音频
#         ),
#         AlignmentTask(
#             chunk_id="c2", text="验证", start_time=2.0, end_time=3.0, 
#             tensor=torch.randn(16000)
#         )
#     ]
    
#     out_path = tmp_path / "mock.TextGrid"
#     # 调用内部缝合逻辑
#     flex_aligner._stitch_and_export(mock_chunks, full_duration=3.5, output_path=str(out_path))
    
#     content = out_path.read_text()
#     # 验证是否正确插入了 1.0s - 2.0s 之间的 NULL
#     assert 'xmin = 1.000000' in content
#     assert 'xmax = 2.000000' in content
#     assert 'text = "NULL"' in content
#     # 验证末尾补齐 3.0s - 3.5s
#     assert 'xmax = 3.500000' in content