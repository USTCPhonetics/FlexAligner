import pytest
import torch
from pathlib import Path
from flexaligner.io import load_audio, load_text
from flexaligner.pipeline import FlexAligner

# 定义资源路径（相对于项目根目录）
ASSETS_DIR = Path(__file__).parent.parent / "assets" / "examples"
WAV_PATH = ASSETS_DIR / "SP01_001.wav"
TXT_PATH = ASSETS_DIR / "SP01_001.txt"

@pytest.fixture
def check_assets():
    """检查测试文件是否存在，否则跳过测试"""
    if not WAV_PATH.exists() or not TXT_PATH.exists():
        pytest.skip(f"测试素材缺失: 请确保文件位于 {ASSETS_DIR}")

def test_io_loading(check_assets):
    """
    测试目标：验证 io 模块对真实文件的兼容性
    """
    # 1. 测试音频加载
    audio = load_audio(str(WAV_PATH))
    assert isinstance(audio, torch.Tensor)
    assert audio.ndim == 1  # 我们之前的物理逻辑要求 squeeze 成单维度
    
    # 2. 测试文本加载
    text = load_text(str(TXT_PATH))
    assert isinstance(text, list)
    assert len(text) > 0
    assert "SP01_001" not in text[0] # 确保只读了内容，没读 ID

def test_pipeline_dry_run(check_assets, monkeypatch):
    """
    测试目标：在有真实输入的情况下，验证流程是否能走到保存那一环
    """
    aligner = FlexAligner()
    
    # 因为我们还没搬运模型逻辑，所以我们暂时 mock 掉 chunker 的 process
    # 但我们保留 io.load_audio 的真实调用
    def mock_find_chunks(self, audio_tensor, text_list):
        from flexaligner.io import AudioChunk
        return [AudioChunk(tensor=audio_tensor[:16000], start_time=0.0, end_time=1.0, text="测试", chunk_id="1")]
    
    from flexaligner.chunker import CTCChunker
    monkeypatch.setattr(CTCChunker, "find_chunks", mock_find_chunks)

    output_tg = "tests/test_output.TextGrid"
    
    # 运行流程
    aligner.align(str(WAV_PATH), str(TXT_PATH), output_tg)
    
    # 验证是否产生了输出（即使现在是 fake 的内容）
    assert Path(output_tg).exists()
    # Path(output_tg).unlink() # 测试完可以删掉