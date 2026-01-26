import pytest
import torch
import shutil
from pathlib import Path
from dataclasses import asdict

# 引入我们写好的核心模块
from flexaligner.pipeline import FlexAligner
from flexaligner.config import AlignmentConfig
from flexaligner.io import load_audio, load_text

# ==========================================
#  测试资源配置
# ==========================================
# 自动定位到项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
MODELS_DIR = PROJECT_ROOT / "models"

# 测试用例文件
WAV_PATH = ASSETS_DIR / "examples" / "SP01_001.wav"
TXT_PATH = ASSETS_DIR / "examples" / "SP01_001.txt"

# 必须的字典文件 (请确保你把原来 chunks2.py 用的资源放到了这里)
LEXICON_PATH = ASSETS_DIR / "dictionaries" / "dict.mandarin.2"
PHONES_PATH = ASSETS_DIR / "dictionaries" / "phones.json"

# ==========================================
#  Fixtures (测试准备工作)
# ==========================================

@pytest.fixture
def check_assets():
    """基础素材检查"""
    if not WAV_PATH.exists() or not TXT_PATH.exists():
        pytest.skip(f"测试音频/文本缺失，跳过集成测试: {WAV_PATH}")

@pytest.fixture
def clean_output():
    """每次测试前清理输出目录，测试后保留(方便人工检查)"""
    output_dir = PROJECT_ROOT / "tests" / "outputs"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    return output_dir

# ==========================================
#  Tests
# ==========================================

def test_io_loading(check_assets):
    """
    [Unit] 测试 IO 模块是否健壮
    """
    # 1. 音频
    audio = load_audio(str(WAV_PATH))
    assert isinstance(audio, torch.Tensor)
    assert audio.ndim == 1  # 必须 squeeze 成 (Time,)
    assert audio.size(0) > 16000  # 至少有一秒吧

    # 2. 文本
    text = load_text(str(TXT_PATH))
    assert isinstance(text, list)
    assert len(text) > 0
    # 简单的冒烟测试：确保没有把 ID 读进去
    assert "SP01" not in text[0] 

def test_pipeline_real_run_ctc(check_assets, clean_output):
    """
    [Integration] 真实跑通 Stage 1 (CTC Chunking)
    不再使用 Mock，而是加载真实模型！
    """
    # 1. 检查模型是否存在 (本地优先策略)
    # 我们假设你的模型在 models/hf_phs 下
    local_model = MODELS_DIR / "hf_phs"
    
    if not local_model.exists():
        pytest.skip(f"本地模型未找到: {local_model}。跳过真实推理测试。")
        
    if not LEXICON_PATH.exists() or not PHONES_PATH.exists():
        pytest.skip(f"字典文件缺失: {LEXICON_PATH}。请将资源放入 assets/dictionaries")

    # 2. 构建真实配置
    # 注意：我们显式指定路径，防止 pytest 运行目录不同导致找不到文件
    config = AlignmentConfig(
        chunk_model_path=str(local_model),
        lexicon_path=str(LEXICON_PATH),
        phone_json_path=str(PHONES_PATH),
        device="cpu",  # 测试用 CPU 即可，兼容性好
        beam_size=5    # 改小一点，跑得快
    )

    # 3. 初始化 Pipeline
    # 注意：chunker 内部是用 dict 初始化的，所以我们要把 dataclass 转 dict
    # (或者你修改 pipeline 让它支持 dataclass，这里先转 dict 最稳)
    aligner_config_dict = asdict(config)
    aligner = FlexAligner(config=aligner_config_dict)

    # 4. 运行对齐
    output_tg = clean_output / "SP01_001.TextGrid"
    
    print("\n[Test] 开始运行 CTC 推理，加载模型可能需要几秒钟...")
    chunks = aligner.align(str(WAV_PATH), str(TXT_PATH), str(output_tg))

    # 5. 验证结果 (Assertions)
    
    # A. 确保确实切出了东西
    assert len(chunks) > 0, "竟然一个 Chunk 都没切出来？模型可能崩了或者静音阈值太高。"
    
    # B. 检查第一个 Chunk 的结构
    first_chunk = chunks[0]
    assert first_chunk.end_time > first_chunk.start_time
    assert len(first_chunk.text) > 0
    assert isinstance(first_chunk.tensor, torch.Tensor)
    
    # C. 检查文件是否生成
    assert output_tg.exists()
    
    print(f"\n[Test] 成功！切出了 {len(chunks)} 个片段。")
    print(f"[Test] 第一个片段: {first_chunk.start_time:.2f}s - {first_chunk.end_time:.2f}s | 内容: {first_chunk.text}")
    
    
def test_pipeline_dynamic_precision(check_assets, clean_output):
    """
    [Integration] 测试开启开启 use_dynamic_hop 后的高精度模式
    """
    local_model = MODELS_DIR / "hf_phs"
    
    # 1. 显式开启 use_dynamic_hop 开关
    config = AlignmentConfig(
        chunk_model_path=str(local_model),
        lexicon_path=str(LEXICON_PATH),
        phone_json_path=str(PHONES_PATH),
        device="cpu",
        use_dynamic_hop=True,  # <--- 开启高精度开关
        frame_hop_s=0.01       # 依然保持 10ms 基础步长
    )

    # 2. 初始化与运行
    aligner = FlexAligner(config=asdict(config))
    output_tg = clean_output / "SP01_001_dynamic.TextGrid"
    
    print("\n[Test] 正在运行高精度模式 (Dynamic Hop)...")
    _chunks = aligner.align(str(WAV_PATH), str(TXT_PATH), str(output_tg))

    # 3. 验证
    assert output_tg.exists()
    print(f"[Test] 高精度 TextGrid 已生成: {output_tg}")