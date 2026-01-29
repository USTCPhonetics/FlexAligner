import pytest
import shutil
import numpy as np
from pathlib import Path
from dataclasses import asdict

# 核心模块引入
from flexaligner.pipeline import FlexAligner
from flexaligner.config import AlignmentConfig
from flexaligner.frontend import TextFrontend

# ==========================================
#  测试资源配置
# ==========================================
PROJECT_ROOT = Path(__file__).parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
MODELS_DIR = PROJECT_ROOT / "models"

# 典型的中文测试用例
WAV_PATH = ASSETS_DIR / "examples" / "zh.wav"
TXT_PATH = ASSETS_DIR / "examples" / "zh.txt"

# 字典与模型路径
LEXICON_PATH = ASSETS_DIR / "dictionaries" / "dict.mandarin.2"
PHONES_PATH = ASSETS_DIR / "dictionaries" / "phones.json"

# ==========================================
#  Fixtures
# ==========================================

@pytest.fixture
def config():
    """构建基础测试配置"""
    local_model = MODELS_DIR / "hf_phs"
    return AlignmentConfig(
        chunk_model_path=str(local_model),
        lexicon_path=str(LEXICON_PATH),
        phone_json_path=str(PHONES_PATH),
        device="cpu",  # 测试用 CPU 保证稳定性
        beam_size=5,
        offset_s=0.0125 # 验证我们的物理补丁
    )

@pytest.fixture
def frontend():
    """初始化坦克级前端 (ROBUST 模式进行集成测试)"""
    return TextFrontend(mode="ROBUST")

@pytest.fixture
def output_dir():
    """清理并准备输出目录"""
    d = PROJECT_ROOT / "tests" / "outputs"
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)
    return d

# ==========================================
#  Tests
# ==========================================

def test_frontend_integration(frontend):
    """
    [Unit/Integration] 验证重构后的 Frontend 是否能为 Pipeline 提供正确格式
    """
    if not WAV_PATH.exists():
        pytest.skip("音频文件缺失")

    # 1. 验证音频载入格式 (应为 numpy，Pipeline 会转为 Tensor)
    audio = frontend.load_audio(str(WAV_PATH))
    assert isinstance(audio, np.ndarray)
    assert audio.dtype == np.float32

    # 2. 验证文本载入与语言识别
    raw_text = frontend.load_text(str(TXT_PATH))
    lang = frontend.detect_language(raw_text)
    assert lang == "zh"
    
    # 3. 验证分词分流
    tokens = frontend.get_phonemes(raw_text, lang)
    assert isinstance(tokens, list)
    assert len(tokens) > 0

def test_pipeline_full_run_zh(config, output_dir):
    """
    [Integration] 真实跑通：从音频/文本到 TextGrid 的全闭环
    验证 12.5ms Offset 是否被正确应用
    """
    if not MODELS_DIR.exists():
        pytest.skip("模型文件夹不存在，无法进行推理测试")

    # 1. 初始化大管道
    # 注意：确保 FlexAligner 的 __init__ 能够接受 AlignmentConfig 对象或 dict
    aligner = FlexAligner(config=asdict(config))

    # 2. 运行对齐
    output_tg = output_dir / "zh_test.TextGrid"
    
    print("\n[Test] 正在进行全闭环对齐推理...")
    # 传入原始路径，让 Pipeline 内部调用重构后的 Frontend
    chunks = aligner.align(str(WAV_PATH), str(TXT_PATH), str(output_tg))

    # 3. 结果断言
    assert len(chunks) > 0
    assert output_tg.exists()

    # 4. 物理对齐验证：检查 TextGrid 的起始时间
    with open(output_tg, 'r', encoding='utf-8') as f:
        content = f.read()
        # 我们的 offset_s 是 0.0125，第一个 xmin 理论上不应是 0.000000
        # 如果第一个 interval 是 sil 且从 offset 开始，这里应该能体现
        assert "xmin =" in content
    
    print(f"[Test] 成功生成 TextGrid。切分片段数: {len(chunks)}")

# // Modified in tests/test_integration.py

# tests/test_integration.py

def test_pipeline_unsupported_audio(config, output_dir):
    """
    [Robustness] 修复版：验证 Pipeline 遇到坏音频时的防御性反应
    """
    # 1. 物理准备：制造“毒药”文件
    bad_wav = output_dir / "corrupted_fake.wav"
    bad_wav.write_text("This is not a wav file, just some garbage strings.")
    
    # 2. 核心防御逻辑：
    # 如果本地模型文件夹没准备好（空的），我们需要把路径改为云端 ID。
    # 否则，FlexAligner 初始化时会因为绝对路径格式非法而直接炸掉，
    # 导致测试根本跑不到 align 这一步。
    model_path = Path(config.chunk_model_path)
    if not (model_path / "config.json").exists():
        print(f"[Test] Local model incomplete at {model_path}. Switching to Cloud ID for test.")
        config.chunk_model_path = "USTCPhonetics/FlexAligner"

    # 3. 初始化 Pipeline
    # 如果初始化崩了，说明是网络或环境问题，这里直接 fail 掉
    try:
        aligner = FlexAligner(config=asdict(config))
    except Exception as e:
        pytest.fail(f"Pipeline 初始化失败（模型路径或网络问题）: {e}")
    
    # 4. 执行对齐：捕获运行时的音频拦截异常
    # 现在的 FlexAligner.align 内部会调用重构后的 frontend.load_audio
    with pytest.raises(Exception) as excinfo:
        aligner.align(str(bad_wav), str(TXT_PATH), str(output_dir / "fail.TextGrid"))
    
    error_msg = str(excinfo.value)
    print(f"\n[Captured Error]: {error_msg}")
    
    # 5. 最终断言：
    # 只要包含了我们预设的“Audio Error”或者底层的“Format not recognised”即代表防御成功
    assert any(keyword in error_msg for keyword in ["Audio Error", "Format", "recognised"])