import pytest
import shutil
import soundfile as sf
import numpy as np
from pathlib import Path
import sys

# ==========================================
# 1. 环境准备
# ==========================================
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))

from flexaligner.frontend import TextFrontend

# ==========================================
# 2. 测试脚手架 (Chaos Generators)
# ==========================================
TEMP_DIR = Path(__file__).parent / "temp_chaos"

@pytest.fixture(scope="module", autouse=True)
def setup_chaos_env():
    """在测试开始前建立混沌实验室，结束后销毁"""
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)
    TEMP_DIR.mkdir(parents=True)
    yield
    # shutil.rmtree(TEMP_DIR) # Debug 时可注释掉

# [关键修改] 参数化 Fixture：让测试自动在 FAST 和 ROBUST 两种模式下各跑一遍
@pytest.fixture(params=["FAST", "ROBUST"])
def frontend(request):
    """
    自动切换 FAST (SoundFile) 和 ROBUST (Librosa) 模式
    """
    mode = request.param
    print(f"\n[Test Setup] Initializing Frontend in {mode} mode...")
    return TextFrontend(mode=mode)

# ==========================================
# 3. 音频鲁棒性测试 (Audio Robustness)
# ==========================================

def test_audio_format_conversion(frontend):
    """测试不同格式输入 (FLAC)"""
    # 构造一个 1 秒的标准正弦波 (440Hz)
    sr_source = 16000
    t = np.linspace(0, 1.0, sr_source)
    src_wav = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    
    # 模拟 FLAC 格式 (FAST 模式的 soundfile 也支持 flac)
    flac_path = TEMP_DIR / "test_standard.flac"
    sf.write(flac_path, src_wav, sr_source)
    
    # 测试加载
    wav_out = frontend.load_audio(str(flac_path))
    
    assert len(wav_out) == sr_source
    # 物理一致性检查
    assert np.allclose(wav_out, src_wav, atol=1e-3)

def test_audio_resampling(frontend):
    """
    [核心物理测试] 采样率不匹配的自动重采样 (44.1k -> 16k)
    FAST 模式下：必须验证 fallback 机制 (scipy) 是否生效
    ROBUST 模式下：验证 librosa 是否生效
    """
    sr_high = 44100
    # 生成 1 秒的音频
    wav_high = np.random.uniform(-1, 1, sr_high).astype(np.float32)
    path = TEMP_DIR / "high_sr.wav"
    sf.write(path, wav_high, sr_high)
    
    # 无论何种模式，load_audio 必须强制输出 16000 采样率
    wav_out = frontend.load_audio(str(path))
    
    # 物理时间锚点：1秒
    expected_length = 16000 
    # 允许 1-2 个采样点的重采样误差
    assert abs(len(wav_out) - expected_length) < 5 
    assert frontend.target_sr == 16000

def test_audio_too_short(frontend):
    """测试物理熔断：音频太短"""
    short_wav = np.zeros(10)
    path = TEMP_DIR / "too_short.wav"
    sf.write(path, short_wav, 16000)
    
    # 统一捕获 RuntimeError (底层 ValueError 被包装)
    with pytest.raises(RuntimeError, match="too short"):
        frontend.load_audio(str(path))

def test_audio_corruption(frontend):
    """测试损坏/伪装文件"""
    fake_path = TEMP_DIR / "fake_audio.wav"
    with open(fake_path, 'w') as f:
        f.write("This is definitely not a RIFF wave file.")
        
    with pytest.raises(RuntimeError):
        frontend.load_audio(str(fake_path))

# ==========================================
# 4. 文本鲁棒性测试 (Text & Encoding)
# ==========================================

def test_text_encoding_hell(frontend):
    """
    测试编码陷阱
    注意：FAST 模式可能不支持极度复杂的编码，这里测试基础兼容性
    """
    content = "甚至出现交易几乎停滞的情况"
    
    # 1. 测试 GB18030 (FAST 模式也应支持基础 GBK)
    gbk_path = TEMP_DIR / "zh_gbk.txt"
    with open(gbk_path, 'wb') as f:
        f.write(content.encode('gb18030'))
        
    loaded_gbk = frontend.load_text(str(gbk_path))
    assert loaded_gbk == content
    
    # 2. 测试 UTF-8 BOM
    bom_path = TEMP_DIR / "zh_bom.txt"
    with open(bom_path, 'wb') as f:
        f.write(content.encode('utf-8-sig'))
        
    loaded_bom = frontend.load_text(str(bom_path))
    assert loaded_bom == content
    assert '\ufeff' not in loaded_bom

def test_text_dirty_cleaning(frontend):
    dirty_zh = "  甚至，出现 交易； 几乎停滞！ 👋 \n\n"
    # 预期：标点去除，保留词间空格
    cleaned = frontend.clean_text(dirty_zh, lang="zh")
    
    assert "，" not in cleaned
    assert "👋" not in cleaned
    assert "甚至" in cleaned
    assert "出现" in cleaned

# ==========================================
# 5. 语言识别与音素化分流 (Logic Routing)
# ==========================================

@pytest.mark.parametrize("text, expected_lang", [
    ("甚至出现交易", "zh"),
    ("Montreal Forced Aligner", "en"),
    ("I love 编程", "zh"), 
    ("12345", "unknown"),
    ("", "unknown")
])
def test_language_detection(frontend, text, expected_lang):
    assert frontend.detect_language(text) == expected_lang

def test_phonemization_dispatch(frontend):
    """验证分流：英文给音素，中文给分词"""
    # 1. 英文
    en_text = "Montreal"
    en_phones = frontend.get_phonemes(en_text, lang="en")
    assert len(en_phones) > 0
    # G2P 结果应为 list
    assert isinstance(en_phones, list)

    # 2. 中文
    zh_text = "甚至出现交易"
    zh_words = frontend.get_phonemes(zh_text, lang="zh")
    # Jieba 结果
    assert zh_words == ["甚至", "出现", "交易"]

# ==========================================
# 6. [新增] SECURE 模式专项测试
# ==========================================

def test_secure_mode_large_file():
    """单独测试 SECURE 模式的防御逻辑"""
    secure_frontend = TextFrontend(mode="SECURE")
    
    # 创建一个伪造的大文件 (>50MB)
    # 为了测试速度，我们 Mock 一下 os.path.getsize 或者创建一个稀疏文件
    large_path = TEMP_DIR / "large_bomb.wav"
    with open(large_path, "wb") as f:
        f.seek(51 * 1024 * 1024) # 51MB
        f.write(b'\0')
        
    with pytest.raises(ValueError, match="too large"):
        secure_frontend._secure_check_audio(str(large_path))