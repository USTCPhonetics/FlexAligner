import pytest
from pathlib import Path
import sys

# 挂载源路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))

from flexaligner.frontend import TextFrontend

# ==========================================
#  Fixtures
# ==========================================

@pytest.fixture
def frontend():
    """默认前端（Word 模式）"""
    return TextFrontend(mode="FAST")

@pytest.fixture
def frontend_g2p():
    """开启 G2P 的前端"""
    # 模拟 Config 传入
    return TextFrontend(config={"use_g2p": True}, mode="FAST")

# ==========================================
#  Tests: 缩写与展开 (The Messy Part)
# ==========================================

@pytest.mark.parametrize("input_str, expected", [
    ("I'm here", "i am here"),
    ("It's a trap", "it is a trap"),
    ("Don't do it", "do not do it"),
    ("I can't fly", "i cannot fly"),
    ("I won't go", "i will not go"),
    ("I'll be back", "i will be back"),
    ("You've got mail", "you have got mail"),
])
def test_en_abbreviation_expansion(frontend, input_str, expected):
    """验证缩写展开逻辑是否准确"""
    cleaned = frontend.clean_text(input_str, lang="en")
    assert cleaned == expected

# ==========================================
#  Tests: 符号与标点
# ==========================================

@pytest.mark.parametrize("input_str, expected", [
    ("Rock & Roll", "rock and roll"),
    ("Me @ Home", "me at home"),
    ("1 + 2", "one plus two"), # 假设你在 frontend 里加了 + 的映射，或者至少看它是否被保留
    ("Hello, World!!!", "hello world"), # 标点应该被抹除
    ("Line-Break", "line break"), # 连字符换空格
    ("User_Name", "user name"),   # 下划线换空格
])
def test_en_special_symbols(frontend, input_str, expected):
    """验证特殊符号的语义化转换或清洗"""
    # 注意：如果你的 frontend 还没加 + 的映射，这个测试会帮你发现问题
    # 目前代码里加了 &, @, +，所以应该能过
    cleaned = frontend.clean_text(input_str, lang="en")
    # 归一化空格比较
    assert " ".join(cleaned.split()) == expected

def test_en_dirty_input(frontend):
    """测试极度脏乱的输入"""
    dirty = "  Hey!!!   I'm...   here & there.  "
    expected = "hey i am here and there"
    cleaned = frontend.clean_text(dirty, lang="en")
    assert cleaned == expected

# ==========================================
#  Tests: 模式切换 (Words vs Phonemes)
# ==========================================

def test_get_phonemes_word_mode(frontend):
    """验证默认模式下返回的是单词列表"""
    text = "The quick brown fox"
    tokens = frontend.get_phonemes(text, lang="en")
    
    assert isinstance(tokens, list)
    assert tokens == ["the", "quick", "brown", "fox"]
    # 确保没有音素符号混进去
    assert "DH" not in tokens 

def test_get_phonemes_g2p_mode(frontend_g2p):
    """验证 G2P 模式下返回的是音素列表"""
    text = "Hello"
    # 这里可能会触发 NLTK 下载，可能会慢一点
    tokens = frontend_g2p.get_phonemes(text, lang="en")
    
    # g2p_en 对 Hello 的输出通常是 ['HH', 'AH0', 'L', 'OW1']
    assert isinstance(tokens, list)
    assert len(tokens) > 1 
    # 验证它是音素而不是单词
    assert "HH" in tokens or "AH0" in tokens
    assert "hello" not in tokens

# ==========================================
#  Tests: 语言探测边界
# ==========================================

def test_detect_language_mixed(frontend):
    """验证混合文本的判定逻辑"""
    assert frontend.detect_language("Pure English") == "en"
    # 中文霸权逻辑验证
    assert frontend.detect_language("I love 中国") == "zh"
    assert frontend.detect_language("123456") == "unknown"