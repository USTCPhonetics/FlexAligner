import os
import re
import numpy as np
import jieba
import nltk
from g2p_en import G2p
from typing import List

# 延迟导入区：核心防御逻辑
# 防止因为缺少非核心库导致 FAST 模式无法启动
try:
    import librosa
except ImportError:
    librosa = None

try:
    import chardet
except ImportError:
    chardet = None

# [Lazy Import] num2words 不需要顶层导入，避免 IO 开销
num2words = None 

class TextFrontend:
    def __init__(self, config=None, mode="FAST"):
        """
        :param config: 全局配置对象
        :param mode: 验证模式 ["FAST", "ROBUST", "SECURE"]
        """
        self.config = config or {}
        self.mode = mode.upper()
        self.target_sr = 16000
        
        # 预加载核心组件 (Jieba 是必须的，但 num2words 只有用到时才加载)
        if self.mode != "FAST": 
            print(f"[Frontend] Initializing in {self.mode} mode...")
            
        jieba.initialize()
        self._g2p = None
        
        # 依赖检查
        self._check_dependencies()

        # 英语清洗规则：缩写展开映射表
        self._en_abbreviations = {
            r"i'm": "i am",
            r"it's": "it is",
            r"don't": "do not",
            r"can't": "cannot",
            r"won't": "will not",
            r"n't": " not",
            r"'ll": " will",
            r"'ve": " have",
            r"'re": " are",
            r"'d": " would",
        }
        
        # 特殊符号语义映射
        self._en_symbols = {
            r"&": " and ",
            r"@": " at ",
            r"\+": " plus ",
        }

    def _check_dependencies(self):
        """根据模式检查必要的外部库"""
        if self.mode in ["ROBUST", "SECURE"]:
            if librosa is None:
                raise ImportError(f"[{self.mode}] 需要 librosa。请运行: pip install librosa")
            if chardet is None:
                raise ImportError(f"[{self.mode}] 需要 chardet。请运行: pip install chardet")

    @property
    def g2p(self):
        if self._g2p is None:
            try:
                nltk.data.find('taggers/averaged_perceptron_tagger_eng')
            except LookupError:
                print("[Frontend] Downloading NLTK resource: averaged_perceptron_tagger_eng...")
                nltk.download('averaged_perceptron_tagger_eng')
            self._g2p = G2p()
        return self._g2p

    def _normalize_numbers(self, text: str) -> str:
        """
        [ROBUST] 延迟加载 num2words 进行数字归一化
        """
        # 1. 尝试动态导入
        global num2words
        if num2words is None:
            try:
                from num2words import num2words as n2w_func
                num2words = n2w_func
            except ImportError:
                # 优雅降级：如果没装库，打印警告并原样返回
                print("[Frontend] Warning: 'num2words' library not found. Skipping number normalization.")
                return text

        def replace_num(match):
            num_str = match.group()
            try:
                # 智能年份判定：如果是 4 位且在常见年份区间
                if len(num_str) == 4 and (num_str.startswith("19") or num_str.startswith("20")):
                    return num2words(num_str, to='year')
                return num2words(num_str)
            except:
                return num_str

        # 正则匹配：只匹配独立的数字单词
        return re.sub(r'\b\d+\b', replace_num, text)

    def _secure_check_audio(self, path: str):
        """[SECURE] 物理层面的防御性检查"""
        if os.path.getsize(path) > 50 * 1024 * 1024:
            raise ValueError("[SECURE] Audio file too large (>50MB)")
            
        with open(path, 'rb') as f:
            header = f.read(4)
            if header not in [b'RIFF', b'ID3\x03', b'fLaC', b'OggS']:
                pass 

    def load_audio(self, path: str) -> np.ndarray:
        if self.mode == "SECURE":
            self._secure_check_audio(path)

        try:
            if librosa is not None:
                wav, _ = librosa.load(path, sr=self.target_sr, mono=True)
            else:
                import soundfile as sf
                wav, sr = sf.read(path)
                if sr != self.target_sr:
                    try:
                        from scipy import signal
                        samples = round(len(wav) * float(self.target_sr) / sr)
                        wav = signal.resample(wav, samples)
                    except ImportError:
                        raise RuntimeError(f"[FAST] Input sr={sr} != {self.target_sr}. 需要 librosa 开启 ROBUST 模式。")
                if wav.ndim > 1:
                    wav = wav.mean(axis=1)
                wav = wav.astype(np.float32)

            if len(wav) == 0: 
                raise ValueError("Audio is empty")
            if len(wav) < self.target_sr * 0.05: 
                raise ValueError("Audio is too short (< 0.05s)")
            return wav
        except Exception as e:
            raise RuntimeError(f"Audio Error ({self.mode}): {str(e)}")

    def load_text(self, path: str) -> str:
        if not os.path.exists(path):
             raise FileNotFoundError(f"Text file not found: {path}")

        if self.mode == "SECURE" and os.path.getsize(path) > 100 * 1024: 
            raise ValueError("[SECURE] Text file too large")

        with open(path, 'rb') as f: raw = f.read()
        if not raw: return ""

        try:
            text = raw.decode('utf-8')
        except UnicodeDecodeError:
            if self.mode == "FAST":
                try: text = raw.decode('gb18030')
                except: raise UnicodeError("[FAST] Decoding failed.")
            else:
                try: text = raw.decode('gb18030')
                except:
                    if chardet:
                        encoding = chardet.detect(raw)['encoding'] or 'utf-8'
                        text = raw.decode(encoding, errors='ignore')
                    else:
                        text = raw.decode('utf-8', errors='ignore')

        return text.replace('\ufeff', '').replace('\r\n', '\n').strip()

    def detect_language(self, text: str) -> str:
        if not text: return "unknown"
        zh_char_count = len(re.findall(r'[\u4e00-\u9fa5]', text))
        en_char_count = len(re.findall(r'[a-zA-Z]', text))
        if zh_char_count > 0: return "zh"
        if en_char_count > 0: return "en"
        return "unknown"

    def clean_text(self, text: str, lang: str) -> str:
        if not text: return ""

        if lang == "en":
            text = text.lower()
            
            # 1. 缩写展开
            for pattern, replacement in self._en_abbreviations.items():
                text = re.sub(pattern, replacement, text)
            
            # 2. 特殊符号展开
            for pattern, replacement in self._en_symbols.items():
                text = re.sub(pattern, replacement, text)

            # 3. 数字归一化 (使用延迟加载的 num2words)
            text = self._normalize_numbers(text)
            
            # 4. 物理清洗
            text = re.sub(r"[^a-z0-9 ]", " ", text)
            return " ".join(text.split())

        elif lang == "zh":
            text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9\s]", " ", text)
            return " ".join(text.split())
        
        return text

    def get_phonemes(self, text: str, lang: str) -> List[str]:
        cleaned_text = self.clean_text(text, lang)
        
        if lang == "en":
            if self.config.get("use_g2p", False):
                return [p for p in self.g2p(cleaned_text) if p.strip() != ""]
            return cleaned_text.split()
                
        elif lang == "zh":
            return jieba.lcut(cleaned_text)
            
        return []