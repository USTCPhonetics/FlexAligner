import os
import re
import numpy as np
import jieba
import nltk
from g2p_en import G2p
from typing import List

# 延迟导入区：核心防御逻辑
try:
    import librosa
except ImportError:
    librosa = None

try:
    import chardet
except ImportError:
    chardet = None

# [Lazy Import] num2words 不需要顶层导入
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
        
        # [Fix] 必须最先初始化缓存，因为 _load_custom_lexicon 依赖它
        self._vocab_cache = set()
        
        # 预加载核心组件
        if self.mode != "FAST": 
            print(f"[Frontend] Initializing in {self.mode} mode...")
            
        # 初始化 jieba
        jieba.initialize()
        
        # 加载自定义词典（如果有）
        # 这一步会用到 self._vocab_cache，所以必须放在上面初始化之后
        if self.config.get("lexicon_path"):
            self._load_custom_lexicon(self.config["lexicon_path"])

        self._g2p = None
        
        # 依赖检查
        self._check_dependencies()

        # 英语清洗规则
        self._en_abbreviations = {
            r"i'm": "i am", r"it's": "it is", r"don't": "do not",
            r"can't": "cannot", r"won't": "will not", r"n't": " not",
            r"'ll": " will", r"'ve": " have", r"'re": " are", r"'d": " would",
        }
        self._en_symbols = {r"&": " and ", r"@": " at ", r"\+": " plus "}

    def _load_custom_lexicon(self, path: str):
        """
        从对齐用的词典文件加载词汇表，确保分词结果都在词典里
        格式: WORD P1 P2 ...
        """
        if not os.path.exists(path): return
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 1:
                    word = parts[0]
                    jieba.add_word(word) # 告诉jieba这个词是合法的
                    self._vocab_cache.add(word)

    def _check_dependencies(self):
        if self.mode in ["ROBUST", "SECURE"]:
            if librosa is None: raise ImportError(f"[{self.mode}] 需要 librosa")
            if chardet is None: raise ImportError(f"[{self.mode}] 需要 chardet")

    @property
    def g2p(self):
        if self._g2p is None:
            try:
                nltk.data.find('taggers/averaged_perceptron_tagger_eng')
            except LookupError:
                print("[Frontend] Downloading NLTK resource...")
                nltk.download('averaged_perceptron_tagger_eng')
            self._g2p = G2p()
        return self._g2p

    def _normalize_numbers(self, text: str) -> str:
        global num2words
        if num2words is None:
            try:
                from num2words import num2words as n2w_func
                num2words = n2w_func
            except ImportError:
                print("[Frontend] Warning: 'num2words' not found.")
                return text

        def replace_num(match):
            num_str = match.group()
            try:
                if len(num_str) == 4 and (num_str.startswith("19") or num_str.startswith("20")):
                    return num2words(num_str, to='year')
                return num2words(num_str)
            except:
                return num_str
        return re.sub(r'\b\d+\b', replace_num, text)

    def _secure_check_audio(self, path: str):
        if os.path.getsize(path) > 50 * 1024 * 1024:
            raise ValueError("[SECURE] Audio file too large")
        with open(path, 'rb') as f:
            if f.read(4) not in [b'RIFF', b'ID3\x03', b'fLaC', b'OggS']: pass 

    def load_audio(self, path: str) -> np.ndarray:
        if self.mode == "SECURE": self._secure_check_audio(path)
        try:
            if librosa is not None:
                wav, _ = librosa.load(path, sr=self.target_sr, mono=True)
            else:
                import soundfile as sf
                from scipy import signal
                wav, sr = sf.read(path)
                if sr != self.target_sr:
                    samples = round(len(wav) * float(self.target_sr) / sr)
                    wav = signal.resample(wav, samples)
                if wav.ndim > 1: wav = wav.mean(axis=1)
                wav = wav.astype(np.float32)
            
            if len(wav) < self.target_sr * 0.05: raise ValueError("Audio too short")
            return wav
        except Exception as e:
            raise RuntimeError(f"Audio Error: {str(e)}")

    def load_text(self, path: str) -> str:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Text file not found: {path}")

        if self.mode == "SECURE" and os.path.getsize(path) > 100 * 1024: 
            raise ValueError("[SECURE] Text file too large")

        with open(path, 'rb') as f:
            raw = f.read()
        
        if not raw: return ""

        text = ""
        # 1. 尝试使用 utf-8-sig (可以自动处理带或不带 BOM 的 UTF-8)
        try:
            text = raw.decode('utf-8-sig')
        except UnicodeDecodeError:
            # 2. 如果失败，且不是 FAST 模式，尝试 chardet
            if self.mode != "FAST" and chardet is not None:
                try:
                    res = chardet.detect(raw)
                    encoding = res['encoding'] or 'utf-8'
                    text = raw.decode(encoding, errors='ignore')
                except:
                    # chardet 失败时的兜底
                    text = raw.decode('gb18030', errors='ignore')
            else:
                # 3. FAST 模式或无 chardet，尝试常见中文编码
                try:
                    text = raw.decode('gb18030')
                except UnicodeDecodeError:
                    # 最后的兜底：忽略错误强制解码
                    text = raw.decode('utf-8', errors='ignore')

        # 4. 统一清洗：去除 BOM (虽然 utf-8-sig 应该去除了，但为了防止其他编码遗留，再洗一次)
        # 去除 Windows 换行符
        # 去除首尾空白
        return text.replace('\ufeff', '').replace('\r\n', '\n').strip()

    def detect_language(self, text: str) -> str:
        if not text: return "unknown"
        if len(re.findall(r'[\u4e00-\u9fa5]', text)) > 0: return "zh"
        if len(re.findall(r'[a-zA-Z]', text)) > 0: return "en"
        return "unknown"

    def clean_text(self, text: str, lang: str) -> str:
        if not text: return ""
        if lang == "en":
            text = text.lower()
            for p, r in self._en_abbreviations.items(): text = re.sub(p, r, text)
            for p, r in self._en_symbols.items(): text = re.sub(p, r, text)
            text = self._normalize_numbers(text)
            text = re.sub(r"[^a-z0-9 ]", " ", text)
            return " ".join(text.split())
        elif lang == "zh":
            # 保留汉字、字母、数字
            text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9\s]", " ", text)
            return " ".join(text.split())
        return text

    def _is_in_vocab(self, word: str) -> bool:
        """检查词是否在词典中"""
        # 1. 优先查缓存
        if word in self._vocab_cache: return True
        
        # 2. 查 Jieba 系统词典 (FREQ > 0 表示在词典里)
        freq = jieba.get_FREQ(word)
        if freq is not None and freq > 0:
            return True
        
        # 3. 兜底：纯数字或纯字母通常认为是合法的
        if re.match(r'^[a-zA-Z0-9]+$', word):
            return True
            
        return False

    def _recursive_split_zh(self, token: str) -> List[str]:
        """
        [ZH] 递归式分词策略
        """
        if not token.strip(): return []

        if len(token) == 1 or self._is_in_vocab(token):
            return [token]

        sub_segs = jieba.lcut(token, cut_all=False, HMM=True)

        if len(sub_segs) == 1 and sub_segs[0] == token:
            return list(token)

        final_tokens = []
        for seg in sub_segs:
            if seg == token:
                 raise ValueError(f"Fatal: Token '{token}' not in lexicon and cannot be split further.")
            final_tokens.extend(self._recursive_split_zh(seg))

        return final_tokens

    def get_phonemes(self, text: str, lang: str) -> List[str]:
        cleaned_text = self.clean_text(text, lang)
        
        if lang == "en":
            if self.config.get("use_g2p", False):
                return [p for p in self.g2p(cleaned_text) if p.strip() != ""]
            return cleaned_text.split()
                
        elif lang == "zh":
            raw_tokens = cleaned_text.split()
            final_tokens = []
            for token in raw_tokens:
                sub_tokens = self._recursive_split_zh(token)
                final_tokens.extend(sub_tokens)
            return final_tokens
            
        return []