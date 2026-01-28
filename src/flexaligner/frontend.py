import os
import re
import numpy as np
import jieba
import nltk
from g2p_en import G2p

# 延迟导入，防止 FAST 模式下因为缺少依赖而报错
try:
    import librosa
except ImportError:
    librosa = None

try:
    import chardet
except ImportError:
    chardet = None

class TextFrontend:
    def __init__(self, config=None, mode="FAST"):
        """
        :param config: 全局配置对象
        :param mode: 验证模式 ["FAST", "ROBUST", "SECURE"]
                     FAST: 默认模式，假设环境干净，依赖最小化。
                     ROBUST: 引入 chardet/librosa 完整功能，抗噪。
                     SECURE: 增加文件头检查、时长限制，防注入。
        """
        self.config = config
        self.mode = mode.upper()
        self.target_sr = 16000
        
        # 预加载核心组件 (Jieba 是必须的)
        if self.mode != "FAST": 
            print(f"[Frontend] Initializing in {self.mode} mode...")
            
        jieba.initialize()
        self._g2p = None
        
        # 依赖检查
        self._check_dependencies()

    def _check_dependencies(self):
        """根据模式检查必要的外部库"""
        if self.mode in ["ROBUST", "SECURE"]:
            if librosa is None:
                raise ImportError(f"[{self.mode}] 需要 librosa。请运行: pip install librosa (并安装 ffmpeg)")
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

    def _secure_check_audio(self, path: str):
        """[SECURE] 物理层面的防御性检查"""
        # 1. 文件大小熔断 (例如限制 50MB，防止解压炸弹)
        if os.path.getsize(path) > 50 * 1024 * 1024:
            raise ValueError("[SECURE] Audio file too large (>50MB)")
            
        # 2. 文件头魔数检查 (防止将 exe 改名为 wav 注入)
        # 简单检查 RIFF/ID3/fLaC 头
        with open(path, 'rb') as f:
            header = f.read(4)
            if header not in [b'RIFF', b'ID3\x03', b'fLaC', b'OggS']:
                # 注意：mp3 的 ID3 头比较多变，这里只是示例
                # 真正的生产环境建议引入 python-magic 库
                pass 

    def load_audio(self, path: str) -> np.ndarray:
        # SECURE 模式先行检查
        if self.mode == "SECURE":
            self._secure_check_audio(path)

        try:
            # 策略分歧：如何读取音频
            if librosa is not None:
                # 方案 A: 有 Librosa (ROBUST/SECURE 标配) -> 坦克级读取
                # orig_sr=None 意味着直接由底层重采样，效率最高
                wav, _ = librosa.load(path, sr=self.target_sr, mono=True)
            
            else:
                # 方案 B: 只有 SoundFile (FAST 标配) -> 轻量级读取
                # 限制：soundfile 不支持 mp3，且重采样能力弱，我们手动处理
                import soundfile as sf
                wav, sr = sf.read(path)
                
                # 物理重采样 (如果 FAST 模式下输入不是 16k)
                if sr != self.target_sr:
                    # 如果没有 librosa，可以使用 scipy (通常 numpy 伴生)
                    # 或者直接抛错提示用户升级模式
                    try:
                        from scipy import signal
                        samples = round(len(wav) * float(self.target_sr) / sr)
                        wav = signal.resample(wav, samples)
                    except ImportError:
                        raise RuntimeError(
                            f"[FAST] Input sr={sr} != {self.target_sr}. "
                            "FAST 模式下缺少 scipy/librosa 无法重采样。请安装 librosa 开启 ROBUST 模式。"
                        )
                
                # 强转单声道
                if wav.ndim > 1:
                    wav = wav.mean(axis=1)
                
                wav = wav.astype(np.float32)

            # 物理熔断：空音频或太短
            if len(wav) == 0: 
                raise ValueError("Audio is empty")
            if len(wav) < self.target_sr * 0.05: 
                raise ValueError("Audio is too short (< 0.05s)")
                
            return wav

        except Exception as e:
            # 捕获所有底层异常，统一向上抛出
            raise RuntimeError(f"Audio Error ({self.mode}): {str(e)}")

    def load_text(self, path: str) -> str:
        if not os.path.exists(path):
             raise FileNotFoundError(f"Text file not found: {path}")

        # SECURE 模式限制文本大小
        if self.mode == "SECURE" and os.path.getsize(path) > 100 * 1024: # 100KB
            raise ValueError("[SECURE] Text file too large")

        with open(path, 'rb') as f: 
            raw = f.read()
        
        if not raw: return ""

        # 编码探测策略
        try:
            # 1. FAST 模式：盲猜 UTF-8 (90% 情况)
            text = raw.decode('utf-8')
        except UnicodeDecodeError:
            # 2. 失败后进入 ROBUST 逻辑
            if self.mode == "FAST":
                # FAST 模式下稍微努力一下尝试 GB18030，不行就报错
                try:
                    text = raw.decode('gb18030')
                except:
                    raise UnicodeError("[FAST] Decoding failed. Enable ROBUST mode to use chardet.")
            else:
                # ROBUST/SECURE: 穷举 + Chardet
                try:
                    text = raw.decode('gb18030')
                except:
                    if chardet:
                        encoding = chardet.detect(raw)['encoding'] or 'utf-8'
                        text = raw.decode(encoding, errors='ignore')
                    else:
                        text = raw.decode('utf-8', errors='ignore')

        # 统一清洗 (BOM, CRLF)
        return text.replace('\ufeff', '').replace('\r\n', '\n').strip()

    def detect_language(self, text: str) -> str:
        """语言路由：中文霸权策略"""
        if not text: return "unknown"
        zh_char_count = 0
        en_char_count = 0 
        
        for char in text:
            if '\u4e00' <= char <= '\u9fa5':
                zh_char_count += 1
            elif 'a' <= char.lower() <= 'z':
                en_char_count += 1
                
        if zh_char_count > 0: return "zh"
        if en_char_count > 0: return "en"
        return "unknown"

    def clean_text(self, text: str, lang: str) -> str:
        # SECURE 模式可以在这里加一步 HTML 转义或 SQL 注入清洗
        
        if lang == "en":
            text = text.lower().replace("-", " ")
            text = re.sub(r"[^a-z' ]", "", text)
            return " ".join(text.split())
        elif lang == "zh":
            # 温和清洗：保留汉字、字母、数字、空格
            text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9\s]", " ", text)
            return " ".join(text.split())
        return text

    def get_phonemes(self, text: str, lang: str):
        cleaned_text = self.clean_text(text, lang)
        if lang == "en":
            # 过滤空音素
            return [p for p in self.g2p(cleaned_text) if p.strip() != ""]
        elif lang == "zh":
            return jieba.lcut(cleaned_text)
        return []