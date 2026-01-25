import torch
import soundfile as sf
import torchaudio.functional as F
from pathlib import Path
from typing import List
from dataclasses import dataclass

@dataclass
class AudioChunk:
    """Stage 1 产出的中间对象：内存中的音频片段"""
    tensor: torch.Tensor  # 具体的音频数值
    start_time: float     # 在原音频中的起始秒数
    end_time: float       # 在原音频中的结束秒数
    text: str            # 该片段对应的转写文本
    chunk_id: str        # 唯一标识符

@dataclass
class AlignmentResult:
    """最终对齐结果的承载对象"""
    segments: List[dict]  # 包含 label, start, end 的列表
    full_duration: float  # 原音频总时长

def load_audio(path: str, target_sr: int = 16000) -> torch.Tensor:
    """
    使用 soundfile 读取音频。
    这是我们为了绕过 torchaudio 兼容性坑而写的'核心补丁'。
    """
    wav_np, sr = sf.read(str(path))
    wav = torch.from_numpy(wav_np).float()
    
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    else:
        wav = wav.t()
        
    if wav.size(0) > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)
        
    if sr != target_sr:
        wav = F.resample(wav, sr, target_sr)
        
    return wav.squeeze(0)  # 返回 (T,)

def load_text(path: str) -> List[str]:
    """
    读取转写文本。
    逻辑：读取文件，按空格切分，并尝试自动过滤掉开头的 ID（如 SP01_001）。
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"找不到文本文件: {path}")
        
    content = p.read_text(encoding="utf-8").strip()
    words = content.split()
    
    # 启发式过滤 ID：如果第一个词包含下划线且后面还有词，通常是 ID
    if len(words) > 1 and "_" in words[0]:
        words = words[1:]
        
    return [w.lower() for w in words]

def export_textgrid(result: AlignmentResult, output_path: str):
    """[待填充] 负责将结果缝合成 TextGrid 并保存"""
    pass