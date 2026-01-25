import os
import torch
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# =================================================
#  路径解析逻辑 (Local First -> HF Fallback)
# =================================================

HF_ORG = "USTCPhonetics"
REPO_NAME = "FlexAligner"

def resolve_model_path(local_folder_name: str, hf_subfolder: Optional[str] = None) -> str:
    """
    智能路径解析器：
    1. 优先检查本地 `models/` 目录下是否存在。
    2. 如果本地不存在，返回 HuggingFace 的 Repo ID。
    """
    # 1. 检查本地项目根目录下的 models 文件夹
    project_root = Path(os.getcwd()) 
    local_path = project_root / "models" / local_folder_name
    
    # [DEBUG] 如果需要调试路径问题，取消下面两行的注释
    # print(f"\n[DEBUG] Looking for model at: {local_path.absolute()}")
    
    if local_path.exists():
        print(f"[Config] Found local model cache: {local_path}")
        return str(local_path)
    
    # 2. 本地没找到，返回 HF ID (云端兜底)
    print(f"[Config] Local model not found at {local_path}")
    print(f"[Config] Fallback to HuggingFace Hub: {HF_ORG}/{REPO_NAME} (subfolder={hf_subfolder})")
    
    # 返回 Repo ID，chunker/aligner 代码会配合 subfolder 参数使用
    return f"{HF_ORG}/{REPO_NAME}" 

# =================================================
#  配置类定义
# =================================================

@dataclass
class AlignmentConfig:
    """
    全局配置类：管理路径、超参数和计算设备
    """
    # --- 1. 资源路径 (核心) ---
    # Stage 1 (CTC) 模型路径
    chunk_model_path: str = field(
        default_factory=lambda: resolve_model_path("hf_phs", hf_subfolder="hf_phs")
    )
    # Stage 2 (CE) 模型路径
    align_model_path: str = field(
        default_factory=lambda: resolve_model_path("ce2", hf_subfolder="ce2")
    )
    
    # 词典路径
    lexicon_path: str = "assets/dictionaries/dict.mandarin.2"
    phone_json_path: str = "assets/dictionaries/phones.json"
    
    # --- 2. Stage 1 (Chunker) 超参数 ---
    # 决定了粗切分的粒度
    beam_size: int = 10         # CTC 解码 Beam Size
    min_chunk_s: float = 1.0    # 最小切片长度
    max_chunk_s: float = 12.0   # 最大切片长度
    max_gap_s: float = 0.35     # 允许的最大单词间隙
    min_words: int = 2          # 最小单词数
    pad_s: float = 0.15         # 边界安全填充
    blank_token: str = "<pad>"  # CTC Blank
    
    # --- 3. Stage 2 (Local Aligner) 超参数 ---
    # 决定了精对齐的准确度 [关键修正区]
    
    # [Fix] 根据 vocab.json，这里的静音符号必须是 "sil" (小写)
    sil_phone: str = "sil"      
    
    # 静音插入策略
    optional_sil: bool = True   # 是否允许词之间插入静音
    sil_cost: float = -0.5      # 插入静音的惩罚 (负数表示抑制静音生成，防止过度切分)
    
    # Viterbi 解码参数
    align_beam_size: int = 400  # 精对齐时的 Beam Size (通常比 Stage 1 大)
    p_stay: float = 0.92        # 自环概率 (控制对齐的平滑度，越大越不容易跳变)
    
    # 物理参数
    # Wav2Vec2 默认 stride=320, sr=16000 => 320/16000 = 0.02s (20ms)
    frame_hop_s: float = 0.01   
    use_dynamic_hop: bool = False
    # --- 4. 运行环境 ---
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    def __post_init__(self):
        """初始化后的校验逻辑"""
        pass

# 实例化一个默认配置供快速调用
default_config = AlignmentConfig()