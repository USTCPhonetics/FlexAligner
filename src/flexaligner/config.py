import os
import torch
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# =================================================
#  设备分配逻辑 (Unified Device Manager)
# =================================================

def get_best_device(requested_device: Optional[str] = None) -> str:
    """
    智能设备探测器：
    1. 优先使用用户在 config 中指定的设备 (如 'cuda:1')。
    2. 如果没指定，探测 CUDA，默认返回 'cuda:0'。
    3. 如果都没有，Fallback 到 'cpu'。
    """
    if requested_device:
        return requested_device
        
    if torch.cuda.is_available():
        # 这里可以扩展：如果要智能选择显存最多的卡，可以在这里写逻辑
        # 目前默认锁定 GPU 0，这是最稳健的极客做法
        return "cuda:0"
    
    return "cpu"

# =================================================
#  路径解析逻辑 (保持你的 Local First 逻辑)
# =================================================

HF_ORG = "USTCPhonetics"
REPO_NAME = "FlexAligner"

# // Modified in src/flexaligner/config.py

def resolve_model_path(local_folder_name: str, hf_subfolder: Optional[str] = None) -> str:
    project_root = Path(os.getcwd()) 
    local_path = project_root / "models" / local_folder_name
    
    # 物理验证：不仅文件夹要存在，核心文件必须配齐
    # 如果只有文件夹名而没文件，Transformers 会误以为是本地模型而报错
    required_files = ["config.json", "preprocessor_config.json"]
    is_valid_local = local_path.exists() and all((local_path / f).exists() for f in required_files)
    
    if is_valid_local:
        return str(local_path.absolute())
    
    # 本地不成立，强制走云端
    return f"{HF_ORG}/{REPO_NAME}"

# =================================================
#  配置类定义
# =================================================

@dataclass
class AlignmentConfig:
    """
    全局配置类：管理路径、超参数和计算设备
    """
    # --- 1. 运行环境 (智能寻址) ---
    # 默认 device 设为 None，在 __post_init__ 中进行智能探测
    device: str = field(default="cuda" if torch.cuda.is_available() else "cpu")
    
    # --- 2. 资源路径 ---
    chunk_model_path: str = field(
        default_factory=lambda: resolve_model_path("hf_phs", hf_subfolder="hf_phs")
    )
    align_model_path: str = field(
        default_factory=lambda: resolve_model_path("ce2", hf_subfolder="ce2")
    )
    
    lexicon_path: str = "assets/dictionaries/dict.mandarin.2"
    phone_json_path: str = "assets/dictionaries/phones.json"
    
    # --- 3. Stage 1 (Chunker) 超参数 ---
    beam_size: int = 10
    min_chunk_s: float = 1.0
    max_chunk_s: float = 12.0
    max_gap_s: float = 0.35
    min_words: int = 2
    pad_s: float = 0.15
    blank_token: str = "<pad>"
    
    # --- 4. Stage 2 (Local Aligner) 物理参数 [核心修正] ---
    sil_phone: str = "sil"
    optional_sil: bool = True
    sil_cost: float = -0.5
    align_beam_size: int = 400
    p_stay: float = 0.92
    
    # 物理真理：stride=160, sr=16000 => 0.01s
    frame_hop_s: float = 0.01  
    
    # 要求的 12.5ms 偏移修正
    offset_s: float = 0.0125  

    def __post_init__(self):
        """初始化后的校验逻辑"""
        # 确保设备分配是最优的
        self.device = get_best_device(self.device)
        print(f"[Config] Global computation device set to: {self.device}")

# 实例化
default_config = AlignmentConfig()