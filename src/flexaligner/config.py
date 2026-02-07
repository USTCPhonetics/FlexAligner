import os
import torch
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# =================================================
#  设备分配逻辑 (Unified Device Manager)
# =================================================

def get_best_device(requested_device: Optional[str] = None) -> str:
    if requested_device:
        return requested_device
    return "cuda:0" if torch.cuda.is_available() else "cpu"

# =================================================
#  资源寻址引擎 (Language-Aware Resolver)
# =================================================

HF_ORG = "USTCPhonetics"
REPO_NAME = "FlexAligner"

def resolve_resource_path(lang: str, stage: str) -> str:
    """
    智能路径解析器：
    1. 优先查找本地路径: models/{lang}/{stage}
    2. 如果本地残缺，返回 HF Repo ID 并带上 subfolder 标识
    """
    project_root = Path(os.getcwd())
    # 这里的 stage 对应云端的 'chunker' 或 'aligner'
    local_path = project_root / "models" / lang / stage
    
    required_files = ["config.json", "preprocessor_config.json"]
    is_valid_local = local_path.exists() and all((local_path / f).exists() for f in required_files)
    
    if is_valid_local:
        return str(local_path.absolute())
    
    # 物理定位：返回 Repo ID，具体的 subfolder 逻辑由 Chunker/Aligner 内部处理
    return f"{HF_ORG}/{REPO_NAME}"

# =================================================
#  配置类定义
# =================================================

@dataclass
class AlignmentConfig:
    """
    FlexAligner 2.0 全局配置类：支持多语种动态切换
    """
    # --- 0. 核心语言标识 ---
    lang: str = "zh" 

    # --- 1. 运行环境 ---
    device: str = field(default=None) 
    
    # --- 2. 资源路径 (动态生成) ---
    chunk_model_path: str = field(default=None)
    align_model_path: str = field(default=None)
    lexicon_path: str = field(default=None)
    phone_json_path: str = field(default=None)

    # --- 3. 算法参数 (Stage 1) ---
    beam_size: int = 10
    min_chunk_s: float = 1.0
    max_chunk_s: float = 12.0
    max_gap_s: float = 0.35
    min_words: int = 2
    pad_s: float = 0.15
    blank_token: str = "<pad>"
    
    # --- 4. 算法参数 (Stage 2) ---
    sil_phone: str = "sil"
    optional_sil: bool = True
    sil_cost: float = -0.5
    align_beam_size: int = 400
    p_stay: float = 0.92
    
    # 🔴 物理真理：修改后的 Wav2Vec2 Stride=1 => 10ms
    frame_hop_s: float = 0.01
    offset_s: float = 0
    boundary_lambda: float = 0.0
    boundary_context_s: float = 0.06
    
    chunks_out_dir : Optional[str] = "chunks_out" # 可选的输出目录参数，默认为 "chunks_out"
    verbose: bool = False # 是否开启详细日志输出
    # src/flexaligner/config.py

    # src/flexaligner/config.py 中的 __post_init__ 关键部分修正

    def __post_init__(self):
        self.device = get_best_device(self.device)
        
        # 自动模型寻址
        if self.chunk_model_path is None:
            self.chunk_model_path = resolve_resource_path(self.lang, "chunker")
        if self.align_model_path is None:
            self.align_model_path = resolve_resource_path(self.lang, "aligner")

        # [同步重命名] 动态词典绑定
        base_asset = Path("assets/dictionaries")
        if self.lang == "zh":
            self.lexicon_path = str(base_asset / "zh.dict") # 修正
            # self.phone_json_path = str(base_asset / "phones.json")
        elif self.lang == "en":
            self.lexicon_path = str(base_asset / "en.dict") # 修正
            
            # 英语音素表逻辑：优先找本地 vocab.json
            if os.path.isdir(self.chunk_model_path):
                vocab_path = Path(self.chunk_model_path) / "vocab.json"
                # self.phone_json_path = str(vocab_path) if vocab_path.exists() else None
            else:
                self.phone_json_path = None # 云端模式将由 Chunker 在运行时补全