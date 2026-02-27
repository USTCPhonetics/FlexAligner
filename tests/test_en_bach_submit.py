import sys
import os
from pathlib import Path
import torch

# [极客防线] 物理切断网络探测，强制本地加载
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

# 确保能导入 src 下的包
sys.path.append(os.path.abspath("./src"))

from flexaligner.pipeline import FlexAligner
from flexaligner.config import AlignmentConfig

def main():
    # ================= 1. 全局资源拓扑定址 =================
    PROJECT_ROOT = Path("/home/wangyiming/projects/FlexAligner") # 建议写绝对路径，防止工作区偏移
    INPUT_DIR = PROJECT_ROOT / "tests/testfiles/en"
    OUTPUT_DIR = INPUT_DIR / "flexaligner-submit"
    
    # [严格对齐 Bash 变量]
    SEG_MODEL_DIR = PROJECT_ROOT / "models_hidden/en/hf_phs"      # 对应 CHUNK_MODEL
    ALIGN_MODEL_DIR = PROJECT_ROOT / "models_hidden/en/ce17.6000" # 对应 ALIGN_MODEL
    DICT_PATH = PROJECT_ROOT / "assets/dictionaries/en.dict"      # 对应 EN_LEXICON
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 Starting Batch Test (FlexAligner Advanced Mode)")
    print(f"   Input:  {INPUT_DIR}")
    print(f"   Output: {OUTPUT_DIR}")
    print("-" * 60)

    # ================= 2. 核心引擎配置 (1:1 物理等效映射) =================
    # 这里的每一个参数，都严格对应你 bash 脚本中的命令行 flag
    config = AlignmentConfig(
        lang="en",
        device="cuda" if torch.cuda.is_available() else "cpu",
        
        # --- 资源路径直连 ---
        chunk_model_path=str(SEG_MODEL_DIR),
        align_model_path=str(ALIGN_MODEL_DIR),
        lexicon_path=str(DICT_PATH),
        
        # --- Stage 1: Chunker 物理参数 (对应 chunks2.py) ---
        min_chunk_s=1.0,
        max_chunk_s=12.0,
        max_gap_s=0.35,
        pad_s=0.15,
        blank_token="<pad>",
        
        # --- Stage 2: Aligner 基础马尔可夫参数 ---
        sil_phone="sil",
        optional_sil=True,
        sil_cost=-0.5,
        align_beam_size=400,
        p_stay=0.92,
        frame_hop_s=0.01,
        
        # 🔴 --- Stage 2: 极客物理装甲 (The Boss Tricks) --- 🔴
        boundary_lambda=200.0,       # 对应 --boundary_lambda 200.0
        boundary_context_s=0.05,     # 对应 --boundary_context_s 0.05
    )
    
    # [动态属性注入]
    # 如果你的 AlignmentConfig dataclass 里还没来得及声明这三个新参数，
    # 我们可以通过动态注入的方式，确保它们被 LocalAligner 捕获。
    config.sil_at_ends = True        # 对应 --sil_at_ends
    config.min_sil_dur_ms = 50.0      # 如果你想开启防碎片，设为 50.0
    config.sil_enter_cost = 0     # 如果你想开启过路费，设为 -1.0

    # 实例化引擎
    try:
        aligner = FlexAligner(config)
        # 将动态属性同步到 config_dict 给底层 LocalAligner 使用
        aligner.config_dict["sil_at_ends"] = config.sil_at_ends
        aligner.config_dict["min_sil_dur_ms"] = config.min_sil_dur_ms
        aligner.config_dict["sil_enter_cost"] = config.sil_enter_cost
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        return

    # ================= 3. 构建任务管线 =================
    tasks = []
    wav_files = sorted(list(INPUT_DIR.glob("*.wav")))
    
    for wav_path in wav_files:
        txt_path = wav_path.with_suffix(".txt")
        out_path = OUTPUT_DIR / f"{wav_path.stem}.TextGrid"
        
        if txt_path.exists():
            tasks.append((str(wav_path), str(txt_path), str(out_path)))
        else:
            print(f"⚠️  Skipping {wav_path.name}: No .txt found.")

    if not tasks:
        print("❌ No tasks found! Check input directory.")
        return

    # ================= 4. 满血点火执行 =================
    aligner.align_batch(tasks)

    print("-" * 60)
    print(f"✅ FlexAligner Advanced Run Complete.")
    print(f"   Now you can diff {OUTPUT_DIR} with legacy_en_tricks!")

if __name__ == "__main__":
    main()