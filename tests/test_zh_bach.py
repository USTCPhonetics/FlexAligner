import sys
import os
from pathlib import Path
import torch

# 1. 确保能导入 src 下的包
sys.path.append(os.path.abspath("./src"))

from flexaligner.pipeline import FlexAligner
from flexaligner.config import AlignmentConfig

def main():
    # ================= 配置区 =================
    PROJECT_ROOT = Path(".")
    INPUT_DIR = PROJECT_ROOT / "tests/testfiles/zh"
    OUTPUT_DIR = INPUT_DIR / "flexaligner" # 输出到 flexaligner 文件夹
    
    # 资源路径
    MODEL_DIR = PROJECT_ROOT / "Toolkit/models/ce2"
    # 注意：Pipeline 内部会自动处理 chunker 模型，只要 config 传对
    # 但 chunks2.py 用的是 hf_phs，这里 pipeline 如果是做 alignment，
    # 需要确认 pipeline 内部 chunker 是否加载了正确的模型。
    # FlexAligner 默认 chunker 模型路径通常也是 align_model_path 
    # 或者你需要显式指定 segmentation_model_path (如果你的 config 支持)
    # 假设目前架构复用同一个 config
    
    DICT_PATH = PROJECT_ROOT / "Toolkit/dictionary/dict.mandarin.2"
    
    # 确保输出目录存在
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 Starting Batch Test (Full Pipeline)")
    print(f"   Input:  {INPUT_DIR}")
    print(f"   Output: {OUTPUT_DIR}")
    print("-" * 60)

    # ================= 1. 初始化引擎 (参数对齐) =================
    # 关键：这里必须传入与 legacy.sh 中 chunks2.py 一致的参数
    config = AlignmentConfig(
        lang="zh",
        device="cuda" if torch.cuda.is_available() else "cpu",
        
        # 资源
        align_model_path=str(MODEL_DIR),
        lexicon_path=str(DICT_PATH),
        
        # --- [Stage 1: Segmentation 参数] (必须与 legacy.sh 一致) ---
        max_gap_s=0.35,      # Legacy: --max_gap_s 0.35
        min_chunk_s=1.0,     # Legacy: --min_chunk_s 1.0
        max_chunk_s=12.0,    # Legacy: --max_chunk_s 12.0
        pad_s=0.15,          # Legacy: --pad_s 0.15
        
        # --- [Stage 2: Alignment 参数] (必须与 legacy.sh 一致) ---
        align_beam_size=400, # Legacy: --beam 400
        p_stay=0.92,         # Legacy: --p_stay 0.92
        frame_hop_s=0.01,    # Legacy: --frame_hop_s 0.01
        optional_sil=True,   # Legacy: --optional_sil
        sil_cost=-0.5,       # Legacy 默认值通常是 -0.5
        
        # 模式
        # validation_mode="FAST"
    )
    
    # 注意：FlexAligner 初始化时会同时加载 Chunker 和 Aligner
    # 确保 Chunker 加载的是分割模型 (hf_phs) 还是 对齐模型 (ce2)？
    # 原版 legacy.sh 中：
    #   Step 1 (Segmentation) 用的是 ./models/hf_phs
    #   Step 2 (Alignment)    用的是 ./models/ce2
    # 
    # 如果 FlexAligner 的 CTCChunker 默认复用 align_model_path，
    # 那么你需要在这里显式指定 model_path 指向 hf_phs，否则切分会变！
    
    # [修正] 显式指定 segmentation model path
    # 如果 AlignmentConfig 支持 model_path 字段作为 chunker 模型：
    SEG_MODEL_DIR = PROJECT_ROOT / "Toolkit/models/hf_phs"
    config.model_path = str(SEG_MODEL_DIR) 

    aligner = FlexAligner(config)

    # ================= 2. 收集任务 =================
    tasks = []
    wav_files = sorted(list(INPUT_DIR.glob("*.wav")))
    
    for wav_path in wav_files:
        txt_path = wav_path.with_suffix(".txt")
        out_path = OUTPUT_DIR / f"{wav_path.stem}.TextGrid"
        
        if txt_path.exists():
            tasks.append((str(wav_path), str(txt_path), str(out_path)))

    if not tasks:
        print("❌ No tasks found!")
        return

    # ================= 3. 执行批量对齐 =================
    aligner.align_batch(tasks)

    print("-" * 60)
    print(f"✅ Test Complete. Results saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()