import sys
import os
from pathlib import Path
import torch
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
# 1. 确保能导入 src 下的包
sys.path.append(os.path.abspath("./src"))

from flexaligner.pipeline import FlexAligner
from flexaligner.config import AlignmentConfig

def main():
    # ================= 配置区 (请根据实际情况修改) =================
    # 定义基础路径
    PROJECT_ROOT = Path(".")
    INPUT_DIR = PROJECT_ROOT / "tests/testfiles/en"
    OUTPUT_DIR = INPUT_DIR / "flexaligner-00"
    SEG_MODEL_DIR = PROJECT_ROOT / "models_hidden/en/hf_phs"
    MODEL_DIR = PROJECT_ROOT / "models_hidden/en/ce2" 
    # DICT_PATH = PROJECT_ROOT / "assets/dictionaries/en.dict"
    # ---------------------------------------------------------
    # [关键] 这里需要指向你的英文模型和词典！
    # 假设你的 Toolkit 里有对应的英文资源，请修改下面的路径
    # ---------------------------------------------------------
    # 示例: 指向英文声学模型 (如 wav2vec2-large-960h 或 Toolkit 内置的英文模型)
    # 如果没有特定的 Stage 1 模型，align_model 和 chunk_model 可以指向同一个
    # MODEL_DIR = PROJECT_ROOT / "Toolkit/models/english_model_placeholder" 
    
    # 示例: 指向英文词典 (如 cmudict)
    # DICT_PATH = PROJECT_ROOT / "Toolkit/dictionary/english.dict" 
    # ---------------------------------------------------------
    
    # 确保输出目录存在
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 Starting Batch Test (English)")
    print(f"   Input:  {INPUT_DIR}")
    print(f"   Output: {OUTPUT_DIR}")
    print("-" * 60)

    # 检查模型路径是否存在，避免盲跑
    # if not Path(MODEL_DIR).exists() or not Path(DICT_PATH).exists():
    #     print(f"⚠️  警告: 模型或词典路径不存在，请修改脚本中的 MODEL_DIR 和 DICT_PATH！")
    #     print(f"   当前设置: {MODEL_DIR}")
    #     print(f"   当前设置: {DICT_PATH}")
    #     # 这里不退出，万一你是用来测试代码逻辑的，但大概率会报错
    
    # ================= 1. 初始化引擎 =================
    config = AlignmentConfig(
        lang="en",  # 设定为英文
        device="cuda" if torch.cuda.is_available() else "cpu",
        
        # # 资源路径
        # align_model_path=str(MODEL_DIR),
        # lexicon_path=str(DICT_PATH),
        
        # 如果需要 G2P (针对词典里没有的词自动注音)，开启此选项
        # 英文通常需要 G2P
        # use_g2p=True,
        
        # --- [物理参数] (根据英文语速特点可能需要微调，这里先沿用标准值) ---
        max_gap_s=0.35,      
        min_chunk_s=1.0,     
        max_chunk_s=12.0,    
        pad_s=0.15,          
        
        align_beam_size=400, 
        p_stay=0.92,         
        frame_hop_s=0.01, # 通常 Wav2Vec2 都是 20ms (0.02) 或 10ms (0.01)，需确认模型配置
        optional_sil=True,   
        
        # validation_mode="FAST"
    )
    
    # 如果 Stage 1 (Chunking) 使用不同的模型，请在这里指定
    # config.model_path = str(PROJECT_ROOT / "Toolkit/models/english_chunker")
    config.model_path = str(SEG_MODEL_DIR) 
    try:
        aligner = FlexAligner(config)
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        return
    aligner.config_dict["align_model_path"] = MODEL_DIR 
    aligner.config_dict["chunk_model_path"] = SEG_MODEL_DIR
    # ================= 2. 收集任务 =================
    tasks = []
    # 扫描所有 wav 文件
    wav_files = sorted(list(INPUT_DIR.glob("*.wav")))
    
    for wav_path in wav_files:
        # 寻找同名 txt 文件
        txt_path = wav_path.with_suffix(".txt")
        out_path = OUTPUT_DIR / f"{wav_path.stem}.TextGrid"
        
        if txt_path.exists():
            tasks.append((str(wav_path), str(txt_path), str(out_path)))
        else:
            print(f"⚠️  Skipping {wav_path.name}: No .txt found.")

    if not tasks:
        print("❌ No tasks found! Check input directory.")
        return

    # ================= 3. 执行批量对齐 =================
    # 这将自动执行 Stage 1 (Chunking) -> Stage 2 (Alignment)
    aligner.align_batch(tasks)

    print("-" * 60)
    print(f"✅ English Test Complete. Results saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()