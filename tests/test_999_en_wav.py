import os
from pathlib import Path
from flexaligner import FlexAligner

# ================= 配置区 =================
# 1. 数据所在目录 (假设脚本在 tests/testfiles/en 下运行，或者你手动指定绝对路径)
# 这里的路径改为你截图中的路径
DATA_DIR = Path("/home/wangyiming/projects/FlexAligner/tests/testfiles/en")
OUTPUT_DIR = DATA_DIR / "output_mfa_new"  # 结果输出到这里

# 2. 确保输出目录存在
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

def main():
    print(f"🚀 Initializing FlexAligner for English (TIMIT)...")
    
    # === 初始化对齐器 ===
    # 关键点：这里 lang="en" 会触发英文逻辑 (包括 OOV 修复等)
    # 关键点：boundary_lambda=1.0 激活你刚写的边界优化算法
    aligner = FlexAligner({
        "device": "cuda",          # 使用 GPU
        "lang": "en",              # 指定英语 (会自动加载英文模型)
        "boundary_lambda": 0.0,    # 🔥 开启边界感知 Viterbi
        "boundary_context_s": 0.06 # 边界窗口 60ms
    })
    
    # === 扫描任务 ===
    tasks = []
    wav_files = list(DATA_DIR.glob("*.wav"))
    
    print(f"📂 Found {len(wav_files)} audio files in {DATA_DIR}")

    for wav_path in wav_files:
        # 构造对应的 .txt 路径
        # TIMIT 文件名示例: DR1_FAKS0_SA1.wav -> DR1_FAKS0_SA1.txt
        txt_path = wav_path.with_suffix(".txt")
        
        # 构造输出的 .TextGrid 路径
        tg_path = OUTPUT_DIR / f"{wav_path.stem}.TextGrid"
        
        if txt_path.exists():
            tasks.append((str(wav_path), str(txt_path), str(tg_path)))
        else:
            print(f"⚠️ Warning: Missing text file for {wav_path.name}, skipping.")

    if not tasks:
        print("❌ No valid tasks found. Please check file paths.")
        return

    # === 执行批量对齐 ===
    print(f"▶️  Starting batch alignment for {len(tasks)} files...")
    results = aligner.align_batch(tasks)
    
    print(f"\n✅ Processing complete!")
    print(f"💾 Results saved to: {OUTPUT_DIR}")
    
    # 简单打印第一个结果的路径，方便确认
    if results:
        print(f"📝 Example output: {tasks[0][2]}")

if __name__ == "__main__":
    main()