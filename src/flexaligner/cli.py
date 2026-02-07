import argparse
import sys
import csv
import time
from pathlib import Path
from dataclasses import asdict
from typing import List, Tuple, Optional

# 导入核心组件
from flexaligner import FlexAligner
from flexaligner.config import AlignmentConfig

# 定义支持的扩展名，用于智能路由
AUDIO_EXTENSIONS = {'.wav', '.mp3', '.flac', '.ogg', '.opus', '.m4a'}
BATCH_EXTENSIONS = {'.csv', '.txt', '.tsv'}

def infer_paths(audio_str: str, text_str: Optional[str] = None, out_str: Optional[str] = None) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    [智能推断逻辑]
    根据你的设想：
    1. Audio: 必需。
    2. Text:  如果有则用；如果没有，推断同名 .txt。不存在则返回 None (用于后续 Skip)。
    3. Output: 如果有则用；如果没有，推断同名 .TextGrid。
    """
    audio_p = Path(audio_str)
    
    # 1. 处理 Text
    if text_str and text_str.strip():
        text_p = Path(text_str)
    else:
        text_p = audio_p.with_suffix(".txt")
    
    # 2. 处理 Output
    if out_str and out_str.strip():
        out_p = Path(out_str)
    else:
        out_p = audio_p.with_suffix(".TextGrid")

    # 3. 存在性检查 (Audio 和 Text 必须存在)
    if not audio_p.exists():
        print(f"⚠️  [Skip] Audio missing: {audio_p}")
        return None, None, None
    
    if not text_p.exists():
        print(f"⚠️  [Skip] Transcript missing: {text_p} (Derived from audio)")
        return None, None, None

    return str(audio_p), str(text_p), str(out_p)

def parse_batch_file(file_path: Path) -> List[Tuple[str, str, str]]:
    """
    解析 CSV/TXT 文件，生成任务列表。
    格式支持：
    Col 1: Audio Path (Required)
    Col 2: Text Path (Optional)
    Col 3: Output Path (Optional)
    """
    tasks = []
    print(f"📂 Parsing batch file: {file_path.name}...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        # 使用 csv reader 处理带逗号的文件名等复杂情况
        # 自动探测分隔符 (支持逗号或制表符)
        line = f.readline()
        f.seek(0)
        dialect = csv.Sniffer().sniff(line) if len(line) > 2 else 'excel'
        reader = csv.reader(f, dialect)

        for row in reader:
            if not row or row[0].startswith("#"): continue
            
            # 提取列
            audio_in = row[0].strip()
            text_in = row[1].strip() if len(row) > 1 else None
            out_in  = row[2].strip() if len(row) > 2 else None
            
            # 智能推断与检查
            a_p, t_p, o_p = infer_paths(audio_in, text_in, out_in)
            
            if a_p and t_p and o_p:
                tasks.append((a_p, t_p, o_p))
                
    return tasks

def print_dashboard(args, config: AlignmentConfig, is_batch: bool, tasks_count: int = 0):
    """[系统预检仪表盘] 打印所有详尽参数"""
    print("\n" + "⚙️ " + "="*58)
    print(f"{' FLEXALIGNER CONFIGURATION DASHBOARD ':=^58}")
    print("="*60)
    
    # 1. 运行环境
    print(f"  [Environment]")
    print(f"    - Mode:        {'📦 BATCH' if is_batch else '🎯 SINGLE'}")
    print(f"    - Device:      {config.device.upper()}")
    print(f"    - Language:    {config.lang if config.lang else 'Auto-Detect'}")
    
    # 2. 物理与算法参数 (显化默认值)
    print(f"\n  [Algorithm Parameters]")
    print(f"    - Max Gap (s): {config.max_gap_s:<10} (Stage 1 split threshold)")
    print(f"    - Beam Size:   {config.beam_size:<10} (Stage 1 search width)")
    print(f"    - Min Chunk:   {getattr(config, 'min_chunk_s', 1.0):<10} s")
    print(f"    - Pad Window:  {getattr(config, 'pad_s', 0.15):<10} s")
    
    # 3. 任务信息
    print(f"\n  [Task Scope]")
    if is_batch:
        print(f"    - Batch File:  {args.input_file}")
        print(f"    - Tasks Loaded:{tasks_count}")
    else:
        print(f"    - Audio In:    {args.input_file}")
        print(f"    - Text In:     {args.transcript_file if args.transcript_file else '(Auto-derived)'}")
        print(f"    - Output:      {args.output if args.output else '(Auto-derived .TextGrid)'}")

    print("="*60 + "\n")


    
def main():
    parser = argparse.ArgumentParser(
        description="🌊 FlexAligner: Robust Signal-to-Symbol Alignment.",
        epilog="Examples:\n  flexaligner audio.wav text.txt\n  flexaligner batch.csv",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # 核心位置参数
    parser.add_argument("input_file", help="Input Audio file OR Batch list (.csv/.txt)")
    parser.add_argument("transcript_file", nargs="?", help="Transcript file (for single mode)")
    
    # 选项
    parser.add_argument("-o", "--output", help="Output path (for single mode)")
    parser.add_argument("-l", "--lang", choices=["zh", "en"], help="Force language (triggers Language Lock)")
    parser.add_argument("--device", default="cpu", help="Compute device (cuda/cpu/mps)")
    
    # 调试与详尽模式 # Modified
    parser.add_argument("-v", "--verbose", action="store_true", help="Print detailed parameters and logs")
    
    # 高级参数
    parser.add_argument("--beam_size", type=int, default=10, help="Stage 1 Beam Size")
    parser.add_argument("--max_gap", type=float, default=0.05, help="Chunk split sensitivity (s)")
    
    args = parser.parse_args()
    
    input_path = Path(args.input_file)
    
    # --- 1. 智能路由 ---
    if input_path.suffix.lower() in BATCH_EXTENSIONS:
        is_batch = True
    elif input_path.suffix.lower() in AUDIO_EXTENSIONS:
        is_batch = False
    else:
        is_batch = not bool(args.transcript_file)

    # --- 2. 初始化配置 ---
    config = AlignmentConfig(
        device=args.device,
        lang=args.lang,
        beam_size=args.beam_size,
        max_gap_s=args.max_gap
    )
    
    # 获取任务列表以供打印
    tasks = []
    if is_batch and input_path.exists():
        tasks = parse_batch_file(input_path)

    # --- 3. 打印仪表盘 (如果开启 verbose) --- # Modified
    if args.verbose:
        print_dashboard(args, config, is_batch, len(tasks))
    else:
        # 非 verbose 模式下的简短输出
        print(f"🌊 FlexAligner (v1.0.0) | {config.device.upper()} | {'Batch' if is_batch else 'Single'}")

    # --- 4. 初始化引擎 ---
    try:
        aligner = FlexAligner(config=asdict(config))
    except Exception as e:
        print(f"❌ Core Init Failed: {e}")
        sys.exit(1)

    t0 = time.time()

    # --- 5. 执行逻辑 ---
    if is_batch:
        if not input_path.exists():
            print(f"❌ Error: Batch file not found: {input_path}")
            sys.exit(1)
        if not tasks:
            print("⚠️  No valid tasks to process.")
            sys.exit(0)
            
        print(f"🚀 Starting pipeline for {len(tasks)} tasks...")
        aligner.align_batch(tasks) # 内部可以根据 config.verbose 决定是否打印每条进度
        
    else:
        # 单文件推断逻辑保持不变
        actual_transcript = args.transcript_file
        if not actual_transcript:
            potential_txt = input_path.with_suffix(".txt")
            if potential_txt.exists():
                actual_transcript = str(potential_txt)
            else:
                print("❌ Error: Transcript file required for single mode.")
                sys.exit(1)

        actual_output = args.output if args.output else str(input_path.with_suffix(".TextGrid"))
            
        if not input_path.exists():
            print(f"❌ Error: Audio file not found: {input_path}")
            sys.exit(1)
            
        try:
            # 这里的 verbose 传给 align 方法，用于打印 Stage 1/2 的细节
            aligner.align(str(input_path), actual_transcript, actual_output, verbose=args.verbose)
            if args.verbose:
                print(f"✨ [Success] Result saved to: {Path(actual_output).absolute()}")
        except Exception as e:
            print(f"\n❌ Alignment Failed: {e}")
            sys.exit(1)

    print(f"\n🏁 Total Runtime: {time.time() - t0:.2f}s")

if __name__ == "__main__":
    main()