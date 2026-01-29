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

def main():
    parser = argparse.ArgumentParser(
        description="🌊 FlexAligner: Robust Signal-to-Symbol Alignment.",
        epilog="Examples:\n  flexaligner audio.wav text.txt\n  flexaligner batch.csv",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # 核心位置参数：Input File (可能是音频，也可能是 CSV)
    parser.add_argument("input_file", help="Input Audio file OR Batch list (.csv/.txt)")
    # 可选位置参数：Transcript (仅单文件模式需要，Batch模式会自动忽略)
    parser.add_argument("transcript_file", nargs="?", help="Transcript file (for single mode)")
    
    # 选项
    parser.add_argument("-o", "--output", help="Output path (for single mode)")
    parser.add_argument("-l", "--lang", choices=["zh", "en"], help="Force language (triggers Language Lock)")
    parser.add_argument("--device", default="cpu", help="Compute device (cuda/cpu/mps)")
    
    # 高级参数
    parser.add_argument("--beam_size", type=int, default=10, help="Stage 1 Beam Size")
    parser.add_argument("--max_gap", type=float, default=0.05, help="Chunk split sensitivity (s)")
    
    args = parser.parse_args()
    
    input_path = Path(args.input_file)
    
    # --- 1. 智能路由 (Smart Routing) ---
    is_batch = False
    
    if input_path.suffix.lower() in BATCH_EXTENSIONS:
        is_batch = True
    elif input_path.suffix.lower() in AUDIO_EXTENSIONS:
        is_batch = False
    else:
        # 后缀无法识别，尝试读取内容或根据第二个参数判断
        # 如果给了 transcript_file，肯定是单文件模式
        if args.transcript_file:
            is_batch = False
        else:
            # 默认为 Batch 尝试解析
            is_batch = True

    # --- 2. 初始化引擎 (一次性) ---
    config = AlignmentConfig(
        device=args.device,
        lang=args.lang,
        beam_size=args.beam_size,
        max_gap_s=args.max_gap
    )
    
    print("\n" + "="*60)
    print(f"🌊 FlexAligner (v1.0.0) | Mode: {'📦 BATCH' if is_batch else '🎯 SINGLE'}")
    print(f"   Device: {config.device.upper()} | Lang: {args.lang if args.lang else 'Auto'}")
    print("="*60)

    try:
        aligner = FlexAligner(config=asdict(config))
    except Exception as e:
        print(f"❌ Core Init Failed: {e}")
        sys.exit(1)

    t0 = time.time()

    # --- 3. 执行逻辑 ---
    if is_batch:
        if not input_path.exists():
            print(f"❌ Error: Batch file not found: {input_path}")
            sys.exit(1)
            
        tasks = parse_batch_file(input_path)
        if not tasks:
            print("⚠️  No valid tasks to process.")
            sys.exit(0)
            
        print(f"✅ Loaded {len(tasks)} valid tasks. Starting pipeline...")
        aligner.align_batch(tasks)
        
    else:
        # 单文件模式：需要更严格的检查
        if not args.transcript_file:
            # 尝试自动推断 transcript
            potential_txt = input_path.with_suffix(".txt")
            if potential_txt.exists():
                print(f"ℹ️  Auto-detected transcript: {potential_txt.name}")
                transcript_path = str(potential_txt)
            else:
                print("❌ Error: Transcript file required for single mode.")
                sys.exit(1)
        else:
            transcript_path = args.transcript_file

        # 推断输出路径
        if args.output:
            output_path = args.output
        else:
            output_path = str(input_path.with_suffix(".TextGrid"))
            
        if not input_path.exists():
            print(f"❌ Error: Audio file not found: {input_path}")
            sys.exit(1)
            
        # 执行单条对齐 (开启 Verbose 仪表盘)
        try:
            aligner.align(str(input_path), transcript_path, output_path, verbose=True)
            print(f"\n✅ Saved to: {Path(output_path).absolute()}")
        except Exception as e:
            print(f"\n❌ Alignment Failed: {e}")
            sys.exit(1)

    print(f"🕒 Total Runtime: {time.time() - t0:.2f}s")

if __name__ == "__main__":
    main()