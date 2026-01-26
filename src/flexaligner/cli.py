import argparse
import sys
import time
from pathlib import Path
from dataclasses import asdict

from flexaligner import FlexAligner
from flexaligner.config import AlignmentConfig, default_config

def main():
    parser = argparse.ArgumentParser(
        description="FlexAligner: A Robust Two-Stage Speech-Text Alignment Framework."
    )
    
    # 核心参数
    parser.add_argument("audio", help="Path to input audio file (.wav)")
    parser.add_argument("transcript", help="Path to transcript file (.txt)")
    
    # 可选参数：如果不传，则自动生成同名 TextGrid
    parser.add_argument("-o", "--output", help="Path to output TextGrid (default: same as audio)")
    
    # 功能开关
    parser.add_argument("--dynamic", action="store_true", 
                        help="Enable dynamic hop for higher precision (compensates for sample rate drift)")
    parser.add_argument("--device", default=default_config.device, help="Compute device (cuda/cpu)")
    parser.add_argument("--beam_size", type=int, default=default_config.beam_size, help="Beam size for Stage 1")
    parser.add_argument("--align_beam", type=int, default=400, help="Beam size for Stage 2")

    args = parser.parse_args()

    # 1. 自动化路径解析
    audio_path = Path(args.audio)
    text_path = Path(args.transcript)
    
    if args.output:
        output_path = Path(args.output)
    else:
        # 默认：/path/to/audio.wav -> /path/to/audio.TextGrid
        output_path = audio_path.with_suffix(".TextGrid")

    # 2. 检查输入是否存在
    if not audio_path.exists():
        print(f"❌ Error: Audio file not found: {audio_path}")
        sys.exit(1)
    if not text_path.exists():
        print(f"❌ Error: Transcript file not found: {text_path}")
        sys.exit(1)

    # 3. 组装配置
    config = AlignmentConfig(
        device=args.device,
        beam_size=args.beam_size,
        align_beam_size=args.align_beam,
        use_dynamic_hop=args.dynamic  # 这里的开关控制了精度模式
    )
    
    # 打印启动信息
    print("="*60)
    print("🚀 FlexAligner (v0.1.0)")
    print(f"   Mode:    {'✨ Dynamic Precision' if args.dynamic else '📜 Baseline (Classic)'}")
    print(f"   Device:  {config.device}")
    print(f"   Output:  {output_path.name}")
    print("="*60)

    t0 = time.time()
    try:
        # 初始化引擎
        aligner = FlexAligner(config=asdict(config))
        
        # 执行对齐
        chunks = aligner.align(str(audio_path), str(text_path), str(output_path))
        
        t_end = time.time()
        print("-" * 60)
        print("✅ Alignment Successful!")
        print(f"   - Chunks: {len(chunks)}")
        print(f"   - Time:   {t_end - t0:.2f}s")
        print(f"   - Saved:  {output_path.absolute()}")
        print("="*60)
        
    except Exception as e:
        print(f"\n[Fatal Error] {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()