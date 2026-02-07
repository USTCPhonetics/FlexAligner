import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from praatio import textgrid
from flexaligner.pipeline import FlexAligner
from flexaligner.config import AlignmentConfig
from dataclasses import asdict

# --- 路径定义 ---
EXAMPLES_DIR = Path("assets/examples")
MFA_DIR = Path("assets/mfa_output")
OUTPUT_DIR = Path("assets/flex_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_intervals(tg_path, tier_name='words'):
    """加载 TextGrid 单词层，过滤静音"""
    try:
        tg = textgrid.openTextgrid(str(tg_path), includeEmptyIntervals=False)
        # 探测正确的层级名 (MFA 可能是 'words', Flex 可能是 'words')
        t_names = tg.tierNames if hasattr(tg, 'tierNames') else list(tg.tierDict.keys())
        target = next((n for n in t_names if n.lower() == tier_name.lower()), t_names[0])
        tier = tg.getTier(target)
        return [e for e in tier.entries if e.label.lower() not in ['sil', 'null', '', 'sp', '<eps>']]
    except Exception as e:
        print(f"   [Error] Loading {tg_path.name}: {e}")
        return []

def run_benchmark():
    # 1. 初始化对齐器 (根据实际物理发现，英语用 10ms, 中文用 10ms)
    # 这里我们创建两个实例以应对不同语种的模型特性
    aligners = {
        "en": FlexAligner(config=asdict(AlignmentConfig(lang="en", frame_hop_s=0.01, device="cpu"))),
        "zh": FlexAligner(config=asdict(AlignmentConfig(lang="zh", frame_hop_s=0.01, device="cpu")))
    }

    # 查找所有音频文件
    audio_exts = ['.wav', '.flac', '.mp3']
    audio_files = [f for f in EXAMPLES_DIR.iterdir() if f.suffix.lower() in audio_exts]
    
    results = []

    print(f"🚀 Starting Benchmark: Found {len(audio_files)} files.")
    print("=" * 90)
    print(f"{'FILENAME':<15} | {'WORDS':<6} | {'MAE (ms)':<10} | {'MAX ERR':<10} | {'STATUS'}")
    print("-" * 90)

    for audio_path in audio_files:
        stem = audio_path.stem
        txt_path = audio_path.with_suffix('.txt')
        mfa_tg = MFA_DIR / f"{stem}.TextGrid"
        flex_tg = OUTPUT_DIR / f"{stem}.TextGrid"

        if not txt_path.exists() or not mfa_tg.exists():
            print(f"{stem:<15} | {'-':<6} | {'-':<10} | {'-':<10} | Missing Files")
            continue

        # A. 执行 FlexAligner
        lang = "zh" if "zh" in stem or any('\u4e00' <= c <= '\u9fff' for c in stem) else "en"
        aligners[lang].align(str(audio_path), str(txt_path), str(flex_tg))

        # B. 数据比对
        mfa_words = load_intervals(mfa_tg)
        flex_words = load_intervals(flex_tg)

        if len(mfa_words) != len(flex_words):
            status = f"Word Mismatch ({len(mfa_words)} vs {len(flex_words)})"
            results.append({"file": stem, "mae": np.nan})
            print(f"{stem:<15} | {len(mfa_words):<6} | {'-':<10} | {'-':<10} | {status}")
            continue

        # 计算误差
        errors = [abs(m.start - f.start) * 1000 for m, f in zip(mfa_words, flex_words)]
        mae = np.mean(errors)
        max_err = np.max(errors)
        
        print(f"{stem:<15} | {len(mfa_words):<6} | {mae:>8.2f}ms | {max_err:>8.2f}ms | OK")
        results.append({"file": stem, "mae": mae, "max": max_err, "words": len(mfa_words), "lang": lang})

    print("-" * 90)
    # 总结
    df = pd.DataFrame(results).dropna()
    if not df.empty:
        total_mae = df['mae'].mean()
        print(f"🏆 GLOBAL MEAN ABSOLUTE ERROR: {total_mae:.2f} ms")
        df.to_csv("benchmark_report.csv", index=False)
        print(f"📝 Full report saved to benchmark_report.csv")

if __name__ == "__main__":
    run_benchmark()