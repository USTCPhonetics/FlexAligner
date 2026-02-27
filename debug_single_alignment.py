import sys
import os
from pathlib import Path
import torch

# [极客防线] 离线锁定
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

PROJECT_ROOT = Path("/home/wangyiming/projects/FlexAligner")
sys.path.append(str(PROJECT_ROOT / "src"))

from flexaligner.pipeline import FlexAligner
from flexaligner.config import AlignmentConfig
from flexaligner.frontend import TextFrontend

# =====================================================================
# 🔬 探针 1：前端文本清洗拦截器 (Frontend Interceptor)
# =====================================================================
original_clean_text = TextFrontend.clean_text

def probe_clean_text(self, text, lang):
    cleaned = original_clean_text(self, text, lang)
    print("\n" + "="*50)
    print(f"🔬 [探针 1] TextFrontend 文本清洗结果")
    print(f"  原始文本: {text}")
    print(f"  清洗输出: {cleaned}")
    print("="*50)
    return cleaned

TextFrontend.clean_text = probe_clean_text

# =====================================================================
# 🔬 探针 2：词典寻址拦截器 (Lexicon Interceptor)
# =====================================================================
# 我们要在字典获取发音的瞬间，看看它到底在查什么词，有没有被静默跳过
def inject_lexicon_probe(aligner):
    if hasattr(aligner, 'aligner') and hasattr(aligner.aligner, 'lexicon'):
        original_get_prons = aligner.aligner.lexicon.get_prons
        
        def probe_get_prons(word):
            try:
                prons = original_get_prons(word)
                # print(f"  [Lexicon] 命中: '{word}' -> {prons}")
                return prons
            except KeyError:
                print(f"  🚨 [探针 2 警报] 词典未命中 (OOV): '{word}' --> 即将被 Epsilon 捷径静默跳过！")
                raise # 继续抛出，让底层的 try-except 捕获，维持原逻辑
                
        aligner.aligner.lexicon.get_prons = probe_get_prons

# =====================================================================
# 🚀 活体解剖主程序
# =====================================================================
def main():
    # 锁定目标样本
    WAV_PATH = PROJECT_ROOT / "tests/testfiles/en/DR1_FAKS0_SA1.wav"
    TXT_PATH = PROJECT_ROOT / "tests/testfiles/en/DR1_FAKS0_SA1.dict2.txt"
    OUTPUT_TG = PROJECT_ROOT / "debug_DR1_FAKS0_SA1.TextGrid"
    
    SEG_MODEL = PROJECT_ROOT / "models_hidden/en/hf_phs"
    ALIGN_MODEL = PROJECT_ROOT / "models_hidden/en/ce17.6000"
    DICT_PATH = PROJECT_ROOT / "assets/dictionaries/dict2"
    
    print(f"🔪 开始活体解剖: {WAV_PATH.name}")

    # 1. 配置引擎 (关闭 G2P，因为输入已经是纯音素)
    config = AlignmentConfig(
        lang="en",
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        chunk_model_path=str(SEG_MODEL),
        align_model_path=str(ALIGN_MODEL),
        lexicon_path=str(DICT_PATH),
        # use_g2p=False,  # 绝对不能开，否则 G2P 会去翻译 "IY1" 这种音素
        sil_phone="sil",
        optional_sil=True,
        sil_cost=-0.5,
        verbose=True # 开启底层日志
    )
    config.raw_phoneme_mode = True
    aligner = FlexAligner(config)
    
    # 注入词典探针
    inject_lexicon_probe(aligner)
    
    # 2. 读取原始标注序列用于最终比对
    with open(TXT_PATH, 'r', encoding='utf-8') as f:
        raw_input_seq = f.read().strip().split()

    # 3. 单点执行对齐
    tasks = [(str(WAV_PATH), str(TXT_PATH), str(OUTPUT_TG))]
    aligner.align_batch(tasks)

    # 4. 🔬 探针 3：提取结果进行对比
    if not OUTPUT_TG.exists():
        print("\n❌ 致命错误：TextGrid 未生成，流水线崩溃。")
        return

    import textgrid
    tg = textgrid.TextGrid.fromFile(str(OUTPUT_TG))
    target_tier = next((t for t in tg if t.name.lower() in ["phones", "words"]), tg[0])
    output_seq = [i.mark for i in target_tier if i.mark and i.mark.strip() not in ["", " "]]

    print("\n" + "="*50)
    print(f"🔬 [探针 3] 最终序列对比 (Input vs Output)")
    print(f"  输入序列长度: {len(raw_input_seq)}")
    print(f"  输出序列长度: {len(output_seq)}")
    
    limit = min(len(raw_input_seq), len(output_seq))
    for i in range(limit):
        inp = raw_input_seq[i] if i < len(raw_input_seq) else "N/A"
        out = output_seq[i] if i < len(output_seq) else "N/A"
        status = "✅" if inp.upper() == out.upper() else "❌ (偏移或丢失)"
        print(f"  [{i:02d}] IN: {inp:<8} | OUT: {out:<8} | {status}")
    print("="*50)

if __name__ == "__main__":
    main()