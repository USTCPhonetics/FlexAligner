import os
from pathlib import Path

def merge_lexicons(custom_dict_path, output_path):
    """
    将手动打磨的 custom_dict 与 g2p 的 CMU 词典融合，生成适用于 MFA 的超级词典。
    """
    merged_lexicon = {} # key: WORD, value: set of phoneme strings

    # 1. 第一优先级：加载你辛苦打磨的 en.dict (含 Buckeye 特种残骸)
    if os.path.exists(custom_dict_path):
        print(f"📖 正在加载手工词典: {custom_dict_path}")
        with open(custom_dict_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    word = parts[0].upper()
                    phones = " ".join(parts[1:])
                    if word not in merged_lexicon:
                        merged_lexicon[word] = set()
                    merged_lexicon[word].add(phones)
        print(f"✅ 已装载 {len(merged_lexicon)} 个手工词条 (包含特种残骸)。")

    # 2. 第二优先级：吸纳 g2p_en 的 CMU 全量词库进行增量补充
    try:
        print("🚀 正在从 g2p_en 提取 CMU 词库...")
        from g2p_en.g2p import G2p
        g2p_inst = G2p()
        cmu_dict = g2p_inst.cmu # 格式: {word: [pron1, pron2]}
        
        added_count = 0
        for word, prons in cmu_dict.items():
            w_upper = word.upper()
            # 如果 custom_dict 里已经有了（比如你手动修过的 WUZH），绝对不覆盖
            if w_upper not in merged_lexicon:
                merged_lexicon[w_upper] = set()
                for p_list in prons:
                    # 剔除标点，合并音素
                    phones = " ".join([p for p in p_list if p.isalnum()])
                    if phones:
                        merged_lexicon[w_upper].add(phones)
                added_count += 1
        print(f"✅ G2P 增量补全完成：补充了 {added_count} 个通用词汇。")
    except ImportError:
        print("❌ 错误：未安装 g2p_en，请先 pip install g2p_en")
        return

    # 3. 物理持久化：输出 buckeye.dict
    # MFA 格式建议：单词和音素之间使用 Tab 键，单词按字母序排列
    print(f"💾 正在锻造终极词典: {output_path}")
    sorted_words = sorted(merged_lexicon.keys())
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for word in sorted_words:
            # 一个单词如果有多种发音，按 MFA 规范需分行书写
            for phones in sorted(merged_lexicon[word]):
                f.write(f"{word}\t{phones}\n")

    print(f"🏁 任务完成！总词条数: {len(merged_lexicon)}")
    print(f"📍 路径: {Path(output_path).absolute()}")

if __name__ == "__main__":
    # 配置你的路径
    CUSTOM_DICT = "assets/dictionaries/en.dict"
    OUTPUT_DICT = "assets/dictionaries/buckeye.dict"
    
    merge_lexicons(CUSTOM_DICT, OUTPUT_DICT)