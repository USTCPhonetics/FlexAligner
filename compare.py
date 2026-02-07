import os
import sys
import re
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict

# ================= 配置区 =================
DIR_A = Path("tests/testfiles/zh/flexaligner")
DIR_B = Path("tests/testfiles/zh/legacy")
TOLERANCE = 1e-4 
# =========================================

@dataclass
class Interval:
    xmin: float
    xmax: float
    text: str

@dataclass
class Tier:
    name: str
    intervals: List[Interval]

@dataclass
class TextGrid:
    xmin: float
    xmax: float
    tiers: Dict[str, Tier] # 修改为字典，方便按名查找

def parse_textgrid(path: Path) -> TextGrid:
    """简易 TextGrid 解析器"""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    xmin = float(re.search(r"xmin\s*=\s*([0-9\.]+)", content).group(1))
    xmax = float(re.search(r"xmax\s*=\s*([0-9\.]+)", content).group(1))

    tiers = {}
    tier_blocks = re.split(r"item\s*\[\d+\]:", content)[1:]
    
    for block in tier_blocks:
        name_match = re.search(r'name\s*=\s*"(.*?)"', block)
        if not name_match: continue
        name = name_match.group(1).strip()
        
        intervals = []
        int_blocks = re.split(r"intervals\s*\[\d+\]:", block)[1:]
        for ib in int_blocks:
            ixmin = float(re.search(r"xmin\s*=\s*([0-9\.]+)", ib).group(1))
            ixmax = float(re.search(r"xmax\s*=\s*([0-9\.]+)", ib).group(1))
            itext_match = re.search(r'text\s*=\s*"(.*?)"', ib)
            itext = itext_match.group(1) if itext_match else ""
            intervals.append(Interval(ixmin, ixmax, itext))
            
        tiers[name] = Tier(name, intervals)
        
    return TextGrid(xmin, xmax, tiers)

def compare_files(path_a, path_b):
    tg1 = parse_textgrid(path_a)
    tg2 = parse_textgrid(path_b)
    
    report = []
    is_identical = True
    max_diff = 0.0
    
    # 1. 比较总时长
    if abs(tg1.xmax - tg2.xmax) > TOLERANCE:
        is_identical = False
        report.append(f"❌ Duration Mismatch: A={tg1.xmax:.4f}s vs B={tg2.xmax:.4f}s")
    
    # 2. 检查 Tier 集合是否一致
    names_a = set(tg1.tiers.keys())
    names_b = set(tg2.tiers.keys())
    
    if names_a != names_b:
        is_identical = False
        report.append(f"❌ Tier sets differ: A has {names_a}, B has {names_b}")
    
    # 3. 对共有的 Tier 进行内容比对
    common_tiers = names_a.intersection(names_b)
    for tname in common_tiers:
        t1 = tg1.tiers[tname]
        t2 = tg2.tiers[tname]
        
        if len(t1.intervals) != len(t2.intervals):
            is_identical = False
            report.append(f"❌ '{tname}' interval count mismatch: A={len(t1.intervals)} vs B={len(t2.intervals)}")
            continue
            
        for j, (int1, int2) in enumerate(zip(t1.intervals, t2.intervals)):
            d_start = abs(int1.xmin - int2.xmin)
            d_end = abs(int1.xmax - int2.xmax)
            current_max = max(d_start, d_end)
            max_diff = max(max_diff, current_max)
            
            if current_max > TOLERANCE:
                is_identical = False
                report.append(f"❌ '{tname}' Time mismatch Int [{j+1}]: Diff={current_max:.6f}s")
            
            if int1.text != int2.text:
                # 针对语音处理的特殊过滤：忽略 NULL 和 <eps> 的命名差异
                label_a = int1.text.replace("NULL", "<eps>")
                label_b = int2.text.replace("NULL", "<eps>")
                if label_a != label_b:
                    is_identical = False
                    report.append(f"❌ '{tname}' Label mismatch Int [{j+1}]: A='{int1.text}' vs B='{int2.text}'")

    if is_identical:
        return True, ["✅ Identical content (ignoring tier order)"], max_diff
    else:
        return False, report, max_diff

def main():
    print("="*60)
    print("⚖️  TextGrid Comparison Tool (Tier-Agnostic)")
    print("="*60)

    files_a = sorted(list(DIR_A.glob("*.TextGrid")))
    total_files, passed_files = 0, 0
    
    for f_a in files_a:
        f_b = DIR_B / f_a.name
        if not f_b.exists(): continue
            
        total_files += 1
        match, msgs, max_err = compare_files(f_a, f_b)
        
        status = "✅ PASS" if match else "❌ FAIL"
        print(f"[{status}] {f_a.name} | Max Delta: {max_err*1000:.4f} ms")
        if not match:
            for m in msgs[:3]: print(f"      {m}")

    print("\n" + "="*60)
    print(f"📊 Summary: {passed_files}/{total_files} files matched.")
    print("="*60)

if __name__ == "__main__":
    main()