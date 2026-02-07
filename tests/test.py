from praatio import textgrid
import numpy as np

def calculate_mae(ref_path, pred_path):
    # 加载 TextGrid
    tg_ref = textgrid.openTextgrid(ref_path, includeEmptyIntervals=False)
    tg_pred = textgrid.openTextgrid(pred_path, includeEmptyIntervals=False)
    
    # 提取 words 层级
    ref_entries = tg_ref.getTier('words').entries
    pred_entries = tg_pred.getTier('words').entries
    
    # 过滤掉静音标签
    ref_words = [e for e in ref_entries if e.label.lower() not in ['sil', 'null', '']]
    pred_words = [e for e in pred_entries if e.label.lower() not in ['sil', 'null', '']]
    
    print(f"\n📏 Comparing boundaries for {len(ref_words)} words...")
    
    errors = []
    for r, p in zip(ref_words, pred_words):
        start_diff = abs(r.start - p.start) * 1000  # 转为 ms
        end_diff = abs(r.end - p.end) * 1000
        errors.append(start_diff)
        print(f"Word: {r.label:<10} | Start Err: {start_diff:>6.2f}ms | End Err: {end_diff:>6.2f}ms")
    
    print("-" * 40)
    print(f"🏆 Mean Absolute Error (MAE): {np.mean(errors):.2f} ms")

if __name__ == "__main__":
    calculate_mae("assets/mfa_output/en.TextGrid", "inspect_result.TextGrid")