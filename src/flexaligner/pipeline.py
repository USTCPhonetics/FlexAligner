from pathlib import Path
from .io import load_audio, load_text
# 延迟导入防止循环依赖，但在函数内部导入是安全的
# from .chunker import CTCChunker
# from .aligner import LocalAligner

class FlexAligner:
    def __init__(self, config: dict = None):
        self.config = config or {}
        
        # 延迟加载核心组件
        from .chunker import CTCChunker
        from .aligner import LocalAligner
        
        # 初始化两个车间
        self.chunker = CTCChunker(self.config)
        self.aligner = LocalAligner(self.config)

    def align(self, audio_path: str, text_path: str, output_path: str):
        audio_tensor = load_audio(audio_path)
        text_list = load_text(text_path)
        audio_duration = audio_tensor.size(0) / 16000.0

        # 1. Chunker 切分
        chunks = self.chunker.find_chunks(audio_tensor, text_list)

        global_phones = []
        global_words = []
        
        # [关键改动] 记录上一个片段的结束时间，初始为 0
        prev_end_time = 0.0

        for chunk in chunks:
            # 2. 检测间隙：如果当前 Chunk 的开始时间 > 上一个的结束时间，说明有 NULL 区
            if chunk.start_time > prev_end_time + 1e-6:
                null_seg = ( "NULL", prev_end_time, chunk.start_time )
                global_phones.append(null_seg)
                global_words.append(null_seg)

            # 3. 局部对齐
            local_result = self.aligner.align_locally(chunk.tensor, chunk.text)
            offset = chunk.start_time
            
            for seg in local_result["phones"]:
                global_phones.append((seg.label, offset + seg.start, offset + seg.end))
                
            for seg in local_result["words"]:
                global_words.append((seg.label, offset + seg.start, offset + seg.end))
            
            # 更新结束时间
            prev_end_time = chunk.end_time

        # 4. 扫尾：如果最后一段离音频结束还有距离，补一个 NULL
        if prev_end_time < audio_duration - 1e-6:
            last_null = ("NULL", prev_end_time, audio_duration)
            global_phones.append(last_null)
            global_words.append(last_null)

        # 5. 导出
        self._export_textgrid(
            output_path, 
            audio_duration, 
            {"phones": global_phones, "words": global_words}
        )
        
        return chunks

    def _export_textgrid(self, path: str, duration: float, tiers_data: dict):
        """
        [工业级导出] 确保格式严格对齐，且文件物理落地
        """
        # 1. 确保路径存在 (转为 Path 对象)
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        
        # 2. 模拟原脚本的格式化
        def fmt(val):
            return f"{val:.6f}"

        def format_tier(name, segments):
            lines = []
            lines.append('        class = "IntervalTier"')
            lines.append(f'        name = "{name}"')
            lines.append('        xmin = 0') 
            lines.append(f'        xmax = {fmt(duration)}') 
            lines.append(f'        intervals: size = {len(segments)}')
            
            for i, (label, start, end) in enumerate(segments):
                lines.append(f'        intervals [{i+1}]:')
                lines.append(f'            xmin = {fmt(start)}')
                lines.append(f'            xmax = {fmt(end)}')
                lines.append(f'            text = "{label}"')
            return lines

        # 3. 构建正文
        lines = [
            'File type = "ooTextFile"',
            'Object class = "TextGrid"',
            '',
            'xmin = 0',
            f'xmax = {fmt(duration)}',
            'tiers? <exists>',
            f'size = {len(tiers_data)}',
            'item []:'
        ]
        
        tier_idx = 1
        for name in ["phones", "words"]:
            if name in tiers_data:
                lines.append(f'    item [{tier_idx}]:')
                lines.extend(format_tier(name, tiers_data[name]))
                tier_idx += 1
                
        # 4. 物理写入
        content = "\n".join(lines) + "\n"
        p.write_text(content, encoding="utf-8")
        
        # 🔴 Debug 打印：确保这行代码被执行了
        print(f"[Pipeline] Successfully wrote TextGrid to: {p.absolute()}")