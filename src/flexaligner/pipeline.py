import torch
import numpy as np
from pathlib import Path
from flexaligner.frontend import TextFrontend

# 延迟导入防止循环依赖
# from .chunker import CTCChunker
# from .aligner import LocalAligner

class FlexAligner:
    def __init__(self, config: dict = None):
        self.config = config or {}
        
        # 1. 初始化坦克级前端 (FAST/ROBUST/SECURE 由 config 控制)
        # 默认为 FAST 模式，如果在 config 里指定了 'validation_mode' 则使用指定的
        mode = self.config.get("validation_mode", "FAST")
        self.frontend = TextFrontend(config=self.config, mode=mode)
        
        # 2. 延迟加载核心组件
        from .chunker import CTCChunker
        from .aligner import LocalAligner
        
        # 3. 初始化 Chunker (Stage 1)
        # Chunker 通常跨语言通用 (基于 Wav2Vec2-Large-XLSR 或特定语言模型)
        self.chunker = CTCChunker(self.config)
        
        # 4. 初始化 Aligner (Stage 2)
        # 这里暂时只初始化一个默认 Aligner，未来可以根据语言做 Dict 缓存
        self.aligner = LocalAligner(self.config)

    def align(self, audio_path: str, text_path: str, output_path: str):
        """
        全闭环对齐流程：
        IO -> 清洗 -> 语种识别 -> Chunker切分 -> Local对齐 -> 合并 -> 导出
        """
        # 1. 前端加载 (自带分级防御和重采样)
        # 返回 numpy array (float32, 16k)
        audio_np = self.frontend.load_audio(audio_path)
        
        # 2. 文本处理
        raw_text = self.frontend.load_text(text_path)
        lang = self.frontend.detect_language(raw_text)
        
        # 获取原始音素/分词列表
        raw_tokens = self.frontend.get_phonemes(raw_text, lang)
        
        # 【关键修复】: 物理过滤所有空元素或纯空格
        # 这一步确保了 text_list 的长度和 Chunker 内部查词典后的列表长度严格一致
        text_list = [t.strip() for t in raw_tokens if t.strip()]
        
        print(f"[Pipeline] Detected language: {lang}")
        print(f"[Pipeline] Raw tokens: {len(raw_tokens)} -> Cleaned tokens: {len(text_list)}")

        # 3. 数据转换 (Numpy -> Tensor)
        # Chunker 和 Aligner 都需要 Tensor 输入
        audio_tensor = torch.from_numpy(audio_np).float()
        audio_duration = audio_tensor.size(0) / 16000.0

        # 4. Stage 1: Chunker 切分
        # 注意：text_list 此时已经是清洗过的列表，绝无空格
        chunks = self.chunker.find_chunks(audio_tensor, text_list)

        global_phones = []
        global_words = []
        
        # 记录上一个片段的结束时间，用于检测静音空隙
        prev_end_time = 0.0

        for chunk in chunks:
            # 5. 检测间隙 (Gap)：如果当前 Chunk 的开始时间 > 上一个的结束时间，补 NULL
            if chunk.start_time > prev_end_time + 1e-6:
                null_seg = ("NULL", prev_end_time, chunk.start_time)
                global_phones.append(null_seg)
                global_words.append(null_seg)

            # 6. Stage 2: 局部对齐 (Local Alignment)
            # chunk.tensor 是切出来的音频片段，chunk.text 是对应的文本片段
            local_result = self.aligner.align_locally(chunk.tensor, chunk.text)
            
            # 计算全局偏移量
            offset = chunk.start_time
            
            # 7. 合并结果 (应用 12.5ms Offset 逻辑是在 align_locally 内部完成的，这里直接叠加)
            for seg in local_result["phones"]:
                global_phones.append((seg.label, offset + seg.start, offset + seg.end))
                
            for seg in local_result["words"]:
                global_words.append((seg.label, offset + seg.start, offset + seg.end))
            
            # 更新游标
            prev_end_time = chunk.end_time

        # 8. 扫尾：补全文件末尾的静音
        if prev_end_time < audio_duration - 1e-6:
            last_null = ("NULL", prev_end_time, audio_duration)
            global_phones.append(last_null)
            global_words.append(last_null)

        # 9. 物理导出
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
        # 1. 确保路径存在
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        
        # 2. 格式化工具闭包
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
                # 转义引号，防止文本破坏 TextGrid 结构
                safe_label = label.replace('"', '""')
                lines.append(f'            text = "{safe_label}"')
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
        # 保证 tier 顺序
        for name in ["phones", "words"]:
            if name in tiers_data:
                lines.append(f'    item [{tier_idx}]:')
                lines.extend(format_tier(name, tiers_data[name]))
                tier_idx += 1
                
        # 4. 物理写入
        content = "\n".join(lines) + "\n"
        p.write_text(content, encoding="utf-8")
        
        print(f"[Pipeline] Successfully wrote TextGrid to: {p.absolute()}")