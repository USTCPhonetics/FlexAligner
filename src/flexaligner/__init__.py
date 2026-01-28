# 定义版本号，这是规范
__version__ = "0.1.0"

# 从内部模块把核心类“提溜”到顶层
from .pipeline import FlexAligner
from .io import AudioChunk, AlignmentResult
from .chunker import CTCChunker
from .aligner import LocalAligner

# 定义当别人 from flexaligner import * 时，哪些东西会被导出
__all__ = [
    "FlexAligner",
    "AudioChunk",
    "AlignmentResult",
    "CTCChunker",
    "LocalAligner",
    "TextFrontend"
]