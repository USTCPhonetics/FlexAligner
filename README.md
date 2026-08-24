<div align="center">

# 🌊 FlexAligner

### Robust Speech–Text Alignment from Signal to Symbol

[![PyPI](https://img.shields.io/pypi/v/flexaligner.svg)](https://pypi.org/project/flexaligner/)
[![Python](https://img.shields.io/badge/Python-3.10--3.14-blue.svg)](https://www.python.org/)
[![CI](https://github.com/USTCPhonetics/FlexAligner/actions/workflows/ci.yml/badge.svg)](https://github.com/USTCPhonetics/FlexAligner/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Models](https://img.shields.io/badge/Models-Hugging_Face-orange.svg)](https://huggingface.co/USTCPhonetics/FlexAligner)

**A neural forced aligner for real-world speech with local mismatches**<br>
**面向真实非受控语音的高容错强制对齐工具**

[English](#english) | [简体中文](#简体中文) | [Models](https://huggingface.co/USTCPhonetics/FlexAligner) | [PyPI](https://pypi.org/project/flexaligner/) | [Issues](https://github.com/USTCPhonetics/FlexAligner/issues)

</div>

---

# English

## Overview

FlexAligner is a two-stage forced-alignment framework built on wav2vec 2.0. It
is designed for speech recordings that may contain noise, hesitations,
untranscribed events, or local disagreement between the audio and transcript.

The current public alpha supports:

- English and Mandarin CPU single-file alignment;
- word- and phone-level Praat TextGrid output;
- complete timeline coverage with explicit `NULL` intervals;
- local, lexicon-first OOV G2P with visible CLI warnings;
- verified model retrieval from Hugging Face or `hf-mirror.com`;
- optional multi-format decoding, resampling, and PCM16 WAV conversion.

GPU inference, corpus/batch alignment, Web services, automatic language
detection, and confidence calibration are not yet available.

## Quick start

### 1. Install the recommended package

For normal use, install **all currently supported languages and audio
capabilities**:

```bash
python -m pip install "flexaligner[inference,en,zh,audio]==0.3.0a1"
```

This recommended installation provides English G2P, Mandarin segmentation and
G2P, audio conversion, and the frozen inference stack. It supports Python
3.10–3.12. On CPU-only Linux, Torch may be installed from its CPU index first:

```bash
python -m pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install "flexaligner[inference,en,zh,audio]==0.3.0a1"
```

Verify the CLI and installed capabilities:

```bash
flexaligner --help
flexaligner capabilities
```

### 2. Prepare the models

```bash
# English
flexaligner models fetch --language en

# Mandarin
flexaligner models fetch --language zh
```

In an interactive terminal, FlexAligner asks for download consent, cache
directory, and source. The default source is `hf-mirror.com`; choose `official`
to use `huggingface.co`. For automation, authorize the download explicitly:

```bash
flexaligner models fetch --language en --yes --model-source mirror
flexaligner models fetch --language zh --yes --model-source official
```

### 3. Align one file

English:

```bash
flexaligner align \
  --language en \
  --audio recording.wav \
  --text-file transcript.txt \
  --lexicon english.dict \
  --output recording.TextGrid
```

Mandarin:

```bash
flexaligner align \
  --language zh \
  --audio mandarin.wav \
  --text-file mandarin.txt \
  --lexicon mandarin.dict \
  --output mandarin.TextGrid
```

A UTF-8 pronunciation dictionary is required. Dictionary entries always take
priority; local G2P is used only for OOV words and emits a structured `WARNING`
for every generated pronunciation.

## Installation options

The recommended command above is the easiest choice. Smaller installations are
available for controlled deployments:

| Installation | Included capability |
|---|---|
| `flexaligner` | Import-safe core, CLI, model-cache management |
| `flexaligner[inference]` | Frozen Torch/Transformers inference stack |
| `flexaligner[en]` | Local English OOV G2P language pack |
| `flexaligner[zh]` | Mandarin `jieba` segmentation and local `pypinyin` G2P |
| `flexaligner[audio]` | PyAV decoding, resampling, and audio conversion |
| `flexaligner[inference,en,zh,audio]` | **Recommended: all current capabilities** |

The core package is importable on Python 3.10–3.14. Actual alignment requires
the `[inference]` extra, which is currently supported on Python 3.10–3.12 due to
the frozen Torch 2.3.1 and Transformers 4.41.2 stack. The separate
`flexaligner-g2p-en` distribution is installed automatically by `[en]`; users do
not need to call it directly.

## CLI behavior

### Model resolution

When model paths are omitted, `flexaligner align` checks the standard Hugging
Face cache. An interactive cache miss starts the same confirmed download flow
as `models fetch`. A non-interactive cache miss fails with `model_cache_miss`
and prints a copyable command; it never downloads implicitly.

```bash
flexaligner models fetch
flexaligner models fetch --language zh
flexaligner models fetch --yes --model-source mirror
flexaligner models fetch --yes --model-source official \
  --model-cache-dir /data/huggingface/hub
```

The downloader pins model release `v0.2.0a1` to immutable commit
`f9ca09d445e5e8981e43eca6a2f5421526ddc59e`, disables implicit token use, and
validates the manifest plus every selected file's size and SHA-256. It never
silently switches endpoints. A damaged cache fails closed; an explicit
`models fetch --yes` may repair it, but the complete validation must pass again.

To stay completely local, provide both `--chunker-model` and `--aligner-model`.
Providing only one is an error.

### Pronunciation and language behavior

The CLI defaults to `--pronunciation-mode g2p`:

- explicit dictionary entries are never replaced or written back;
- each generated OOV pronunciation produces a structured warning;
- `--pronunciation-mode lexicon` enables strict dictionary-only behavior;
- English local G2P accepts normalized ASCII English words;
- Mandarin uses `jieba` without crossing user-supplied whitespace boundaries;
- Mandarin G2P produces tone-free initial/final phones compatible with the
  current model, but does not claim polyphonic-word disambiguation;
- the current Mandarin model uses `sil` and does not contain `sph`;
- an evident text/dictionary/model language mismatch fails with
  `language_mismatch` before inference.

Important or ambiguous pronunciations should always be supplied explicitly in
the dictionary.

### Audio input

The default contract is strict 16 kHz mono PCM16 WAV. The `[audio]` extra adds
explicit conversion and opt-in preprocessing:

```bash
flexaligner audio convert input.flac canonical.wav
flexaligner align ... --audio-policy auto-resample   # WAV input only
flexaligner align ... --audio-policy multi-format    # PyAV decoding
```

No conversion occurs silently under the default policy.

### Output and safety

- TextGrid output contains continuous `words` and `phones` tiers.
- Uncovered leading, internal, and trailing regions are labeled `NULL`.
- Transcript word order is preserved.
- Existing output files are not overwritten.
- Literal transcript text may be passed with `--text` instead of `--text-file`.
- `--num-threads` sets Torch's process-global CPU thread count.

## Python API

The Python API is intentionally stricter than the interactive CLI: callers
provide validated local models and explicitly select optional fallback behavior.

```python
from pathlib import Path

from flexaligner import (
    AlignmentOptions,
    AlignmentRequest,
    FlexAligner,
    LocalModelBundle,
    PronunciationMode,
    TextGridOutput,
)

models = LocalModelBundle(
    chunker_dir=Path("/local/models/en/chunker"),
    aligner_dir=Path("/local/models/en/aligner"),
)

with FlexAligner(models=models, lexicon_path=Path("english.dict")) as aligner:
    result = aligner.align(
        AlignmentRequest(
            audio_path=Path("recording.wav"),
            transcript=Path("transcript.txt").read_text(encoding="utf-8"),
            output=TextGridOutput(path=Path("recording.TextGrid")),
            utterance_id="recording",
        ),
        options=AlignmentOptions(pronunciation_mode=PronunciationMode.G2P),
    )

print(result.output_sha256)
```

## How it works

```mermaid
graph LR
    A[Audio + Transcript] --> B[Stage 1: CTC Chunking]
    B --> C[Reliable Ordered Chunks]
    C --> D[Stage 2: Pronunciation Graph]
    D --> E[Two-pass Viterbi Decoding]
    E --> F[Words + Phones + NULL TextGrid]
```

1. **CTC chunking** finds reliable transcript anchors and divides long-form
   speech into ordered local chunks.
2. **Local alignment** builds a constrained pronunciation graph and applies
   two-pass Viterbi decoding to estimate word and phone boundaries.

## Source installation

```bash
git clone https://github.com/USTCPhonetics/FlexAligner.git
cd FlexAligner

python -m pip install -e packages/flexaligner-g2p-en
python -m pip install -e ".[inference,en,zh,audio]"
```

---

# 简体中文

## 项目简介

FlexAligner 是一个基于 wav2vec 2.0 的两阶段强制对齐框架，面向真实语料中
常见的噪音、停顿、未转写声音事件，以及音频与文本的局部不一致。

当前公开 alpha 版提供：

- 英语和普通话 CPU 单文件对齐；
- 词级和音素级 Praat TextGrid 输出；
- 使用明确的 `NULL` 区间完整覆盖音频时间轴；
- 词典优先的本地 OOV G2P，并在 CLI 中明确警告；
- 从 Hugging Face 或 `hf-mirror.com` 下载并严格校验固定模型；
- 可选的多格式解码、重采样和 PCM16 WAV 转换。

GPU 推理、语料库/批处理对齐、Web 服务、自动语言识别和置信度校准尚未提供。

## 快速开始

### 1. 安装推荐版本

普通用户建议直接安装**当前全部语言和音频能力**：

```bash
python -m pip install "flexaligner[inference,en,zh,audio]==0.3.0a1"
```

该命令包含英语 G2P、普通话分词与 G2P、音频转换以及固定版本的推理依赖，
适用于 Python 3.10–3.12。CPU-only Linux 可先从 PyTorch CPU 索引安装 Torch：

```bash
python -m pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install "flexaligner[inference,en,zh,audio]==0.3.0a1"
```

验证安装：

```bash
flexaligner --help
flexaligner capabilities
```

### 2. 准备模型

```bash
# 英语
flexaligner models fetch --language en

# 普通话
flexaligner models fetch --language zh
```

在交互终端中，程序会询问是否同意下载、缓存目录和下载源。默认使用
`hf-mirror.com`，也可选择 `official` 访问 `huggingface.co`。自动化环境必须显式授权：

```bash
flexaligner models fetch --language en --yes --model-source mirror
flexaligner models fetch --language zh --yes --model-source official
```

### 3. 对齐单个文件

英语：

```bash
flexaligner align \
  --language en \
  --audio recording.wav \
  --text-file transcript.txt \
  --lexicon english.dict \
  --output recording.TextGrid
```

普通话：

```bash
flexaligner align \
  --language zh \
  --audio mandarin.wav \
  --text-file mandarin.txt \
  --lexicon mandarin.dict \
  --output mandarin.TextGrid
```

必须提供 UTF-8 发音词典。词典中已有的发音始终优先；本地 G2P 只处理 OOV，
并且每次生成发音都会输出结构化 `WARNING`。

## 安装组合

| 安装项 | 包含的能力 |
|---|---|
| `flexaligner` | 可安全 import 的核心、CLI 和模型缓存管理 |
| `flexaligner[inference]` | 固定版本的 Torch/Transformers 推理栈 |
| `flexaligner[en]` | 本地英语 OOV G2P 语言包 |
| `flexaligner[zh]` | 普通话 `jieba` 分词和本地 `pypinyin` G2P |
| `flexaligner[audio]` | PyAV 解码、重采样和音频转换 |
| `flexaligner[inference,en,zh,audio]` | **推荐：当前全部能力** |

基础包可在 Python 3.10–3.14 上 import。真正对齐需要 `[inference]`；由于当前固定
Torch 2.3.1 和 Transformers 4.41.2，推理支持 Python 3.10–3.12。`[en]` 会自动安装
独立的 `flexaligner-g2p-en` 发行包，用户无需单独调用它。

## CLI 行为说明

### 模型下载与缓存

未指定模型路径时，`flexaligner align` 首先检查标准 Hugging Face 缓存。交互缓存缺失会
进入经用户确认的下载流程；非交互环境不会自行下载，而是返回 `model_cache_miss`
并给出可复制的命令。

下载器将模型 release `v0.2.0a1` 固定到不可变提交
`f9ca09d445e5e8981e43eca6a2f5421526ddc59e`，禁止隐式使用 token，并校验 manifest、
每个文件的大小和 SHA-256。下载源之间不会静默切换。残缺或被篡改的缓存会
关闭式失败；显式执行 `models fetch --yes` 可尝试修复，但必须重新通过全部校验。

完全离线使用时，必须同时提供 `--chunker-model` 和 `--aligner-model`；只提供一项会报错。

### 发音、分词与语言校验

CLI 默认使用 `--pronunciation-mode g2p`：

- 词典中的条目不会被替换或回写；
- 每个由 G2P 生成的 OOV 发音都会输出结构化警告；
- `--pronunciation-mode lexicon` 用于严格词典模式；
- 英语 G2P 接受规范化后的 ASCII 英语词；
- 普通话使用 `jieba` 分词，不跨越用户已给定的空格边界；
- 普通话 G2P 生成无声调 initial/final，但不承诺多音字消歧；
- 当前普通话模型使用 `sil`，不包含 `sph`；
- 明显的文本、词典或模型语言错配会在推理前返回 `language_mismatch`。

对重要发音或多音字，应在词典中明确给出发音。

### 音频输入

默认输入契约是严格的 16 kHz 单声道 PCM16 WAV。`[audio]` 提供显式转换和预处理：

```bash
flexaligner audio convert input.flac canonical.wav
flexaligner align ... --audio-policy auto-resample   # 仅 WAV
flexaligner align ... --audio-policy multi-format    # PyAV 解码
```

默认策略下不会静默转换音频。

### 输出与安全边界

- TextGrid 包含连续的 `words` 和 `phones` tier；
- 开头、内部和尾部的未覆盖区域都标记为 `NULL`；
- 保持输入文本的词序；
- 不覆盖已有输出文件；
- 可使用 `--text` 代替 `--text-file` 直接传入文本；
- `--num-threads` 设置 Torch 的进程全局 CPU 线程数。

## 算法概要

1. **CTC 宏观切分：** 寻找可靠的文本锚点，将长音频分成顺序一致的局部片段。
2. **局部微观对齐：** 构建受约束的发音图，通过两遍 Viterbi 解码估计词和音素边界。

Python API 示例、源码安装方式以及更完整的技术行为见上方英文章节。

---

## Roadmap / 路线图

- [x] Two-stage CTC and pronunciation-graph alignment engine
- [x] English and Mandarin CPU single-file alignment
- [x] Complete `words` and `phones` timeline coverage with `NULL`
- [x] Verified model retrieval and local English/Mandarin OOV G2P
- [x] Optional audio decoding, resampling, and conversion
- [x] PyPI public alpha `0.3.0a1`
- [ ] GPU inference and corpus/batch processing
- [ ] Web service and confidence calibration

## Authors & Affiliation / 作者与机构

```text
Yiming Wang (王一鸣) - University of Science and Technology of China (USTC)
Jiahong Yuan (袁家宏) - University of Science and Technology of China (USTC)
```

## Citation / 引用

```bibtex
@misc{flexaligner2026,
  title        = {FlexAligner: Robust Speech--Text Alignment via CTC Chunking and Local Cross-Entropy Alignment},
  author       = {Wang, Yiming and Yuan, Jiahong},
  year         = {2026},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/USTCPhonetics/FlexAligner}},
  organization = {University of Science and Technology of China}
}
```

## License / 许可证

FlexAligner is released under the [MIT License](LICENSE). The optional
`flexaligner-g2p-en` distribution contains a checkpoint derived from `g2p-en`
2.1.0 under Apache-2.0, together with its license and provenance notice. The
minimal MIT-licensed `flexaligner` wheel does not contain that asset.

FlexAligner 使用 [MIT License](LICENSE)。可选的 `flexaligner-g2p-en` 发行包包含源自
`g2p-en` 2.1.0 的 checkpoint，依 Apache-2.0 分发，并随包提供许可证和来源说明。
最小的 MIT 许可 `flexaligner` wheel 不包含该资产。

<div align="center"><sub>Built by USTCPhonetics.</sub></div>
