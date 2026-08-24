<div align="center">

# 🌊 FlexAligner

### Robust Speech-Text Alignment from Signal to Symbol

[![Laboratory](https://img.shields.io/badge/Laboratory-USTC_Phonetics-red.svg)](http://phonetics.ustc.edu.cn/)
[![Python](https://img.shields.io/badge/Python-3.10--3.14-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Model](https://img.shields.io/badge/Model-Wav2Vec2.0-orange.svg)](https://huggingface.co/USTCPhonetics)

**A Neural-Based Forced Alignment Framework for "Wild" Real-World Data**
<br>
**面向真实非受控数据的深度学习强鲁棒性对齐工具**

[**English**](#-introduction) | [**简体中文**](#-简介)

</div>

---

## 📖 Introduction

**FlexAligner** is a robust speech-text alignment framework built upon
**wav2vec 2.0**. It is designed for real-world linguistic data, where audio
signals and textual transcriptions may contain noise, hesitations, untranscribed
events, or local mismatches.

FlexAligner decomposes forced alignment into two stages:

1. **Macro-Segmentation (CTC Chunking):** Uses a CTC acoustic model to locate
   reliable transcript anchors and divide long-form audio into ordered chunks.
2. **Micro-Alignment (Local Alignment):** Uses a constrained pronunciation
   graph and two-pass Viterbi decoding to estimate word and phone boundaries
   within each chunk.

### 🌟 Key Features

* **🛡️ Tolerance to Mismatch:** Uncovered portions of the audio timeline are
  represented explicitly as `NULL` intervals instead of being silently forced
  into neighboring words or phones.
* **🎯 Word and Phone Boundaries:** Produces Praat TextGrid files with continuous
  `words` and `phones` tiers while preserving the input word order.
* **🔒 Local and Reproducible:** Explicit model paths remain fully local. When
  model paths are omitted, the CLI validates the selected pinned language bundle
  in the Hugging Face cache and asks before downloading it. Language-specific OOV
  pronunciations are generated locally and always reported as CLI warnings.
* **📦 Python Package and CLI:** Provides a typed Python API and a command-line
  interface for single-file alignment.

The `0.3.0a1` preview provides **English and Mandarin CPU single-file
alignment**, optional local OOV G2P language packs, `jieba` segmentation for
Mandarin, and an optional audio conversion frontend. GPU, batch processing, Web
services, and confidence calibration remain reserved interfaces.

---

## 🌏 简介

**FlexAligner** 是一个基于 **wav2vec 2.0** 的语音—文本对齐框架，面向真实语言材料中
常见的噪音、停顿、未转写声音事件以及音频与文本局部不一致等问题。

FlexAligner 将强制对齐分为两个阶段：

1. **宏观切分（CTC Chunking）：** 使用 CTC 声学模型寻找可靠的文本锚点，并将长音频
   划分为顺序一致的局部片段。
2. **微观对齐（Local Alignment）：** 在每个片段内构建受约束的发音图，通过两遍
   Viterbi 解码估计词和音素边界。

### 🌟 核心优势

* **🛡️ 容错设计：** 对未被词或音素覆盖的时段使用明确的 `NULL` 区间表示，避免将其
  静默挤压到相邻标签中。
* **🎯 词与音素边界：** 输出 Praat TextGrid，`words` 和 `phones` 两层连续覆盖完整
  音频时间轴，同时保持输入词序。
* **🔒 本地与可复现：** 显式模型路径始终只在本地使用。未指定模型路径时，CLI 会先
  校验 Hugging Face 默认缓存中所选语言的固定模型；缓存缺失时必须获得用户确认才会
  下载。不同语言的 OOV 发音均在本地生成，CLI 每次都会明确输出 warning。
* **📦 Python 包与 CLI：** 提供带类型定义的 Python API 和单文件命令行接口。

`0.3.0a1` 预览版提供**英语与普通话 CPU 单文件对齐**、可选的本地 OOV G2P 语言包、
普通话 `jieba` 分词，以及可选音频转换前端。GPU、批处理、Web 服务和置信度校准仍为
预留接口。

---

## 🏗️ Architecture

```mermaid
graph TD
    Input[Input: PCM16 WAV + Transcript] --> Lexicon[Local Pronunciation Lexicon];
    Lexicon --> B[Stage 1: CTC Chunking];
    B --> C{Reliable Ordered Chunks};
    C --> D[Stage 2: Pronunciation Graph];
    D --> E[Two-pass Viterbi Decoding];
    E --> F[Words + Phones + NULL TextGrid];
```

## 🚀 Installation

The import-safe core package targets Python 3.10–3.14. The frozen inference
extra pins Torch 2.3.1 and Transformers 4.41.2 and is installable on Python
3.10–3.12. Real-model release evidence currently covers only Linux x86_64 with
Python 3.10.8; Python 3.13–3.14 are core-only.

Install the `0.3.0a1` preview with English local OOV G2P and the frozen
inference stack:

```bash
python -m pip install "flexaligner[inference,en]==0.3.0a1"
```

For a CPU-only Linux environment, install the frozen Torch build from its CPU
index first:

```bash
python -m pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install "flexaligner[inference,en]==0.3.0a1"
```

To work from a reviewed source checkout:

```bash
git clone https://github.com/USTCPhonetics/FlexAligner.git
cd FlexAligner

python -m pip install -e ".[inference]"
```

The minimal package does not contain the English G2P checkpoint and does not
install Mandarin segmentation/G2P or audio conversion dependencies. Add only
the incremental capabilities you need. For a source checkout, install the
English language pack before selecting `[en]`:

```bash
# English local OOV G2P: separately packaged checkpoint
python -m pip install -e packages/flexaligner-g2p-en
python -m pip install -e ".[inference,en]"

# Mandarin alignment: inference stack + jieba + pypinyin
python -m pip install -e ".[inference,zh]"

# Add explicit decoding, resampling and PCM16 WAV conversion
python -m pip install -e ".[inference,zh,audio]"
```

The package does not include acoustic-model weights. The CLI can retrieve the
selected pinned `en` or `zh` bundle into the standard Hugging Face cache after
explicit confirmation. A pronunciation dictionary is still required; local G2P
is used only for words missing from that dictionary.

基础包支持 Python 3.10–3.14。首个 alpha 的推理依赖固定为 Torch 2.3.1 和
Transformers 4.41.2，仅支持在 Python 3.10–3.12 上安装；真实模型发布验证目前仅覆盖
Linux x86_64 与 Python 3.10.8。Python 3.13–3.14 暂时只承诺基础包和接口可用。
wheel 不包含声学模型权重。最小安装也不包含英语 G2P checkpoint、`jieba`、
`pypinyin` 或 PyAV；英语本地 G2P 使用 `[en]` extra 和独立
`flexaligner-g2p-en` distribution，普通话能力使用 `[zh]` extra，转码与重采样使用
`[audio]` extra。CLI 可在用户明确确认后，把固定的 `en` 或 `zh` 模型下载到标准
Hugging Face 缓存。发音词典仍必须由用户提供；本地 G2P 只处理词典 OOV，已有词典
条目始终优先。

## 💻 Usage

### 1. Command Line Interface (CLI)

Align one English 16 kHz mono PCM16 WAV file. If both model options are omitted,
the CLI first checks the default Hugging Face cache. On a cache miss in an
interactive terminal, it asks for download consent, cache directory (press
Enter for the default), and source (the default is `hf-mirror.com`; choose
`official` for `huggingface.co`):

```bash
flexaligner align \
  --audio recording.wav \
  --text-file transcript.txt \
  --lexicon english.dict \
  --output recording.TextGrid \
  --chunk-metadata recording.alignment.json \
  --num-threads 1
```

To fetch models before alignment, including in automation, run:

```bash
flexaligner models fetch
flexaligner models fetch --yes --model-source mirror
flexaligner models fetch --yes --model-source official \
  --model-cache-dir /data/huggingface/hub
flexaligner models fetch --language zh --yes --model-source official
```

The downloader pins release `v0.2.0a1` to the immutable commit
`f9ca09d445e5e8981e43eca6a2f5421526ddc59e`, requests only the twelve files for
the selected language, disables implicit token use, and validates the built-in manifest
hash plus every file size and SHA-256. It never falls back from one endpoint to
another. A non-interactive cache miss fails with `model_cache_miss` unless
`--yes` explicitly authorizes downloading. An incomplete or hash-invalid cache
fails closed without `--yes`; explicit `models fetch --yes` force-downloads the
pinned files and accepts the repaired cache only after the complete 12-file
size and SHA-256 validation passes again.

To bypass cache resolution and all network behavior, provide both
`--chunker-model` and `--aligner-model`. Providing only one is an error.

The CLI defaults to `--pronunciation-mode g2p`. Explicit dictionary entries are
never replaced or written back. Each generated OOV pronunciation emits one
structured `WARNING` on stderr with the word, occurrence indices, phones, and
G2P engine version. Use `--pronunciation-mode lexicon` for strict dictionary-only
behavior; an OOV then fails before inference. English OOV generation requires
the `[en]` extra. Its separately packaged checkpoint performs no network access
and supports normalized ASCII English words only. If the extra is absent, only
an actual English OOV request fails with `optional_dependency_missing`.

Mandarin alignment uses `--language zh`. The `[zh]` extra segments unspaced text
with `jieba`; user-supplied whitespace remains a hard segmentation boundary.
The current Mandarin model uses `sil` and has no `sph` output category, so the
Mandarin Stage 2 graph never creates optional `sph` states:

```bash
flexaligner align \
  --language zh \
  --audio mandarin.wav \
  --text-file mandarin.txt \
  --lexicon mandarin.dict \
  --output mandarin.TextGrid
```

The Mandarin G2P fallback uses local, tone-free initial/final phones compatible
with the current model vocabulary. Polyphonic-word quality is not claimed;
provide an explicit dictionary entry whenever pronunciation matters.

The optional `[audio]` extra provides explicit conversion and opt-in alignment
policies while the default remains strict PCM16 WAV:

```bash
flexaligner audio convert input.flac canonical.wav
flexaligner align ... --audio-policy auto-resample   # WAV input only
flexaligner align ... --audio-policy multi-format    # explicit PyAV decoding
```

未指定两项模型参数时，CLI 会先检查默认 Hugging Face 缓存。交互终端发现缓存缺失后，
依次询问是否下载、缓存目录（直接回车使用默认目录）和下载来源（默认使用国内可访问的
`hf-mirror.com`，也可选择 `official`）。非交互环境不会自行下载，必须提前执行
`flexaligner models fetch --yes`。显式同时提供 `--chunker-model` 和
`--aligner-model` 时，会完全绕过缓存与网络；只提供其中一项会报错。
残缺或 hash 不匹配的 cache 在普通命令下会关闭式失败；只有显式执行
`flexaligner models fetch --yes` 才会强制重新下载，且必须再次通过完整 12 文件的
size 与 SHA-256 校验后才能使用。

CLI 默认使用 `--pronunciation-mode g2p`。显式词典条目不会被覆盖或写回文件；每个由
G2P 生成的 OOV 发音都会在 stderr 输出一条结构化 `WARNING`，包含词、出现位置、音素
和引擎版本。需要严格词典模式时使用 `--pronunciation-mode lexicon`，此时 OOV 会在
推理前失败。英语 OOV 生成需要 `[en]` extra；checkpoint 位于独立的
`flexaligner-g2p-en` distribution，不联网，当前只接受规范化后的 ASCII 英语词。
未安装 `[en]` 时，只有真实发生英语 OOV G2P 请求才返回
`optional_dependency_missing`；完整词典对齐不受影响。

普通话使用 `--language zh`。`[zh]` extra 通过 `jieba` 对无空格文本分词，用户已有的
空格边界不会被跨越；本地 `pypinyin` G2P 只为词典 OOV 生成与当前模型兼容的无声调
initial/final，并输出结构化 warning。多音字质量不作保证，重要发音应写入用户词典。
当前普通话模型包含 `sil`、不包含 `sph`，因此普通话 Stage 2 不构建任何可选 `sph`
状态。若文本、词典或模型与 `--language` 明显不匹配，程序会在推理前返回
`language_mismatch`。

`[audio]` extra 提供显式 `audio convert`、`auto-resample` 和 `multi-format` 能力；
默认输入契约仍是严格的 16 kHz 单声道 PCM16 WAV，不会静默转换。

Literal transcript text may be passed with `--text` instead of `--text-file`.
`--num-threads` configures Torch's process-global CPU thread count for the
inference lifetime; it is not isolated to a single aligner instance.
Use the capability command to inspect the installed preview:

```bash
flexaligner capabilities
flexaligner capabilities --json
```

### 2. Python API

```python
from pathlib import Path

from flexaligner import (
    AlignmentRequest,
    AlignmentOptions,
    FlexAligner,
    LocalModelBundle,
    PronunciationMode,
    TextGridOutput,
)

models = LocalModelBundle(
    chunker_dir=Path("/local/models/en/chunker"),
    aligner_dir=Path("/local/models/en/aligner"),
)

with FlexAligner(
    models=models,
    lexicon_path=Path("english.dict"),
) as aligner:
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

### Input Requirements / 输入要求

* Audio must be uncompressed 16 kHz mono PCM16 WAV unless an optional audio
  policy is explicitly selected.
* The transcript and dictionary must be UTF-8 and match `--language`. CLI G2P
  mode may fill dictionary OOVs with warnings; strict mode and the Python default
  require full coverage.
* Models must resolve to validated local directories before inference; the
  dictionary must be available locally.
* Output paths use no-clobber semantics: an existing output file is not
  overwritten.

* 除非显式选择 `[audio]` 提供的策略，音频必须是未压缩的 16 kHz 单声道 PCM16 WAV。
* 文本和词典必须为 UTF-8，并与 `--language` 一致。CLI G2P 模式可在 warning 后填补
  词典 OOV；严格模式和 Python API 默认模式仍要求词典完整覆盖。
* 推理开始前模型必须解析为已校验的本地目录；词典必须提前保存在本地。
* 输出采用不覆盖已有文件的策略；若目标文件已存在，程序不会将其覆盖。
* `--num-threads` 会设置 Torch 的进程全局 CPU 线程数，并非仅对单个 aligner 实例生效。

## 🗓️ Roadmap

- [x] **Core Alignment Engine:** Two-stage CTC chunking and local alignment.
- [x] **English CPU Single-File Alignment:** CLI, Python API, and validated
  TextGrid output.
- [x] **Continuous TextGrid Coverage:** `words` and `phones` tiers use `NULL`
  intervals to cover the complete timeline.
- [x] **Mandarin Development Path:** Current models, `jieba` segmentation,
  lexicon-first local G2P, `sil`-only Stage 2, and real-sample validation.
- [ ] **GPU and Batch Processing:** Accelerated and high-throughput workflows.
- [x] **Optional Audio Frontend:** Explicit PyAV decoding, resampling and
  canonical PCM16 WAV conversion in the development tree.
- [x] **Validated English Model Retrieval:** Pinned cache lookup, confirmed
  download, mirror/official selection, and manifest/hash verification.
- [x] **Local English OOV G2P:** Lexicon-first ARPAbet fallback with structured
  CLI warnings and strict vocabulary validation.
- [x] **PyPI Public Alpha:** `flexaligner==0.2.0a1` is published with validated
  model retrieval and local English OOV G2P.

## 👨‍💻 Authors & Affiliation

```text
Yiming Wang (王一鸣) - University of Science and Technology of China (USTC)

Jiahong Yuan (袁家宏) - University of Science and Technology of China (USTC)
```

## 📜 Citation

If you use FlexAligner in your research, please cite:

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

## 📄 License

FlexAligner is released under the [MIT License](LICENSE). Please refer to the
repository's `LICENSE` file for the authoritative license and copyright notice.
The optional `flexaligner-g2p-en` distribution contains the checkpoint derived
from `g2p-en` 2.1.0 under Apache-2.0, together with its license and provenance
notice. The minimal MIT-licensed `flexaligner` wheel does not contain that asset.

<div align="center"><sub>Built by USTCPhonetics.</sub></div>
