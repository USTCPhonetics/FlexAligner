# FlexAligner: Robust Speech-Text Alignment

<div align="center">

**A Deep Learning Based Forced Alignment Tool for the "Wild" Real-World Data** **面向真实世界数据的深度学习强鲁棒性强制对齐工具**

[**English**](#introduction) | [**简体中文**](#简介)

</div>

---

## Authors & Affiliation
* **Yiming Wang (王一鸣)** - *University of Science and Technology of China (USTC)*
* **Jiahong Yuan (袁家宏)** - *University of Science and Technology of China (USTC)*

---

## Introduction

**FlexAligner** is a robust speech-text alignment framework designed to supersede traditional HMM-GMM approaches (like MFA) in the deep learning era. Unlike traditional aligners that force-align every frame even when transcription is missing or incorrect, FlexAligner utilizes a **CTC-based Chunking** mechanism to intelligently detect mismatches.

### Why FlexAligner? (Key Features)

1.  **Tolerance to Mismatch (Transcription Drift):**
    * **MFA:** Forces alignment even if the text is missing, causing "domino effect" errors.
    * **FlexAligner:** Automatically detects unrelated audio segments (e.g., laughter, noise, missing words) and labels them as `<NULL>` or silence.
2.  **High-Resolution Boundaries:**
    * Leverages **wav2vec 2.0** representations and frame-level Cross-Entropy classification for precise boundary detection.
3.  **Easy to Use (WIP):**
    * Designed for linguistics students and researchers. No complex Kaldi environment setup required.

---

## 简介

**FlexAligner** 是一个面向深度学习时代的强鲁棒性语音-文本对齐框架。旨在解决传统 HMM-GMM 方法（如 MFA）在处理非理想数据时的痛点。不同于传统对齐器强制将每一帧音频分配给文本（即使文本缺失或错误），FlexAligner 利用 **CTC Chunking** 机制智能感知“转写-音频”的不匹配。

### 核心优势

1.  **转写缺失与错配的鲁棒性：**
    * **传统 MFA：** 即使文本漏词，也会强行拉伸音素填满音频，导致时间戳完全错误。
    * **FlexAligner：** 能够自动检测出音频中多余的片段（如笑声、噪音、或漏记的词），并将其标记为 `<NULL>`，保证剩余部分的精准对齐。
2.  **高精度边界：**
    * 结合 **wav2vec 2.0** 的深层特征与帧级分类（Cross-Entropy），实现超越传统帧移限制的对齐精度。
3.  **开箱即用（开发中）：**
    * 专为语言学研究者设计，无需配置复杂的 Kaldi 环境，未来将提供一键运行版本。

---

## Installation / 安装

> ⚠️ **Note:** This project is currently under active refactoring. The installation method below is for the upcoming release.
>
> ⚠️ **注意：** 本项目目前正在进行重构。以下安装方式适用于即将发布的版本。

### From Source (Recommended for Devs)

```bash
git clone [https://github.com/USTCPhonetics/FlexAligner.git](https://github.com/USTCPhonetics/FlexAligner.git)
cd FlexAligner
# Recommended: Use uv or conda to create a virtual environment
pip install -e .