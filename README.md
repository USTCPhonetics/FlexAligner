<div align="center">

# 🌊 FlexAligner

### Robust Speech-Text Alignment from Signal to Symbol

[![Laboratory](https://img.shields.io/badge/Laboratory-USTC_Phonetics-red.svg)](http://phonetics.ustc.edu.cn/)
[![Python](https://img.shields.io/badge/Python-3.10--3.14-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**A two-stage forced-alignment project for real-world speech**  
**面向真实语音数据的两阶段强制对齐项目**

</div>

> **Rebuild status (2026-08-11):** this repository is a clean implementation
> baseline. It is not published on PyPI, and the real English alignment path has
> not yet passed its staged acceptance gates. The capability table below is the
> authoritative public status; a placeholder is an importable contract that
> fails explicitly, not a supported feature.

## Introduction / 简介

FlexAligner keeps the original product goal of aligning imperfect real-world
recordings and transcripts while rebuilding the implementation around explicit,
testable contracts. The target algorithm has two stages:

1. **Macro segmentation / 宏观切分:** a CTC model locates transcript anchors
   and forms ordered chunks.
2. **Local alignment / 局部对齐:** a constrained phone graph and two-pass
   Viterbi procedure estimate phone and word boundaries inside each chunk.

The first real product path under development is deliberately narrow: English,
CPU, one 16 kHz mono PCM16 WAV, an explicit transcript, a local pronunciation
lexicon, and two local model directories. Inputs that do not satisfy the
contract fail instead of silently dropping words. Successful output will be a
validated, atomically written Praat TextGrid.

FlexAligner 保留原项目“CTC 宏观定位 + 局部精细对齐”的产品目标，
但新代码库不继承旧核心的宽松回退行为。当前第一条真实实现链路
仅面向英语、CPU、单文件、本地模型和本地词典；其他能力不会被静默
伪装成已支持。

## Capability matrix

The package exposes machine-readable capability discovery through
`flexaligner capabilities --json`. Status values are `available`,
`placeholder`, and `unavailable`.

| Capability ID | Public status | Current meaning |
|---|---|---|
| `api.python` | `available` | Typed Python package surface is present; algorithm stages remain separately gated. |
| `cli` | `available` | Version, capability discovery, and explicit placeholder commands are present. |
| `capabilities.discovery` | `available` | Human-readable and JSON capability reports are present. |
| `alignment.single_file.en.cpu` | `placeholder` | Target MVP; no accepted production alignment implementation yet. |
| `language.zh` | `placeholder` | Mandarin is outside this milestone's real implementation. |
| `device.gpu` | `placeholder` | CPU is the only planned MVP execution device. |
| `alignment.batch` | `placeholder` | No batch execution or manifest recovery. |
| `integration.web` | `placeholder` | No Web/API service implementation. |
| `models.auto_download` | `placeholder` | Models must remain explicit and local; no automatic download. |
| `audio.multi_format` | `placeholder` | MVP accepts strict WAV only. |
| `audio.auto_resample` | `placeholder` | MVP does not convert or resample audio. |
| `text.zh_segmentation` | `placeholder` | No Chinese segmentation implementation. |
| `pronunciation.g2p.default` | `placeholder` | OOV words fail; no default G2P fallback. |
| `confidence.calibration` | `placeholder` | Any emitted reference score is explicitly uncalibrated. |

An `available` interface is not evidence that the later alignment algorithm has
passed. Acceptance status is tracked separately in `ACCEPTANCE.md`.

## Installation for development

There is currently no authorized PyPI release. Install only from a reviewed
local checkout:

```bash
python -m pip install --upgrade "pip>=25.1"
python -m pip install -e .
```

Install the heavy local Hugging Face inference adapter only when working on the
real alignment path:

```bash
python -m pip install -e ".[inference]"
```

Developer groups use the standardized dependency-group interface:

```bash
python -m pip install --group ci
```

Package import and capability discovery must not download models or access the
network.

## Current CLI contract

The implemented Stage 1 discovery surface is:

```bash
flexaligner --version
flexaligner capabilities
flexaligner capabilities --json
```

The reserved single-file command shape is shown below for interface review. In
the current Stage 1 package it raises `FeatureNotAvailableError` and must not
create an official output:

```bash
flexaligner align \
  --audio recording.wav \
  --text-file transcript.txt \
  --lexicon english.dict \
  --chunker-model /local/models/en/chunker \
  --aligner-model /local/models/en/aligner \
  --output recording.TextGrid
```

`--text "..."` may replace `--text-file`; they are mutually exclusive.
`batch`, `serve`, and `models fetch` are also explicit placeholders.

Machine-readable failures use a stable envelope:

```json
{
  "code": "feature_not_available",
  "message": "The requested capability is not available in this build.",
  "context": {}
}
```

## Public Python surface

Stage 1 reserves these imports:

```python
from flexaligner import (
    AlignmentOptions,
    AlignmentRequest,
    AlignmentResult,
    Capability,
    CapabilityId,
    CapabilityReport,
    CapabilityStatus,
    FeatureNotAvailableError,
    FlexAligner,
    FlexAlignerError,
    LocalModelBundle,
    TextGridOutput,
    __version__,
    get_capabilities,
)
```

The real alignment methods remain unavailable until their algorithm,
differential, output, and frozen-model acceptance gates pass.

## Validation and release policy

- Fast test execution is model-free and socket-denied; dependency installation
  and the vulnerability query remain separate networked CI steps.
- Model E2E is offline, uses content-addressed external assets, and may never
  turn missing assets into a passing skip.
- Wheels and sdists exclude model weights, recordings, TextGrid results, and
  local caches.
- A production release must be tag-only, build once, test that exact artifact,
  pass the frozen English model E2E, and then receive protected-environment
  approval.
- PyPI Trusted Publishing is not configured and publishing is not authorized.

The dedicated E2E runner label, local manifest path, and offline dependency
wheelhouse are deployment `[TBD]` values. The workflow fails closed when they
are absent; it never downloads a missing model.

## Scope and roadmap

- [x] Governance, clean repository, package/API design, and explicit capability states.
- [ ] Characterize the frozen local algorithm reference with model-free tests.
- [ ] Implement Stage 1 CTC chunking with differential parity.
- [ ] Implement Stage 2 graph/Viterbi/redecode with differential parity.
- [ ] Implement strict local CPU inference and validated TextGrid output.
- [ ] Pass the frozen English real-model E2E.
- [ ] Resolve package ownership/version/license/release `[TBD]` items.
- [ ] Publish to PyPI only after separate authorization.

## Authors and affiliation

- Yiming Wang (王一鸣), University of Science and Technology of China (USTC)
- Jiahong Yuan (袁家宏), University of Science and Technology of China (USTC)

## Citation

The following citation is retained from the fixed upstream README as project
metadata; publication details should be rechecked before formal use:

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

## Provenance and license

Product identity, authorship, the two-stage description, and citation were
adapted from the upstream README at
[`USTCPhonetics/FlexAligner@c5361efe`](https://github.com/USTCPhonetics/FlexAligner/tree/c5361efe4b5d8ad02574dae1bd7caa89ed3e4af0).
The fixed README snapshot has SHA-256
`665dd93bc04a802a9233b4f868e58ed43606bcb5e5eaf934a460bad01d1c280c`.

`LICENSE` preserves the text of that fixed commit's MIT license snapshot; the
upstream source file's SHA-256 is
`b1f12d62c29df3906f7a05b2a18e4faed00876170b75d0c50e2cf50317e00ee7`.
Exact carry-over and attribution remain tracked as `TBD-LIC-001` until the main
project audit closes that question.
