# FlexAligner English G2P language pack

This distribution contains the pinned, offline English G2P checkpoint used by
FlexAligner. It is installed through the main package's `en` extra:

```bash
python -m pip install "flexaligner[en]==0.3.0a1"
```

The checkpoint originates from `g2p-en==2.1.0`, is redistributed under
Apache-2.0, and is verified by FlexAligner against SHA-256
`b8af35e4596d8dd5836dfd3fe9b2ba4f97b9c311efe8879544cbcfcbd566d8c6`
before use. This package performs no network access and is not a standalone
command-line application.

## 中文说明

本 distribution 保存 FlexAligner 英语本地 G2P 使用的固定 checkpoint，通过主包的
`en` extra 安装。checkpoint 来源于 `g2p-en==2.1.0`，依 Apache-2.0 再分发；每次加载
前由 FlexAligner 核验固定 SHA-256。本包不联网，也不提供独立命令行应用。
