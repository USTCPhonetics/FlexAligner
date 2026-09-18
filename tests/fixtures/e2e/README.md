# Release E2E 测试夹具

## 英语夹具

本目录保存现有 OpenPhonetics 合成英语 release-E2E 夹具所使用的、已经批准且冻结的
仓库内词典。决定 `D-033` 只批准它作为 release-E2E 测试证据。它不是默认 G2P
实现、语言学规范发音或可分发模型资产。

Transcript 为：

```text
This synthetic example shows openphonetics word and phone alignment.
```

Provenance：

- `THIS`、`SYNTHETIC`、`EXAMPLE`、`SHOWS`、`WORD`、`AND`、`PHONE` 和
  `ALIGNMENT` 按源文件顺序保留
  `/Users/yiyi0369/projects/openphonetics/word.dict` 中所有匹配行；该文件
  SHA-256 为
  `f6548978de94dfdcfa4c4503c0d3983fd1a4a59fe6497c4f1e1d490fd08a801b`。
- `openphonetics OW1 P AH0 N F AH0 N EH1 T IH0 K S` 冻结自当前
  OpenPhonetics `README.md` 的示例（SHA-256
  `f205aae389de76d6bbd9817e39095a567dea2fb2c533faee64bfaf29a6838017`）和
  `PRODUCT_REQUIREMENTS.md`（SHA-256
  `b048e5a82e8266eb088ce84d5b2b2b1a2658f52ed1ff983fc295e1060c29bd62`）。
- 源目录没有 Git metadata，因此无法确定源 commit。本项目以 hash 为权威，
  不推测 revision。

`asset_manifest.json` 以 `FLEXALIGNER_E2E_ASSET_ROOT` 为根目录。当前本地夹具布局
应把它设为 `/Users/yiyi0369/projects`。可信 runner 可以在其他绝对根目录下复现相同
相对布局。缺少变量、文件或 hash 不匹配时必须报告 `MODEL_E2E_BLOCKED`，不得变成
通过的 skip。

决定 `D-033` 以 `release-e2e-fixture-only` 范围批准此 fixture 专用 OOV 发音。
该批准只允许关闭式 release E2E 使用冻结行，本身不构成公共 G2P 的授权或通用发音
标准。公共本地英语 OOV G2P 后来由独立决定 D-048--D-049 启用，其 checkpoint 输出、
warning 和模型词表校验另行测试，不以本 fixture 的冻结发音作为准确率金标准。

## 普通话夹具

`mandarin_asset_manifest.json` 冻结 `0.4.0a1` 普通话发布路径所需的 18 项资产：
schema 2 `model_manifest.json`、`zh/chunker`、`zh/aligner`、预先提供的 AISHELL-1
参考词典 `zh/word.dict`，以及 macOS `Tingting zh_CN` 合成的“日月无一”和
“日月星辰”PCM16 WAV/UTF-8 文本。词典只作为
预先提供的参考词典使用，不声明所有权；模型据提供方说明基于 AISHELL-1 公开数据
训练，本夹具不扩大该来源声明或推导额外权利归属。

模型 revision 固定为
`6720609c167c8ddcd09ea4cb9223aeb569150df8`，模型 manifest SHA-256 固定为
`7ab9057cba1b51fb91ab434f893d4915358729bcc411e3aceca9e0b67883df36`。测试必须在
Linux x86_64 / Python 3.10.8 的冻结运行时中，禁用网络并逐项校验资产 hash；缺失、
漂移或运行时不一致均以 `MODEL_E2E_BLOCKED` 失败，不允许 skip。

该夹具验证内置系统词典、带调音素和 `ri -> r iz4` 的发布契约；它不是中文 G2P
质量集，也不构成多音字、轻声、儿化或专名准确率声明。
