# 英语 release E2E 测试夹具

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
该批准允许关闭式 release E2E 使用冻结行，但不会启用公共默认 G2P 能力，也不建立
通用发音标准。
