# 冻结英语真实模型 E2E 报告

> 日期：2026-08-11（Asia/Shanghai）
> 本地 approved-fixture exact-wheel E2E：PASS
> protected remote E2E：NOT_RUN / BLOCKED

## 1. 验证对象

| 项目 | 冻结值 |
|---|---|
| wheel | `flexaligner-0.1.0.dev0-py3-none-any.whl` |
| wheel SHA-256 | `a33dcc22f8023e4b4a7905bf7ab78bd827e4576a3fae368f24850b89f0ac9558` |
| reference SHA-256 | `9ed4e21e615718ddfd10930359f55769fb27a0d284599cce45a3fc755e835de1` |
| manifest | `openphonetics-english-synthetic-release-e2e-v1`，`status=approved`，decision `D-033` |
| 外部资产 | 16/16 路径与 SHA-256 通过 |
| Python / NumPy | 3.10.8 / 2.2.6 |
| Torch / Transformers | 2.3.1 / 4.41.2 |

实际导入来自仓库外
`/tmp/flexaligner-stage8-final-wheel-site.uy5PIZ/flexaligner/__init__.py`，不是
`src/` 或 editable checkout。模型在 `HF_HUB_OFFLINE=1`、
`TRANSFORMERS_OFFLINE=1`、禁 socket、CPU 单线程条件下运行；没有下载或修改
外部资产。approved manifest 的 16/16 路径、hash 与精确 runtime preflight 通过。

## 2. 结果

| 核验项 | 结果 |
|---|---|
| 新实现真实模型运行 | PASS |
| 当前权威 reference 运行 | PASS |
| TextGrid 字节比较 | 完全一致 |
| TextGrid SHA-256 | `ddbe0fecbbd7fc32442bd7b81ccb6257e391ab81970d398eb236de46a50e415f` |
| 新 metadata SHA-256 | `c6c5b035be5aeb3727996538c37c168e5af0c5591b08b38befeaace5a9f36140` |
| 标准化词序 | 9/9 完整，含 `openphonetics` |
| 共享 confidence / pronunciation | 数值与顺序精确一致 |
| 置信度声明 | raw、`calibrated=false`，没有伪装成校准概率 |
| 临时/正式输出事务 | PASS，无遗留 `.tmp` |

对应测试是 `tests/e2e/test_english_frozen.py`。D-033 后重建的确切 wheel 验证
结果为 `1 passed, 676 deselected`；fast suite 显式排除该 marker，缺少环境或
资产时测试会报告
`MODEL_E2E_BLOCKED`，不会用 skip 伪造通过。

## 3. 冲突与限制

- OpenPhonetics 既有 `english_synthetic.TextGrid` 的 SHA-256 是
  `78a69bf46cd3c5ef54cae8b879cf45720132373f4f030513e83212c557ce3e00`，
  与当前权威脚本和新实现共同产生的 `ddbe0fec…e415f` 不同。按项目事实
  纪律，既有文件只作为 legacy candidate 留档，不进入当前 oracle。
- 当前 fixture 词典已经包含 `openphonetics`，所以技术上不再是 OOV。D-033
  只批准该发音作 release-E2E fixture；它不是规范发音、默认 G2P、准确率金标准
  或模型分发许可。工作流的 `--require-approved` fail-closed 逻辑继续保留。
- 新 metadata 使用版本化 schema；reference metadata 字段名不同。只比较
  两边共同定义的词索引、词、发音、置信度及 log 置信度，均精确一致。
- 输出保留 reference 的约 18 ms 尾部 gap，属于 `TBD-ALG-001`，本次没有
  静默修正。
- 两个 Hugging Face bundle 均出现 weight-normalization 键迁移/重新初始化
  警告；当前冻结结果可复现，但广泛模型兼容性仍为 `TBD-INF-001`。

## 4. 发布判定

本地 approved-fixture exact-wheel E2E 已通过。公共发布仍为 `BLOCKED`：manifest
中的 repo-local fixture 路径依赖当前 `flexaligner-rebuild` 目录布局
（`TBD-E2E-002`），目标 GitHub runner、离线 wheelhouse、protected environment
和完整远程矩阵都尚未配置或运行。不得把本地 PASS 外推成 protected remote PASS。
