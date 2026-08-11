# FlexAligner 重建最终验收报告

> 日期：2026-08-11（Asia/Shanghai）
> 工程基线：ACCEPT
> 公共 PyPI 发布：NO-GO

## 1. 总结

本地干净代码重建已经完成。英语、CPU、单文件、严格本地模型/词典、16 kHz
单声道 PCM16 WAV 的真实链路已实现；NumPy Stage 1/Stage 2、懒加载本地推理、
API/CLI、TextGrid 事务与未经校准的 raw confidence 均有测试证据。普通话和
用户指定的九项未来能力保持显式占位，没有静默回退。

当前仓库可以生成并安装 pip 包候选，但还不能公开发布。阻断项来自尚未完成
的远程/账户/批准条件，而不是未完成的本地核心实现。

## 2. 验收摘要

| 范围 | 结论 | 核心证据 |
|---|---|---|
| 治理、reference、Stage 1/2 | PASS | hash guard、三方 parity、独立 invariant/resource tests |
| 严格 pipeline/API/CLI/TextGrid | PASS | 676 项 fast tests；92.31% branch coverage |
| 格式、lint、严格类型 | PASS | Ruff 75 files；mypy 20 source/script files |
| wheel/sdist | PASS | Twine strict、wheel contents、distribution inventory、仓外安装 |
| 候选真实模型工程 E2E | PASS | exact wheel；16/16 assets；TextGrid 与 reference 字节一致 |
| 十项未来能力 | PLACEHOLDER | 类型化 pre-I/O failure；无下载、转换、回退或端口绑定 |
| Python 3.10–3.14 全远程矩阵 | BLOCKED | 本地仅验证 3.10 与部分 3.12；仓库无 remote/Actions 结果 |
| approved-fixture release E2E | BLOCKED | manifest 仍为 `candidate`；发布工作流 fail-closed |
| PyPI 发布 | NO-GO | owner/name/version/Trusted Publisher/授权均未完成 |

完整逐项状态见 `ACCEPTANCE.md`；真实模型证据见
`REAL_MODEL_E2E_REPORT.md`。

## 3. 最终发行候选

| Artifact | SHA-256 |
|---|---|
| `flexaligner-0.1.0.dev0-py3-none-any.whl` | `882337e536bda28814293f803cd88f62ce4d3f137183aeb8c1396799b1199d32` |
| `flexaligner-0.1.0.dev0.tar.gz` | `93b5fb63560c6fa014fbb3a0994c271c22c69757302551ef0a2da79879c51a82` |

这些 artifact 位于临时审计目录，不是已发布文件；仓库原有 `dist/` 中的旧
Stage 3 产物没有被当作最终证据，也没有覆盖或删除。

## 4. 最小发布前计划

1. 审批或替换候选 `openphonetics` 发音，并通过单独决定把 manifest 改为
   `approved`。
2. 确认 GitHub 历史/remote、PyPI 名称与所有者、首个稳定版本，以及
   README/LICENSE 最终归属文本。
3. 配置自托管 E2E runner、冻结 asset root、离线 wheelhouse、受保护的
   `pypi` environment 和 Trusted Publisher。
4. 在目标 remote 跑完 Python 3.10–3.14、Linux/Windows/macOS、wheel smoke
   与 dependency-audit；任何失败均回到本地修复，不能降低断言。
5. 通过所有门禁后再创建稳定 tag；只有用户另行明确授权，发布 job 才可执行。

## 5. 未执行的外部动作

- 未配置或修改 Git remote；
- 未创建 GitHub 仓库、PR、tag 或 release；
- 未注册 PyPI 项目、Trusted Publisher 或环境变量；
- 未上传 wheel/sdist；
- 未下载模型或修改冻结资产。
