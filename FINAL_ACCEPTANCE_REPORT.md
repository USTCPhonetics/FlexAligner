# FlexAligner 重建最终验收报告

> 日期：2026-08-11（Asia/Shanghai）
> 工程基线：ACCEPT
> 公共 PyPI 发布：NO-GO

## 1. 总结

本地干净代码重建已经完成。英语、CPU、单文件、严格本地模型/词典、16 kHz
单声道 PCM16 WAV 的真实链路已实现；NumPy Stage 1/Stage 2、懒加载本地推理、
API/CLI、TextGrid 事务与未经校准的 raw confidence 均有测试证据。普通话和
用户指定的九项未来能力保持显式占位，没有静默回退。

当前仓库可以生成并安装 pip 包候选，但还不能公开发布。用户已经选定 public alpha、
GitHub 直接替代策略、包名/owner、上游身份文本与 fixture-only 发音；剩余阻断项是
尚未完成的 alpha 元数据/依赖契约、远程执行、账户配置和逐项外部授权。

## 2. 验收摘要

| 范围 | 结论 | 核心证据 |
|---|---|---|
| 治理、reference、Stage 1/2 | PASS | hash guard、三方 parity、独立 invariant/resource tests |
| 严格 pipeline/API/CLI/TextGrid | PASS | 676 项 fast tests；92.31% branch coverage |
| 格式、lint、严格类型 | PASS | Ruff 76 files；mypy 20 source/script files |
| wheel/sdist | PASS | Twine strict、wheel contents、distribution inventory、仓外安装 |
| approved-fixture 本地 E2E | PASS | D-033；当前 exact wheel；TextGrid 非 `NULL` interval 与 reference 一致且双 tier 全覆盖 |
| 十项未来能力 | PLACEHOLDER | 类型化 pre-I/O failure；无下载、转换、回退或端口绑定 |
| Python 3.10–3.14 全远程矩阵 | BLOCKED | 本地仅验证 3.10 与部分 3.12；仓库无 remote/Actions 结果 |
| protected remote release E2E | BLOCKED | runner/wheelhouse 未配置；manifest repo-local 路径尚不可移植 |
| PyPI 发布 | NO-GO | alpha metadata/inference contract、remote、Trusted Publisher 与外部授权未完成 |

完整逐项状态见 `ACCEPTANCE.md`；真实模型证据见
`REAL_MODEL_E2E_REPORT.md`。

## 3. Stage 8 开发审计产物

| Artifact | SHA-256 |
|---|---|
| `flexaligner-0.1.0.dev0-py3-none-any.whl` | `0938914d1418ec0f4b61dc628e7ead89502a40a09692c0cb655133e0bb8a1253` |
| `flexaligner-0.1.0.dev0.tar.gz` | `85beeadaa7a24394d58081ee8bc045b08bd704fcef772a67ff317bb8a3fc72a9` |

这些 artifact 位于临时审计目录，不是已发布文件；仓库原有 `dist/` 中的旧
Stage 3 产物没有被当作最终证据，也没有覆盖或删除。

## 4. 最小发布前计划

1. 实施 `0.1.0a1` 元数据并完成 D-034 要求的窄推理依赖/运行时契约验证。
2. 修复 `TBD-E2E-002`：让 repo-local fixture 与外部模型资产使用可移植的根目录
   语义，再在目标 runner 复验。
3. 在任何直接替代远端 `main` 前，重新核对目标 commit、建立可恢复快照，并取得
   独立外部授权；本轮策略批准不是 force-push 授权。
4. 配置自托管 E2E runner、冻结 asset root、离线 wheelhouse、受保护的
   `pypi` environment 和 Trusted Publisher。
5. 在目标 remote 跑完 Python 3.10–3.14、Linux/Windows/macOS、wheel smoke
   与 dependency-audit；任何失败均回到本地修复，不能降低断言。
6. 通过所有门禁后，分别请求 tag 与 PyPI upload 授权；二者互不蕴含。

## 5. 未执行的外部动作

- 未配置或修改 Git remote；
- 未创建 GitHub 仓库、PR、tag 或 release；
- 未注册 PyPI 项目、Trusted Publisher 或环境变量；
- 未上传 wheel/sdist；
- 未下载模型或修改冻结资产。
