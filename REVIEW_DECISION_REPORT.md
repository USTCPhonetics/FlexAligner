# FlexAligner 用户审阅与决策报告

> 日期：2026-08-11（Asia/Shanghai）
>
> 文档状态：**REVIEWED — D-039/D-040 已实施；D-036--D-038 待实施；长音频性能未通过**
>
> 本地工程基线：主 agent 基于测试证据判定 `ACCEPT`
>
> 公共发布：`NO-GO`

## 1. 结论先行

干净代码基线的重建工作已经完成；当前不需要再讨论“另起炉灶还是修旧框架”。
用户已经明确选择干净基线重建，英语 CPU 单文件链路也已形成可安装、可测试的
pip 包候选。

本报告最初提出了以下六组发布与产品边界供审阅：

1. 首个公开版本是开发者预览，还是生产级产品；
2. 新历史如何接入 GitHub；
3. PyPI 包名、所有者和首发版本；
4. LICENSE、版权、作者、品牌和引用文本；
5. 是否批准候选发音只作为 release-E2E 测试夹具；
6. 是否接受建议的 v0.1 支持边界。

用户已于 2026-08-11 完成批阅；结果见下表。远程 CI、真实 runner、dependency
audit 和 Trusted Publisher 是否真正通过，
属于后续必须执行的验证，不是可以由用户主观批准为 `PASS` 的事项。即使本文所有
方案获批，也不自动授权 push、改变默认分支、创建 tag 或上传 PyPI。

| 决策 | 用户批阅结果 | 决策记录 |
|---|---|---|
| R-01 | `PUBLIC_ALPHA`，首个公开版本 `0.1.0a1` | D-029 |
| R-02 | `REPLACE_MAIN_HISTORY`，直接替代现有 GitHub 主历史 | D-030 |
| R-03 | distribution `flexaligner`；owner `ustcphonetics` | D-031 |
| R-04 | 固定远端身份文本：README 提供作者/品牌/引用，LICENSE 提供 MIT/版权行 | D-032 |
| R-05 | `APPROVE_FIXTURE_ONLY` | D-033 |
| R-06 | `ACCEPT_PREVIEW_BUNDLE` | D-034 |
| R-07 | 算法与资源修正已批准，代码/测试仍待实施 | D-036--D-040 |

本次回复没有授予任何外部操作权限，D-013 继续生效。

## 2. 本次审阅依据

本文按项目事实权威顺序，以当前代码和治理文件为准；没有从旧会话补全状态。

| 项目 | 2026-08-11 当前事实 |
|---|---|
| 本地仓库 | 本轮决定落档前的基线为 `main@b594b9c`，共 16 个提交；本报告把它作为审阅前快照，不冒充提交后的 HEAD |
| 本地 remote / tag | 均未配置或创建 |
| Stage 8 变更范围 | 本轮统一落档 16 个跟踪文件；另有一个不属于本轮的未跟踪 `.DS_Store`，不纳入提交 |
| 远端只读复核 | `USTCPhonetics/FlexAligner` 的 `main`/HEAD 仍为 `c5361efe…`，`dev` 为 `ea3b5836…` |
| 包元数据 | distribution 已选定为 `flexaligner`；当前版本 `0.1.0.dev0`；`Pre-Alpha` |
| PyPI 名称探测 | 官方 JSON 端点当前返回 404；这只表示未发现公开项目，**不证明名称一定可注册或归本项目所有** |
| 工程门禁 | 676 项禁网 fast tests；92.31% branch coverage；Ruff、strict mypy、包审计通过 |
| approved-fixture 本地 E2E | exact wheel、16/16 资产、冻结运行时通过；新旧 TextGrid 字节一致 |
| E2E 发布状态 | D-033 approved 配置通过 16/16 preflight 与 exact-wheel 本地 E2E；protected remote E2E 尚未运行 |
| 外部变更 | GitHub、PyPI、模型资产均未修改 |

工程验收的详细证据见 `FINAL_ACCEPTANCE_REPORT.md`、`ACCEPTANCE.md` 和
`REAL_MODEL_E2E_REPORT.md`。这里的 `ACCEPT` 是主 agent 的工程判定，不是
“用户已经验收”或“已经批准发布”。

## 3. 已完成批阅的决定

### R-01 — 首个公开版本定位

**用户决定：`PUBLIC_ALPHA`。** 将首个公开版本定位为研究/开发者预览版：严格英语、
CPU、单文件、用户自备本地模型与词典；不声称生产安全、广泛模型兼容、对齐准确率
或行为纠错完成。

| 选项 | 含义 | 影响 |
|---|---|---|
| `PUBLIC_ALPHA`（推荐） | 门禁通过后先发布 `0.1.0a1`，保持 Alpha / Pre-Alpha 语义 | 当前算法限制可以披露后延期；需调整稳定版专用 release guard，再完整复验 |
| `INTERNAL_ONLY` | 继续保留 `0.1.0.dev0`，只分发本地 wheel | 不需要立即配置 PyPI；其他公开发布决定可延期 |
| `PUBLIC_FINAL` | 首发使用 PEP 440 final `0.1.0` | 必须退出当前 Pre-Alpha 定位；若同时声称生产可用，资源、模型、推理矩阵、线程副作用、provenance、输出一致性等须升级为阻断项 |

**不决定时的默认行为：** 保持 `0.1.0.dev0`，不发布。

### R-02 — GitHub 仓库与历史接入

**用户决定：`REPLACE_MAIN_HISTORY`。** 重建历史将直接替代现有
`USTCPhonetics/FlexAligner` 主历史，不采用本报告原推荐的双历史 merge/graft。
这是 GitHub 接入策略决定，不是 force-push、改变默认分支或删除远端引用的执行授权。
执行前仍须只读核对精确 remote/branch/commit，建立可恢复快照，并获得单独外部授权。

| 选项 | 优点 | 代价/风险 |
|---|---|---|
| `PRESERVE_BOTH_INTEGRATION_PR`（原推荐，未选择） | 不 force-push；旧项目和重建过程都可追溯；可回滚 | 需要一次明确的双历史集成、tree equality 和 PR 审计 |
| `NEW_REPOSITORY` | 新仓库保持完全独立历史 | 需要确定新 URL；旧项目迁移与用户发现路径要另行处理 |
| `REPLACE_MAIN_HISTORY`（用户选择） | Git 图最简 | 会改写现有项目入口；破坏性最高，必须另行明确授权 |

执行前必须再次 fetch/核对远端最新状态。批准策略不等于授权 push 或切换默认分支。

**不决定时的默认行为：** 本地仓库不配置 remote。

### R-03 — PyPI 名称、所有者和首发版本

**用户决定：** distribution 使用 `flexaligner`，owner/organization 使用
`ustcphonetics`；所有门禁通过前保持 `0.1.0.dev0`，首个公开预览使用
`0.1.0a1`。PyPI 名称和 owner 的实际可创建/控制状态仍须在外部配置时验证。

原审阅建议是由机构/项目团队控制的 PyPI
账号或组织持有；所有门禁通过前保持 `0.1.0.dev0`，首个公开预览使用
`0.1.0a1`。只有明确退出 Alpha / Pre-Alpha 定位后，才使用 final `0.1.0`。

批阅结果：

- PyPI distribution：`flexaligner`
- PyPI owner 或 organization：`ustcphonetics`
- 首个公开版本：`0.1.0a1`

当前 [PyPI project page](https://pypi.org/project/flexaligner/) 和 JSON 端点返回
404，但这不是名称所有权证明。名称仍可能因为已注册但无 release、相似名称或管理
限制而不可用；最终可注册性只能由另行授权的 project creation / Trusted Publisher
配置或首次发布确认。PyPI 名称会被规范化，因此 `FlexAligner`、`flex-aligner` 和
`flex_aligner` 不是三个独立候选，见
[PyPA name normalization](https://packaging.python.org/en/latest/specifications/name-normalization/)。
任何名称、版本或许可变更后，都要重新构建并重新验证确切 artifact，不能给旧 wheel
改名后发布。

**不决定时的默认行为：** `flexaligner` 只作为本地候选名，不注册、不上传。

### R-04 — LICENSE、署名、品牌和引用

当前文本包含：

- MIT License；
- `Copyright (c) 2026 WANG Yiming`；
- `Yiming Wang` 与 `Jiahong Yuan` 两位 package/README 作者；
- USTC / USTC Phonetics 品牌与单位表述；
- 指向 `USTCPhonetics/FlexAligner` 的项目 URL 和引用条目；
- 对上游固定提交 `c5361efe…` 的 README/许可来源说明。

**用户决定：`APPROVE_CURRENT_TEXT`。** 固定远端 README 提供品牌、两位作者、
USTC affiliation 和 citation；同一提交的远端 LICENSE 提供 MIT 正文及
`Copyright (c) 2026 WANG Yiming`。远端 README 本身没有版权行，因此本文不把作者
或 `ustcphonetics` owner 静默改写成版权主体。

审阅时提供的互斥选项是：

- `APPROVE_CURRENT_TEXT`：确认当前文本可公开使用；或
- `REVISE`：逐项给出版权主体、作者、单位、品牌、项目 URL、引用的精确修改；或
- `LEGAL_REVIEW`：先保持 NO-GO，等待机构/法律审阅。

用户选择 `APPROVE_CURRENT_TEXT`，精确来源边界按 D-032 记录。

**不决定时的默认行为：** 不公开发布。

### R-05 — 候选 E2E 发音批准

候选项是：

```text
openphonetics OW1 P AH0 N F AH0 N EH1 T IH0 K S
```

该行有当前 OpenPhonetics README 与 PRODUCT_REQUIREMENTS 的文件哈希作为来源；
本地源目录没有 Git metadata。使用它时，新实现与权威 reference 的真实模型
TextGrid 已达到字节一致。

**用户决定：`APPROVE_FIXTURE_ONLY`。** 仅批准它作为冻结的 release-E2E 测试材料。
该批准不表示它是语言学规范发音，不启用默认 G2P，不构成准确率金标准，也不把模型
随包发布。

| 选项 | 结果 |
|---|---|
| `APPROVE_FIXTURE_ONLY`（用户选择） | 已同步审计 `status`、`fixture_id` 和 approval provenance；本地 exact-wheel 门禁已重跑通过 |
| `REPLACE_FIXTURE` | 用户提供替代发音或更权威的版本化来源；重新冻结并完整重跑 |
| `KEEP_CANDIDATE` | 继续作为工程证据；公开 release gate 保持 BLOCKED |

### R-06 — v0.1 支持边界包

**用户决定：`ACCEPT_PREVIEW_BUNDLE`。** 一次性确认以下边界，避免让没有对比实验的
算法问题阻塞开发者预览：

1. 唯一真实链路仍是英语、CPU、单文件、16 kHz mono PCM16 WAV、本地模型、
   本地全覆盖词典；普通话及其他九项未来能力，共十项能力继续为显式 placeholder。
2. wheel 不携带模型，也不自动下载；首版是“代码包 + 用户自备经验证的本地资产”。
3. 基础包以 Python 3.10–3.14 为远程 fast-CI 目标；真实推理目前只有冻结的
   Python 3.10.8 / NumPy 2.2.6 / Torch 2.3.1 / Transformers 4.41.2 组合的
   approved-fixture 本地 E2E 证据。protected remote 门禁通过前，不把它写成
   广泛发布支持认证。
4. `[inference]` 当前的 `torch>=2.3,<3`、`transformers>=4.41,<6` 是 pip 解析器
   实际会使用的范围，不能只靠 README 把它降级为“不支持”。公开 alpha 前，默认采用
   `FROZEN_INFERENCE_CONTRACT`：主 agent 必须收窄实际依赖/运行时契约到有证据的
   窗口，并验证至少一条公共索引安装解析路径；如果无法形成诚实可安装的窄契约，
   就从 alpha 中移除该 public extra，而不是静默保留宽范围。广泛兼容矩阵延期。
5. 不建立宽泛 legacy Python/CLI 兼容层；只有用户列出真实调用入口后才加适配器。
6. 保留 reference parity 行为并明确披露：冻结 E2E fixture 中实测约 18 ms 尾
   gap、固定 10 ms Stage 2 stride、同词连续相同 phone 的折叠、简化
   repeated-label CTC recurrence。
7. confidence 仍是 raw、未校准；final phone provenance 缺失时保持 `None`，不伪造。
8. 词汇 `sil` / `null` 继续在模型加载前类型化拒绝。
9. 当前没有批准的默认资源上限；开发者预览不得声称可安全处理不受信任的任意长输入。
10. Torch CPU thread count 是进程全局副作用，公开发布前必须补充明确文档；
    若要求宿主进程隔离，则应在发布前改实现并验证。

如果选择 `PUBLIC_FINAL` 并同时要求生产可用，不能直接接受这组延期边界；至少资源
上限、推理兼容、线程隔离、模型 provenance 和连续覆盖需要重新提升为发布阻断项。

**不决定时的默认行为：** 当前实现不变，但不公开发布。

以上第 6、9 项是 2026-08-11 的历史批准口径。2026-08-23 后续算法审阅已经由
D-036--D-040 明确取代冲突部分：不再把 gap、连续相同 phone 折叠、简化 repeated-label
recurrence 或“无默认资源上限”作为发布目标。固定 10 ms 也由 D-037 收窄为必须验证的
名义 stride 契约。保留这段原文用于审计历史，不表示当前仍批准延期这些修正。

### R-07 — 后续算法与资源批阅

用户已接受以下目标：

| 决定 | 已批准目标 | 当前实施状态 |
|---|---|---|
| D-036 | phones/words 两层以大写 `NULL` 填充开头、内部和末尾 gap，保持非 `NULL` 边界并严格覆盖完整音频 | `NOT_RUN` |
| D-037 | v0.1 只接受 16 kHz 下卷积总 stride 160 samples；时间戳固定为 `frame_index * 0.01` | `NOT_RUN` |
| D-038 | 同词连续相同 phone 以至少 `(word_index, phone_index)` 的 occurrence 身份区分 | `NOT_RUN` |
| D-039 | 相邻相同 Stage 1 目标必须由至少一个 CTC blank 帧分隔 | `PASS`（代码级与短自然语音 E2E） |
| D-040 | 900 s、200,000,000 trellis cells、每请求累计 200,000,000 candidate transition evaluations；beam width 400；word/phone/graph 默认值未在本次固定 | `PASS`（保险丝与短 E2E）；900 s 性能未通过 |

D-040 的两个 200,000,000 数值是待实测复核的 alpha 初始门槛，不是性能或成功 SLA。
用户指定的 `english_natural` 真实 E2E 已取得 11,546 cells、281,411 work，并与给定
TextGrid 字节一致；`example1` 真实 E2E 取得 83,368 cells、219,135 work 和 3 chunks。
D-040 的代码保险丝可以通过，但当前样本不支持 900 s 性能声明。完整证据见
`ALPHA_RESOURCE_VALIDATION.md`。

## 4. 已解决算法项与仍未决技术事项

原报告把 `TBD-ALG-001..005` 建议为 alpha 后实验；D-036--D-040 已经取代该建议：

| 原 ID | 解决结果 | 决定 | 实施/验收 |
|---|---|---|---|
| `TBD-ALG-001` | TextGrid 两层 `NULL` 全覆盖 | D-036 | `NOT_RUN` |
| `TBD-ALG-002` | 只验证名义 10 ms stride | D-037 | `NOT_RUN` |
| `TBD-ALG-003` | 连续相同 phone 保留 occurrence 身份 | D-038 | `NOT_RUN` |
| `TBD-ALG-004` | 标准 repeated-target blank 约束 | D-039 | `PASS` |
| `TBD-ALG-005` | D-040 alpha 初始保险丝；200M/200M | D-040 | `PASS`（代码/短 E2E）；长音频性能 `FAIL` |

以下问题仍未解决：

| ID | 当前保守行为 | 何时重新决策 |
|---|---|---|
| `TBD-OUT-001` | 进程内回滚；不承诺两文件掉电原子性 | 要求 power-loss crash consistency 时 |
| `TBD-API-002` | 未知 phone provenance 返回 `None` | 设计可验证的 fixed-state 身份方案时 |
| `TBD-PROV-001` | 公开结果暂不伪造 model fingerprint | 确定版本化 manifest digest schema 时 |
| `TBD-INF-001` | 仅保留冻结模型/运行时的候选工程 E2E 证据 | 要公开承诺广泛 Hugging Face 兼容时 |
| `TBD-THREAD-001` | 设置请求线程数，不承诺恢复宿主全局值 | 嵌入式 API 需要线程隔离时 |
| `TBD-TEXT-001` | `sil` / `null` 类型化拒绝 | 设计不与词面冲突的内部标签身份时 |

## 5. 不是“决策”，而是必须通过的执行门禁

以下项目不能通过审阅意见改成 PASS：

1. 在目标 GitHub remote 实际运行 Ubuntu Python 3.10–3.14、Windows/macOS
   端点矩阵、wheel/sdist smoke 和 dependency audit；
2. 配置并实际运行自托管离线 E2E runner、asset root 和 wheelhouse；
3. 使用批准后的 manifest 验证 exact built wheel；
4. 配置受保护的 `pypi` environment、审批人和 Trusted Publisher；
5. 名称/版本/许可最终化后重新 build once，并让同一 artifact 通过全部门禁。
6. 最终化 `[inference]` 的实际 resolver/runtime 契约，并验证公共索引安装解析路径；
   hosted fast CI 和基础 dependency audit 不能替代真实推理依赖验证。
7. 为 D-036--D-040 保留 before/after 证据并运行全部新增边界/关闭式失败测试；
8. 使用新的经审阅长音频 fixture 完成 D-040 的长输入资源实测；
9. 重新生成修正后的 E2E candidate，字段级审阅后才可接受新 golden，并对 exact wheel
   重跑完整 fast、包审计和 approved-fixture E2E。

任何失败都必须回到实现或缩窄支持声明，不能降低断言、更新 golden 掩盖差异，
也不能由用户口头“批准为通过”。

## 6. 外部操作授权必须逐项拆分

本文的审阅回复默认是 `AUTH_NONE`。以下授权彼此独立，任何一项都不蕴含其他项：

| 授权代码 | 外部动作 |
|---|---|
| `AUTH_PUSH_REVIEW_BRANCH` | 向指定 GitHub 仓库推送集成分支 |
| `AUTH_OPEN_PR` | 创建面向指定分支的 PR |
| `AUTH_CHANGE_DEFAULT_BRANCH` | 修改 GitHub 默认分支或保护规则 |
| `AUTH_CONFIGURE_RUNNER` | 配置 self-hosted runner、asset root 与离线 wheelhouse |
| `AUTH_CONFIGURE_GITHUB_RELEASE_ENV` | 配置 GitHub protected `pypi` environment 与审批人 |
| `AUTH_CONFIGURE_PYPI_PROJECT` | 创建或配置指定 PyPI project、owner 与 collaborator |
| `AUTH_CONFIGURE_TRUSTED_PUBLISHER` | 绑定指定 GitHub workflow 与 PyPI project 的 Trusted Publisher |
| `AUTH_CREATE_TAG` | 在指定审计通过的 commit 创建并推送指定 tag |
| `AUTH_PYPI_PUBLISH` | 上传已验收的 exact artifact 到指定 PyPI project |

用户可以现在只完成产品决策，保持全部外部授权为 `NO`。例如，授权 tag 并不自动
授权上传 PyPI；授权 runner 也不授权修改 Trusted Publisher。

## 7. 已经决定，不应再次询问

- 采用干净代码基线重建，而不是继续修补旧核心；
- 当前本地 `align_single_cpu.py` 是算法行为权威；
- 普通话暂时占位；
- GPU、批处理、Web、自动模型下载、多格式音频、自动重采样、中文分词、默认
  G2P、置信度校准只留接口并显式失败；
- 不自动下载、不静默回退、不用缺资产的 skip 伪造 E2E 成功；
- TextGrid 全覆盖、10 ms stride 验证、phone occurrence、标准重复目标 blank 约束和
  D-040 初始资源保险丝已经决定，不应再次作为算法选择提问；
- 未经另行授权不修改远端、不打 tag、不上传 PyPI。

## 8. 已发现并在 Stage 8 修订的治理记录

以下事实滞后已经在本轮按权威顺序修订，不需要用户重新决定：

1. `OPEN_QUESTIONS.md` 中 `TBD-LIC-001` 的“抓取并在提交前审计”已经完成；
   D-032 又按用户批阅关闭了最终身份文本问题。
2. `TBD-ALG-001` 的 characterize 已完成；D-034 选择在 alpha 披露并延期纠正。
3. `DECISIONS.md` 的 D-017 曾写 real alignment E2E pending；当前 D-033
   approved exact-wheel 本地 E2E 已通过，protected remote E2E 仍单独阻断。
4. 工具 pins 和覆盖率门槛已经由 D-015、D-027 与 `pyproject.toml` 固定；
   `TBD-CI-001` 现在只剩远程矩阵的实际执行证据。
5. 2026-08-23 的 D-036--D-040 又取代了 D-034 中 gap、stride、相同 phone、简化
   recurrence 和无默认资源上限的冲突口径；当前只完成决定落档，实施和验证仍为
   `NOT_RUN`。

这些冲突按权威顺序以当前代码、`STATE.md` 和最终实验输出为准；本文没有静默合并
旧描述。关闭结果记录在 `OPEN_QUESTIONS.md` 的 resolved ledger。

## 9. 批阅结果落档

批阅结果已经拆分写入独立的 `DECISIONS.md`、`STATE.md` 与
`OPEN_QUESTIONS.md`：前者记录 D-029--D-040，第二份记录当前实施/验证状态，
第三份列出仍未解决的问题并保留已关闭问题的 resolved ledger。任何远端替代、
GitHub 设置、tag 或 PyPI 上传仍须按第 6 节逐项获得独立授权。
