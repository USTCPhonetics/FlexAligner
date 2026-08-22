# 已接受决定

本文档用于区分用户明确接受的决定与可逆的实施选择。实施选择不得被改写成
用户授权。

| ID | 决定 | 来源 | 状态 |
|---|---|---|---|
| D-001 | 从干净代码基线重建，而不是在远端旧核心上原地修补 | 用户当前消息 | 已接受 |
| D-002 | 大体保留远端 README 和产品身份，但所有能力与命令都必须依据新实现重新审计 | 用户当前及此前消息 | 已接受；身份、作者和许可由 D-032 固定，能力与命令文本以重建实现为准 |
| D-003 | 以当前本地 `align_single_cpu.py` 作为核心算法权威 | 当前项目指令及本地文件 | 已接受 |
| D-004 | 普通话在本里程碑中只保留占位 | 用户当前消息 | 已接受 |
| D-005 | GPU、批处理、Web、自动模型下载、多格式音频、自动重采样、中文分词、默认 G2P 和置信度校准只建立接口 | 用户当前消息 | 已接受 |
| D-006 | 以可发布的 pip/PyPI 包为目标，并建立严格 CI/CD 测试 | 用户当前消息 | 已接受 |
| D-007 | 对边界明确的工作使用多个子 agent 并行处理，由主 agent 审计、合并、调度和最终验收 | 用户当前消息 | 已接受 |
| D-008 | 使用用户给出的事实权威顺序和冲突处理纪律 | 用户提供的治理文本 | 已接受 |
| D-009 | 在 `/Users/yiyi0369/projects/flexaligner-rebuild` 建立可逆的本地新仓库 | 主 agent 对“新建本地仓库”的路径选择 | 有效实施选择 |
| D-010 | 第一条真实产品链路为英语、CPU、单文件、严格本地模型/词典/WAV | 普通话占位要求和当前 reference/资产 | 有效实施选择；公共 API 冻结前仍可调整 |
| D-011 | 占位能力调用抛出类型化 `FeatureNotAvailableError`，不得静默回退 | 主 agent 的安全/API 设计 | 有效实施选择 |
| D-012 | 在改变行为前先完成特征化并达到 reference 等价 | 主 agent 根据“以本地逻辑为准”建立的迁移保护 | 有效实施选择 |
| D-013 | 未经单独明确授权，不发布到 PyPI，也不修改远端仓库 | 当前范围和外部写入安全要求 | 有效保护规则 |
| D-014 | 使用 `src/` 包布局和单一权威版本来源 | 主 agent 的打包设计；与当前 PyPA 指南一致 | 有效实施选择 |
| D-015 | 在 Stage 1 经审阅的 88% 基线后，将首个分支覆盖率门槛设为 85%；后续只能保持或提高，不得静默降低 | 主 agent 本地门禁证据；Python 3.10.8 上 50 项 Stage 1 测试 | 有效实施选择 |
| D-016 | 将权威脚本逐字节保存为测试证据，同时禁止生产代码导入，并从 wheel/sdist 排除 | Stage 2 可移植性与包边界审计 | 有效实施选择 |
| D-017 | 离线模型预检必须绑定已提交的候选 manifest 和精确运行时；缺少或不匹配时关闭式失败 | Stage 2 E2E 预检审计 | 保护规则继续有效；候选阶段 E2E 以及 D-033 后的 approved exact-wheel 复跑均在本地通过 |
| D-018 | Stage 1 使用仅依赖 NumPy 的内部核心，精确核算稠密 trellis，并允许调用方提供预分配 cell 上限 | Stage 3 等价与资源审计 | 有效实施选择；安全默认值仍为 TBD-ALG-005 |
| D-019 | 在毫秒取整前明确拒绝非有限 chunk 边界，并统一抛出稳定的 `ValueError` | Stage 3 主 agent 审计 | 已接受的无效输入安全修正；有效 reference 行为不变 |
| D-020 | Stage 2 使用仅依赖 NumPy 的图/beam 核心，同时保留完整终态、稳定平局、逐帧 bias、进入代价和相同 phone 的当前行为 | Stage 4 等价、精确 DP 与主 agent 交叉审计 | 有效实施选择；相同 phone 修正仍为 TBD-ALG-003 |
| D-021 | 保留两种已特征化的时长换算：Viterbi 静音锁定使用 `round`，短 gap 剪枝使用 `ceil` | 当前 reference 行为与 Stage 4 差分证据 | 已接受的等价行为；10 ms 帧移下 65 ms 分别对应 6 个锁定帧和 7 帧剪枝阈值 |
| D-022 | 本地词表出现重复 JSON member、Chunker tokenizer 映射与 `vocab.json` 不同，或 posterior 不是有限且归一化的对数概率时，必须关闭式失败 | Stage 5 adapter/pipeline 交叉审计 | 已接受的无效模型安全修正；有效 reference 输入不变 |
| D-023 | 使用同目录原子 no-clobber 硬链接发布输出，随后重新验证文件身份、精确字节和语义；不得覆盖并发产物 | Stage 5 输出事务交叉审计 | 有效实施选择；跨文件崩溃一致性仍为 TBD-OUT-001 |
| D-024 | 在模型加载前拒绝 transcript 中的 `sil` 和 `null`，因为当前 reference tier 格式将它们作为保留标签 | Stage 5 词身份审计 | 已接受的明确限制；无冲突身份方案仍为 TBD-TEXT-001 |
| D-025 | Chunker 与 Aligner 模型 session 不得重叠，必须只使用 CPU 和本地文件；先释放 Chunker，再加载 Aligner | Stage 5 生命周期审计和当前用户范围 | 有效实施选择 |
| D-026 | 当英语冻结 manifest 为 `status=candidate` 时，只把它视为工程证据；release E2E 必须在 manifest 明确为 `approved` 后才能通过 | Stage 6 E2E 与发布审计 | 历史候选阶段由 D-033 关闭；approval 关闭式保护仍有效，approved exact-wheel 已在本地通过 |
| D-027 | 精确固定 Hatchling，并在安装固定 package group 后使用 `--no-isolation` 构建 | Stage 6 可复现性审计 | 有效 CI/发布选择；当前 backend 为 1.32.0 |
| D-028 | 使用当前 reference/新实现双跑输出作为 E2E oracle；既有 OpenPhonetics TextGrid 只作为带 hash 的 legacy candidate 保留 | D-003 权威规则与 Stage 6 字节比较 | 已接受的冲突处理；没有把旧输出预期静默合并进来 |
| D-029 | 将首个公开版本定位为 `PUBLIC_ALPHA`，PEP 440 版本为 `0.1.0a1` | 用户审阅消息 | 已接受；在批准的元数据修改完成并复验前，`pyproject.toml` 仍保持开发版本 `0.1.0.dev0` |
| D-030 | 使用 `REPLACE_MAIN_HISTORY` 接入 GitHub：直接替代现有 `USTCPhonetics/FlexAligner` 的 main 历史，不合并或嫁接无关历史 | 用户审阅消息 | 已接受的策略；不构成配置 remote、push、强制更新分支、修改默认分支、创建 tag 或取消可恢复性的授权 |
| D-031 | PyPI distribution 使用 `flexaligner`，owner/organization 使用 `ustcphonetics` | 用户审阅消息 | 已接受；PyPI 实际可用性、控制权和 Trusted Publisher 配置仍须外部验证 |
| D-032 | 许可、版权、作者、单位和引用身份采用原远端固定快照：MIT；`Copyright (c) 2026 WANG Yiming`；Yiming Wang 与 Jiahong Yuan，USTC | 用户审阅消息；固定上游 README 和同提交 LICENSE，提交为 `c5361efe4b5d8ad02574dae1bd7caa89ed3e4af0` | 已接受；README 提供 MIT 标识和作者，同提交 LICENSE 提供精确版权行；不复活旧仓库中未经支持的能力声明 |
| D-033 | 只批准 `openphonetics OW1 P AH0 N F AH0 N EH1 T IH0 K S` 作为冻结的 release-E2E 测试夹具发音 | 用户审阅消息 | 已接受并通过 approved exact wheel 本地复验；它不是规范发音、默认 G2P、准确率金标准或模型分发许可 |
| D-034 | 接受下述 v0.1 公开预览支持边界（`ACCEPT_PREVIEW_BUNDLE`） | 用户审阅消息 | 已接受；允许 alpha 披露并延期研究问题，但不把实施、CI、远端 E2E 或发布门禁自动改成通过 |
| D-035 | 对外发布项目的根 `README.md` 保持现有中英双语接口；给项目维护者使用的治理、计划、验收、资源和测试说明文档一律使用中文 | 用户当前消息 | 已接受；翻译前英文版由提交 `31becaf` 和 `docs/archive/2026-08-22-english-docs.md` 索引保存 |

## 已接受的 v0.1 公开预览边界（D-034）

批准的 `0.1.0a1` 边界如下：

1. 唯一实现的对齐链路是严格英语、CPU、单文件、16 kHz 单声道 PCM16 WAV、
   本地模型和本地全覆盖词典。普通话及其他九项未来能力保持显式占位。
2. wheel 不携带模型，也不自动下载。用户必须提供兼容的本地资产。
3. Python 3.10--3.14 仍是远端 fast CI 目标。真实模型证据只来自冻结组合：
   Python 3.10.8、NumPy 2.2.6、Torch 2.3.1、Transformers 4.41.2；
   不据此暗示更广泛支持。
4. 公开 alpha 前，必须把实际 `[inference]` 解析/运行时契约收窄到有证据的范围，
   并通过公共索引安装路径验证；否则应移除公开 extra。广泛 Hugging Face 矩阵延期。
5. 不提供宽泛的旧 Python/CLI 兼容层。只有确认真实调用方后才添加具体 adapter。
6. 保留并披露 reference 等价行为：冻结夹具约 18 ms 尾部 gap、固定 10 ms Stage 2
   stride、同一词内连续相同 phone 折叠，以及简化的重复标签 CTC recurrence；
   不在本次静默修复。
7. 置信度保持 raw 且未经校准。缺失的最终 phone provenance 保持 `None`，不得推断。
8. 词项 `sil` 和 `null` 继续在模型加载前产生类型化输入错误。
9. 不声称存在已批准的默认资源上限，也不把预览版描述成可安全处理任意长度的
   不可信输入。
10. Torch CPU 线程数修改是进程全局行为，公开发布前必须文档化；不声称宿主进程隔离。

## 待决定的算法问题

下列问题尚未获准改变行为：

| ID | 问题 |
|---|---|
| TBD-ALG-001 | MVP 是否以及如何填补 TextGrid 的内部/尾部 gap，同时保持非 `NULL` 边界 |
| TBD-ALG-002 | Stage 2 是只验证当前 10 ms stride，还是动态推导 stride |
| TBD-ALG-003 | 是否区分同一词内连续相同 phone 的图状态 |
| TBD-ALG-004 | Stage 1 是否加入标准的重复目标 CTC blank 约束 |
| TBD-ALG-005 | 最终时长、token、trellis cell 和 beam work 上限 |

任何解决方案都需要专门测试和新的明确决定记录。D-034 对 `0.1.0a1` 的接受，
只表示允许披露并延期这些问题，不表示接受任何改变行为的答案。
