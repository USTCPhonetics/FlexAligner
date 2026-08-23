# 验收矩阵

> 审阅者：主 agent
>
> 规则：只有主 agent 在重新运行对应证据后，才能把某一行改为 `PASS`。
>
> 状态词表：`NOT_RUN`、`IN_PROGRESS`、`PASS`、`FAIL`、`BLOCKED`、
> `PLACEHOLDER`、`N/A`。
>
> 2026-08-23 边界：既有 `PASS` 是 D-036--D-040 之前的迁移/工程基线证据。新增决定
> 取代冲突的发布目标，但不会倒写历史测试结果；修正项必须由下列新增行独立复验。

## A. Stage 0 — 仓库与治理

| ID | 验收条件 | 状态 | 证据/命令 | 审阅备注 |
|---|---|---|---|---|
| S0-001 | 新本地 Git 仓库位于 `main` 且没有 remote | PASS | `git branch --show-current` → `main`；`git remote -v` → 空 | 2026-08-11 验证 |
| S0-002 | 生产代码实施前已经记录实施计划 | PASS | 初始 tree 只有治理文件；没有 `src/` 或包 metadata | 2026-08-11 验证 |
| S0-003 | 治理文档在事实权威、范围和占位能力上保持一致 | PASS | 主 agent 对六份 Markdown 的跨文档审计 | 用户接受事实与实施选择分离 |
| S0-004 | Reference 文件存在且 SHA-256 与冻结值一致 | PASS | `shasum -a 256 /Users/yiyi0369/projects/flexaligner/align_single_cpu.py` | `9ed4e21e...e835de1`，2026-08-11 验证 |
| S0-005 | 所有实质未决选择都以 `[TBD]`/未决问题明确可见 | PASS | `rg 'TBD'`；`OPEN_QUESTIONS.md`；`DECISIONS.md` 算法表 | 实质未知项映射到稳定 ID |
| S0-006 | 仅治理文件基线通过空白/diff 检查并已提交 | PASS | `git diff --check`；根提交 `833306e` | 2026-08-11 提交治理基线 |

## B. Stage 1 — 包、接口与 CI/CD

| ID | 验收条件 | 状态 | 证据/命令 | 审阅备注 |
|---|---|---|---|---|
| S1-001 | `src/` 包只有一个权威版本，导入时没有模型/网络副作用 | PASS | `test_import_safety.py`；隔离 wheel 从 `site-packages` 导入；metadata 版本 `0.1.0.dev0` | 不导入 Torch/Transformers，不写 cwd |
| S1-002 | Editable 安装与 wheel 安装都提供 `flexaligner` CLI | PASS | `.venv` editable 安装；`/tmp/flexaligner-wheel-smoke.Q5d5Vd` wheel 安装 | `pip check`、版本和 capability 通过 |
| S1-003 | CLI help、version 和 capability discovery 输出确定 | PASS | `tests/test_cli.py`；9 项测试 | 人类可读与 JSON 输出稳定 |
| S1-004 | 所有要求的未来能力都有可导入契约 | PASS | 公共符号与完整 14 项 capability report | Future enum 不表示能力可用 |
| S1-005 | 占位调用抛出类型化不可用错误，不静默回退 | PASS | `tests/test_placeholders.py`、`test_capabilities.py`、CLI 测试 | guard 先于输入/模型/输出工作 |
| S1-006 | 已定义格式、lint、strict type、test/coverage、build 和 wheel smoke job | PASS | 提交 `5702f0a`；本地 Ruff/mypy/50 tests/88.16% coverage/build；`ci.yml` | 远端矩阵仍为 `TBD-CI-001` |
| S1-007 | Release workflow 受事件/environment 保护；只有 publish job 有 OIDC 写权限 | PASS | `tests/test_workflow_security.py`；5 项测试；YAML 解析 | 静态策略通过；未尝试远端发布 |
| S1-008 | 包 metadata、README 和许可文件存在且一致 | PASS | `twine check --strict`；`check-wheel-contents`；`audit_dist.py` | 已复核 README/LICENSE 上游源 hash |

## C. Stage 2 — 特征化与 oracle

| ID | 验收条件 | 状态 | 证据/命令 | 审阅备注 |
|---|---|---|---|---|
| S2-001 | CI 保护冻结 reference hash 或等价的 vendored oracle manifest | PASS | 字节 `cmp`；SHA `9ed4e21e…835de1`；7 项 guard tests；新 dist 审计 | Vendored reference 只用于测试，禁止进入 wheel/sdist |
| S2-002 | Stage 1 reference 语义有无模型数组测试 | PASS | `test_stage1_reference.py`：30 passed | 独立 NumPy 和小规模穷举 oracle |
| S2-003 | Stage 2 图/path/剪枝/二次解码语义有无模型测试 | PASS | `test_stage2_reference.py`：26 passed | 精确 DP oracle 加具名相同 phone 限制 |
| S2-004 | 输入、失败、词序、TextGrid 和原子写语义有测试 | PASS | 输入 19 passed；TextGrid/输出 6 passed；Stage 1 覆盖负例 | 严格 PCM16/16 kHz/单声道；没有 skip/xfail |
| S2-005 | 差分框架报告字段级不一致并禁止随意更新 golden | PASS | 差分测试 4 passed；特征化规则审计 | A/B/C 类分开；没有 update-golden 入口 |
| S2-006 | 真实模型 fixture manifest 记录 hash、版本和 provenance | PASS | 16/16 hash 加精确 3.10.8/2.2.6/2.3.1/4.41.2 运行时预检 | D-033 approved fixture；本地 exact-wheel 对齐证据另见 Q-007 |
| S2-007 | 缺少 E2E 资产产生 `BLOCKED`/明确预检失败，不得变成通过的 skip | PASS | `test_verify_model_assets.py`：7 passed；负例子进程非零退出 | Workflow 绑定已提交 manifest 和精确运行时检查 |
| S2-008 | D-036--D-040 每项修正都有绑定决定 ID 的 before/after 特征化和独立 oracle | NOT_RUN | 待新增具名测试与字段级差分报告 | 不得覆盖 reference fixture 或用新 golden 代替旧行为证据 |

## D. Stage 3 — Stage 1 实现

| ID | 验收条件 | 状态 | 证据/命令 | 审阅备注 |
|---|---|---|---|---|
| S3-001 | 第一条发音与 Chunker 重音数字移除匹配 reference | PASS | 三方等价测试：123 passed | 覆盖首个 variant/顺序/失败上下文 |
| S3-002 | D-039 之前的 trellis/backtrace（含目标提前完成）匹配 reference | PASS | reference + NumPy oracle + 固定种子穷举 | 历史迁移证据；其中“重复目标当前行为”已被 D-039 取代，不是新发布目标 |
| S3-003 | Word emission confidence 匹配未经校准的 reference 定义 | PASS | emission/平均帧/phone-to-word 数值等价 | 保留概率几何平均语义 |
| S3-004 | ±0.3 s anchor、严格 `<0.2 s` 合并和毫秒网格匹配 reference | PASS | 边界/取整/尾部裁剪等价与负例 | D-019 下 NaN/Inf 在取整前明确失败 |
| S3-005 | 每个 word index 按顺序恰好覆盖一次，否则对齐失败 | PASS | 重复/缺失/乱序/word mismatch 不变量 | 重复 word label 仍按 index 区分 |
| S3-006 | D-039 标准重复目标 CTC blank 约束已实施 | PASS | 同词、跨词、去除 ARPAbet 重音后重复、2 帧失败/3 帧 blank 分隔测试；fast 717 passed | 有意偏离冻结 reference；before 行为仍由具名 parity 测试保存 |
| S3-007 | D-040 Stage 1 默认保险丝在工作分配前生效 | PASS | 默认 900 s/200M 契约测试；精确 cell 等值通过/超值在 `numpy.full` 前失败；短 E2E 实测 11,546 与 83,368 cells | 尚无接近 900 s/200M 的真实 E2E，不作性能声明 |

## E. Stage 4 — Stage 2 实现

| ID | 验收条件 | 状态 | 证据/命令 | 审阅备注 |
|---|---|---|---|---|
| S4-001 | 多发音 `sil`/`sph` 图路径匹配已接受的 reference 图语义 | PASS | 80 项等价 + 81 项独立 Stage 2 测试；64 配置只读交叉审计 | 验证六种内部路径、边界 SPH 和多发音路径 |
| S4-002 | Beam Viterbi score、边界对比和 transition 代价匹配 reference | PASS | reference/精确 DP 差分；固定种子随机交叉审计；Stage 2 分支覆盖 90.63% | 固定稳定 `>` 平局、最终 move、逐帧 bias 和一次性进入代价 |
| S4-003 | 不完整终态 path 明确失败 | PASS | 窄 beam 与非法/不完整 path 负例 | 不把部分对齐作为成功返回 |
| S4-004 | 内部短 `sil`/`sph` 剪枝匹配 65/50 ms 阈值 | PASS | 专用锁定/剪枝边界测试 | Viterbi 使用 `round`（65 ms → 6）；prune 使用 `ceil`（65 ms → 7） |
| S4-005 | 固定状态第二次解码匹配 reference | PASS | reference 等价、精确 DP 不变量和参数捕获回归 | 第二遍明确清零 minimum lock 和两个进入代价 |
| S4-006 | 相邻重复词保留不同 word index | PASS | 重复 `go go` phone/word 分段测试 | Word 身份按 index，不只按 label |
| S4-007 | D-038 之前的相同 phone 状态限制已特征化 | PASS | 具名当前行为测试；D-020 | 历史迁移证据；折叠行为已被 D-038 取代，不是新发布目标 |
| S4-008 | D-038 连续相同 phone occurrence 在两遍解码和分段中保持独立身份 | PASS | `(word_index, pronunciation_index, phone_index)` 经图、path、fixed state、phone segment、分块合并和公共 API 传播测试 | 同 label 不合并；特殊 `sil`/`sph`/`NULL` 身份为 `None`；TBD-API-002 已解决 |
| S4-009 | D-040 Stage 2 图状态及累计 beam work 保险丝关闭式生效 | PASS | 10k graph states 在物化前检查精确边界；200M work 覆盖 start/stay/successor/terminal、跨 decode/两遍共享和 pipeline 无残留 | 默认 10k states/200M work，beam 400；长样本性能仍待验收 |

## F. Stage 5 — Pipeline、CLI 与输出

| ID | 验收条件 | 状态 | 证据/命令 | 审阅备注 |
|---|---|---|---|---|
| S5-001 | 强制执行严格 WAV/text/词典/模型/词表验证 | PASS | 50 项严格输入测试；posterior/model 负例集成测试；完整 673 项门禁 | 包含重复 JSON key、精确 Chunker token→ID 映射和保留标签预检 |
| S5-002 | 推理只用 CPU/本地文件，并惰性导入模型依赖 | PASS | `test_hf_local.py` 加导入安全测试；46 项 adapter/生命周期测试 | `local_files_only=True`、`trust_remote_code=False`；顶层不导入 Torch/Transformers |
| S5-003 | Chunker 与 Aligner 模型顺序加载，不同时保留 | PASS | `test_model_lifecycle.py`；插桩集成 trace | `chunk.load→infer→close→align.load→infer→close`，含失败清理 |
| S5-004 | Pipeline 保留完整归一化输入词序 | PASS | 24 项集成测试加 Stage 1/2 不变量 | 重复词按 index 区分；不完整或变化序列失败 |
| S5-005 | TextGrid 临时写入、回读验证并以不覆盖方式原子发布 | PASS | 34 项 TextGrid/事务测试 | No-clobber 硬链接提交；inode/字节/语义后验证；保留 TBD-OUT-001 |
| S5-006 | 失败运行不留下正式成功产物 | PASS | 竞态、篡改、symlink 和失败注入测试 | 回滚只删除仍由本次调用拥有的产物 |
| S5-007 | Confidence metadata 明确未经校准 | PASS | pipeline/API/metadata 集成测试 | Raw score 为有限 `[0,1]`；`calibrated=false`，不替换为校准 score |
| S5-008 | D-036 phones/words 两层以 `NULL` 严格覆盖完整音频 | PASS | 合成三类 gap/亚 epsilon 测试；`english_natural`、`example1` 和 approved release fixture 真实复测 | 两层覆盖 `[0, audio_duration]`；所有旧非 `NULL` 边界保持不变 |
| S5-009 | D-037 v0.1 Aligner 名义 10 ms stride 契约关闭式生效 | PASS | `conv_stride` 必须为正整数序列且乘积 160；实际配置 `[5,2,2,2,2,2,1]` 通过；非 160/畸形配置失败 | 观察到的 `seconds_per_frame` 不要求精确等于 0.01；Stage 2 时间戳仍固定 0.01 |
| S5-010 | D-040 默认保险丝经 Python API/CLI 统一应用且不可被无意绕过 | PASS | 默认 900 s/10k phones/200M cells/10k graph states/200M work；窄上限产生 `resource_limit_exceeded` 且无正式输出；fast 717 passed | 保险丝不是延迟、吞吐或 900 s 成功 SLA |
| S5-011 | 使用 `english_natural.wav` / `.txt` 实测复核 D-036--D-040 | PASS | 5.015 s、12 words；新 SHA-256 `02ac0a42...e205d`；双 tier 全覆盖且非 `NULL` 边界不变 | 资源计数沿用同输入已插桩结果 11,546 cells/281,411 work；不证明 900 s 性能 |
| S5-013 | 使用 `example1.wav` / `.txt` 实测多 chunk 对齐 | PASS | 49.0413125 s、10 words、3 chunks；新 SHA-256 `54d77a81...c1e3a`；双 tier 全覆盖 | 资源计数沿用同输入已插桩结果 83,368 cells/219,135 work；词序和非 `NULL` 边界守恒 |

## G. 用户要求的占位接口

| ID | 能力 | 状态 | 证据/命令 | 审阅备注 |
|---|---|---|---|---|
| P-001 | 普通话 | PLACEHOLDER | `language.zh` capability/API tests | 已验证类型化 pre-I/O failure |
| P-002 | GPU | PLACEHOLDER | `device.gpu` capability/API tests | 不回退到 CPU |
| P-003 | 批处理 | PLACEHOLDER | `alignment.batch` tests | 已验证不消费 iterable |
| P-004 | Web | PLACEHOLDER | `integration.web` CLI/API tests | 不导入 framework，不绑定端口 |
| P-005 | 自动模型下载 | PLACEHOLDER | `models.auto_download` tests | 网络访问前失败 |
| P-006 | 多格式音频 | PLACEHOLDER | `audio.multi_format` tests | 当前只严格实现 WAV |
| P-007 | 自动重采样 | PLACEHOLDER | `audio.auto_resample` tests | 不隐式转换 |
| P-008 | 中文分词 | PLACEHOLDER | `text.zh_segmentation` report/require tests | 普通话保持占位 |
| P-009 | 默认 G2P | PLACEHOLDER | `pronunciation.g2p.default` tests | 不提供 OOV 回退 |
| P-010 | 置信度校准 | PLACEHOLDER | `confidence.calibration` tests | Raw score 契约保持未经校准 |

## H. Stage 6 — 质量、包与真实模型证据

| ID | 验收条件 | 状态 | 证据/命令 | 审阅备注 |
|---|---|---|---|---|
| Q-001 | Formatter/linter 通过且没有忽略新增违规 | PASS | Ruff 0.16.2：76 个维护文件已格式化；全部检查通过 | 不可变 reference 排除格式化并由 hash 保护 |
| Q-002 | Strict static type check 通过 | PASS | mypy 2.3.0：20 个配置的 `src`/`scripts` 文件无问题 | Strict 配置 |
| Q-003 | Fast tests 在每个支持的 Python 版本上通过 | BLOCKED | 本地 3.10.8 和部分 3.12.12 均为 676 passed、1 E2E deselected | 无 remote；3.11/3.13/3.14 与 Linux/Windows 矩阵无当前结果 |
| Q-004 | 覆盖率达到已记录且不可降低的门槛 | PASS | 分支覆盖率 92.31%；D-015 门槛 85% | Python 3.10.8，676 fast tests，E2E marker deselected |
| Q-005 | 当前源码构建 wheel/sdist 并通过 metadata 检查 | PASS | Hatchling 1.32.0、`--no-isolation`；Twine strict/check-wheel/audit 通过 | wheel `0938914d...a1253`；sdist `85beea...72a9` |
| Q-006 | 构建的 wheel 在仓库源码树外安装并运行 | PASS | `/tmp/flexaligner-wheel-site-after.ZXqAVU`；外部导入路径和 exact-wheel CLI E2E | 使用 exact wheel `0938914d...a1253` |
| Q-007 | 英语真实模型 E2E 使用冻结资产通过，或如实标为 `BLOCKED` | PASS | D-033 approved manifest；源码 E2E `1 passed`；exact-wheel TextGrid `d15265c2...2d32a` | 非 `NULL` interval 匹配 reference 且新输出全覆盖；protected remote runner 仍为 NOT_RUN/BLOCKED |
| Q-008 | 测试或包导入不执行未声明网络请求 | PASS | 676 fast tests 和 D-033 exact-wheel E2E 均禁 socket；离线模型环境 | 依赖安装是单独的联网准备步骤 |
| Q-009 | D-036--D-040 修正后的 exact wheel 通过完整 fast、包审计和 approved-fixture E2E | NOT_RUN | 待构建新的 content-addressed artifact 并记录测试数、覆盖率、依赖/模型/input/output hash | Q-005--Q-008 是修正前 artifact 的历史证据，不可直接充当本行证据 |

## I. Stage 7 — 最终审计

| ID | 验收条件 | 状态 | 证据/命令 | 审阅备注 |
|---|---|---|---|---|
| F-001 | 所有必要行均为 `PASS`，有意延期的未来能力为 `PLACEHOLDER` | BLOCKED | 主 agent 验收审计 | S2-008、S3-006--007、S4-008--009、S5-008--011、Q-009 尚未运行；原远端/alpha 阻断仍存在 |
| F-002 | README 声明映射到测试/证据，不把占位能力宣传为支持 | PASS | README/runtime capability 交叉审计；占位测试 | 英语链路可用；十项未来能力保持占位 |
| F-003 | `STATE`、决定和未决问题与已验证仓库状态一致 | PASS | D-036--D-040 跨文档审计；`ALPHA_RESOURCE_VALIDATION.md`；git diff check | D-036--D-040 已实施；长样本性能仍未验收；二者明确分开 |
| F-004 | 工作树、提交、remote 和未执行发布步骤报告准确 | PASS | Stage 8 `git status`、`git log`、`git remote -v`、release guard 审计 | 只有本地 main；无 remote/tag/GitHub/PyPI 修改或发布；无关 `.DS_Store` 保持未跟踪 |
