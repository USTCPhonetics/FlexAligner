# FlexAligner 干净基线重建项目档案

> 文档状态：工程验收完成；用户发布决策已落档；公开发布阻断
> 建立日期：2026-08-11（Asia/Shanghai）
> 当前阶段：Stage 8 — 批阅决定实施与复验

## 1. 当前目标

在 `/Users/yiyi0369/projects/flexaligner-rebuild` 建立新的本地 Git 仓库，
以可发布 PyPI 包为目标，用干净代码重新实现 FlexAligner。第一条真实实现
链路为英语、CPU、单文件、本地模型、本地词典和严格 WAV 输入；核心算法以
当前 `/Users/yiyi0369/projects/flexaligner/align_single_cpu.py` 为准。

普通话、GPU、批处理、Web、自动模型下载、多格式音频、自动重采样、中文
分词、默认 G2P、置信度校准在本里程碑只建立稳定接口和显式不可用行为。

## 2. 项目事实治理纪律

对于项目事实，使用以下权威顺序：

1. 当前消息明确提供的信息
2. 当前上传的代码、配置、数据和实验输出
3. `STATE.md`
4. `DECISIONS.md`
5. `OPEN_QUESTIONS.md`
6. 旧会话内容

若旧会话与文件冲突：

- 以当前文件和 `STATE.md` 为准；
- 明确指出冲突；
- 不得静默合并；
- 不得从旧会话复活已否定的假设。

除非用户明确要求：

- 不将 EXPLORE 或 TEACH 会话中的内容视为已接受决策；
- 不根据模糊历史自行补全项目状态。

## 3. 权威输入快照

### 3.1 本地算法参考

| 项目 | 当前已核实事实 |
|---|---|
| 路径 | `/Users/yiyi0369/projects/flexaligner/align_single_cpu.py` |
| SHA-256 | `9ed4e21e615718ddfd10930359f55769fb27a0d284599cce45a3fc755e835de1` |
| 规模 | 2,548 行 |
| 用途 | 算法语义参考、行为 oracle；生产包不得 import |
| 当前限制 | 连续覆盖、固定 Stage 2 stride、重复 phone state、Stage 1 重复标签约束、资源上限等问题尚未解决 |

“以本地逻辑为准”不是逐行复制，也不是把已知限制静默宣布为正确行为。
实施顺序是先特征化、再等价迁移；行为更改必须单独记录决定。

### 3.2 远端项目快照

| 项目 | 当前已核实事实 |
|---|---|
| 仓库 | `https://github.com/USTCPhonetics/FlexAligner` |
| 默认分支快照 | `main@c5361efe4b5d8ad02574dae1bd7caa89ed3e4af0` |
| README 来源 | 上述固定提交的 `README.md` |
| 用途 | README/品牌/许可来源和差异比较；旧核心代码不进入新实现 |
| `dev` | 存在独立快照，未审计、未合并、不是当前需求来源 |

### 3.3 新仓库当前事实

| 项目 | 当前状态 |
|---|---|
| 路径 | `/Users/yiyi0369/projects/flexaligner-rebuild` |
| Git | 新建本地仓库，默认分支 `main` |
| remote | 未配置；用户选择直接替代现有 `USTCPhonetics/FlexAligner` 主历史，但未授权执行 |
| PyPI 项目 | distribution 选择为 `flexaligner`，owner/organization 选择为 `ustcphonetics`；外部存在性/控制权未验证 |
| 首个公开版本 | `PUBLIC_ALPHA` / `0.1.0a1`；当前 `pyproject.toml` 仍为 `0.1.0.dev0`，待实施复验 |
| 发布 | 未授权、未执行 |
| 生产代码 | 严格英语 CPU 单文件管线、API/CLI、本地推理、TextGrid 事务及未来能力占位已接通 |
| 测试 | 676 项禁网 fast tests 通过；分支覆盖率 92.31% |
| reference | 仓内字节冻结并 hash guard；禁止进入生产 import、wheel 或 sdist |
| E2E 资产 | 16 项资产与精确运行时通过；D-033 fixture-only approved exact-wheel 本地 E2E 通过；protected remote 门禁仍未运行 |
| Stage 1 资源 | 精确 O(TN) 估算和显式 cell limit 已验证；安全默认值 `[TBD-ALG-005]` |
| Stage 2 资源 | O(TB(1+d)) / O(TB) 复杂度已审计；`beam=400` 不是已验证安全上限 |

## 4. 核心算法契约摘要

### Stage 1

- 整段音频运行 Chunker CTC，秒/帧按实际时长除以输出帧数计算；
- 每词使用词典第一条发音，Chunker phone 去除 ARPAbet 重音数字；
- 稠密 trellis，允许从完成目标的最优时间点提前回溯；
- 词置信度是 phone emission 概率的几何平均，未经校准；
- word anchor 左右扩展 0.3 秒，gap 严格小于 0.2 秒才合并；
- chunk 边界落到毫秒网格，所有 word index 按序恰好覆盖一次。

### Stage 2

- 每个 chunk 独立推理，词典所有发音进入图；
- 词边界可选择 epsilon、`sil`、`sph` 及其既有组合；
- 第一遍 Beam-Viterbi 保持当前 stay/move、逐帧 bias、独立 enter cost 和
  boundary contrast 语义；
- 内部短 `sil`/`sph` 经过阈值剪枝，再以固定状态序列第二遍重解码；
- 必须达到完整合法终态；OOV、未知 phone 或未消费完整文本均显式失败。

### 输出

- 局部结果平移/裁剪至全局时间轴；
- words/phones tier 保持输入词序；
- 只允许合并相邻 `NULL`；
- 输出经临时写入、回读验证后以原子 no-clobber 操作发布；可选 metadata
  与 TextGrid 的跨文件崩溃一致性仍为 `[TBD-OUT-001]`。

更完整的迁移和验收规则见 `IMPLEMENTATION_PLAN.md` 与 `ACCEPTANCE.md`。

## 5. 能力状态表达

新包中的能力状态必须区分：

- `available`：有真实实现并通过相应验收；
- `placeholder`：接口已存在，调用会产生类型化“尚不可用”错误；
- `unavailable`：当前构建不支持或缺少已声明依赖/资产。

不得把 `placeholder` 写成“支持”，也不得使用静默回退伪造成功。

## 6. 已知冲突

| ID | 冲突 | 当前处置 |
|---|---|---|
| C-001 | 旧会话曾描述 reference 已补齐连续 gap；当前脚本没有相邻无 gap/终点覆盖的严格保证 | 当前脚本优先；旧描述不进入基线，修复为 `[TBD-ALG-001]` |
| C-002 | 旧普通话实验曾使用 `optional_sph=False`；当前脚本默认 `True` | 普通话本里程碑仅占位，不继承旧实验特例 |
| C-003 | 远端 README/代码描述多格式、批量、GPU、G2P 等能力；当前核心范围不实现 | README 能力表按新包实测重写；这些能力只留占位接口 |
| C-004 | 远端核心与本地脚本虽同为“两阶段”，但终止、切块、图、评分和失败语义不同 | 不静默混合；新核心只以本地脚本为迁移参考 |
| C-005 | 旧问题把 `openphonetics` 写成缺词典；当前 fixture 词典已包含该发音并完成工程 E2E | 当前文件优先；技术 OOV 已消失，且 D-033 已批准该发音仅作 release-E2E fixture |
| C-006 | OpenPhonetics 既有 TextGrid SHA 为 `78a69bf4…e3e00`，当前权威脚本与新实现输出均为 `ddbe0fec…e415f` | 旧文件只保留为 hashed legacy candidate，不作为 oracle，也不与当前输出静默合并 |
| C-007 | 审阅报告曾推荐保留新旧双历史；当前用户明确选择直接替代 GitHub 主历史 | 当前用户消息优先，采用 D-030；不折中为 merge/graft，也不把策略批准解释为外部执行授权 |
| C-008 | 用户要求许可/版权/作者采用远端 README 身份，但远端 README 没有精确版权行 | README 提供作者/品牌/引用；同提交 LICENSE 提供 MIT 正文与 `Copyright (c) 2026 WANG Yiming`；不推定版权转让给组织 |

## 7. 文档职责

| 文件 | 职责 |
|---|---|
| `IMPLEMENTATION_PLAN.md` | 阶段、并行分工、门禁和完成定义 |
| `ACCEPTANCE.md` | 每个可核验条件、状态、命令和证据 |
| `STATE.md` | 当前已验证状态、正在进行的阶段和下一动作 |
| `DECISIONS.md` | 已接受决定、实施决定及其依据 |
| `OPEN_QUESTIONS.md` | 未决项、默认占位和阻塞范围 |
| `REAL_MODEL_E2E_REPORT.md` | 冻结资产、确切 wheel、reference 差分及发布阻断证据 |
| `FINAL_ACCEPTANCE_REPORT.md` | 工程验收结论、发行候选和最小发布前计划 |
| `REVIEW_DECISION_REPORT.md` | 用户批阅选项、D-029..D-034 结果和外部授权边界 |
