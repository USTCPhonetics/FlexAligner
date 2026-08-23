# FlexAligner 干净代码重建实施计划

> 状态：D-039/D-040 已实施并通过短自然语音 E2E；D-036--D-038 待实施；公共发布仍被阻断
>
> 建立日期：2026-08-11（Asia/Shanghai）
>
> 本地仓库：`/Users/yiyi0369/projects/flexaligner-rebuild`
>
> 算法 reference：`/Users/yiyi0369/projects/flexaligner/align_single_cpu.py`
>
> Reference SHA-256：`9ed4e21e615718ddfd10930359f55769fb27a0d284599cce45a3fc755e835de1`

## 1. 目标

建立新的干净 Python 代码库，可作为 PyPI 包分发，第一条真实产品链路为：

```text
英语 transcript + 本地词典 + 本地 CTC 模型 + 16 kHz 单声道 PCM16 WAV
    -> 仅 CPU 的两阶段强制对齐
    -> 经验证、原子 no-clobber 发布的 TextGrid
```

在任何改变行为的修正前，实现必须先特征化当前 `align_single_cpu.py` reference 的算法
语义。Reference 是迁移 oracle 和 before 证据来源，生产代码不得导入它。D-036--D-040
已经批准五项有意修正；这些修正实施后必须保留字段级 before/after 证据，不能再以
reference parity 代替修正后的验收。

仓库还必须为未来工作暴露有意设计并有文档的扩展边界，但不得把尚未实现的能力伪装
成可用。

## 2. 完成定义

只有同时满足以下条件，目标才算完成：

1. 本地仓库包含可复现的 `src/` 布局 Python 包、可安装 wheel/sdist、单一权威版本
   来源和可工作的 CLI。
2. CPU/单文件/英语链路使用干净模块和明确契约实现 Stage 1、Stage 2、pipeline 与
   TextGrid 输出。
3. Fast CI 不下载外部模型即可通过：格式化、lint、strict typing、unit/characterization
   tests、覆盖率、包构建、metadata 验证和隔离 wheel 安装 smoke。
4. 单独门禁的真实模型 E2E 要么针对冻结本地资产 manifest 通过，要么以精确缺失证据
   记录为 `BLOCKED`；不得静默 skip 后仍报告通过。
5. 普通话、GPU、批处理、Web、自动模型下载、多格式音频、自动重采样、中文分词、
   默认 G2P 和置信度校准具有可导入契约和明确 capability 状态；实施前调用必须抛出
   类型化 `FeatureNotAvailableError`。
6. `ACCEPTANCE.md` 每一行都包含状态、命令/证据和审阅备注。必要项为 `PASS`；
   有意保留的占位项为 `PLACEHOLDER`，不能写成 `PASS` 或“已实现”。
7. `STATE.md`、`DECISIONS.md`、`OPEN_QUESTIONS.md` 准确描述最终验证状态，
   不从旧会话复活假设。

## 3. 项目事实纪律

项目事实按以下权威顺序使用：

1. 用户当前消息明确提供的信息；
2. 当前上传/本地代码、配置、数据和实验输出；
3. `STATE.md`；
4. `DECISIONS.md`；
5. `OPEN_QUESTIONS.md`；
6. 旧会话内容。

旧会话与文件冲突时：

- 以当前文件和 `STATE.md` 为准；
- 明确记录冲突；
- 不静默合并两个版本；
- 不从会话历史复活已否定假设。

除非用户明确要求，否则 EXPLORE/TEACH 材料不视为已接受决定，也不使用模糊历史补全
项目状态。

文档语言遵循 D-035：根目录对外 `README.md` 保持中英双语；内部治理、计划、验收、
资源、reference 和测试说明使用中文。命令、API、路径、错误码和状态值保持原样。

## 4. 已接受范围

### 4.1 首个包中真实实现

- 干净仓库和包身份；
- Python 3.10+（最终支持上限由远端 CI 矩阵确认）；
- 仅 CPU 执行；
- 每次调用处理一个音频和一个 transcript；
- 英语、空白分词 transcript；
- 严格本地词典全覆盖；
- 严格本地 Chunker/Aligner 模型路径；
- 严格 16 kHz、单声道、未压缩 PCM16 WAV 输入；
- Stage 1 CTC 宏观定位/切块；
- Stage 2 发音图与两遍 Viterbi 对齐；
- 严格输入、词表、path 完整终态和词序失败；
- words/phones TextGrid tier；
- 临时写入、回读验证和原子 no-clobber 发布；
- 可选的未经校准 Stage 1 confidence metadata，明确命名语义且绝不表述成校准概率；
- Python API 和 `flexaligner` / `python -m flexaligner` CLI；
- 包构建、安装、CI 和受保护 release workflow。

### 4.2 只提供稳定占位

下列能力在本里程碑中必须有明确契约和 capability discovery，但没有生产实现：

| 能力 | 占位契约 | 当前必要行为 |
|---|---|---|
| 普通话 | `Language.ZH` 加 capability guard | 报告 placeholder；类型化失败 |
| GPU | `Device.CUDA/MPS` 加 capability guard | 报告 placeholder；明确只支持 CPU |
| 批处理 | `align_batch()` 加 capability guard | 报告 placeholder；不得消费 iterable |
| Web | `integration.web` capability 加 `serve` CLI | 报告 placeholder；不导入 framework 或绑定端口 |
| 自动模型下载 | `ModelResolution.AUTO_DOWNLOAD` 加 `models fetch` CLI | 只接受本地路径；网络访问前失败 |
| 多格式音频 | `AudioPolicy.MULTI_FORMAT` | 只提供严格 WAV/PCM16 decoder；其他格式失败 |
| 自动重采样 | `AudioPolicy.AUTO_RESAMPLE` | 只验证；不隐式转换 |
| 中文分词 | 带普通话 guard 的 `text.zh_segmentation` capability | 只做英语空白归一化 |
| 默认 G2P | `PronunciationMode.G2P` | 只用词典；OOV 保持错误 |
| 置信度校准 | `CalibrationMode.CALIBRATED` 加 raw-score schema | 只提供未经校准标记；校准请求失败 |

占位方法不得使用 `pass`、返回空成功值、静默回退、下载资产、修改输入或声称可用。

### 4.3 明确不在范围内

- 模型训练或 fine-tuning；
- 远端托管或部署；
- 修改 GitHub 默认分支或发布到 PyPI；
- 与远端旧实现所有 API 的全面向后兼容；
- 冻结 fixture 未支持的准确率或可靠性声明；
- 本里程碑的普通话 E2E；
- 自动恢复 OOV、缺失 phone 或不完整 Viterbi path。

## 5. 算法迁移规则

迁移与修正是两条分离的工作轨：

1. **特征化 reference 行为。**为已接受语义写确定性测试，并明确标记已知缺陷/限制。
2. **按等价迁移。**新模块在冻结数组和冻结真实模型 artifact 上必须在批准容差内匹配
   reference。
3. **审计等价。**任何不一致都停止该阶段；修改 golden 不是修复。
4. **只通过决定修正。**改变行为的修正需要新的 `DECISIONS.md` 条目、前后比较和
   专用测试。

下列已知问题已由后续明确决定解决；D-039/D-040 已实施并通过短自然语音 E2E，
D-036--D-038 尚未完成：

- D-036：phones/words 两层以 `NULL` 填充开头、内部和末尾 gap，严格覆盖完整音频；
- D-037：v0.1 只接受 16 kHz 下卷积总 stride 为 160 samples 的 Aligner，并固定使用
  10 ms 时间网格；
- D-038：同词连续相同 phone 按至少 `(word_index, phone_index)` 的 occurrence 身份区分；
- D-039：相邻相同 Stage 1 目标必须由至少一个 CTC blank 帧分隔（已实施）；
- D-040：alpha 初始默认保险丝为 900 s、单次 Stage 1 trellis 200,000,000 cells、
  每请求累计 200,000,000 次 beam candidate transition evaluations，beam width 保持
  400（已实施；`english_natural` 和 `example1` E2E 通过，长音频性能尚未验收）。
  words、phone targets 和
  Stage 2 图状态只保留调用方显式限制或实测后另行决定，不属于本次批准的默认值。

这些改变必须绑定决定 ID、专用测试和经审阅的 E2E golden。D-036--D-038 在对应验收项
成为 `PASS` 前，只能描述为“已决定、待实施”。

## 6. 目标仓库结构

```text
.github/workflows/
  ci.yml
  release.yml
src/flexaligner/
  __init__.py
  __main__.py
  api.py
  capabilities.py
  contracts.py
  errors.py
  ports.py
  core/
    stage1.py
    stage2.py
  textgrid.py
  pipeline.py
  cli.py
  adapters/
    wav_pcm16.py
    lexicon_file.py
    hf_local.py
tests/
  unit/
  characterization/
  integration/
  e2e/
  fixtures/
README.md
LICENSE
pyproject.toml
project.md
IMPLEMENTATION_PLAN.md
ACCEPTANCE.md
STATE.md
DECISIONS.md
OPEN_QUESTIONS.md
```

审计时可以减少模块数量，但依赖方向固定：

```text
contracts / errors / capabilities
               ^
adapters -> ports <- pipeline -> stage1 / stage2 / textgrid
                          ^
                         api
                          ^
                         cli
```

`stage1.py` 和 `stage2.py` 只处理明确数组和 domain record，不加载模型、不解析 CLI
参数、不写文件。只有 `adapters/hf_local.py` 可以导入 `transformers`；包导入和
capability discovery 不得导入 Torch、打开模型或访问网络。

### 6.1 预期稳定公共接口

首个公共接口有意保持很小：

- 一个惰性 `FlexAligner` engine，提供 `align()`、`capabilities()`、`close()` 和
  context manager 契约；
- 不可变、仅 keyword 的 `AlignmentRequest`、`AlignmentOptions`、
  `LocalModelBundle`、`TextGridOutput`、`AlignmentResult` record；
- 带版本的 `CapabilityReport`；
- 类型化 `FlexAlignerError` 层级和稳定机器可读字符串错误码。

算法函数和 adapter Protocol 在 v1 保持内部。Enum 可以命名 `ZH`、`CUDA`、
`AUTO_DOWNLOAD`、`G2P` 等未来选项，但选择后必须调用 capability guard，并在消费输入、
创建文件、加载模型或联网前失败。

Raw confidence 单独存储，`kind=chunker_emission_geometric_mean`、
`calibrated=false`。未来 calibrator 可以增加 calibrated score，但不得覆盖或重新解释
raw score。

### 6.2 占位失败顺序

- 普通话/GPU 请求在检查输入/模型前失败，且不得降级为英语/CPU；
- `align_batch()` 在消费 iterable 前失败；
- `serve` 在导入 Web framework 或绑定端口前失败；
- model fetch 在发起网络请求前失败；
- 不支持的音频/重采样请求在调用 ffmpeg 或修改 sample 前失败；
- G2P 在修改 OOV transcript 前失败；词典模式 OOV 保持独立输入错误；
- 校准 confidence 请求必须失败，不得重标记 raw score。

## 7. 工作分配与合并纪律

主 agent 负责范围、仓库状态、共享契约修改、集成、验收证据和最终签字。

并行工作拆分为边界明确的工作流：

| 工作流 | 初始职责 | 合并规则 |
|---|---|---|
| A — 打包/CI | 包 metadata、依赖拆分、CI 与 release gate | 主 agent 核验官方指南并运行全部门禁 |
| B — 架构/API | domain 契约与未来能力占位 | 主 agent 审计导入、行为和 API 稳定性 |
| C — 测试/oracle | 特征化矩阵、fixture、差分框架 | 主 agent 对照当前文件验证所有预期 |
| 主 agent — 治理/core | 计划、决定、状态、集成和算法迁移 | 只有主 agent 可以标记验收行 |

Agent 不得并发编辑同一文件。共享文件系统中的工作只有在主 agent 重跑对应门禁后才
可接受。

### 7.1 Stage 5 并行执行分工

主 agent 冻结共享 Protocol 并创建 `adapters/__init__.py` 后，Stage 5 使用以下独占
范围：

| 工作流 | 独占生产文件 | 独占测试 |
|---|---|---|
| A — 严格输入 | `adapters/wav_pcm16.py`、`adapters/lexicon_file.py` | `tests/unit/test_wav_pcm16.py`、`tests/unit/test_lexicon_file.py` |
| B — 本地推理 | `adapters/hf_local.py` | `tests/unit/test_hf_local.py` |
| C — TextGrid/输出 | `textgrid.py` | `tests/unit/test_textgrid.py`、`tests/unit/test_output_transaction.py` |
| 主 agent — 集成 | `ports.py`、`pipeline.py`、公共 API/CLI/contracts/capabilities/errors 和包导出 | pipeline、生命周期、API/CLI 与占位回归测试 |

合并和审计顺序固定为：严格输入、本地推理、TextGrid/输出、pipeline、API、CLI，最后
提升 capability。推理 factory 暴露不重叠的 Chunker/Aligner context manager；pipeline
必须先退出 Chunker context，再进入 Aligner context。Fast tests 使用 fake session，
保持无模型、禁网。

所有未来选项在 I/O 前 guard。只有完整 pipeline 门禁通过后，单一实现链路才提升为
`available`。对可选 metadata 和 TextGrid，所有 artifact 先暂存/验证；metadata 先提交，
TextGrid 最后作为成功标志；进程内失败回滚本次调用创建的 artifact。普通文件系统无法
保证两个文件在崩溃/断电下原子提交，该限制记录为 `TBD-OUT-001`。

## 8. 分阶段执行

### Stage 0 — 仓库、治理和可执行计划

交付物：

- 新本地 Git 仓库；
- 本计划；
- 初始 `project.md`、`STATE.md`、`DECISIONS.md`、`OPEN_QUESTIONS.md`；
- 带稳定 ID 和 `NOT_RUN` 状态的 `ACCEPTANCE.md`；
- 在不复制旧核心代码的情况下记录 reference/远端 provenance。

门禁：范围完整、未知项有 `[TBD]`/未决记录、reference SHA 匹配，且没有把生产实现
描述成已经完成。

### Stage 1 — 包与接口骨架

交付物：`pyproject.toml`、`src/` 布局和权威版本；公共 domain record、类型化错误和
capability registry；可导入占位接口和明确失败；CLI help/version/capabilities；初始
README 能力表；严格 CI 和受保护 release workflow。

门禁：editable/wheel 安装可用；占位有测试；包导入没有模型/网络副作用；任何 PR 或
普通 branch push 都不能发布；只有 release job 获得 `id-token: write`。

### Stage 2 — Reference 特征化与测试框架

交付物：reference hash guard；纯数组 Stage 1/2 特征化案例；输入/失败/TextGrid/原子
写测试；差分 record 与 golden 更新规则；带 hash/provenance 的冻结 E2E asset manifest。

门禁：fast tests 无模型/无网络；每项迁移行为在实现前有特征化测试；已知限制具名，
不被归一化掉；缺资产时 E2E 不能报告通过。

### Stage 3 — Stage 1 实现

交付物：首条发音选择与重音数字移除；可提前结束的 CTC trellis/backtrace；D-039
标准重复目标 blank 约束；word emission confidence；anchor、严格合并边界和毫秒 chunk
取整；完整有序 word-index 覆盖检查；D-040 Stage 1 默认资源保险丝。

门禁：除 D-039 的具名偏离外，合成数组等价精确或处于明确记录容差内；同词、跨词和
去除 ARPAbet 重音后形成的相邻重复目标都有 before/after 测试；900 s 和
200,000,000 trellis cells 的边界与分配前关闭式失败通过；复杂度、实际 cell 核算和
资源行为重新测量。

### Stage 4 — Stage 2 实现

交付物：多发音 phone DAG；可选 `sil`/`sph` gap path；D-038 occurrence 身份；保留
stay/move、frame bias、进入代价和边界对比语义的 beam Viterbi；完整终态强制；内部
短状态剪枝和固定序列第二次解码；state 到 phone/word 分段；D-040 图规模与累计 beam
work 保险丝。

门禁：图/path/剪枝/第二遍特征化测试通过；不完整 path 明确失败；相邻重复词和同词
连续相同 phone occurrence 都保持不同实例；每请求 transition evaluation 计数跨所有
chunk、第一遍和固定序列第二遍累计，精确边界允许，下一次将超出 200,000,000 时
关闭式失败。Stage 2 图状态默认上限不由 D-040 固定。

### Stage 5 — 推理、Pipeline、CLI 与输出

交付物：严格 WAV/text/词典/模型验证；D-037 名义 10 ms stride 验证；惰性本地
Hugging Face 推理 adapter；顺序 Chunker/Aligner 加载和释放；D-036 全覆盖
local-to-global TextGrid；Python API 和单文件 CLI；结构化且未经校准的 metadata。

门禁：失败不留正式输出；输出写到同目录临时文件、回读验证后不覆盖原子发布；两层
TextGrid 在保留所有非 `NULL` 边界的同时严格覆盖 `[0, audio_duration]`；只接受
16 kHz 下名义卷积总 stride 160 samples，时间戳使用 `frame_index * 0.01`；所有 D-040
默认保险丝经公共 API/CLI 生效；归一化的非特殊词序与输入完全一致；不联网；核心保持
仅 CPU。

### Stage 6 — 包、真实模型 E2E 与发布演练

交付物：完整 fast 质量套件；构建并检查 metadata 的 wheel/sdist；在隔离环境安装
wheel 并在源码树外 smoke；冻结英语真实模型 E2E；带 owner/environment `[TBD]` 的
受保护 Trusted Publishing workflow；最终 README 能力/限制表。

门禁：所有必要 fast gate 本地通过；E2E 记录精确 Python/依赖/模型/词典/输入 hash；
publish workflow 只构建一次并发布同一上传 artifact；未经用户授权和账户配置不实际
上传 PyPI。

### Stage 7 — 主 agent 最终审计

交付物：完整 `ACCEPTANCE.md`；最终 `STATE.md`、决定和未决问题；diff/stat 与仓库
状态审阅；面向用户的发布准备报告。

门禁：必要验收行不得为 `NOT_RUN`、`FAIL` 或虚假 `PASS`；占位与实现能力可区分；
剩余 `[TBD]` 要么是发布阻断，要么是已记录非阻断限制；准确报告工作树状态。

## 9. CI/CD 规则

“严格 CI/CD”表示以下独立且可检查的门禁：

1. 格式检查；
2. lint 和导入卫生；
3. strict static type check；
4. 带分支覆盖率门槛的 fast unit/characterization tests；
5. 从干净源码构建包；
6. sdist/wheel metadata 检查；
7. 在新环境安装构建 wheel 并执行导入/CLI smoke；
8. 可选/手动真实模型 E2E，必须显式预检资产；
9. 只从已批准 GitHub Release/tag 和受保护 `pypi` environment 使用 OIDC Trusted
   Publishing 发布；
10. release job 消费此前构建 artifact，不 checkout 源码，也不执行任意构建步骤。

初始工具为 `ruff`、`mypy`、`pytest`、`coverage`、`build`、`twine`。精确版本和
覆盖率门槛在首个绿色基线前记录为 `TBD-CI-001`；之后只能提高门槛，或通过正式决定
修改。

第三方 GitHub Action 在任何远端使用前固定到不可变 commit SHA；可在注释中标明
人类可读 major tag。

### 9.1 测试与证据层

| 层 | 必要证据 | 常规触发 | 门禁 |
|---|---|---|---|
| C0 reference guard | reference hash/provenance；生产导入和 wheel 排除 | 每次变更 | 阻断 |
| C1 无模型核心 | 纯数组、契约、失败、TextGrid 和占位测试 | 每次变更 | 阻断 |
| C2 包 smoke | 构建、metadata、干净 wheel 安装、导入和 CLI | 支持的 Python 矩阵 | 阻断 |
| C3 posterior 等价 | 固定合成 posterior Stage 1/2 差分报告 | 每次变更/main | 阻断 |
| C4 真实模型 E2E | 离线英语 reference/新实现双跑与完整 hash | 可信 runner/release | 发布阻断；不得成为通过的 skip |
| C5 资源 | D-040 上限边界/关闭式失败、耗时、峰值 RSS、累计 work 和模型生命周期证据 | nightly/release candidate | 阻断 |

测试采用三条分离 oracle 轨道：

- **A — 当前行为等价：**精确可观察 reference 行为，包括具名已知特性；
- **B — 独立不变量：**数学小案例 oracle、词/顺序、有限值、输出和失败保证；
- **C — 已批准修正：**只有绑定决定 ID 的改变。

轨道 C 必须同时证明目标改变和所有无关轨道 A 行为不回归。

### 9.2 Golden 与差分规则

- CI 没有 `--update-golden` 路径；
- generator 只写 `candidate/`，绝不覆盖已接受基线；
- 作为 oracle 的 fixture、posterior、有效词典、模型目录和环境必须 content-addressed；
- 更新基线需要决定 ID、字段级旧/新语义 diff、provenance 修改和 README/API 影响；
- 不得通过放宽容差、删除断言、`skip` 或 `xfail` 解决差异；
- 等价报告记录首个不同字段及双方值；
- 大型外部模型不提交 Git；完整 manifest 和 hash 作为证据；
- reference 与新实现始终写入不同临时目录；
- Fast CI 不下载模型、不联网。

## 10. 证据与状态词表

每项验收只使用一种状态：

- `NOT_RUN`：没有当前证据；
- `IN_PROGRESS`：正在实施或验证；
- `PASS`：已成功运行声明的命令/检查并记录证据；
- `FAIL`：检查已运行且失败；
- `BLOCKED`：因具名前置条件不可用而不能运行；
- `PLACEHOLDER`：接口存在，且不可用行为已验证；
- `N/A`：由已接受范围决定排除。

不得只通过削弱断言或重新生成预期输出，把 `FAIL` 改为 `PASS`。

## 11. 已审阅决定与剩余未知项

2026-08-11 用户审阅解决了初始包/历史/身份选择：

- D-029：首个公开版本 `0.1.0a1`（`PUBLIC_ALPHA`）；
- D-030：直接替代现有 `USTCPhonetics/FlexAligner` main 历史；
- D-031：distribution `flexaligner`，预期 owner `ustcphonetics`；
- D-032：使用固定上游 README 和同提交 LICENSE 身份；
- D-033：指定发音只批准为 release-E2E fixture；
- D-034：接受已披露 v0.1 预览边界，不提供宽泛旧别名；
- D-035：根公共 README 保持中英双语，内部项目文档使用中文。

2026-08-23 后续算法审阅又接受：

- D-036：TextGrid 两层以 `NULL` 实现完整时间覆盖；
- D-037：v0.1 只验证名义 10 ms Stage 2 stride；
- D-038：连续相同 phone 按 occurrence 身份区分；
- D-039：标准重复目标 CTC blank 约束；
- D-040：900 s、200,000,000 trellis cells 与累计 200,000,000 beam work 等 alpha
  初始资源保险丝；代码级与短/中等长度 E2E 已验证，长音频性能仍待新 fixture。

D-036--D-040 取代 D-034 中与其冲突的旧行为保留/延期口径，但决定本身不构成实现或
测试通过。

剩余问题是执行项或 alpha 后研究项：

- `[TBD-CI-001]` 实际远端 Python/操作系统矩阵证据；
- `[TBD-REMOTE-001]` 单独授权、可恢复的直接历史替代；
- `[TBD-E2E-002]` 外部资产与 repo-local fixture 词典的远端可移植拆分；
- `[TBD-REL-001]` 精确 protected environment、审批人和 Trusted Publisher；
- D-036--D-040 的代码实施、before/after 审计、资源边界测试和修正后 E2E 复验。

已解决选择不授予外部操作权限，也不把未运行门禁变成通过。

## 12. 立即执行顺序

1. 为 D-036--D-040 增加 before/after、边界、累计资源和失败原子性测试；
2. 实施 D-036--D-040，重跑 fast 门禁并审阅修正后的 approved-fixture E2E golden；
3. 选择新的经审阅长音频 fixture，记录总耗时、各阶段耗时、峰值 RSS、trellis cells、
   图状态、累计 transition evaluations 和 chunk 数，复核 D-040 alpha 初始门槛；
4. 实施 `0.1.0a1` metadata 和 D-034 冻结推理契约，重建并测试 exact artifact；
5. 在远端 E2E 前去除 manifest 对本地目录名的依赖；
6. 直接替代远端 `main` 前取得单独授权并创建已验证恢复快照；
7. 配置受保护 `pypi` environment、Trusted Publisher、自托管 E2E runner、asset root
   和离线 wheelhouse；
8. 运行声明的 Python/操作系统矩阵、依赖审计和 protected 模型 E2E；
9. 所有发布阻断项清除后，再分别请求 D-013 要求的 tag 和 PyPI 发布授权。
