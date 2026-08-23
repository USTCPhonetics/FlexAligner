# 当前状态

> 最后更新：2026-08-23（Asia/Shanghai）

## 当前阶段

- Stage 0（仓库、治理和可执行计划）：完成。
- Stage 1（包与接口骨架）：完成，提交 `5702f0a`。
- Stage 2（reference 特征化与测试框架）：完成，提交 `e582dd4`。
- Stage 3（Stage 1 实现）：完成，提交 `ec7bd2d`。
- Stage 4（Stage 2 实现）：完成，提交 `d65ab6a`。
- Stage 5（推理、pipeline、CLI 与输出）：完成，提交 `6c1d4eb`。
- Stage 6（包、真实模型 E2E 与发布演练）：完成，提交 `e694645`；当时 approved
  fixture 发布门禁如实保持阻断。
- Stage 7（主 agent 最终审计）：完成。工程基线已接受；在记录的阻断项解决前，
  公共发布为 `NO-GO`。
- Stage 8（已审阅发布决定与实施复验）：进行中。用户已接受 public alpha 产品决定，
  本地 approved-fixture exact-wheel 复跑通过；alpha 元数据/推理约束、manifest
  可移植性和远端发布门禁尚未完成。
- Stage 9（算法审阅修正）：D-039 标准重复目标 CTC 和 D-040 900 s/200M/200M
  保险丝已实施并通过 fast/quality 门禁；D-036--D-038 与修正后 exact-wheel E2E
  尚未完成。当前真实 E2E 只覆盖 5.015 s 和 49.0413125 s，不声称 900 s 性能可用。

## 已验证的当前事实

- `/Users/yiyi0369/projects/flexaligner-rebuild` 是新建本地 Git 仓库，分支为 `main`。
- 没有配置 Git remote。
- `IMPLEMENTATION_PLAN.md` 在生产实现前已经建立。
- 仅治理文件的根提交为 `833306e`。
- 权威算法 reference 已从
  `/Users/yiyi0369/projects/flexaligner/align_single_cpu.py` 逐字节冻结到
  `reference/`，预期 SHA-256 为
  `9ed4e21e615718ddfd10930359f55769fb27a0d284599cce45a3fc755e835de1`。
- `src/` 包、公共契约、用户要求的占位能力、README/LICENSE、严格本地质量门禁和
  受保护 workflow 均已实现。
- Stage 1 包骨架证据：50 项禁网测试通过；分支覆盖率 88.16%，高于 85% 门槛；
  Ruff 与 strict mypy 通过。
- Stage 1 包骨架分发审计通过：wheel
  `f3b6e47f305532329ba1f9c9b7dd6281c165b0bd92ad4cc9b684e33a56db9449`，
  sdist `3867f2c6db38b9b5fcd9d8d3a4fae8d9a27a3202b9f2354ab22324561dca5353`。
- 该 Stage 1 wheel 在源码树外安装并通过 `pip check`、版本、导入路径、CLI 和
  capability smoke。
- Stage 2 共有 92 项无模型特征化/oracle 测试：Stage 1 为 30 项、Stage 2 为
  26 项、输入/失败为 19 项、TextGrid/输出为 6 项、reference guard 为 7 项、
  差分框架为 4 项。
- Stage 2 门禁时，149 项 fast tests 在禁 socket 条件下通过；Ruff、strict mypy、
  actionlint、reference 字节比较和 `git diff --check` 均通过。
- Stage 2 门禁时，当时仍为 candidate 的 E2E manifest 记录了 16 项资产和精确的
  Python/NumPy/Torch/Transformers 版本；冻结的 OpenPhonetics 本地环境通过全部
  hash 和运行时版本检查。该阶段只有资产预检证据；D-033 批准和后续真实对齐证据
  见下文。
- 当前重建分发包排除 `reference/` 和 tests。Stage 3 后重新构建审计通过：wheel
  `8bf982789427ff8dc904278f6a5b563fb8cfd25fb79d11b8111b91f6f067868f`，
  sdist `6d83cb0cc49c3d2ee75b2bd0d7c21f3b1ffe85bdcd6ccaea0efc10c0498be364`。
- 干净的 NumPy Stage 1 核心已实现，不导入 reference、Torch 或 Transformers。
  通过 123 项三方等价测试和 76 项独立不变量/资源测试；当时完整测试为 348 项，
  分支覆盖率 94.75%。
- 已实现稠密 trellis 精确资源核算和调用方提供的预分配 cell 上限。D-040 已批准
  并实施包级默认 200M cell 保险丝；短自然语音实测为 11,546 cells，详见
  `STAGE1_RESOURCE_REPORT.md` 和 `ALPHA_RESOURCE_VALIDATION.md`。
- Stage 3 wheel 在源码树外 `/tmp/flexaligner-stage3-smoke.TPMkh3` 安装，并通过
  `pip check`、包/core 导入、CLI、capability 和资源估算 smoke。
- 干净的 NumPy Stage 2 核心已实现，不导入 reference、Torch 或 Transformers。
  通过 80 项 reference 等价测试和 81 项独立不变量/资源测试；当时完整测试为
  509 项、分支覆盖率 92.85%，Stage 2 模块分支覆盖率 90.63%。
- 独立只读交叉审计检查了全部 64 种边界/内部 SIL/SPH flag 组合，以及 100 个固定
  种子随机小图，并同时与冻结 reference 和独立精确 DP oracle 比较；未发现生产缺陷。
- Stage 2 资源复杂度和限制记录于 `STAGE2_RESOURCE_REPORT.md`。继承的 `beam=400`
  保持不变；D-040 已批准请求级累计 200,000,000 次 beam work 保险丝，但当前已验证
  代码基线尚未实施和实测该保险丝。
- 严格英语 CPU 单文件 pipeline 已实现，包含严格 transcript/词典/WAV/模型/词表
  检查、惰性本地推理，以及不重叠的顺序 Chunker/Aligner session。
- TextGrid 和可选 metadata 会先暂存并回读验证。发布使用原子 no-clobber 硬链接、
  提交后 inode/字节/语义验证和按所有权回滚；跨文件崩溃一致性仍为
  `TBD-OUT-001`。
- Stage 5 主 agent 验证通过 673 项禁 socket 测试，分支覆盖率 92.31%；Ruff 检查/
  格式化 71 个文件，strict mypy 检查 20 个源文件且无错误。
- Transcript 中的 `sil` 和 `null` 会在模型加载前失败，因为当前 tier 标签保留这些
  身份；未来身份方案为 `TBD-TEXT-001`。
- D-036--D-040 合并后的 wheel 和 sdist 通过 Twine strict、check-wheel-contents 和仓库
  inventory 审计，SHA-256 分别为 `0938914d...a1253` 和 `85beeaad...72a9`。
- 该 exact wheel 从源码树外 `/tmp/flexaligner-wheel-site-after.ZXqAVU` 导入并完成
  D-033 approved 英语 CLI E2E；TextGrid SHA-256 为 `d15265c2...2d32a`。它与 reference
  的所有非 `NULL` interval/边界一致，并按 D-036 额外实现双 tier 全覆盖。
- Fast tests 在本地 Python 3.10.8 和部分匹配的 Python 3.12.12 环境中通过
  （676 passed，1 个真实模型 marker deselected）。完整 Python/操作系统矩阵尚未
  运行，记录为 `TBD-CI-001`。
- 用户只批准冻结的 `openphonetics` 发音作为 release-E2E fixture。仓库内 manifest
  为 `approved`，记录 D-033 和 `release-e2e-fixture-only`，通过 16/16 精确运行时
  预检。重建 exact wheel 通过禁 socket E2E（`1 passed, 676 deselected`），因此
  本地 Q-007 为 PASS；protected remote E2E 仍为 NOT_RUN。
- 首个公开版本选择为 `PUBLIC_ALPHA` / `0.1.0a1`。`pyproject.toml` 仍为
  `0.1.0.dev0`；版本/classifier 和 exact artifact 复验仍属于 Stage 8 工作。
- distribution 选择为 `flexaligner`，PyPI owner/organization 为
  `ustcphonetics`。实际可用性、控制权和 publisher 绑定尚未外部验证。
- GitHub 策略选择为对现有 `USTCPhonetics/FlexAligner` 使用
  `REPLACE_MAIN_HISTORY`。没有接受 merge/graft，但也没有授权或配置远端替代操作。
- MIT 许可、版权、作者、单位和引用身份已按固定原远端 README 和同提交 LICENSE
  快照 `c5361efe4b5d8ad02574dae1bd7caa89ed3e4af0` 批准。
- v0.1 public preview 支持边界由 D-034 接受；披露的限制仍是限制，不是静默修正。
- D-036--D-040 已解决 TBD-ALG-001--005，并明确取代 D-034 中冲突的旧行为保留口径。
  五项均已实施：双 tier `NULL` 全覆盖、Aligner 名义 160-sample stride 验证、稳定
  phone occurrence provenance、标准 repeated-target blank 约束，以及
  900 s/10k/200M/10k/200M 保险丝。禁网非模型测试为 717 passed、1 deselected，分支
  覆盖率 92.42%；Ruff、strict mypy 和 diff check 通过。
- 用户指定的 `english_natural.wav`（5.015 s、12 words）在 D-036--D-040 合并后真实
  E2E 通过；新 TextGrid SHA-256 为 `02ac0a42...e205d`。两层均以 `NULL` 精确覆盖，
  所有旧非 `NULL` interval 及边界保持不变；旧字节一致结果仅保留为 before 证据。
- `example1.wav`（49.0413125 s、10 words）合并后真实 E2E 通过；新 TextGrid SHA-256
  为 `54d77a81...c1e3a`，3 chunks、词序守恒、无 overlap、双 tier 全覆盖，并保持所有
  旧非 `NULL` interval 及边界。
- approved release fixture 已重生 D-036 golden，TextGrid SHA-256 为
  `d15265c2...2d32a`；真实 model E2E `1 passed`。与冻结 reference 不再要求整体字节
  相同，而是要求所有非 `NULL` interval/边界相同，并额外要求新输出严格全覆盖。
- D-035 固定文档语言：根目录对外 `README.md` 保持中英双语；内部项目文档使用中文。
  中文化前英文文档由提交 `31becaf` 和
  `docs/archive/2026-08-22-english-docs.md` 保存。
- 未修改任何外部服务、GitHub 仓库或 PyPI 项目。

## 当前工作

- 选择新的经审阅长音频 fixture，重测接近 900 s 时的实际 cells/work、耗时和峰值 RSS；
  D-036--D-040 的功能实施已经完成。
- Stage 8 已应用并在本地复验 D-033 release fixture；仍需实施 `0.1.0a1` 元数据和
  D-034 要求的窄推理契约。
- approved fixture 仍存在仓库本地路径可移植性问题，记录为 `TBD-E2E-002`；
  尚未声称 protected remote E2E 通过。
- 远端历史替代、CI 执行、runner/environment 配置、tag 创建和发布仍未配置，
  也未取得外部操作授权。

## 当前能力状态

| 能力 | 状态 |
|---|---|
| Python API / CLI / capability discovery | 可用，已通过 Stage 5 测试 |
| 英语 CPU 单文件对齐 | 可用；D-033 approved-fixture exact-wheel E2E 在本地通过；protected remote E2E 仍为 NOT_RUN |
| 普通话 | 占位；类型化失败已验证 |
| GPU | 占位；类型化失败已验证 |
| 批处理 | 占位；类型化失败和不消费输入 iterable 已验证 |
| Web | 占位；类型化失败已验证 |
| 自动模型下载 | 占位；联网前失败已验证 |
| 多格式音频 | 占位；类型化失败已验证 |
| 自动重采样 | 占位；类型化失败已验证 |
| 中文分词 | 占位；类型化失败已验证 |
| 默认 G2P | 占位；类型化失败已验证 |
| 置信度校准 | 占位；类型化失败已验证 |

## 下一道门禁

下一道门禁是 public alpha 发布准备：

1. 实施 `0.1.0a1` 元数据和其余已接受预览契约，然后重建并复验 exact alpha artifact；
2. 去除 `TBD-E2E-002` 记录的 repo-local fixture 路径依赖；
3. 直接替代远端 `main` 前，取得单独外部授权和已验证恢复快照
   （`TBD-REMOTE-001`）；
4. 运行完整远端 Python/操作系统矩阵、依赖审计和离线 E2E；
5. 配置并验证 Trusted Publishing、受保护 environment 和离线 E2E runner；
6. 在任何远端 push、默认分支修改、tag 创建或 PyPI 发布前分别请求授权。

## 已知阻断项

Stage 7 工程基线保持接受，但公共发布仍为 `NO-GO`。阻断项包括：尚未完成的 Stage 8
alpha 元数据/推理工作、`TBD-CI-001`、`TBD-REMOTE-001`、`TBD-E2E-002`、
`TBD-REL-001`、没有配置 remote/tag，以及没有外部操作授权。行为修正、输出、
provenance、兼容性和资源问题继续作为已披露的预览限制，不得静默标记为已修复。
