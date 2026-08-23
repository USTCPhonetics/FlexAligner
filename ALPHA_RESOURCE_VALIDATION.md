# v0.1 Alpha 资源门槛验证报告

> 日期：2026-08-23（Asia/Shanghai）
>
> 决定：D-039、D-040
>
> 结论：代码级保险丝和 `english_natural` 真实 E2E 通过；900 s 长输入能力仍未验收

## 1. 已批准门槛

- `max_audio_seconds = 900.0`
- `max_trellis_cells = 200_000_000`
- `max_beam_work_units = 200_000_000`
- `beam = 400`

`max_beam_work_units` 定义为每个请求实际访问的 beam candidate 数，累计全部 chunk、
第一遍和固定序列第二遍解码。start、stay、每个 successor 访问和 terminal 检查都计入。
它是异常有限搜索的关闭式保险丝，不是模型推理超时、延迟或成功 SLA。

## 2. 指定验证输入

| 项目 | 已验证值 |
|---|---|
| 音频 | `/Users/yiyi0369/Desktop/USTC_ICRLS_DISK/Buckeye_Evaluation/input_wav_text/s0101a.wav` |
| 音频 SHA-256 | `cb8be0fe755aeef4949357036915411c44a75bd2dced98e0688334a5da612493` |
| 音频格式 | 16 kHz、单声道、PCM16、9,969,854 frames |
| 音频时长 | 623.115875 s；位于 900 s 门槛内 |
| 文本 | `/Users/yiyi0369/Desktop/USTC_ICRLS_DISK/Buckeye_Evaluation/input_wav_text/s0101a.txt` |
| 文本 SHA-256 | `8d9c0088b074924f7e3b6efff96f255f6a7435d1887b6d492fb2d4c4b0a8ad7b` |
| 项目归一化词数 | 998 words；370 unique words |
| 词典 | `/Users/yiyi0369/projects/openphonetics/word.dict`；指定文本全覆盖 |
| 模型 | `/Users/yiyi0369/projects/openphonetics/models/en/{chunker,aligner}` |
| 运行时 | Python 3.12.12、NumPy 2.2.6、Torch 2.3.1、Transformers 4.41.2 |

本报告只绑定上表精确路径和 hash。其他同名 `s0101a.txt` 文件的词数、hash 和展开结果
不得混入本次事实。

## 3. Stage 1 静态精确核算

使用项目归一化、`word.dict` 第一条发音和 Chunker 的卷积配置：

- 目标 phone 数：3,372；
- 相邻重复目标：32 处，D-039 会要求 blank 分隔；
- Chunker 配置 `conv_stride=[5,2,2,2,2,2,2]`，对 9,969,854 samples 的预期输出为
  31,155 frames；
- trellis 为 `(31,155 + 1) * (3,372 + 1) = 105,089,188 cells`；
- float32 trellis 为 420,356,752 bytes（约 400.88 MiB），低于 200,000,000 cells。

这证明指定输入按模型配置计算不会触发 Stage 1 cell 门槛，但不等于模型已实际输出
该 posterior，也不等于完整对齐通过。

## 4. 真实运行结果

运行采用源码树当前实现、严格离线环境、CPU、`num_threads=1`，输出写入独立临时目录
`/tmp/flexaligner-s0101a-benchmark.e3bkoR`。观测结果：

| 指标 | 结果 |
|---|---:|
| Wall time | 1,370.879 s（约 22.85 min） |
| 峰值 RSS | 5,884,919,808 bytes（约 5.48 GiB，macOS `ru_maxrss`） |
| 停止位置 | Chunker `Wav2Vec2ForCTC` encoder 前向 |
| 实际 trellis 分配 | 0；尚未进入 `build_trellis` |
| 实际 beam work | 0；尚未进入 Stage 2 |
| 正式 TextGrid/metadata | 未生成 |

达到预先声明的观测窗口后，由主 agent 发送 `KeyboardInterrupt` 安全停止临时 benchmark。
输入文件未修改，未发布任何正式产物。临时 `metrics.json` 保存了原始 wall/RSS/空计数；
其中 `status=unknown` 是因为 `KeyboardInterrupt` 继承 `BaseException`、未进入脚本的
`except Exception`，本报告将其规范记录为“人工中止、性能未通过”。

模型加载还产生 Hugging Face parametrization 权重名称转换警告。该运行时的库版本号
与冻结组合除 Python 3.12.12 外一致，但本次不是已批准的 Python 3.10.8 exact-wheel
release E2E，不能替代 Q-007 历史证据。

## 5. 代码级验证

- D-039：相邻重复目标必须经过 blank；覆盖同词、跨词和去除 ARPAbet 重音后重复；
- 200M trellis：保留精确 Python integer 预分配核算，超限在 `numpy.full` 前抛出
  `ResourceLimitError`；
- 200M beam work：精确边界、小一单位超限、跨解码累计、静音锁 successor 访问、
  两遍共享、pipeline 类型化失败且无正式输出均有测试；
- fast tests：`696 passed, 1 deselected`；
- Ruff check/format：通过；strict mypy：通过；`git diff --check`：通过。

## 6. `english_natural` 真实 E2E

用户随后指定下列短自然语音作为本轮真实门槛测试：

| 项目 | 已验证值 |
|---|---|
| WAV | `/Users/yiyi0369/Documents/openphonetics/examples/english_natural.wav` |
| WAV SHA-256 | `8726d02772cf09965511311be2f8f623c117de31065db072ba17871de72a0914` |
| 格式/时长 | 16 kHz、单声道、PCM16；5.015 s |
| TXT | `/Users/yiyi0369/Documents/openphonetics/examples/english_natural.txt` |
| TXT SHA-256 | `211a0126541989925ffe6a1692c95346859cecd95dc1cc9c5fb9569e81334375` |
| 规模 | 12 words、45 phones、2 处相邻重复目标；词典全覆盖 |
| 给定 TextGrid SHA-256 | `c8d8ef7b0e9acd03d19e5fd80855f5cceda85df1aed5e66ecc325469b85a121f` |

同一离线模型/运行时、CPU、`num_threads=1` 的真实运行结果：

| 指标 | 实测值 | 占门槛比例 |
|---|---:|---:|
| Wall time | 42.215 s | 不适用 |
| 峰值 RSS | 2,249,392,128 bytes（约 2.09 GiB） | 不适用 |
| Stage 1 frames / targets | 250 / 45 | 不适用 |
| Trellis cells | 11,546 | 约 0.0058% of 200M |
| 累计 beam work | 281,411 | 约 0.1407% of 200M |
| Chunk 数 | 1 | 不适用 |
| 输出 TextGrid SHA-256 | `c8d8ef7b0e9acd03d19e5fd80855f5cceda85df1aed5e66ecc325469b85a121f` | 不适用 |

新输出与用户提供的 TextGrid 逐字节相同。两层分别有 51 和 18 个 interval，词序守恒，
无 overlap；但 phones/words 都在 `[4.751, 4.773]` 保留 22 ms 内部 gap。这是 D-036
尚未实施的明确 before 证据，不能因字节等价而把“全时轴连续覆盖”标成通过。

运行产物位于 `/tmp/flexaligner-english-natural.EzUMfD`；`metrics.json` SHA-256 为
`8a60259712f4b65519e662804693a0b141f2229ee41e0f506160a2013c79e008`。

## 7. 验收结论与下一步

| 项目 | 状态 | 结论 |
|---|---|---|
| 900 s 输入 header 门槛 | `PASS` | 指定 623.115875 s 输入在门槛内 |
| 200M trellis cell 保险丝 | `PASS`（代码级） | 指定输入静态精确值 105,089,188；真实分配尚未发生 |
| 200M transition 保险丝 | `PASS`（代码级） | 精确累计和关闭式失败测试通过 |
| `english_natural` 完整 E2E | `PASS` | 11,546 cells、281,411 work；输出与给定 TextGrid 字节一致 |
| D-036 连续覆盖 | `FAIL/待实施` | 两层均存在同一 22 ms 内部 gap |
| `s0101a` 长音频完整 E2E | `FAIL` | 22.85 min 内未完成 Chunker，不能取得实际 cells/work/输出 |

在公开 alpha 前至少需要选择并验证一种方案：对 Chunker 做有重叠的流式/分块前向，
或建立明确的可取消任务与 wall-time 门槛。完成优化后必须用同一 hash 输入重新运行，记录
实际 trellis cells、累计 transition evaluations、分阶段耗时、峰值 RSS、chunk 数和
完整 TextGrid 验收；不得只用静态估算关闭本项。
