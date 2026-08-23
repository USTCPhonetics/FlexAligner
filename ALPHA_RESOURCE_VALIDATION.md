# v0.1 Alpha 资源门槛验证报告

> 日期：2026-08-23（Asia/Shanghai）
>
> 决定：D-039、D-040
>
> 结论：D-036--D-040 代码级门禁和三个真实模型 E2E 通过；900 s 长输入能力仍未验收

## 1. 已批准门槛

- `max_audio_seconds = 900.0`
- `max_phone_tokens = 10_000`
- `max_trellis_cells = 200_000_000`
- `max_stage2_graph_states = 10_000`
- `max_beam_work_units = 200_000_000`
- `beam = 400`

`max_beam_work_units` 定义为每个请求实际访问的 beam candidate 数，累计全部 chunk、
第一遍和固定序列第二遍解码。start、stay、每个 successor 访问和 terminal 检查都计入。
它是异常有限搜索的关闭式保险丝，不是模型推理超时、延迟或成功 SLA。

## 2. 代码级验证

- D-039：相邻重复目标必须经过 blank；覆盖同词、跨词和去除 ARPAbet 重音后重复；
- 200M trellis：保留精确 Python integer 预分配核算，超限在 `numpy.full` 前抛出
  `ResourceLimitError`；
- 10k Stage 2 graph states：在 emitting graph 物化前逐状态关闭式检查；
- 200M beam work：精确边界、小一单位超限、跨解码累计、静音锁 successor 访问、
  两遍共享、pipeline 类型化失败且无正式输出均有测试；
- 禁网 fast tests：`717 passed, 1 deselected`，branch coverage `92.42%`；
- Ruff check/format：通过；strict mypy：通过；`git diff --check`：通过。

## 3. `english_natural` 真实 E2E

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
| 本轮 wall time | 11.38 s | 不适用 |
| 本轮峰值 RSS | 3,076,325,376 bytes（约 2.87 GiB） | 不适用 |
| Stage 1 frames / targets | 250 / 45 | 不适用 |
| Trellis cells | 11,546 | 约 0.0058% of 200M |
| 累计 beam work | 281,411 | 约 0.1407% of 200M |
| Chunk 数 | 1 | 不适用 |
| 输出 TextGrid SHA-256 | `02ac0a42924d342806c3a6f6845710014da9952fd3773ce7ee322cdaca9e205d` | 不适用 |

新输出按 D-036 有意不再与用户提供的旧 TextGrid 逐字节相同。两层分别有 51 和 18 个
interval，各含 2 个 `NULL`；词序守恒、无 overlap，并严格覆盖整个时轴。与 before
输出逐 interval 对比确认：所有非 `NULL` 标签和边界完全不变。

本轮运行产物位于 `/tmp/flexaligner-english-natural-after.dJeypx`。11,546 cells 和
281,411 work 来自同一输入、同一模型的既有插桩运行；本轮用于复核 D-036/D-037 的
真实执行未改变 Stage 1/Stage 2 搜索输入。

## 4. `example1` 真实 E2E

用户指定 `/Users/yiyi0369/projects/xiantutorial/modules/03_flexaligner/data/test/` 中的
`example1.wav` / `example1.txt` 继续验证：

| 项目 | 已验证值 |
|---|---|
| WAV SHA-256 | `22145f74ade6ec1c60241b583b86b4287938ce288d1cba0de4dc3d29f4f8cb27` |
| TXT SHA-256 | `761cae0292326b4b3962608341177b3720bdb515c53a2a1695cb56c4220bde83` |
| 格式/规模 | 16 kHz、单声道、PCM16；49.0413125 s；10 words、33 phones |
| 词典/重复目标 | `word.dict` 全覆盖；相邻重复目标 0 |
| 本轮 wall time / 峰值 RSS | 25.18 s / 3,032,956,928 bytes（约 2.82 GiB） |
| Stage 1 | 2,451 frames、83,368 cells（约 0.0417% of 200M） |
| Stage 2 | 3 chunks、累计 219,135 work（约 0.1096% of 200M） |
| 输出 TextGrid SHA-256 | `54d77a81ddee4b8dc4a5dd8ca95b4f3140da887145003199553fc2998ffc1e3a` |

归一化 lexical word 顺序与输入完全一致，无 overlap。输出两层各有 4 个 `NULL`，
补齐开头以及原有的三个未覆盖区间并严格全覆盖；所有 before 非 `NULL` interval 和
边界逐项保持不变。

本轮运行产物位于 `/tmp/flexaligner-example1-after.wcd0k2`。83,368 cells 和 219,135
work 来自同一输入、同一模型的既有插桩运行。

## 5. 验收结论与下一步

| 项目 | 状态 | 结论 |
|---|---|---|
| 900 s 输入 header 门槛 | `PASS`（代码级） | 默认值和 header 前置拒绝逻辑通过测试；尚无接近上限的 E2E |
| 200M trellis cell 保险丝 | `PASS`（代码级） | 精确核算和分配前关闭式失败测试通过 |
| 200M transition 保险丝 | `PASS`（代码级） | 精确累计和关闭式失败测试通过 |
| 10k phone/graph-state 保险丝 | `PASS`（代码级） | 精确边界和物化前关闭式失败测试通过 |
| `english_natural` 完整 E2E | `PASS` | 11,546 cells、281,411 work；双 tier 全覆盖且非 `NULL` 边界不变 |
| `example1` 完整 E2E | `PASS` | 83,368 cells、219,135 work；3 chunks、词序守恒、双 tier 全覆盖 |
| approved release E2E | `PASS` | 修正后 golden `d15265c2...2d32a`；1 passed；非 `NULL` 区间匹配冻结 reference |
| D-036 连续覆盖 | `PASS` | 合成边界测试和三个真实模型输出均通过 |

在公开 alpha 前仍需用经过审阅的长音频 fixture 记录实际 trellis cells、累计 transition
evaluations、分阶段耗时、峰值 RSS、chunk 数和完整 TextGrid 验收；不得只用静态估算
声称 900 s 性能已经通过。
