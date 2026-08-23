# Stage 2 Beam-Viterbi 资源报告

> 状态：D-040 已实施并通过代码级与短自然语音 E2E；D-038 待实施，长音频性能待解决
>
> 日期：2026-08-11（Asia/Shanghai）
>
> 范围：发音图、CPU beam 解码、backpointer、分段、剪枝和固定序列二次解码；
> 不包含模型推理

## 复杂度契约

设：

- `T` 为 posterior 帧数；
- `B` 为保留的 beam key 上限；
- `d` 为 emitting state 的最大 successor 数；
- `S` 和 `E` 分别为图中 emitting state 数和 successor arc 数；
- `V` 为 posterior 词表大小。

每个活跃 `(state, silence_lock)` key 会考虑一次 stay 和最多 `d` 次 move。因此解码器
最多执行以下数量的 transition score 计算：

```text
T * B * (1 + d)
```

算法时间上界为 `O(T * B * (1 + d))`。具体 top-`B` 选择还可能增加排序/选择开销，
本报告不隐藏该实现常数。保留 backpointer 为 `O(TB)`，当前活跃 frontier 为 `O(B)`，
图存储为 `O(S + E)`，对齐后的 state/phone path 为 `O(T)`。输入 posterior 矩阵
为 `O(TV)`，不计入下方 beam/backpointer 数量。

`PhoneGraph` 为每条 emitting edge 保存一条 state record，并具体保存 predecessor 和
successor ID。因此图规模包含 `S` 条 state record 和 `E` 条有向 successor 关系；
互为对应的 predecessor 表示仍属于同一线性量级，不包括 epsilon closure 前的临时节点。

## 规模示例

下表是根据公式得到的精确操作/key 上界，不是经过测量的安全工作负载：

| 帧数 `T` | Beam `B` | 最大出度 `d` | Transition 计算 `T*B*(1+d)` | 保留 backpointer key `T*B` |
|---:|---:|---:|---:|---:|
| 1,000 | 100 | 3 | 400,000 | 100,000 |
| 10,000 | 400 | 4 | 20,000,000 | 4,000,000 |
| 100,000 | 400 | 6 | 280,000,000 | 40,000,000 |

本报告不为基于字典的 beam key/backpointer 给出字节估算：Python object、tuple、整数、
字典容量和分配器开销依赖具体运行时。未来内存声明必须在已接受硬件上测量峰值 RSS，
不能用猜测的对象大小相乘。

## D-040 默认保险丝与计数口径

D-040 已批准并实施 v0.1 Stage 2 alpha 初始默认保险丝：每个对齐请求累计最多
200,000,000 次 beam candidate transition evaluations；beam width 保持 400。精确
计数、跨 chunk/两遍共享和关闭式失败的代码级测试通过；`english_natural` 和 `example1`
分别实测 281,411 与 219,135 work。详见 `ALPHA_RESOURCE_VALIDATION.md`。
Stage 2 emitting states 的包级默认上限未由 D-040 固定。

`beam candidate transition evaluation` 定义为：对一个活跃 beam key 实际准备计算的
一次 stay 候选，或沿一条合法 successor arc 实际准备计算的一次 move 候选。计数规则：

1. 第一遍图解码与固定序列第二遍解码都计入；
2. 一个 API/CLI 请求的所有 chunk 共用同一个单调累计计数器，不得逐 chunk 或逐遍重置；
3. 过滤掉且从未准备评分的非法候选不计入；已经准备读取 posterior/entry cost 并比较
   score 的候选必须计入，不得因其随后未进入 top-`B` 而扣除；
4. 精确累计到 200,000,000 允许完成；下一批候选会使总数超限时，必须在评分该批候选
   前抛出 `ResourceLimitError`，并报告当前值、请求增量和上限；
5. 该计数是防失控保险丝，不是 CPU 指令数、墙钟时间、内存上限或延迟 SLA。

调用方显式更窄的正 work 限制可以收紧默认值，但不得用 `None` 无意关闭包级保险丝。

## 图与路径不变量

独立门禁验证：

1. state/successor ID 均在范围内，predecessor/successor 关系内部一致；
2. 每个解码帧只能停留在同一 state，或沿已声明 arc 移动；
3. path 从 start state 开始，并在完整 end state 结束；
4. `aligned_phone_ids[t]` 等于 `state_path[t]` 对应的 phone ID；
5. 提取的 segment 时长为正、顺序正确、连续，并覆盖 `[0, T)`；
6. 相邻重复 word label 通过 word index 保持不同身份；
7. beam 宽度至少覆盖全部可达 `(state, silence_lock)` key 时，结果与穷举动态规划一致；
   破坏性的窄 beam 剪枝必须关闭式失败，不能返回非终态 path。

静音锁定、一次性 `sil`/`sph` 进入代价、边界对比、65/50 ms 内部剪枝阈值以及固定
序列二次解码分别测试，避免一个 score 项掩盖另一个项。

两种时长换算有意保留不同契约：Viterbi 静音锁定使用 Python `round`（10 ms 帧移下
65 ms 为 6 帧），解码后短 gap 剪枝使用 `ceil`（相同阈值删除 6 帧、保留 7 帧）。
测试明确区分这两个边界。

## 验收边界

配置值 `beam=400` 继续作为 D-040 接受的 v0.1 beam width，但它本身仍**不是**经过
实证确认的运行时、内存或对齐质量安全/充分保证。D-040 的 200,000,000 累计
transition work 是关闭式保险丝；尚未建立包级图状态、独立可达 key、backpointer
字节或峰值 RSS 上限。

本阶段既有记录提供修正前的确定性小图正确性证据；D-040 又新增并通过了精确边界、
跨 chunk/两遍共享计数和超限无正式输出测试。D-038 occurrence 身份仍未实施。
当前真实 E2E 尚未接近 900 s 或 200M，不能据此声称 900 s 输入成功或 200M 是经过
长样本验证的性能门槛；详见 `ALPHA_RESOURCE_VALIDATION.md`。

## 验证记录

所有命令均在本地 Python 3.10.8 上运行，不使用模型资产或网络：

- 专用独立不变量：`81 passed`；
- reference 等价加独立不变量：`161 passed`；
- 完整测试：`509 passed`；
- 分支覆盖率：`92.85%`，满足 `85%` 门槛；
- 仓库 Ruff check：通过；
- 本测试模块没有 `skip` 或 `xfail` 标记。

精确 DP 检查使用固定种子小型 DAG 并保留全部可达 key；不会导入或执行冻结的
reference 实现。结果只证明已覆盖小状态空间内的确定性正确性，是 D-038/D-040 之前
的历史证据，不构成后续决定的实施验收。
