# Stage 1 稠密 trellis 资源报告

> 状态：D-039/D-040 已实施并通过代码级与短自然语音 E2E；长音频性能待解决
>
> 日期：2026-08-11（Asia/Shanghai）
>
> 范围：仅 Stage 1 CTC 稠密 trellis；不包含模型推理和 emission 存储

## 复杂度契约

以下公式描述 D-039 之前、当前已验证的简化 recurrence。设 `T` 为 emission 帧数，
`N` 为目标 phone token 数。稠密动态规划表形状为 `(T + 1, N + 1)`：

```text
cells = (T + 1) * (N + 1)
bytes = cells * dtype.itemsize
```

构表过程会对每个可达 frame/token 对执行一次 stay/emit 更新，因此时间复杂度为
`O(TN)`。保留完整表用于回溯的空间复杂度也是 `O(TN)`。回溯本身最多使用
`O(T + N)` 时间，并保存 `O(N)` 个 emission 点。向量化实现还可能产生临时的
`O(N)` 行；这些临时对象和输入 emission 矩阵 `(T, V)` 不计入下方 trellis 字节数。

## 精确 trellis 大小

括号内显示使用二进制单位，整数 byte 数才是权威值。

| 帧数 `T` | 目标数 `N` | Cell 数 `(T+1)(N+1)` | float32 字节 | float64 字节 |
|---:|---:|---:|---:|---:|
| 100 | 20 | 2,121 | 8,484 (0.008 MiB) | 16,968 (0.016 MiB) |
| 1,000 | 100 | 101,101 | 404,404 (0.386 MiB) | 808,808 (0.771 MiB) |
| 10,000 | 1,000 | 10,011,001 | 40,044,004 (38.189 MiB) | 80,088,008 (76.378 MiB) |
| 100,000 | 5,000 | 500,105,001 | 2,000,420,004 (1.863 GiB) | 4,000,840,008 (3.726 GiB) |

这些估算只描述分配大小，不表示较大的示例可以安全运行。进程峰值内存还包括模型、
emission、Python 对象、NumPy 临时对象和分配器开销。

## D-039 重复目标修正的核算要求

D-039 已批准标准重复目标 CTC blank 约束：相邻相同目标必须由至少一个 blank 帧分隔，
且同词、跨词及去除 ARPAbet 重音后形成的重复一视同仁。该决定有意偏离上述冻结
reference 行为，现已实施并通过具名 before/after 与独立穷举测试。

当前 repeat-aware recurrence 保持 `(T + 1, N + 1)` trellis：进入重复目标时从
`t-2` 状态一次计入 separator blank 和当前 emit，backtrace 同步跳过该 blank 帧。
因此 cell 公式仍为 `(T + 1)(N + 1)`，但最少所需帧数变为 `N + R`，其中 `R` 是相邻
重复目标数。D-040 上限在分配前按该实际矩阵精确核算。

## 上限策略与已验证不变量

D-040 已批准并实施 v0.1 alpha 初始默认保险丝：音频最长 900 s、单次 Stage 1
trellis 最多 200,000,000 cells。精确边界和分配前失败测试通过；`english_natural`
和 `example1` 分别实测 11,546 与 83,368 cells。words 和 phone targets 的包级默认值
未由 D-040 固定，仍可使用调用方显式限制。尚无接近 900 s/200M 的真实 E2E。

修正后的实施门禁要求在昂贵工作/分配前失败：

1. 从已验证 WAV header 拒绝 `duration_s > 900`；
2. 验证维度、ID、有限 score 和请求的上限；
3. 使用 Python 整数计算 D-039 修正后实际 trellis cell 数；
4. 如果 `cells > 200_000_000`，抛出 `ResourceLimitError`；精确等于上限允许继续；
5. 只有全部检查通过后才分配稠密 trellis。

若调用方提供比默认值更窄的正限制，应采用更窄限制；不得通过传入 `None` 无意关闭
包级默认保险丝。错误必须带稳定的 `resource_limit_exceeded` code 和实际值/上限 context。

下列既有不变量证据只验证 D-040 之前的显式调用方上限。新门禁必须继续使用分配器
monkeypatch，分别证明每个默认边界的等值通过、超值失败，以及超限时没有调用
`numpy.full` 或进入后续模型/对齐工作。

## 本地验证

最终无模型审计运行于 CPython 3.10.8：

- 使用固定种子 `20260811`、`T=1..6`、`N=1..min(T, 3)`，同时检查 float32 和
  float64，并与独立穷举 stay/emit 结果比较；
- 等于精确上限时允许分配，`cells > max_trellis_cells` 时在 monkeypatch 的
  `numpy.full` 运行前抛出 `resource_limit_exceeded`；
- 无效维度、dtype、ID、shape、NaN/无穷、非有限 chunk 边界以及不完整/乱序的
  word 覆盖均为硬失败；
- 专用不变量文件通过 `76` 项测试，没有 skip 或 xfail；
- 完整测试通过 `348` 项，分支覆盖率 `94.75%`，高于配置的 `85%` 门槛。

命令：

```text
python -m pytest -q tests/core/test_stage1_invariants.py
python -m pytest -q
python -m pytest -q --cov=flexaligner --cov-report=term-missing
ruff check tests/core/test_stage1_invariants.py
ruff format --check tests/core/test_stage1_invariants.py STAGE1_RESOURCE_REPORT.md
```

## 验收边界

2026-08-11 数值表仍是修正前历史证据；上方公式已按当前 D-039 实现复核。D-039/D-040
代码级门禁通过，但不证明 900 s 输入一定成功、满足 SLA 或适合任意不可信输入。
当前真实 E2E 只覆盖 5.015 s 和 49.0413125 s；900 s 性能仍待新的经审阅长音频 fixture
验证，详见 `ALPHA_RESOURCE_VALIDATION.md`。
