# Stage 1 稠密 trellis 资源报告

> 状态：Stage 3 无模型审计完成
>
> 日期：2026-08-11（Asia/Shanghai）
>
> 范围：仅 Stage 1 CTC 稠密 trellis；不包含模型推理和 emission 存储

## 复杂度契约

设 `T` 为 emission 帧数，`N` 为目标 phone token 数。稠密动态规划表形状为
`(T + 1, N + 1)`：

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

## 上限策略与已验证不变量

`TBD-ALG-005` 尚未解决：本阶段**没有**建立适用于整个包的安全默认时长、phone 数、
trellis cell、峰值 RSS 或 beam work 上限。当前唯一接受的 Stage 1 保护是调用方提供
正数 `max_trellis_cells`。

实施门禁要求在分配前失败：

1. 验证维度、ID、有限 score 和请求的上限；
2. 使用 Python 整数计算精确 cell 数；
3. 如果 `cells > max_trellis_cells`，抛出类型化资源上限错误；
4. 只有通过后才分配稠密 trellis。

不变量测试会 monkeypatch trellis 分配器，证明超过显式上限时会在调用
`numpy.full` 前抛错。`None` 表示调用方没有提供上限，不得把它描述成安全默认值。

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

本报告支持可复现的复杂度核算和显式调用方保护，但没有解决 `TBD-ALG-005`，没有
建立生产安全默认值，也不对长录音作可靠性声明。确定默认值需要在已接受的硬件和
工作负载范围上测量峰值 RSS 与耗时，然后记录明确决定。
