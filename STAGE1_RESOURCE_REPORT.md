# Stage 1 dense-trellis resource report

> Status: Stage 3 model-free audit complete
> Date: 2026-08-11 (Asia/Shanghai)
> Scope: the dense Stage 1 CTC trellis only; model inference and emission storage
> are not included

## Complexity contract

Let `T` be the number of emission frames and `N` the number of target phone
tokens. The dense dynamic-programming table has shape `(T + 1, N + 1)`:

```text
cells = (T + 1) * (N + 1)
bytes = cells * dtype.itemsize
```

Building the table performs one stay/emit update per reachable frame/token pair,
so its time complexity is `O(TN)`. Retaining the full table for backtrace has
`O(TN)` space complexity. Backtrace itself is at most `O(T + N)` time and stores
`O(N)` emitted points. A vectorized implementation may also create temporary
`O(N)` rows; those temporaries and the input emission matrix `(T, V)` are outside
the trellis-byte figures below.

## Exact trellis sizes

The byte counts below use binary units only for the parenthetical display; the
integer byte counts are authoritative.

| Frames `T` | Targets `N` | Cells `(T+1)(N+1)` | float32 bytes | float64 bytes |
|---:|---:|---:|---:|---:|
| 100 | 20 | 2,121 | 8,484 (0.008 MiB) | 16,968 (0.016 MiB) |
| 1,000 | 100 | 101,101 | 404,404 (0.386 MiB) | 808,808 (0.771 MiB) |
| 10,000 | 1,000 | 10,011,001 | 40,044,004 (38.189 MiB) | 80,088,008 (76.378 MiB) |
| 100,000 | 5,000 | 500,105,001 | 2,000,420,004 (1.863 GiB) | 4,000,840,008 (3.726 GiB) |

These estimates describe allocation size, not a claim that the larger examples
are safe to run. Peak process memory also includes the model, emissions, Python
objects, NumPy temporaries, and allocator overhead.

## Limit policy and verified invariant

`TBD-ALG-005` remains unresolved: this stage does **not** establish a safe
package-wide default for duration, phone count, trellis cells, peak RSS, or beam
work. The only accepted Stage 1 guard is a caller-supplied positive
`max_trellis_cells` value.

The implementation gate is fail-before-allocation:

1. validate dimensions, IDs, finite scores, and the requested limit;
2. compute the exact Python-integer cell count;
3. if `cells > max_trellis_cells`, raise the typed resource-limit error;
4. only then allocate the dense trellis.

The invariant tests monkeypatch the trellis allocator and prove that an exceeded
explicit limit raises before `numpy.full` is invoked. `None` means that no
caller limit was supplied; it must not be described as a safe default.

## Local verification

The final model-free audit ran on CPython 3.10.8:

- fixed seed `20260811`, `T=1..6`, `N=1..min(T, 3)`, and both float32 and
  float64 were compared against independent exhaustive stay/emit enumeration;
- the exact-limit case allocates, while `cells > max_trellis_cells` raises
  `resource_limit_exceeded` before the monkeypatched `numpy.full` can run;
- invalid dimensions, dtypes, IDs, shapes, NaN/infinity, non-finite chunk
  boundaries, and incomplete/out-of-order word coverage are hard failures;
- the dedicated invariant file passed `76` tests with no skip or xfail;
- the complete suite passed `348` tests, with `94.75%` branch coverage against
  the configured `85%` gate.

Commands:

```text
python -m pytest -q tests/core/test_stage1_invariants.py
python -m pytest -q
python -m pytest -q --cov=flexaligner --cov-report=term-missing
ruff check tests/core/test_stage1_invariants.py
ruff format --check tests/core/test_stage1_invariants.py STAGE1_RESOURCE_REPORT.md
```

## Acceptance boundary

This report supports reproducible complexity accounting and an explicit caller
guard. It does not resolve `TBD-ALG-005`, establish production-safe defaults, or
make a reliability claim for long recordings. Resolving the default requires
measured peak RSS and timing across accepted hardware and workload envelopes,
followed by a recorded decision.
