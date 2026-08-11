# Stage 2 beam-Viterbi resource report

> Status: Stage 4 independent invariant/resource audit complete
> Date: 2026-08-11 (Asia/Shanghai)
> Scope: pronunciation graph, CPU beam decoding, backpointers, segmentation,
> pruning, and fixed-sequence re-decode; model inference is excluded

## Complexity contract

Let:

- `T` be the number of posterior frames;
- `B` be the retained beam-key limit;
- `d` be the maximum emitting-state successor count;
- `S` and `E` be the graph's emitting-state and successor-arc counts;
- `V` be the posterior vocabulary size.

Each active `(state, silence_lock)` key considers one stay and at most `d`
moves. The decoder therefore performs at most the following transition-score
evaluations:

```text
T * B * (1 + d)
```

The requested algorithmic bound is `O(T * B * (1 + d))` time. Concrete
top-`B` selection may add ordering/selection overhead; that implementation
constant is not hidden by this report. Retained backpointers are `O(TB)`, the
current active frontier is `O(B)`, and graph storage is `O(S + E)`. The aligned
state/phone path is `O(T)`. The input posterior matrix remains `O(TV)` and is
outside the beam/backpointer counts below.

`PhoneGraph` stores one state record per emitting edge and materializes both
predecessor and successor IDs. Thus the graph-size claim counts `S` state
records plus `E` directed successor relations (and the reciprocal predecessor
representation remains the same linear-order term), rather than the temporary
epsilon-construction nodes used before closure.

## Scale illustrations

These are exact upper-bound operation/key counts from the formulas, not measured
safe workloads:

| Frames `T` | Beam `B` | Max outdegree `d` | Transition evaluations `T*B*(1+d)` | Retained backpointer keys `T*B` |
|---:|---:|---:|---:|---:|
| 1,000 | 100 | 3 | 400,000 | 100,000 |
| 10,000 | 400 | 4 | 20,000,000 | 4,000,000 |
| 100,000 | 400 | 6 | 280,000,000 | 40,000,000 |

No byte estimate is claimed for dictionary-backed beam keys/backpointers:
Python object, tuple, integer, dictionary-capacity, and allocator overhead are
runtime-dependent. A future memory claim must measure peak RSS on accepted
hardware rather than multiplying records by a guessed object size.

## Graph and path invariants

The independent gate checks that:

1. state and successor IDs are in range and predecessor/successor relations are
   internally consistent;
2. every decoded frame stays in the same state or follows one declared arc;
3. paths begin in a start state and finish in a complete end state;
4. `aligned_phone_ids[t]` equals the phone ID of `state_path[t]`;
5. extracted segments are positive, ordered, contiguous, and cover `[0, T)`;
6. adjacent repeated word labels remain distinct through their word indices;
7. a beam at least as wide as all reachable `(state, silence_lock)` keys agrees
   with exhaustive dynamic programming, while destructive narrow pruning fails
   closed rather than returning a nonterminal path.

Silence locks, one-time `sil`/`sph` entry costs, boundary contrast, 65/50 ms
internal pruning thresholds, and fixed-sequence re-decode are tested as separate
properties so one score term cannot mask another.

The two duration conversions intentionally have different characterized
contracts: Viterbi silence locking uses Python `round` (65 ms at a 10 ms hop is
6 frames), whereas post-decode short-gap pruning uses `ceil` (the same 65 ms
threshold drops 6 frames and keeps 7). The tests keep those boundaries
separate.

## Acceptance boundary

The profile value `beam=400` is inherited parity behavior. It is **not** an
empirically established safe or sufficient upper bound for runtime, memory, or
alignment quality. `TBD-ALG-005` remains unresolved, including approved limits
for frames, graph size, reachable keys, transition work, and peak RSS.

This stage provides deterministic small-graph correctness evidence only. A safe
default requires measured timing and peak RSS over accepted hardware and
workload envelopes, plus a recorded decision. The report must not be used to
claim that long recordings are bounded safely by `beam=400`.

## Verification record

All commands ran locally on Python 3.10.8 without model assets or network
access:

- dedicated independent invariants: `81 passed`;
- reference parity plus independent invariants: `161 passed`;
- complete suite: `509 passed`;
- branch coverage: `92.85%` (`85%` ratchet satisfied);
- repository Ruff check: passed;
- this test module contains no `skip` or `xfail` markers.

The exact-DP checks use fixed-seed small DAGs and retain every reachable key;
they do not import or execute the frozen reference implementation. These
results establish deterministic correctness only within the exercised small
state spaces and do not resolve the resource acceptance boundary above.
