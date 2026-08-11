# STATE

> Last updated: 2026-08-11 (Asia/Shanghai)

## Current stage

`Stage 0 — repository, governance and executable plan` is in progress.

## Verified current state

- `/Users/yiyi0369/projects/flexaligner-rebuild` is a newly initialized local
  Git repository on branch `main`.
- No Git remote is configured.
- `IMPLEMENTATION_PLAN.md` has been created before production implementation.
- The authoritative algorithm reference remains external at
  `/Users/yiyi0369/projects/flexaligner/align_single_cpu.py` with expected
  SHA-256 `9ed4e21e615718ddfd10930359f55769fb27a0d284599cce45a3fc755e835de1`.
- No production package, tests, README snapshot, license snapshot, build artifact
  or release exists yet.
- No external service, GitHub repository or PyPI project has been changed.

## Active work

- Main agent: governance baseline, acceptance matrix and Stage 0 audit.
- Parallel stream A: packaging and strict CI/CD design review.
- Parallel stream B: architecture and placeholder-interface design review.
- Parallel stream C: characterization, differential and E2E test design review.

## Current capability state

| Capability | State |
|---|---|
| English CPU single-file alignment | planned; not implemented |
| Mandarin | planned placeholder; not implemented |
| GPU | planned placeholder; not implemented |
| Batch | planned placeholder; not implemented |
| Web | planned placeholder; not implemented |
| Automatic model download | planned placeholder; not implemented |
| Multi-format audio | planned placeholder; not implemented |
| Automatic resampling | planned placeholder; not implemented |
| Chinese segmentation | planned placeholder; not implemented |
| Default G2P | planned placeholder; not implemented |
| Confidence calibration | planned placeholder; not implemented |

## Next gate

Stage 0 passes only after:

1. all governance files exist and agree on scope;
2. the current reference hash is independently rechecked;
3. the acceptance matrix has stable IDs and truthful initial states;
4. `git diff --check` and a cross-document audit pass;
5. the governance-only baseline is committed locally.

After that, Stage 1 may start.

## Known blockers

None for Stage 0. PyPI ownership, final remote history strategy, the frozen E2E
effective lexicon and behavior-changing algorithm decisions remain open but do
not block the package/interface skeleton.

