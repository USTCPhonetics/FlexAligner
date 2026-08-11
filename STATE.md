# STATE

> Last updated: 2026-08-11 (Asia/Shanghai)

## Current stage

`Stage 0 — repository, governance and executable plan` is complete.

`Stage 1 — package and interface skeleton` is complete at commit `5702f0a`.

`Stage 2 — reference characterization and test harness` is complete at commit
`e582dd4`.

`Stage 3 — Stage 1 implementation` is in progress.

## Verified current state

- `/Users/yiyi0369/projects/flexaligner-rebuild` is a newly initialized local
  Git repository on branch `main`.
- No Git remote is configured.
- `IMPLEMENTATION_PLAN.md` has been created before production implementation.
- The governance-only root commit is `833306e`.
- The authoritative algorithm reference is frozen byte-for-byte under
  `reference/` from `/Users/yiyi0369/projects/flexaligner/align_single_cpu.py`,
  with expected
  SHA-256 `9ed4e21e615718ddfd10930359f55769fb27a0d284599cce45a3fc755e835de1`.
- The `src/` package, public contracts, requested placeholders, README/LICENSE,
  strict local quality gates and guarded workflows are implemented.
- Local Stage 1 evidence: 50 tests passed with sockets disabled; branch coverage
  88.16% against the 85% ratchet; Ruff and strict mypy passed.
- Final Stage 1 distribution audit passed for wheel
  `f3b6e47f305532329ba1f9c9b7dd6281c165b0bd92ad4cc9b684e33a56db9449`
  and sdist
  `3867f2c6db38b9b5fcd9d8d3a4fae8d9a27a3202b9f2354ab22324561dca5353`.
- The final wheel was installed outside the source tree and passed `pip check`,
  version, import-path, CLI and capability smoke tests.
- Stage 2 has 92 model-free characterization/oracle tests: 30 Stage 1, 26
  Stage 2, 19 input/failure, 6 TextGrid/output, 7 reference guards and 4
  differential-harness tests.
- The full fast suite passes 149 tests with sockets disabled. Ruff, strict mypy,
  actionlint, reference byte comparison and `git diff --check` pass.
- A committed candidate E2E manifest records 16 assets and exact Python/NumPy/
  Torch/Transformers versions. All hashes and exact runtime versions pass in
  the frozen local OpenPhonetics environment; this is asset preflight evidence,
  not yet a real alignment E2E.
- Current rebuilt distributions exclude `reference/` and tests. Fresh build
  audit passed for wheel `f3b6e47f305532329ba1f9c9b7dd6281c165b0bd92ad4cc9b684e33a56db9449`
  and sdist `d87c8a081b2edc9b3b8f256850e330f8099b53321a74dc0c5175841762f7d801`.
- No external service, GitHub repository or PyPI project has been changed.

## Active work

- Main agent: Stage 3 production Stage 1 integration and gate audit.
- Parallel streams: clean NumPy Stage 1 core, production differential tests and
  complexity/resource-limit evidence in disjoint files.

## Current capability state

| Capability | State |
|---|---|
| Python API / CLI / capability discovery | available and Stage 1 tested |
| English CPU single-file alignment | placeholder pending algorithm stages |
| Mandarin | placeholder; typed failure verified |
| GPU | placeholder; typed failure verified |
| Batch | placeholder; typed failure and non-consumption verified |
| Web | placeholder; typed failure verified |
| Automatic model download | placeholder; pre-network failure verified |
| Multi-format audio | placeholder; typed failure verified |
| Automatic resampling | placeholder; typed failure verified |
| Chinese segmentation | placeholder; typed failure verified |
| Default G2P | placeholder; typed failure verified |
| Confidence calibration | placeholder; typed failure verified |

## Next gate

Stage 3 requires:

1. a clean NumPy implementation of first-pronunciation selection, stress
   stripping, trellis/backtrace and emission confidence;
2. anchor construction, strict merge boundary and millisecond rounding;
3. exact ordered word-index coverage validation;
4. production/reference/independent-oracle differential tests;
5. measured dense-trellis resource behavior, with `TBD-ALG-005` remaining an
   explicit limitation unless a default limit is accepted.

## Known blockers

None for model-free Stage 3 work. PyPI ownership, remote history, real-model E2E
execution and behavior-changing algorithm decisions remain open but do not
block parity implementation.
