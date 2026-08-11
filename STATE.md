# STATE

> Last updated: 2026-08-11 (Asia/Shanghai)

## Current stage

`Stage 0 — repository, governance and executable plan` is complete.

`Stage 1 — package and interface skeleton` is complete at commit `5702f0a`.

`Stage 2 — reference characterization and test harness` is in progress.

## Verified current state

- `/Users/yiyi0369/projects/flexaligner-rebuild` is a newly initialized local
  Git repository on branch `main`.
- No Git remote is configured.
- `IMPLEMENTATION_PLAN.md` has been created before production implementation.
- The governance-only root commit is `833306e`.
- The authoritative algorithm reference remains external at
  `/Users/yiyi0369/projects/flexaligner/align_single_cpu.py` with expected
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
- No external service, GitHub repository or PyPI project has been changed.

## Active work

- Main agent: Stage 2 reference provenance, characterization and integration.
- Parallel streams: pure Stage 1 arrays, Stage 2 graph/Viterbi arrays, and
  reference/output/fixture guards in disjoint files.

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

Stage 2 requires:

1. a portable, hash-guarded reference snapshot/provenance manifest that is not
   included in the wheel;
2. model-free characterization of accepted Stage 1 and Stage 2 behavior;
3. independent small-case invariants and field-level differential helpers;
4. TextGrid/current-gap behavior and atomic-output tests;
5. a frozen candidate E2E asset manifest whose missing prerequisites fail
   explicitly rather than skip.

## Known blockers

None for model-free Stage 2 work. PyPI ownership, remote history, real-model E2E
deployment and behavior-changing algorithm decisions remain open but do not
block characterization.
