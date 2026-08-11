# STATE

> Last updated: 2026-08-11 (Asia/Shanghai)

## Current stage

`Stage 0 — repository, governance and executable plan` is complete.

`Stage 1 — package and interface skeleton` is complete at commit `5702f0a`.

`Stage 2 — reference characterization and test harness` is complete at commit
`e582dd4`.

`Stage 3 — Stage 1 implementation` is complete at commit `ec7bd2d`.

`Stage 4 — Stage 2 implementation` is complete at commit `d65ab6a`.

`Stage 5 — inference, pipeline, CLI and output` is in progress.

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
- Stage 1 package-skeleton evidence: 50 tests passed with sockets disabled; branch coverage
  88.16% against the 85% ratchet; Ruff and strict mypy passed.
- The Stage 1 package-skeleton distribution audit passed for wheel
  `f3b6e47f305532329ba1f9c9b7dd6281c165b0bd92ad4cc9b684e33a56db9449`
  and sdist
  `3867f2c6db38b9b5fcd9d8d3a4fae8d9a27a3202b9f2354ab22324561dca5353`.
- That Stage 1 wheel was installed outside the source tree and passed `pip check`,
  version, import-path, CLI and capability smoke tests.
- Stage 2 has 92 model-free characterization/oracle tests: 30 Stage 1, 26
  Stage 2, 19 input/failure, 6 TextGrid/output, 7 reference guards and 4
  differential-harness tests.
- At the Stage 2 gate, the fast suite passed 149 tests with sockets disabled;
  Ruff, strict mypy, actionlint, reference byte comparison and
  `git diff --check` also passed.
- A committed candidate E2E manifest records 16 assets and exact Python/NumPy/
  Torch/Transformers versions. All hashes and exact runtime versions pass in
  the frozen local OpenPhonetics environment; this is asset preflight evidence,
  not yet a real alignment E2E.
- Current rebuilt distributions exclude `reference/` and tests. Fresh build
  audit after Stage 3 passed for wheel
  `8bf982789427ff8dc904278f6a5b563fb8cfd25fb79d11b8111b91f6f067868f`
  and sdist `6d83cb0cc49c3d2ee75b2bd0d7c21f3b1ffe85bdcd6ccaea0efc10c0498be364`.
- The clean NumPy Stage 1 core is implemented with no reference, Torch or
  Transformers import. It passes 123 three-way parity tests and 76 independent
  invariant/resource tests; the full suite passes 348 tests at 94.75% branch
  coverage.
- Dense trellis resource accounting and a caller-supplied pre-allocation cell
  limit are implemented. No package-wide safe default is claimed; that remains
  `TBD-ALG-005` and is documented in `STAGE1_RESOURCE_REPORT.md`.
- The Stage 3 wheel was installed outside the source tree at
  `/tmp/flexaligner-stage3-smoke.TPMkh3` and passed `pip check`, package/core
  import, CLI, capabilities and resource-estimate smoke tests.
- The clean NumPy Stage 2 core is implemented with no reference, Torch or
  Transformers import. It passes 80 reference-parity tests and 81 independent
  invariant/resource tests; the full suite passes 509 tests at 92.85% branch
  coverage, and the Stage 2 module reaches 90.63% branch coverage.
- A separate read-only cross-audit exercised all 64 boundary/internal SIL/SPH
  flag combinations and 100 fixed-seed random small decodes against both the
  frozen reference and an independent exact-DP oracle without finding a
  production defect.
- Stage 2 resource complexity and limitations are recorded in
  `STAGE2_RESOURCE_REPORT.md`. The inherited `beam=400` is not claimed as an
  empirically safe bound; `TBD-ALG-005` remains open.
- No external service, GitHub repository or PyPI project has been changed.

## Active work

- Main agent: Stage 5 architecture, integration order and acceptance audit.
- Next parallel streams: strict input/TextGrid output, lazy local-only model
  inference, and pipeline/API/CLI integration in disjoint files.

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

Stage 5 requires:

1. strict transcript, lexicon, WAV, model-directory and vocabulary validation;
2. a lazy CPU-only Hugging Face adapter using local files only;
3. an instrumented proof that the Chunker is released before the Aligner loads;
4. clean Stage 1/Stage 2 orchestration with complete word-index coverage;
5. merged words/phones tiers and atomically written, read-back-validated
   TextGrid output;
6. Python API and single-file CLI success/failure contracts with explicitly
   uncalibrated metadata.

## Known blockers

None for model-free Stage 5 implementation. The inference extra and frozen local
assets are available for later integration evidence, but fast tests must remain
model-free and network-disabled. PyPI ownership, remote history, real-model E2E
execution and behavior-changing algorithm decisions remain open.
