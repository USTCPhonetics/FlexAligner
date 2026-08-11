# STATE

> Last updated: 2026-08-11 (Asia/Shanghai)

## Current stage

`Stage 0 — repository, governance and executable plan` is complete.

`Stage 1 — package and interface skeleton` is complete at commit `5702f0a`.

`Stage 2 — reference characterization and test harness` is complete at commit
`e582dd4`.

`Stage 3 — Stage 1 implementation` is complete at commit `ec7bd2d`.

`Stage 4 — Stage 2 implementation` is complete at commit `d65ab6a`.

`Stage 5 — inference, pipeline, CLI and output` is complete at commit `6c1d4eb`.

`Stage 6 — package, real-model E2E and release rehearsal` is complete at commit
`e694645`, with the approved-fixture release gate truthfully blocked.

`Stage 7 — main-agent final audit` is complete. The engineering baseline is
accepted; public release is `NO-GO` until the recorded blockers are resolved.

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
  the frozen local OpenPhonetics environment. At the Stage 2 gate this was only
  asset preflight evidence; the Stage 6 real alignment evidence is recorded
  below.
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
- The strict English CPU single-file pipeline is implemented with strict
  transcript/lexicon/WAV/model/vocabulary checks, lazy local-only inference and
  sequential non-overlapping Chunker/Aligner sessions.
- TextGrid and optional metadata are staged and read-back validated. Publication
  uses an atomic no-clobber hard link, postcommit inode/byte/semantic validation
  and ownership-aware rollback; cross-file crash consistency remains
  `TBD-OUT-001`.
- Stage 5 main-agent verification passes 673 socket-denied tests at 92.31%
  branch coverage; Ruff checks/formats 71 files and strict mypy checks 20 source
  files without errors.
- Transcript words `sil` and `null` fail before model loading because current
  tier labels reserve those identities; a future identity scheme is
  `TBD-TEXT-001`.
- The final Stage 6 wheel and sdist pass Twine strict, check-wheel-contents and
  the repository inventory audit. Their SHA-256 values are respectively
  `882337e536bda28814293f803cd88f62ce4d3f137183aeb8c1396799b1199d32`
  and `93b5fb63560c6fa014fbb3a0994c271c22c69757302551ef0a2da79879c51a82`.
- That exact wheel imports from an external `site-packages`, passes `pip check`,
  version/CLI/capability smoke and the frozen candidate English E2E with sockets
  disabled. The new and reference TextGrid bytes are identical at SHA-256
  `ddbe0fecbbd7fc32442bd7b81ccb6257e391ab81970d398eb236de46a50e415f`.
- Fast tests pass on local Python 3.10.8 and partially matched Python 3.12.12
  environments (676 passed, one real-model marker deselected). The full declared
  Python/OS matrix has not run and is `TBD-CI-001`.
- The E2E manifest remains `candidate`. The release workflow now requires
  `approved` and therefore fails closed on `TBD-E2E-001`; engineering success is
  not reported as release approval.
- No external service, GitHub repository or PyPI project has been changed.

## Active work

- No implementation stage is active.
- Awaiting user/external decisions for remote history, package ownership/version,
  fixture approval and release infrastructure.

## Current capability state

| Capability | State |
|---|---|
| Python API / CLI / capability discovery | available and Stage 5 tested |
| English CPU single-file alignment | available; candidate E2E passed, approved-fixture release gate blocked |
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

The next gate is public release readiness:

1. approve or replace the candidate fixture pronunciation;
2. choose the GitHub history/remote, PyPI owner/name and stable version;
3. run the complete remote Python/OS matrix and dependency audit;
4. configure Trusted Publishing, protected environment and offline E2E runner;
5. request separate authorization before any tag/publish mutation.

## Known blockers

No engineering implementation blocker remains. Public release is blocked by
`TBD-CI-001`, `TBD-E2E-001`, `TBD-PKG-001..003`, `TBD-LIC-001`,
`TBD-REL-001` and the absence of a configured remote. Behavior-changing
algorithm, output, provenance, compatibility and resource questions remain
documented limitations rather than silently accepted fixes.
