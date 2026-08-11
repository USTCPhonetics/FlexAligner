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

`Stage 8 — reviewed release decisions and implementation verification` is in
progress. The user has accepted the public-alpha product choices and the local
approved-fixture exact-wheel rerun passes; alpha metadata/inference constraints,
manifest portability and remote release gates are not yet complete.

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
- At the Stage 2 gate, the then-candidate committed E2E manifest recorded 16
  assets and exact Python/NumPy/Torch/Transformers versions. All hashes and exact
  runtime versions passed in the frozen local OpenPhonetics environment. That
  gate supplied only asset preflight evidence; D-033 approval and the later real
  alignment evidence are recorded below.
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
- The Stage 8 refreshed wheel and sdist pass Twine strict,
  check-wheel-contents and the repository inventory audit. Their SHA-256 values
  are respectively
  `a33dcc22f8023e4b4a7905bf7ab78bd827e4576a3fae368f24850b89f0ac9558`
  and `268e34970404c8c2c361a09358fc581db1185e23079eb9377f25dd4dc205569e`.
- That exact wheel imports from the external target directory
  `/tmp/flexaligner-stage8-final-wheel-site.uy5PIZ`, passes version/import/package
  smoke and the D-033 approved English E2E with sockets disabled. The new and
  reference TextGrid bytes are identical at SHA-256
  `ddbe0fecbbd7fc32442bd7b81ccb6257e391ab81970d398eb236de46a50e415f`.
- Fast tests pass on local Python 3.10.8 and partially matched Python 3.12.12
  environments (676 passed, one real-model marker deselected). The full declared
  Python/OS matrix has not run and is `TBD-CI-001`.
- The user approved the frozen `openphonetics` pronunciation only as a
  release-E2E fixture. The repository-local manifest is `approved`, records
  D-033 and `release-e2e-fixture-only`, and passes 16/16 exact-runtime preflight.
  The rebuilt exact wheel passes the socket-denied E2E (`1 passed, 676
  deselected`), so local Q007 is PASS. Protected remote E2E remains NOT_RUN.
- The first public release is selected as the `PUBLIC_ALPHA` version
  `0.1.0a1`. `pyproject.toml` still records `0.1.0.dev0`; version/classifier and
  exact-artifact verification remain Stage 8 implementation work.
- The selected distribution is `flexaligner`, with PyPI owner/organization
  `ustcphonetics`. Availability, control and publisher binding have not been
  externally verified.
- The selected GitHub strategy is `REPLACE_MAIN_HISTORY` for the existing
  `USTCPhonetics/FlexAligner` project. No merge/graft strategy is accepted, but
  no remote replacement operation is authorized or configured.
- MIT license, copyright, authorship, affiliation and citation identity are
  approved from the fixed original remote README and linked LICENSE snapshot at
  `c5361efe4b5d8ad02574dae1bd7caa89ed3e4af0`.
- The v0.1 public-preview support boundary is accepted as D-034. Its disclosed
  limitations remain limitations rather than silently accepted corrections.
- No external service, GitHub repository or PyPI project has been changed.

## Active work

- Stage 8 has applied and locally reverified the D-033 release fixture. It still
  must implement `0.1.0a1` metadata and the D-034 narrow inference contract.
- The approved fixture still has a repository-local path portability issue
  tracked as `TBD-E2E-002`; no protected remote E2E pass is claimed yet.
- Remote history replacement, CI execution, runner/environment configuration,
  tag creation and publishing remain unconfigured and unauthorized external
  operations.

## Current capability state

| Capability | State |
|---|---|
| Python API / CLI / capability discovery | available and Stage 5 tested |
| English CPU single-file alignment | available; D-033 approved-fixture exact-wheel E2E passes locally; protected remote E2E remains NOT_RUN |
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

The next gate is public-alpha release readiness:

1. implement `0.1.0a1` metadata and the rest of the accepted preview contract,
   then rebuild and reverify the exact alpha artifacts;
2. remove the repo-local fixture path dependency tracked by `TBD-E2E-002`;
3. obtain separate external authorization and a verified recovery snapshot
   before directly replacing remote `main` (`TBD-REMOTE-001`);
4. run the complete remote Python/OS matrix, dependency audit and offline E2E;
5. configure and verify Trusted Publishing, the protected environment and the
   offline E2E runner;
6. request separate authorization before any remote push, default-branch change,
   tag creation or PyPI publication.

## Known blockers

The Stage 7 engineering baseline remains accepted, but public release remains
`NO-GO`. It is blocked by unfinished Stage 8 alpha metadata/inference work,
`TBD-CI-001`, `TBD-REMOTE-001`, `TBD-E2E-002`, `TBD-REL-001`, the absence of a
configured remote/tag and the absence of external-operation authorization.
Behavior-changing algorithm, output, provenance, compatibility and resource
questions remain disclosed preview limitations rather than silently accepted
fixes.
