# ACCEPTANCE MATRIX

> Reviewer: main agent
> Rule: only the main agent changes a row to `PASS` after rerunning its evidence.
> Status vocabulary: `NOT_RUN`, `IN_PROGRESS`, `PASS`, `FAIL`, `BLOCKED`,
> `PLACEHOLDER`, `N/A`.

## A. Stage 0 — repository and governance

| ID | Acceptance condition | Status | Evidence / command | Reviewer note |
|---|---|---|---|---|
| S0-001 | New local Git repository exists on `main` and has no remote | PASS | `git branch --show-current` → `main`; `git remote -v` → empty | Verified 2026-08-11 |
| S0-002 | Implementation plan was recorded before production code | PASS | initial tree contains governance files only; no `src/` or package metadata | Verified 2026-08-11 |
| S0-003 | Governance documents agree on fact authority, scope and placeholders | PASS | main-agent cross-document audit of six Markdown files | Accepted/user facts separated from implementation choices |
| S0-004 | Reference file exists and SHA-256 matches the frozen value | PASS | `shasum -a 256 /Users/yiyi0369/projects/flexaligner/align_single_cpu.py` | `9ed4e21e...e835de1`, verified 2026-08-11 |
| S0-005 | All unresolved material choices are visible as `[TBD]`/open questions | PASS | `rg 'TBD'`; `OPEN_QUESTIONS.md`; algorithm table in `DECISIONS.md` | Material unknowns mapped to stable IDs |
| S0-006 | Governance-only baseline passes whitespace/diff checks and is committed | PASS | `git diff --check`; root commit `833306e` | Governance baseline committed 2026-08-11 |

## B. Stage 1 — package, interfaces and CI/CD

| ID | Acceptance condition | Status | Evidence / command | Reviewer note |
|---|---|---|---|---|
| S1-001 | `src/` package has one canonical version and imports without model/network side effects | PASS | `test_import_safety.py`; isolated wheel import from `site-packages`; metadata version `0.1.0.dev0` | No Torch/Transformers import or cwd write |
| S1-002 | Editable install and built-wheel install both expose `flexaligner` CLI | PASS | `.venv` editable install; `/tmp/flexaligner-wheel-smoke.Q5d5Vd` final-wheel install | `pip check`, version and capabilities passed |
| S1-003 | CLI help, version and capability discovery are deterministic | PASS | `tests/test_cli.py`; 9 tests | Human and JSON output stable |
| S1-004 | All requested future capabilities have importable contracts | PASS | public symbols plus complete 14-entry capability report | Future enums do not imply availability |
| S1-005 | Placeholder calls raise typed non-availability errors and do not silently fall back | PASS | `tests/test_placeholders.py`, `test_capabilities.py`, CLI tests | Guards precede input/model/output work |
| S1-006 | Format, lint, strict type, test/coverage, build and wheel-smoke jobs are defined | PASS | commit `5702f0a`; local Ruff/mypy/50-test/88.16%-coverage/build checks; `ci.yml` | Remote matrix remains `TBD-CI-001` |
| S1-007 | Release workflow is event/environment guarded; only publish job has OIDC write permission | PASS | `tests/test_workflow_security.py`; 5 tests; YAML parse | Static policy pass; no remote publish attempted |
| S1-008 | Package metadata, README and license files are present and consistent | PASS | `twine check --strict`; `check-wheel-contents`; `audit_dist.py` | README/LICENSE upstream source hashes rechecked |

## C. Stage 2 — characterization and oracle

| ID | Acceptance condition | Status | Evidence / command | Reviewer note |
|---|---|---|---|---|
| S2-001 | CI guards the frozen reference hash or an equivalent vendored oracle manifest | PASS | byte `cmp`; SHA `9ed4e21e…835de1`; 7 guard tests; fresh dist audit | Vendored reference is test-only and denied from wheel/sdist |
| S2-002 | Stage 1 reference semantics have model-free array tests | PASS | `test_stage1_reference.py`: 30 passed | Independent NumPy and exhaustive small-case oracles |
| S2-003 | Stage 2 graph/path/prune/redecode semantics have model-free tests | PASS | `test_stage2_reference.py`: 26 passed | Exact-DP oracle plus named equal-phone limitation |
| S2-004 | Input, failure, word-order, TextGrid and atomic-write semantics have tests | PASS | input 19 passed; TextGrid/output 6 passed; Stage 1 coverage negatives | Strict PCM16/16 kHz/mono and no skip/xfail |
| S2-005 | Differential harness reports field-level mismatches and forbids casual golden refresh | PASS | differential tests 4 passed; characterization policy audit | A/B/C classes separated; no update-golden entry point |
| S2-006 | Real-model fixture manifest records hashes, versions and provenance | PASS | 16/16 hashes plus exact 3.10.8/2.2.6/2.3.1/4.41.2 runtime preflight | Candidate fixture; real alignment E2E remains Q-007 |
| S2-007 | Missing E2E assets produce `BLOCKED`/explicit preflight failure, never a passing skip | PASS | `test_verify_model_assets.py`: 7 passed; negative subprocess exits nonzero | Workflow binds committed manifest and exact runtime check |

## D. Stage 3 — Stage 1 implementation

| ID | Acceptance condition | Status | Evidence / command | Reviewer note |
|---|---|---|---|---|
| S3-001 | First pronunciation and Chunker stress stripping match reference | PASS | three-way parity suite: 123 passed | First variant/order and failure context covered |
| S3-002 | Trellis/backtrace, including early target completion, match reference | PASS | reference + NumPy oracle + fixed-seed exhaustive tests | Tie-stay and repeated-target current behavior locked |
| S3-003 | Word emission confidence matches the uncalibrated reference definition | PASS | emission/avg-frame/phone-to-word numeric parity | Geometric-mean probability semantics preserved |
| S3-004 | ±0.3 s anchors, strict `<0.2 s` merge and millisecond grid match reference | PASS | boundary/round/tail-clamp parity and negatives | NaN/Inf now fail explicitly before rounding under D-019 |
| S3-005 | Every word index is covered exactly once in order or alignment fails | PASS | duplicate/missing/out-of-order/word-mismatch invariants | Repeated word labels remain distinct by index |

## E. Stage 4 — Stage 2 implementation

| ID | Acceptance condition | Status | Evidence / command | Reviewer note |
|---|---|---|---|---|
| S4-001 | Multi-pronunciation `sil`/`sph` graph paths match accepted reference graph semantics | PASS | 80 parity + 81 independent Stage 2 tests; 64-config read-only cross-audit | Six internal paths, boundary SPH and multi-pronunciation paths verified |
| S4-002 | Beam Viterbi scoring, boundary contrast and transition costs match reference | PASS | reference/exact-DP differential; fixed-seed random cross-audit; Stage 2 branch coverage 90.63% | Stable `>` ties, final move, per-frame bias and one-time enter costs locked |
| S4-003 | Incomplete end paths fail explicitly | PASS | narrow-beam and illegal/incomplete-path negative tests | No partial alignment is returned as success |
| S4-004 | Short internal `sil`/`sph` pruning matches 65/50 ms thresholds | PASS | dedicated lock/prune boundary tests | Viterbi uses `round` (65 ms → 6); prune uses `ceil` (65 ms → 7) |
| S4-005 | Fixed-state second decode matches reference | PASS | reference parity, exact-DP invariant and captured-argument regression | Second pass explicitly clears minimum-lock and both enter costs |
| S4-006 | Adjacent repeated words retain distinct word indices | PASS | repeated `go go` phone/word segmentation tests | Word identity is index-based, not label-only |
| S4-007 | Equal-phone state limitation is characterized and not silently changed | PASS | named current-behavior test; D-020 | Same-phone states within one word remain collapsed pending TBD-ALG-003 |

## F. Stage 5 — pipeline, CLI and output

| ID | Acceptance condition | Status | Evidence / command | Reviewer note |
|---|---|---|---|---|
| S5-001 | Strict WAV/text/lexicon/model/vocabulary validation is enforced | PASS | 50 strict input tests; posterior/model negative integration tests; full 673-test gate | Includes duplicate JSON keys, exact Chunker token→ID mapping and reserved-label preflight |
| S5-002 | Inference is CPU-only, local-only and lazily imports model dependencies | PASS | `test_hf_local.py` plus import-safety tests; 46 adapter/lifecycle tests | `local_files_only=True`, `trust_remote_code=False`; no top-level Torch/Transformers import |
| S5-003 | Chunker and Aligner models are loaded sequentially rather than retained together | PASS | `test_model_lifecycle.py`; instrumented integration trace | `chunk.load→infer→close→align.load→infer→close`, including failure cleanup |
| S5-004 | Pipeline preserves complete normalized input word order | PASS | 24 integration tests plus Stage 1/2 invariants | Repeated words remain index-distinct; incomplete or changed sequences fail |
| S5-005 | TextGrid is temporary-written, read back, validated and atomically published without overwrite | PASS | 34 TextGrid/transaction tests | No-clobber hard-link commit; inode/bytes/semantic postvalidation; TBD-OUT-001 retained |
| S5-006 | Failed runs leave no official success artifact | PASS | race, tamper, symlink and failure-injection tests | Rollback removes only artifacts still owned by that invocation |
| S5-007 | Confidence metadata is explicitly uncalibrated | PASS | pipeline/API/metadata integration tests | Raw scores are finite `[0,1]`; `calibrated=false`, no calibrated score substitution |

## G. Requested placeholder interfaces

| ID | Capability | Status | Evidence / command | Reviewer note |
|---|---|---|---|---|
| P-001 | Mandarin | PLACEHOLDER | `language.zh` capability/API tests | Typed pre-I/O failure verified |
| P-002 | GPU | PLACEHOLDER | `device.gpu` capability/API tests | No CPU fallback |
| P-003 | Batch | PLACEHOLDER | `alignment.batch` tests | Iterable non-consumption verified |
| P-004 | Web | PLACEHOLDER | `integration.web` CLI/API tests | No framework import or port bind |
| P-005 | Automatic model download | PLACEHOLDER | `models.auto_download` tests | Fails before network access |
| P-006 | Multi-format audio | PLACEHOLDER | `audio.multi_format` tests | Strict WAV implementation is a later stage |
| P-007 | Automatic resampling | PLACEHOLDER | `audio.auto_resample` tests | No implicit conversion |
| P-008 | Chinese segmentation | PLACEHOLDER | `text.zh_segmentation` report/require tests | Mandarin remains placeholder |
| P-009 | Default G2P | PLACEHOLDER | `pronunciation.g2p.default` tests | No OOV fallback |
| P-010 | Confidence calibration | PLACEHOLDER | `confidence.calibration` tests | Raw score contract remains uncalibrated |

## H. Stage 6 — quality, package and real-model evidence

| ID | Acceptance condition | Status | Evidence / command | Reviewer note |
|---|---|---|---|---|
| Q-001 | Formatter and linter pass with no ignored new violations | PASS | Ruff 0.16.2: 71 maintained files formatted; all checks passed | Immutable reference excluded and hash-guarded |
| Q-002 | Strict static type check passes | PASS | mypy 2.3.0: no issues in 20 configured `src`/`scripts` files | Strict config |
| Q-003 | Fast tests pass on every supported Python version | NOT_RUN | CI matrix/local available versions | |
| Q-004 | Coverage meets the recorded non-decreasing threshold | PASS | branch coverage 92.31%; D-015 threshold 85% | 673 tests, Python 3.10.8, sockets disabled |
| Q-005 | Wheel and sdist build from clean source and pass metadata checks | PASS | fresh Stage 3 Hatchling build; Twine/check-wheel/audit all pass | wheel `8bf98278...7868f`; sdist `6d83cb0c...be364` |
| Q-006 | Built wheel installs and runs outside repository source tree | PASS | `/tmp/flexaligner-stage3-smoke.TPMkh3`; `pip check`, site-packages core import, CLI | Exact Stage 3 wheel used |
| Q-007 | English real-model E2E passes with frozen assets, or is truthfully `BLOCKED` | NOT_RUN | E2E report and hashes | |
| Q-008 | No test or package import performs an undeclared network request | PASS | `pytest --disable-socket`; import-safety subprocess; offline env | Dependency installation is a separate networked setup step |

## I. Stage 7 — final audit

| ID | Acceptance condition | Status | Evidence / command | Reviewer note |
|---|---|---|---|---|
| F-001 | All required rows are `PASS`; intentional future features are `PLACEHOLDER` | NOT_RUN | acceptance audit script/manual review | |
| F-002 | README claims map to tests/evidence and do not advertise placeholders as support | NOT_RUN | README capability audit | |
| F-003 | `STATE`, decisions and open questions match the verified repository state | NOT_RUN | cross-document audit | |
| F-004 | Working tree, commits, remotes and unexecuted publication steps are reported exactly | NOT_RUN | final Git/release audit | |
