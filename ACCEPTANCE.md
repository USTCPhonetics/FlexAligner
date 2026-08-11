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
| S1-001 | `src/` package has one canonical version and imports without model/network side effects | NOT_RUN | isolated import smoke | |
| S1-002 | Editable install and built-wheel install both expose `flexaligner` CLI | NOT_RUN | clean virtual environments | |
| S1-003 | CLI help, version and capability discovery are deterministic | NOT_RUN | CLI snapshots/tests | |
| S1-004 | All requested future capabilities have importable contracts | NOT_RUN | public API tests | |
| S1-005 | Placeholder calls raise typed non-availability errors and do not silently fall back | NOT_RUN | placeholder behavior tests | |
| S1-006 | Format, lint, strict type, test/coverage, build and wheel-smoke jobs are defined | NOT_RUN | workflow audit and local equivalent | |
| S1-007 | Release workflow is event/environment guarded; only publish job has OIDC write permission | NOT_RUN | workflow security audit | |
| S1-008 | Package metadata, README and license files are present and consistent | NOT_RUN | build metadata / `twine check` | |

## C. Stage 2 — characterization and oracle

| ID | Acceptance condition | Status | Evidence / command | Reviewer note |
|---|---|---|---|---|
| S2-001 | CI guards the frozen reference hash or an equivalent vendored oracle manifest | NOT_RUN | hash-guard test | |
| S2-002 | Stage 1 reference semantics have model-free array tests | NOT_RUN | characterization test report | |
| S2-003 | Stage 2 graph/path/prune/redecode semantics have model-free tests | NOT_RUN | characterization test report | |
| S2-004 | Input, failure, word-order, TextGrid and atomic-write semantics have tests | NOT_RUN | characterization test report | |
| S2-005 | Differential harness reports field-level mismatches and forbids casual golden refresh | NOT_RUN | harness tests / policy audit | |
| S2-006 | Real-model fixture manifest records hashes, versions and provenance | NOT_RUN | manifest validation | |
| S2-007 | Missing E2E assets produce `BLOCKED`/explicit preflight failure, never a passing skip | NOT_RUN | negative E2E preflight | |

## D. Stage 3 — Stage 1 implementation

| ID | Acceptance condition | Status | Evidence / command | Reviewer note |
|---|---|---|---|---|
| S3-001 | First pronunciation and Chunker stress stripping match reference | NOT_RUN | unit/differential tests | |
| S3-002 | Trellis/backtrace, including early target completion, match reference | NOT_RUN | synthetic log-prob tests | |
| S3-003 | Word emission confidence matches the uncalibrated reference definition | NOT_RUN | numeric tests | |
| S3-004 | ±0.3 s anchors, strict `<0.2 s` merge and millisecond grid match reference | NOT_RUN | boundary tests | |
| S3-005 | Every word index is covered exactly once in order or alignment fails | NOT_RUN | property/negative tests | |

## E. Stage 4 — Stage 2 implementation

| ID | Acceptance condition | Status | Evidence / command | Reviewer note |
|---|---|---|---|---|
| S4-001 | Multi-pronunciation `sil`/`sph` graph paths match accepted reference graph semantics | NOT_RUN | graph tests | |
| S4-002 | Beam Viterbi scoring, boundary contrast and transition costs match reference | NOT_RUN | path/score tests | |
| S4-003 | Incomplete end paths fail explicitly | NOT_RUN | negative path tests | |
| S4-004 | Short internal `sil`/`sph` pruning matches 65/50 ms thresholds | NOT_RUN | threshold tests | |
| S4-005 | Fixed-state second decode matches reference | NOT_RUN | two-pass tests | |
| S4-006 | Adjacent repeated words retain distinct word indices | NOT_RUN | repeated-word tests | |
| S4-007 | Equal-phone state limitation is characterized and not silently changed | NOT_RUN | known-limitation test/decision | |

## F. Stage 5 — pipeline, CLI and output

| ID | Acceptance condition | Status | Evidence / command | Reviewer note |
|---|---|---|---|---|
| S5-001 | Strict WAV/text/lexicon/model/vocabulary validation is enforced | NOT_RUN | input matrix tests | |
| S5-002 | Inference is CPU-only, local-only and lazily imports model dependencies | NOT_RUN | adapter tests/integration trace | |
| S5-003 | Chunker and Aligner models are loaded sequentially rather than retained together | NOT_RUN | instrumented integration test | |
| S5-004 | Pipeline preserves complete normalized input word order | NOT_RUN | integration/differential tests | |
| S5-005 | TextGrid is temporary-written, read back, validated and atomically replaced | NOT_RUN | failure-injection tests | |
| S5-006 | Failed runs leave no official success artifact | NOT_RUN | failure-injection tests | |
| S5-007 | Confidence metadata is explicitly uncalibrated | NOT_RUN | schema/docs tests | |

## G. Requested placeholder interfaces

| ID | Capability | Status | Evidence / command | Reviewer note |
|---|---|---|---|---|
| P-001 | Mandarin | NOT_RUN | capability/API tests | must remain placeholder |
| P-002 | GPU | NOT_RUN | capability/API tests | must remain placeholder |
| P-003 | Batch | NOT_RUN | capability/API tests | must remain placeholder |
| P-004 | Web | NOT_RUN | capability/API tests | must remain placeholder |
| P-005 | Automatic model download | NOT_RUN | capability/API tests | local resolver may be real |
| P-006 | Multi-format audio | NOT_RUN | capability/API tests | WAV PCM16 only is real |
| P-007 | Automatic resampling | NOT_RUN | capability/API tests | strict validator only is real |
| P-008 | Chinese segmentation | NOT_RUN | capability/API tests | English whitespace only is real |
| P-009 | Default G2P | NOT_RUN | capability/API tests | lexicon only is real |
| P-010 | Confidence calibration | NOT_RUN | capability/API tests | uncalibrated metadata only is real |

## H. Stage 6 — quality, package and real-model evidence

| ID | Acceptance condition | Status | Evidence / command | Reviewer note |
|---|---|---|---|---|
| Q-001 | Formatter and linter pass with no ignored new violations | NOT_RUN | `ruff format --check`; `ruff check` | |
| Q-002 | Strict static type check passes | NOT_RUN | `mypy` | |
| Q-003 | Fast tests pass on every supported Python version | NOT_RUN | CI matrix/local available versions | |
| Q-004 | Coverage meets the recorded non-decreasing threshold | NOT_RUN | coverage report | |
| Q-005 | Wheel and sdist build from clean source and pass metadata checks | NOT_RUN | `python -m build`; `twine check` | |
| Q-006 | Built wheel installs and runs outside repository source tree | NOT_RUN | isolated wheel smoke | |
| Q-007 | English real-model E2E passes with frozen assets, or is truthfully `BLOCKED` | NOT_RUN | E2E report and hashes | |
| Q-008 | No test or package import performs an undeclared network request | NOT_RUN | network-denial tests/audit | |

## I. Stage 7 — final audit

| ID | Acceptance condition | Status | Evidence / command | Reviewer note |
|---|---|---|---|---|
| F-001 | All required rows are `PASS`; intentional future features are `PLACEHOLDER` | NOT_RUN | acceptance audit script/manual review | |
| F-002 | README claims map to tests/evidence and do not advertise placeholders as support | NOT_RUN | README capability audit | |
| F-003 | `STATE`, decisions and open questions match the verified repository state | NOT_RUN | cross-document audit | |
| F-004 | Working tree, commits, remotes and unexecuted publication steps are reported exactly | NOT_RUN | final Git/release audit | |
