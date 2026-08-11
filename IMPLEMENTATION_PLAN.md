# FlexAligner clean-room rebuild implementation plan

> Status: ACCEPTED EXECUTION PLAN — Stage 5 in progress
> Established: 2026-08-11 (Asia/Shanghai)
> Local repository: `/Users/yiyi0369/projects/flexaligner-rebuild`
> Algorithm reference: `/Users/yiyi0369/projects/flexaligner/align_single_cpu.py`
> Reference SHA-256: `9ed4e21e615718ddfd10930359f55769fb27a0d284599cce45a3fc755e835de1`

## 1. Objective

Build a new, clean Python codebase that can be distributed as a PyPI package and
whose first implemented product path is:

```text
English transcript + local lexicon + local CTC models + 16 kHz mono PCM16 WAV
    -> CPU-only two-stage forced alignment
    -> validated, atomically written TextGrid
```

The implementation must preserve the accepted algorithm semantics of the current
`align_single_cpu.py` reference before any behavior-changing correction is made.
The reference is an oracle and evidence source; production code must not import it.

The repository must also expose deliberate, documented extension boundaries for
future work without pretending that those capabilities are implemented.

## 2. Definition of done

The goal is complete only when all of the following are true:

1. The local repository has a reproducible `src/`-layout Python package, an
   installable wheel and sdist, one canonical version source, and a working CLI.
2. The CPU/single-file/English path implements Stage 1, Stage 2, pipeline and
   TextGrid output using clean modules and explicit contracts.
3. Fast CI gates pass without external model downloads: formatting, linting,
   strict typing, unit/characterization tests, coverage, package build, metadata
   validation, and isolated wheel-install smoke tests.
4. A separately gated real-model E2E test either passes against a frozen local
   asset manifest or is recorded as `BLOCKED` with exact missing evidence. It may
   never be silently skipped while still being reported as passed.
5. Mandarin, GPU, batch, Web, automatic model download, multi-format audio,
   automatic resampling, Chinese segmentation, default G2P and confidence
   calibration have importable contracts and explicit capability states, but
   calls fail with a typed `FeatureNotAvailableError` until implemented.
6. Every row in `ACCEPTANCE.md` contains status, command/evidence and reviewer
   notes. Required rows are `PASS`; intentional placeholders are `PLACEHOLDER`,
   not `PASS` or “implemented”.
7. `STATE.md`, `DECISIONS.md` and `OPEN_QUESTIONS.md` describe the final verified
   state without reviving assumptions from old conversations.

## 3. Project fact discipline

Project facts use this authority order:

1. Explicit information in the current user message.
2. Current uploaded/local code, configuration, data and experiment output.
3. `STATE.md`.
4. `DECISIONS.md`.
5. `OPEN_QUESTIONS.md`.
6. Old conversation content.

If old conversation content conflicts with files:

- current files and `STATE.md` win;
- the conflict is recorded explicitly;
- the two versions are not silently merged;
- rejected assumptions are not revived from conversation history.

Unless the user explicitly requests otherwise, EXPLORE or TEACH material is not
an accepted decision, and vague history is not used to fill project state.

## 4. Accepted scope

### 4.1 Implement in the first package

- clean repository and package identity;
- Python 3.10+ `[TBD: final supported upper-version matrix after CI]`;
- CPU-only execution;
- one audio file and one transcript per invocation;
- English, whitespace-tokenized transcript;
- strict local lexicon coverage;
- strict local Chunker and Aligner model paths;
- strict 16 kHz, mono, uncompressed PCM16 WAV input;
- Stage 1 CTC macro localization/chunking;
- Stage 2 pronunciation graph plus two-pass Viterbi alignment;
- strict input, vocabulary, path-completion and word-order failures;
- words/phones TextGrid tiers;
- temporary write, read-back validation and atomic replacement;
- optional uncalibrated Stage 1 confidence metadata whose semantics are named
  explicitly and never presented as calibrated probability;
- Python API and `flexaligner` / `python -m flexaligner` CLI;
- package build, install, CI and guarded release workflows.

### 4.2 Stable placeholders only

The following must have explicit contracts and capability discovery but no
production implementation in this milestone:

| Capability | Placeholder contract | Required behavior now |
|---|---|---|
| Mandarin | language/profile boundary | report unavailable; typed failure |
| GPU | execution backend boundary | report unavailable; CPU remains explicit |
| Batch | batch request/result boundary | report unavailable; never loop silently |
| Web | service adapter boundary | report unavailable; no server dependency |
| Automatic model download | model resolver boundary | local-path resolver only; remote resolver fails |
| Multi-format audio | audio decoder boundary | WAV/PCM16 decoder only; other formats fail |
| Automatic resampling | audio transform boundary | strict validator only; no implicit conversion |
| Chinese segmentation | tokenizer boundary | English whitespace tokenizer only |
| Default G2P | pronunciation provider boundary | lexicon provider only; OOV remains an error |
| Confidence calibration | calibrator boundary | identity/uncalibrated marker only; calibrated request fails |

Placeholder methods must not use `pass`, return empty success values, silently
fall back, download assets, change input data or claim availability.

### 4.3 Explicitly out of scope

- model training or fine-tuning;
- remote hosting or deployment;
- changing the GitHub default branch or publishing a package to PyPI;
- backward compatibility with every API in the remote legacy implementation;
- accuracy or reliability claims not supported by frozen fixtures;
- Mandarin E2E validation in this milestone;
- automatic recovery from OOV, missing phones or incomplete Viterbi paths.

## 5. Algorithm migration rule

Migration and correction are separate tracks:

1. **Characterize reference behavior.** Write deterministic tests for accepted
   semantics and explicitly mark known defects/limitations.
2. **Migrate for parity.** New modules must match the reference on frozen arrays
   and frozen real-model artifacts within approved tolerances.
3. **Audit parity.** Any mismatch stops the stage; changing a golden file is not
   a repair.
4. **Correct only by decision.** A behavior-changing correction needs a new
   `DECISIONS.md` entry, a before/after comparison and dedicated tests.

Known issues that remain separate decisions:

- `[TBD-ALG-001]` continuous interval coverage and frame-grid tail gaps;
- `[TBD-ALG-002]` validation/dynamic handling of Stage 2 frame stride;
- `[TBD-ALG-003]` identity of consecutive equal phone states in one word;
- `[TBD-ALG-004]` repeated-label blank constraint in Stage 1 CTC;
- `[TBD-ALG-005]` approved limits for duration, words, phones, trellis cells and
  beam work.

These items must be tested and visible. They may not be silently “fixed” during
module extraction.

## 6. Target repository shape

```text
.github/workflows/
  ci.yml
  release.yml
src/flexaligner/
  __init__.py
  __main__.py
  api.py
  capabilities.py
  contracts.py
  errors.py
  ports.py
  stage1.py
  stage2.py
  textgrid.py
  pipeline.py
  cli.py
  adapters/
    wav_pcm16.py
    lexicon_file.py
    hf_local.py
tests/
  unit/
  characterization/
  integration/
  e2e/
  fixtures/
README.md
LICENSE
pyproject.toml
project.md
IMPLEMENTATION_PLAN.md
ACCEPTANCE.md
STATE.md
DECISIONS.md
OPEN_QUESTIONS.md
```

The exact module count may be reduced during audit, but dependency direction is
fixed:

```text
contracts / errors / capabilities
               ^
adapters -> ports <- pipeline -> stage1 / stage2 / textgrid
                          ^
                         api
                          ^
                         cli
```

`stage1.py` and `stage2.py` operate on explicit arrays and domain records. They
must not load models, parse CLI arguments or write files. Only
`adapters/hf_local.py` may import `transformers`; package import and capability
discovery must not import Torch, open a model or access the network.

### 6.1 Intended stable public surface

The first public surface is intentionally small:

- one lazy `FlexAligner` engine with `align()`, `capabilities()`, `close()` and a
  context-manager contract;
- immutable, keyword-only `AlignmentRequest`, `AlignmentOptions`,
  `LocalModelBundle`, `TextGridOutput` and `AlignmentResult` records;
- a versioned `CapabilityReport`;
- a typed `FlexAlignerError` hierarchy with stable machine-readable string codes.

Algorithm functions and adapter Protocols remain internal in v1. Enumerations may
name future options such as `ZH`, `CUDA`, `AUTO_DOWNLOAD` or `G2P`, but selecting
one must call the capability guard and fail before consuming input, creating
files, loading models or accessing a network.

The raw confidence result is stored separately with
`kind=chunker_emission_geometric_mean` and `calibrated=false`. A future
calibrator may add calibrated scores; it must not overwrite or reinterpret the
raw score.

### 6.2 Placeholder failure ordering

- Mandarin/GPU requests fail before input/model inspection and never downgrade
  to English/CPU.
- `align_batch()` fails before consuming its iterable.
- `serve` fails before importing a Web framework or binding a port.
- model fetch fails before a network request.
- unsupported audio/resampling requests fail before invoking ffmpeg or changing
  samples.
- G2P fails before changing an OOV transcript; lexicon-only OOV remains a
  distinct input error.
- calibrated-confidence requests fail instead of relabeling raw scores.

## 7. Work allocation and merge discipline

The main agent owns scope, repository state, edits touching shared contracts,
integration, acceptance evidence and final sign-off.

Parallel work is split into bounded streams:

| Stream | Initial responsibility | Merge rule |
|---|---|---|
| A — packaging/CI | package metadata, dependency split, CI and release gates | main agent verifies official guidance and runs all gates |
| B — architecture/API | domain contracts and future-capability placeholders | main agent audits imports, behavior and API stability |
| C — tests/oracle | characterization matrix, fixtures, differential harness | main agent verifies every expectation against current files |
| Main — governance/core | plan, decisions, state, integration and algorithm migration | only main marks acceptance rows |

Agents do not edit the same file concurrently. Work is reviewed from the shared
filesystem and accepted only after the main agent reruns the relevant gate.

### 7.1 Stage 5 parallel execution allocation

Stage 5 uses the following disjoint ownership after the main agent freezes the
shared Protocols and creates `adapters/__init__.py`:

| Stream | Exclusive production files | Exclusive tests |
|---|---|---|
| A — strict input | `adapters/wav_pcm16.py`, `adapters/lexicon_file.py` | `tests/unit/test_wav_pcm16.py`, `tests/unit/test_lexicon_file.py` |
| B — local inference | `adapters/hf_local.py` | `tests/unit/test_hf_local.py` |
| C — TextGrid/output | `textgrid.py` | `tests/unit/test_textgrid.py`, `tests/unit/test_output_transaction.py` |
| Main — integration | `ports.py`, `pipeline.py`, public API/CLI/contracts/capabilities/errors and package exports | pipeline, lifecycle, API/CLI and placeholder regression tests |

Merge and audit order is strict input, local inference, TextGrid/output,
pipeline, API, CLI, then capability promotion. The inference factory exposes
non-overlapping Chunker and Aligner context managers; the pipeline must exit the
Chunker context before entering the Aligner context. Fast tests use fake
sessions and remain model-free and network-disabled.

All requested future options remain guarded before I/O. The single implemented
path is promoted to `available` only after the complete pipeline gate passes.
For optional metadata plus TextGrid, all artifacts are staged and validated,
metadata is committed first and TextGrid last as the success marker, and
in-process failures roll back artifacts created by that invocation. A normal
filesystem cannot guarantee atomic commit across two files during a crash or
power loss; this limitation is tracked as `TBD-OUT-001`.

## 8. Staged execution

### Stage 0 — repository, governance and executable plan

Deliverables:

- new local Git repository;
- this plan;
- initial `project.md`, `STATE.md`, `DECISIONS.md`, `OPEN_QUESTIONS.md`;
- `ACCEPTANCE.md` with stable IDs and `NOT_RUN` statuses;
- reference/remote provenance recorded without copying legacy core code.

Gate:

- all accepted scope appears in governance documents;
- all unknowns are `[TBD]` or open questions;
- the reference SHA matches the current file;
- no production implementation has been presented as complete.

### Stage 1 — package and interface skeleton

Deliverables:

- `pyproject.toml`, `src/` layout and canonical version;
- public domain records, typed errors and capability registry;
- importable placeholder interfaces with explicit failures;
- CLI help/version/capabilities commands;
- initial README capability table;
- strict CI and guarded release workflow.

Gate:

- clean editable and wheel installs work;
- placeholders are covered by tests;
- package import has no model/network side effect;
- no workflow can publish on a pull request or ordinary branch push;
- only the release job receives `id-token: write`.

### Stage 2 — reference characterization and test harness

Deliverables:

- reference hash guard;
- pure-array Stage 1 and Stage 2 characterization cases;
- input/failure/TextGrid/atomic-write tests;
- differential comparison records and golden-update policy;
- frozen E2E asset manifest with hashes and provenance.

Gate:

- fast tests need no model and no network;
- every migrated reference behavior has a pre-existing characterization test;
- known limitations are named rather than normalized away;
- E2E cannot report pass when an asset is absent.

### Stage 3 — Stage 1 implementation

Deliverables:

- first-pronunciation selection and stress stripping;
- CTC trellis/backtrace with early finish;
- word emission confidence;
- anchors, strict merge boundary and millisecond chunk rounding;
- complete ordered word-index coverage checks.

Gate:

- synthetic array parity is exact or within an explicitly recorded numeric
  tolerance;
- strict failure cases pass;
- complexity/resource behavior is measured and `[TBD-ALG-005]` is resolved or
  remains an explicit release limitation.

### Stage 4 — Stage 2 implementation

Deliverables:

- multi-pronunciation phone DAG;
- optional `sil`/`sph` gap paths;
- beam Viterbi with current stay/move, frame bias, enter cost and boundary
  contrast semantics;
- complete-end-state enforcement;
- short internal-state pruning and fixed-sequence second decode;
- state-to-phone/word segmentation.

Gate:

- graph, path, pruning and second-pass characterization tests pass;
- incomplete paths fail explicitly;
- adjacent repeated word instances remain distinct;
- `[TBD-ALG-003]` is not silently changed.

### Stage 5 — inference, pipeline, CLI and output

Deliverables:

- strict WAV/text/lexicon/model validation;
- lazy local Hugging Face inference adapter;
- sequential Chunker/Aligner loading and release;
- local-to-global merge and validated TextGrid;
- Python API and single-file CLI;
- structured, uncalibrated metadata.

Gate:

- failures leave no official output;
- output is written to a temporary sibling, read back, validated and atomically
  replaced;
- normalized non-special word sequence equals the input exactly;
- no network request occurs;
- core execution remains CPU-only.

### Stage 6 — package, real-model E2E and release rehearsal

Deliverables:

- complete fast quality suite;
- wheel and sdist built and metadata checked;
- wheel installed in an isolated environment and smoke-tested outside source;
- frozen English real-model E2E evidence;
- guarded Trusted Publishing workflow with owner/environment fields `[TBD]`;
- final README capability/limitation table.

Gate:

- every required fast gate passes locally;
- E2E evidence identifies exact Python/dependency/model/dictionary/input hashes;
- publish workflow builds once, then publishes the exact uploaded artifact;
- no actual PyPI upload occurs without explicit user authorization and account
  configuration.

### Stage 7 — main-agent final audit

Deliverables:

- completed `ACCEPTANCE.md`;
- final `STATE.md`, decisions and open questions;
- diff/stat and repository-status review;
- user-facing release-readiness report.

Gate:

- no required acceptance row is `NOT_RUN`, `FAIL` or falsely marked `PASS`;
- placeholders remain distinguishable from implemented features;
- all remaining `[TBD]` items are either release blockers or documented
  non-blocking limitations;
- the working tree state is reported exactly.

## 9. CI/CD policy

“Strict CI/CD” means separate, inspectable gates:

1. formatting check;
2. lint and import hygiene;
3. strict static type checking;
4. fast unit and characterization tests with branch coverage threshold;
5. package build from clean source;
6. sdist/wheel metadata check;
7. install the built wheel in a fresh environment and run import/CLI smoke;
8. optional/manual real-model E2E with explicit asset preflight;
9. release only from an approved GitHub Release/tag and protected `pypi`
   environment using OIDC Trusted Publishing;
10. release job consumes previously built artifacts and has no source checkout or
    arbitrary package build step.

Initial tool choices are `ruff`, `mypy`, `pytest`, `coverage`, `build` and
`twine`; exact versions and the coverage threshold are `[TBD-CI-001]` until the
first green baseline is measured. The threshold may only ratchet upward or change
through a recorded decision.

Third-party GitHub Actions will be pinned to immutable commit SHAs before any
remote use. Human-readable major tags may be noted in comments.

### 9.1 Test and evidence layers

| Layer | Required evidence | Normal trigger | Gate |
|---|---|---|---|
| C0 reference guard | reference hash/provenance; production import and wheel exclusion | every change | blocking |
| C1 no-model core | pure-array, contracts, failure, TextGrid and placeholder tests | every change | blocking |
| C2 package smoke | build, metadata, clean wheel install, import and CLI | supported Python matrix | blocking |
| C3 posterior parity | fixed synthetic posterior Stage 1/2 differential report | every change/main | blocking |
| C4 real-model E2E | offline English reference/new double run with full hashes | trusted runner/release | release blocking; never a passing skip |
| C5 resources | limits, timing, peak RSS and model-lifetime evidence | nightly/release candidate | release policy `[TBD]` |

Tests use three distinct oracle tracks:

- **A — current behavior parity:** exact observable reference behavior, including
  named quirks;
- **B — independent invariants:** mathematical small-case oracle, word/order,
  finite values, output and failure guarantees;
- **C — approved corrections:** only changes tied to a recorded decision ID.

Track C must prove both the intended change and non-regression of every unrelated
Track A behavior.

### 9.2 Golden and differential policy

- CI has no `--update-golden` path.
- A generator writes only to `candidate/`, never over an accepted baseline.
- Every fixture, posterior, effective lexicon, model directory and environment
  used as an oracle is content-addressed.
- Updating a baseline requires a decision ID, field-level old/new semantic diff,
  provenance changes and README/API impact.
- Differences cannot be resolved by wider tolerance, deleted assertions, `skip`
  or `xfail`.
- A parity report records the first differing field and both values.
- Large external models are not committed to Git; their complete manifest and
  hashes are evidence.
- The reference and new implementation always write to different temporary
  directories.
- Fast CI performs no model download and no network access.

## 10. Evidence and status vocabulary

Each acceptance row uses exactly one status:

- `NOT_RUN`: no current evidence;
- `IN_PROGRESS`: implementation or verification is underway;
- `PASS`: the stated command/check was run successfully and evidence is recorded;
- `FAIL`: the check ran and failed;
- `BLOCKED`: the check cannot run because a named prerequisite is unavailable;
- `PLACEHOLDER`: interface exists, and non-availability behavior is verified;
- `N/A`: excluded by an accepted scope decision.

No row may change from `FAIL` to `PASS` merely by weakening its assertion or
regenerating expected output.

## 11. Initial unknowns

- `[TBD-PKG-001]` final PyPI distribution name and owning organization/account.
  `flexaligner` currently appears unregistered, but availability is not reserved.
- `[TBD-PKG-002]` final public repository URL and whether this new local Git
  history will replace or be merged into the existing remote history.
- `[TBD-PKG-003]` final project version for the first public release.
- `[TBD-LIC-001]` exact README/LICENSE carry-over and attribution text after a
  source snapshot is copied and audited.
- `[TBD-API-001]` compatibility aliases, if any, for the remote legacy API.
- `[TBD-CI-001]` coverage threshold and fully pinned developer-tool versions.
- `[TBD-E2E-001]` fixture-specific English lexicon containing `openphonetics` and
  its approved provenance/hash.
- `[TBD-REL-001]` PyPI Trusted Publisher repository, workflow and environment
  configuration.
- `[TBD-ALG-001..005]` behavior corrections listed in section 5.

Unknowns remain visible placeholders. They do not prevent work that has no
dependency on the unresolved choice.

## 12. Immediate execution order

1. Complete Stage 0 documents and audit their consistency.
2. Commit the governance-only baseline.
3. Merge the three parallel design reviews into Stage 1 files.
4. Run and record the Stage 1 gates.
5. Continue one stage at a time; update `STATE.md` and `ACCEPTANCE.md` before
   moving the active stage forward.
