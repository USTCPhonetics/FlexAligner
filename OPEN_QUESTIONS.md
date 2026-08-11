# OPEN QUESTIONS

Open questions are not implicit permissions. Work proceeds only where the answer
does not materially change the result; otherwise the affected path remains a
placeholder or explicit limitation.

| ID | Open question | Current placeholder/default | Blocks |
|---|---|---|---|
| TBD-CI-001 | Does the declared Python 3.10–3.14/OS matrix pass on the target GitHub runners with the committed exact tool pins? | local 3.10.8 and partial 3.12.12 pass; no remote/Actions result exists | Remote matrix confirmation |
| TBD-REMOTE-001 | When will direct replacement of the existing `USTCPhonetics/FlexAligner` main history receive separate external authorization, and what verified recovery snapshot will be retained immediately before replacement? | `REPLACE_MAIN_HISTORY` is an accepted strategy only; no remote, push, force-update, default-branch change or history deletion is authorized | Remote history replacement |
| TBD-E2E-002 | How will the approved E2E manifest remove its repository-local `flexaligner-rebuild/tests/fixtures/e2e/english_synthetic.dict` path dependency and prove portability on the target remote runner? | keep the approved fixture local and fail closed; local Q-007 is limited to the verified exact-wheel run, and protected remote E2E must not be reported as passed until a portable layout and actual rerun are verified | Remote/release E2E portability |
| TBD-REL-001 | Which exact workflow identity, protected `pypi` environment, approvers and Trusted Publisher binding in `USTCPhonetics/FlexAligner` will be used? | the target project and package owner are selected, but no remote environment or publisher is configured or authorized | Actual PyPI upload |
| TBD-ALG-001 | Continuous coverage correction? | behavior is characterized; preserve the current gap behavior for alpha | Behavior-corrected claim |
| TBD-ALG-002 | Stage 2 stride policy? | preserve current 0.01 s for parity, report mismatch | Dynamic timing claim |
| TBD-ALG-003 | Consecutive equal phone-state identity? | preserve current behavior for parity | Corrected phone identity claim |
| TBD-ALG-004 | Standard repeated-label CTC constraint? | preserve current simplified recurrence for parity | Algorithm correction claim |
| TBD-ALG-005 | Resource limits? | add conservative tested limits only after measurement | Production safety claim |
| TBD-OUT-001 | What crash-consistency protocol should cover optional metadata plus TextGrid across two files? | stage and validate both; commit metadata first and TextGrid last; roll back process-visible failures | Multi-artifact crash/power-loss atomicity claim |
| TBD-API-002 | How should final phone intervals retain fixed-state word/phone provenance? | expose `word_index=None`, `phone_index=None`; do not infer missing provenance | Phone-to-word provenance claim |
| TBD-PROV-001 | What canonical model-directory fingerprint format belongs in public provenance? | leave `model_fingerprints` empty; E2E manifest retains asset hashes | Public reproducibility schema |
| TBD-INF-001 | What exact `[inference]` resolver/runtime window can the public alpha install truthfully? | the current broad ranges are not approved support; narrow to an evidenced window and test public-index resolution, or remove the public extra | Public-alpha inference contract |
| TBD-THREAD-001 | Should an alignment restore the process-global Torch thread count after execution? | set only the requested positive CPU thread count inside the inference lifecycle; document process-global scope before alpha | Thread-isolation claim; alpha documentation |
| TBD-TEXT-001 | How should lexical words named `sil` or `null` coexist with the current special TextGrid labels? | reject them with a typed input error before model loading | Aligning those two English tokens |

## Resolved by the 2026-08-11 user review

Resolved questions are retained here only as a closure ledger. They are not open
items and do not grant external-operation authorization.

| Former ID | Resolution | Decision |
|---|---|---|
| TBD-PKG-001 | PyPI distribution `flexaligner`; owner/organization `ustcphonetics` | D-031 |
| TBD-PKG-002 | Directly replace the existing GitHub main history; do not merge or graft unrelated histories | D-030 |
| TBD-PKG-003 | First public version is the alpha `0.1.0a1`; the working tree remains `0.1.0.dev0` until implementation | D-029 |
| TBD-LIC-001 | Use the fixed original remote README and linked LICENSE snapshot for MIT, copyright, authorship, affiliation and citation identity | D-032 |
| TBD-API-001 | No broad legacy compatibility layer in v0.1 preview; reopen only for an identified real caller | D-034 |
| TBD-E2E-001 | Approve the stated pronunciation only as a frozen release-E2E fixture | D-033 |

## Non-questions fixed by current scope

- Mandarin does not need a real-model E2E in this milestone.
- No placeholder capability may trigger an automatic download or conversion.
- Missing model assets are not grounds for claiming an E2E pass.
- The local reference file overrides conflicting old-session descriptions.
