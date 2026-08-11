# OPEN QUESTIONS

Open questions are not implicit permissions. Work proceeds only where the answer
does not materially change the result; otherwise the affected path remains a
placeholder or explicit limitation.

| ID | Open question | Current placeholder/default | Blocks |
|---|---|---|---|
| TBD-PKG-001 | What is the final PyPI distribution name and owner? | use local candidate `flexaligner`; do not publish | Public release only |
| TBD-PKG-002 | Will this clean local history replace, merge into or be grafted onto the existing GitHub history? | no remote; keep provenance links | Remote integration only |
| TBD-PKG-003 | What version should the first public release use? | development version `0.1.0.dev0` proposed | Public release metadata |
| TBD-LIC-001 | Exact README and LICENSE carry-over/attribution text? | fetch fixed remote snapshot, audit before commit | README/license completion |
| TBD-API-001 | Which legacy Python/CLI entry points need compatibility aliases? | no broad compatibility layer | Legacy callers only |
| TBD-CI-001 | Do the exact tool pins and Python 3.10–3.14 matrix pass on every declared GitHub runner? | local 3.10.8 and partial 3.12.12 pass; no remote/Actions result exists | Remote matrix confirmation |
| TBD-E2E-001 | Will the fixture-only `openphonetics OW1 P AH0 N F AH0 N EH1 T IH0 K S` pronunciation be approved as release evidence? | candidate engineering E2E passes; `--require-approved` blocks release | Official real-model E2E |
| TBD-REL-001 | What GitHub repository/workflow/environment will be registered as PyPI Trusted Publisher? | release workflow remains guarded and non-runnable for publish | Actual PyPI upload |
| TBD-ALG-001 | Continuous coverage correction? | characterize current gap behavior first | Behavior-corrected MVP claim |
| TBD-ALG-002 | Stage 2 stride policy? | preserve current 0.01 s for parity, report mismatch | Dynamic timing claim |
| TBD-ALG-003 | Consecutive equal phone-state identity? | preserve current behavior for parity | Corrected phone identity claim |
| TBD-ALG-004 | Standard repeated-label CTC constraint? | preserve current simplified recurrence for parity | Algorithm correction claim |
| TBD-ALG-005 | Resource limits? | add conservative tested limits only after measurement | Production safety claim |
| TBD-OUT-001 | What crash-consistency protocol should cover optional metadata plus TextGrid across two files? | stage and validate both; commit metadata first and TextGrid last; roll back process-visible failures | Multi-artifact crash/power-loss atomicity claim |
| TBD-API-002 | How should final phone intervals retain fixed-state word/phone provenance? | expose `word_index=None`, `phone_index=None`; do not infer missing provenance | Phone-to-word provenance claim |
| TBD-PROV-001 | What canonical model-directory fingerprint format belongs in public provenance? | leave `model_fingerprints` empty; E2E manifest retains asset hashes | Public reproducibility schema |
| TBD-INF-001 | Which Hugging Face processor/model architecture combinations are a supported public matrix? | validate the current AutoProcessor/AutoModelForCTC contract strictly; frozen bundles emit weight-normalization migration warnings; do not claim a broad matrix | Broad model compatibility claim |
| TBD-THREAD-001 | Should an alignment restore the process-global Torch thread count after execution? | set only the requested positive CPU thread count inside the inference lifecycle; document process-global scope | Thread-isolation claim |
| TBD-TEXT-001 | How should lexical words named `sil` or `null` coexist with the current special TextGrid labels? | reject them with a typed input error before model loading | Aligning those two English tokens |

## Non-questions fixed by current scope

- Mandarin does not need a real-model E2E in this milestone.
- No placeholder capability may trigger an automatic download or conversion.
- Missing model assets are not grounds for claiming an E2E pass.
- The local reference file overrides conflicting old-session descriptions.
