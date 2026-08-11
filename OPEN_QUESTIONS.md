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
| TBD-CI-001 | Do the exact tool pins and Python 3.10–3.14 matrix pass on every declared GitHub runner? | local Python 3.10 passes; coverage ratchet is fixed at 85% by D-015 | Remote matrix confirmation |
| TBD-E2E-001 | What approved effective lexicon supplies the OOV word `openphonetics` in the English synthetic fixture? | E2E reports `BLOCKED`, never silently skips | Official real-model E2E |
| TBD-REL-001 | What GitHub repository/workflow/environment will be registered as PyPI Trusted Publisher? | release workflow remains guarded and non-runnable for publish | Actual PyPI upload |
| TBD-ALG-001 | Continuous coverage correction? | characterize current gap behavior first | Behavior-corrected MVP claim |
| TBD-ALG-002 | Stage 2 stride policy? | preserve current 0.01 s for parity, report mismatch | Dynamic timing claim |
| TBD-ALG-003 | Consecutive equal phone-state identity? | preserve current behavior for parity | Corrected phone identity claim |
| TBD-ALG-004 | Standard repeated-label CTC constraint? | preserve current simplified recurrence for parity | Algorithm correction claim |
| TBD-ALG-005 | Resource limits? | add conservative tested limits only after measurement | Production safety claim |

## Non-questions fixed by current scope

- Mandarin does not need a real-model E2E in this milestone.
- No placeholder capability may trigger an automatic download or conversion.
- Missing model assets are not grounds for claiming an E2E pass.
- The local reference file overrides conflicting old-session descriptions.
