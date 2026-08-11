# DECISIONS

This file separates explicit user decisions from reversible implementation
choices. An implementation choice is not rewritten as user authorization.

| ID | Decision | Source | Status |
|---|---|---|---|
| D-001 | Rebuild from a clean code baseline rather than repair the remote core in place | Current user message | Accepted |
| D-002 | Keep the remote README/product identity broadly, but re-audit every capability and command against the new implementation | Current and preceding user message | Accepted; exact text `[TBD]` |
| D-003 | Use the current local `align_single_cpu.py` as the core algorithm authority | Current project instruction and local file | Accepted |
| D-004 | Mandarin is a placeholder in this milestone | Current user message | Accepted |
| D-005 | GPU, batch, Web, automatic model download, multi-format audio, automatic resampling, Chinese segmentation, default G2P and confidence calibration receive interfaces only | Current user message | Accepted |
| D-006 | Build toward a publishable pip/PyPI package with strict CI/CD tests | Current user message | Accepted |
| D-007 | Parallelize bounded work with sub-agents; main agent audits, merges, schedules and signs acceptance | Current user message | Accepted |
| D-008 | Use the stated fact-authority and conflict-handling discipline | User-provided governance text | Accepted |
| D-009 | Create the reversible local repository at `/Users/yiyi0369/projects/flexaligner-rebuild` | Main-agent path choice implementing “new local repository” | Active implementation choice |
| D-010 | First implemented product path is English, CPU, single file, strict local models/lexicon/WAV | Mandarin-placeholder requirement plus current reference/assets | Active implementation choice; reversible before public API freeze |
| D-011 | Placeholder calls fail with typed `FeatureNotAvailableError`; they do not silently fall back | Main-agent safety/API design | Active implementation choice |
| D-012 | Characterize and reach reference parity before behavior-changing corrections | Main-agent migration guard derived from “local logic is authoritative” | Active implementation choice |
| D-013 | Do not publish to PyPI or mutate a remote repository without separate explicit authorization | Current scope and external-write safety | Active guard |
| D-014 | Use a `src/` package layout and one canonical version source | Main-agent packaging design; aligned with current PyPA guidance | Active implementation choice |
| D-015 | Set the first branch-coverage ratchet to 85% after a reviewed 88% Stage 1 baseline; future changes may hold or raise it but not silently lower it | Main-agent local gate evidence, 45 tests on Python 3.10.8 | Active implementation choice |

## Pending algorithm decisions

The following are not accepted behavior changes yet:

| ID | Question |
|---|---|
| TBD-ALG-001 | Whether and how the MVP fills every TextGrid internal/tail gap while preserving non-`NULL` boundaries |
| TBD-ALG-002 | Whether Stage 2 should only validate the current 10 ms stride or dynamically derive it |
| TBD-ALG-003 | Whether to distinguish consecutive equal phone graph states inside one word |
| TBD-ALG-004 | Whether to add the standard repeated-target CTC blank constraint in Stage 1 |
| TBD-ALG-005 | Final duration, token, trellis-cell and beam-work limits |

Any resolution requires dedicated tests and an explicit new decision row.
