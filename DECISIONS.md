# DECISIONS

This file separates explicit user decisions from reversible implementation
choices. An implementation choice is not rewritten as user authorization.

| ID | Decision | Source | Status |
|---|---|---|---|
| D-001 | Rebuild from a clean code baseline rather than repair the remote core in place | Current user message | Accepted |
| D-002 | Keep the remote README/product identity broadly, but re-audit every capability and command against the new implementation | Current and preceding user message | Accepted; identity, authorship and license fields are fixed by D-032, while capability and command text remains bound to the rebuilt implementation |
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
| D-015 | Set the first branch-coverage ratchet to 85% after a reviewed 88% Stage 1 baseline; future changes may hold or raise it but not silently lower it | Main-agent local gate evidence, 50 Stage 1 tests on Python 3.10.8 | Active implementation choice |
| D-016 | Vendor the authoritative script byte-for-byte as test evidence, while forbidding production imports and excluding it from wheel/sdist | Stage 2 portability and package-boundary audit | Active implementation choice |
| D-017 | Bind offline model preflight to the committed candidate manifest and exact recorded runtime; missing or mismatched prerequisites fail closed | Stage 2 E2E-preflight audit | Active guard retained; the candidate E2E and the later D-033 approved exact-wheel rerun both pass locally |
| D-018 | Implement Stage 1 as a NumPy-only internal core with exact dense-trellis accounting and an optional caller-supplied pre-allocation cell limit | Stage 3 parity and resource audit | Active implementation choice; safe default remains TBD-ALG-005 |
| D-019 | Reject non-finite internal chunk boundaries explicitly before millisecond rounding with a stable `ValueError` | Stage 3 main-agent audit | Accepted safety correction for invalid input only; valid reference behavior unchanged |
| D-020 | Implement Stage 2 as a NumPy-only graph/beam core while preserving complete-end, stable-tie, per-frame-bias, enter-cost and equal-phone current behavior | Stage 4 parity, exact-DP and main-agent cross-audit | Active implementation choice; equal-phone correction remains TBD-ALG-003 |
| D-021 | Keep the two characterized duration conversions distinct: Viterbi silence locking uses `round`, while short-gap pruning uses `ceil` | Current reference behavior and Stage 4 differential evidence | Accepted parity behavior; 65 ms at 10 ms is respectively 6 locked frames and a 7-frame prune threshold |
| D-022 | Fail closed when a local vocabulary has duplicate JSON members, the Chunker tokenizer mapping differs from `vocab.json`, or a posterior is not finite normalized log probability | Stage 5 adapter/pipeline cross-audit | Accepted invalid-model safety correction; valid reference inputs are unchanged |
| D-023 | Publish output with a same-directory atomic no-clobber hard link, then revalidate identity, exact bytes and semantics; never overwrite a concurrent artifact | Stage 5 output transaction cross-audit | Active implementation choice; cross-file crash consistency remains TBD-OUT-001 |
| D-024 | Reject transcript tokens `sil` and `null` before model loading because the current reference tier format uses them as reserved labels | Stage 5 word-identity audit | Accepted explicit limitation; a non-conflicting identity scheme remains TBD-TEXT-001 |
| D-025 | Keep Chunker and Aligner model sessions non-overlapping, CPU-only and local-only; release the Chunker before loading the Aligner | Stage 5 lifecycle audit and current user scope | Active implementation choice |
| D-026 | Treat the frozen English manifest as engineering evidence while `status=candidate`; a release E2E must fail closed unless the manifest is explicitly `approved` | Stage 6 E2E and release audit | Historical candidate phase closed by D-033; the fail-closed approval guard remains active and the approved exact-wheel rerun passes locally |
| D-027 | Pin Hatchling exactly and build with `--no-isolation` after the pinned package group is installed | Stage 6 reproducibility audit | Active CI/release choice; current backend is 1.32.0 |
| D-028 | Use the current reference/new double-run output as the E2E oracle; retain the pre-existing OpenPhonetics TextGrid only as a hashed legacy candidate | D-003 authority rule plus Stage 6 byte comparison | Accepted conflict resolution; no legacy-output expectations were merged |
| D-029 | Position the first public release as `PUBLIC_ALPHA` with PEP 440 version `0.1.0a1` | Current user review message | Accepted; `pyproject.toml` remains at the development version `0.1.0.dev0` until the approved metadata change is implemented and reverified |
| D-030 | Integrate the clean rebuild into GitHub using `REPLACE_MAIN_HISTORY`: directly replace the existing `USTCPhonetics/FlexAligner` main history rather than merge or graft unrelated histories | Current user review message | Accepted strategy; it is not authorization to configure a remote, push, force-update a branch, change the default branch, create a tag or remove recoverability |
| D-031 | Use `flexaligner` as the PyPI distribution name and `ustcphonetics` as its owner/organization | Current user review message | Accepted selection; actual PyPI availability, control and Trusted Publisher configuration remain external verification steps |
| D-032 | Use the original remote repository snapshot for license, copyright, authorship, affiliation and citation identity: MIT; `Copyright (c) 2026 WANG Yiming`; Yiming Wang and Jiahong Yuan, USTC | Current user review message; fixed upstream README and linked LICENSE at `c5361efe4b5d8ad02574dae1bd7caa89ed3e4af0` | Accepted; the README supplies the MIT identity/authors and its linked LICENSE supplies the exact copyright line; unsupported legacy capability claims are not revived |
| D-033 | Approve `openphonetics OW1 P AH0 N F AH0 N EH1 T IH0 K S` only as the frozen release-E2E test-fixture pronunciation | Current user review message | Accepted and locally reverified with the approved exact wheel; this is not a canonical pronunciation, default G2P, accuracy gold standard or permission to distribute model assets |
| D-034 | Accept the v0.1 public-preview support boundary bundle recorded below (`ACCEPT_PREVIEW_BUNDLE`) | Current user review message | Accepted; it defers disclosed research questions for the alpha but does not convert implementation, CI, remote E2E or publishing gates into passes |

## Accepted v0.1 public-preview boundary (D-034)

The approved `0.1.0a1` boundary is:

1. The only implemented alignment path is strict English, CPU, single-file,
   16 kHz mono PCM16 WAV, local models and a local full-coverage lexicon.
   Mandarin and the other nine future capabilities remain explicit placeholders.
2. The wheel carries no model and performs no automatic download. Users supply
   compatible, locally available assets.
3. Python 3.10--3.14 remains the remote fast-CI target. The only frozen real-model
   runtime evidence currently comes from Python 3.10.8, NumPy 2.2.6, Torch 2.3.1
   and Transformers 4.41.2; broader support is not implied.
4. Before public alpha, the actual `[inference]` resolver/runtime contract must be
   narrowed to an evidenced window and checked through a public-index install
   path, or the public extra must be removed. A broad Hugging Face matrix is
   deferred.
5. No broad legacy Python/CLI compatibility layer is included. A concrete adapter
   requires an identified real caller.
6. Reference-parity behavior remains disclosed: the frozen fixture's roughly
   18 ms tail gap, fixed 10 ms Stage 2 stride, same-word consecutive equal-phone
   collapse and simplified repeated-label CTC recurrence are not silently fixed.
7. Confidence remains raw and uncalibrated. Missing final phone provenance remains
   `None` rather than being inferred.
8. Lexical words `sil` and `null` remain typed pre-model input errors.
9. No approved default resource limit is claimed, and the preview is not described
   as safe for arbitrary-length untrusted input.
10. Torch CPU thread-count changes are process-global and must be documented before
    public release; host-process isolation is not claimed.

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
For `0.1.0a1`, D-034 explicitly accepts these as disclosed, deferred questions;
it does not accept any behavior-changing answer to them.
