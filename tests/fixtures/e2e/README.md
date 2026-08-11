# English real-model E2E candidate

This directory records a frozen, repository-local lexicon for the existing
OpenPhonetics synthetic English fixture. It is test evidence, not a default G2P
implementation and not a claim that the pronunciation is linguistically
canonical.

The transcript is:

```text
This synthetic example shows openphonetics word and phone alignment.
```

Provenance:

- `THIS`, `SYNTHETIC`, `EXAMPLE`, `SHOWS`, `WORD`, `AND`, `PHONE`, and
  `ALIGNMENT` retain all matching rows, in source order, from
  `/Users/yiyi0369/projects/openphonetics/word.dict`, whose SHA-256 is
  `f6548978de94dfdcfa4c4503c0d3983fd1a4a59fe6497c4f1e1d490fd08a801b`.
- `openphonetics OW1 P AH0 N F AH0 N EH1 T IH0 K S` is frozen from the
  example in the current OpenPhonetics `README.md` (SHA-256
  `f205aae389de76d6bbd9817e39095a567dea2fb2c533faee64bfaf29a6838017`)
  and `PRODUCT_REQUIREMENTS.md` (SHA-256
  `b048e5a82e8266eb088ce84d5b2b2b1a2658f52ed1ff983fc295e1060c29bd62`).
- The source directory has no Git metadata, so a source commit cannot be
  established. Hashes, not an inferred revision, are authoritative here.

`asset_manifest.json` uses `FLEXALIGNER_E2E_ASSET_ROOT` as its root. For the
current local candidate layout, set it to `/Users/yiyi0369/projects`. A trusted
runner may reproduce that relative layout under another absolute root. Missing
variables, files, or hash mismatches must report `MODEL_E2E_BLOCKED`; they must
never become a passing skip.

The approval status of this fixture-specific OOV pronunciation remains tracked
as `TBD-E2E-001`. It may support a provisional engineering E2E, but it does not
activate the public default-G2P capability.
