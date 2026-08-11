# FlexAligner algorithm reference snapshot

`align_single_cpu.py` is an immutable, byte-for-byte evidence snapshot of:

```text
/Users/yiyi0369/projects/flexaligner/align_single_cpu.py
```

Verified snapshot identity:

- SHA-256: `9ed4e21e615718ddfd10930359f55769fb27a0d284599cce45a3fc755e835de1`
- Lines: `2548`
- Bytes: `96230`
- Copy mode: byte-for-byte; no formatting or source edits

The current local source and this snapshot are the behavior oracle for parity
migration. The remote repository snapshot
`USTCPhonetics/FlexAligner main@c5361efe4b5d8ad02574dae1bd7caa89ed3e4af0`
is the README, identity, license-provenance and comparison source; its historical
core implementation is not the new algorithm authority.

This file is evidence, not production code:

- `src/flexaligner` must never import or execute it;
- wheels and source distributions must exclude `reference/`;
- a snapshot change requires a new verified hash and an explicit decision;
- characterization tests may import it only through the isolated loader, which
  stubs Torch and Transformers and restores `sys.modules` afterward.

An older session described continuous TextGrid gap validation as already fixed.
The current authoritative snapshot does not reject leading, internal or trailing
gaps when intervals otherwise remain ordered and in bounds. Characterization
records that conflict as a known limitation; it does not silently repair it.
