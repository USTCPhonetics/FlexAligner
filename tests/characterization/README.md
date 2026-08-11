# Characterization and oracle policy

The immutable file in `reference/` is test evidence only. Production code must
never import it, and neither wheel nor sdist may contain it.

Tests are kept in three distinct classes:

1. **A — reference parity:** characterizes behavior of the current authoritative
   local script, including named quirks. Rebuild code must match until a separate
   decision approves a correction.
2. **B — independent correctness:** small exhaustive or hand-derived oracles
   assert invariants without treating the reference implementation as proof.
3. **C — approved correction:** a decision record names the old behavior, the
   desired behavior, compatibility impact, and new tests before production
   behavior diverges.

There is no `--update-golden`, snapshot rewrite, or comparator-side acceptance
mode. A mismatch must show the first differing field/index. Expected data may be
changed only in the same reviewed change that updates `DECISIONS.md`, and a
failing test must never be made green merely by regenerating output.
