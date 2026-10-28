# Scoped lint policy

These allowances retain the already exercised interfaces and allocation layout.
They do not suppress warnings for the crate or the complete causal namespace.

- Shared fixture files are intentionally mounted separately in test suites so
  their private relative imports and per-suite fixture state stay isolated.
  Individual declarations allow duplicate_mod; suites that use only part of a
  full fixture also allow dead_code on that fixture module. No production module
  receives either allowance through this mechanism.
- Step::Committed retains its diagnostic metadata; production accounting reads
  canonical State counters. State::validate remains a complete Dataset-specific
  inspection convenience while execution checks generic sources separately.
- Forward.latent remains an unused diagnostic return in the same temporary
  layout. Its field-level dead_code allowance applies in library and test builds.
- Items used only by tests (the assay observation publisher and its file helpers)
  are `#[cfg(test)]`, not `allow(dead_code)`.
- Divisibility expressions use is_multiple_of. The anchor divisor is 4; core run
  validation requires positive checkpoint_every and eval_every.

Archived entries (owner-chain Stop/EvaluationReport layouts) are in
`../../archive/src/causal/AGENTS-history.md`.

[Clippy's duplicate_mod implementation](https://github.com/rust-lang/rust-clippy/blob/rust-1.94.0/clippy_lints/src/duplicate_mod.rs)
honors the lint level of each module declaration. These changes require the
targeted final lint check; this policy document is not a claim that it passed.
