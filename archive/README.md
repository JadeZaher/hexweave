# Archive: superseded code and records

On 2026-10-27 the repository was consolidated so the core training path is the
build (`hexweave train`, run by `scripts/core_owner.py`; see
`../src/causal/AGENTS.md` §Core path). This is a strangler-fig cut: the core path
replaced the older paths first, then everything only they reached moved here
with `git mv`, so history follows each file. Paths below `archive/` mirror the
original paths. Baseline before the cut: commit `a896ac7`; last commit with
these files in place: `8e7a0eb`.

## What moved, and why

| Original location | Files | What it was |
| --- | ---: | --- |
| `src/causal/` | 43 | The v3 owned-study chain (owner requests, launch receipts, runner, DiskStore persistence and its ledger, release/seal/retention, prompt evaluation, assay study binding), the anchor-only comparison, learning-gap and predictor diagnostics, persistence tests and profiler, and `PERSISTENCE.md` |
| `src/pulse/`, `src/pulse_cli.rs` | 13 | The bounded `pulse` ternary experiment and its command |
| `src/text/`, `src/text_cli.rs` | 19 | The disk-local `text` experiment (T1/T2 credit campaigns) and its command |
| `scripts/` | 30 | The v3 Python owner, six-arm retained-text study, pulse and text owners and campaigns, and the `speed_loop.py` comparator |
| `tests/` | 15 | Rust and Python tests of the code above |
| `docs/` | 10 | Pulse, text, training-pulse, T1 and T2 protocol and result records |

Extracted rather than moved (new files here; the kept part stays in place):
`src/causal/AGENTS-history.md` (historical sections of `src/causal/AGENTS.md`
and one `LINT.md` entry), `scripts/AGENTS-history.md` (historical sections of
`scripts/AGENTS.md`) and `src/causal/assay_contracts_checkpoint.rs` (the
DiskStore checkpoint test split out of `src/causal/assay_contracts.rs`).

Kept in place on purpose: `docs/*CONTRACT*.md` (SHA-pinned, byte-exact),
`datasets/**` (including `datasets/tiny`, which only the archived text code
reads), and the text owner's Windows helpers, renamed
`scripts/owner_resources.py` and `scripts/owner_win32.py` for the core owner.

## Archived code does not build in place

Nothing under `archive/` is compiled, tested or link-checked: Cargo, `unittest`
discovery and `scripts/check_structure.py` scan only `src`, `tests`, `scripts`,
`bench`, `examples`, root and `docs/` Markdown. Relative paths inside archived
files are as written at their original location. The cut also removed live
seams the archived code used: `hash_file` moved to `child_io`, `Terminal` to
`generation`, `checkpoint_state` kept only the state record, and items that
became dead were deleted (`config` protocol pins and owned-config validation,
most of `storage_paths`, `TrainingSource::verify_boundary`, the dense
`Predictor::train`).

## Restoring

For one file, either move it back (`8e7a0eb` content), creating the parent first
because `git mv` will not create it (for example `src/pulse/` or `src/text/`):

```powershell
New-Item -ItemType Directory -Force (Split-Path <original path>) | Out-Null
git mv archive/<original path> <original path>
```

or take the pre-consolidation baseline (identical except
`src/causal/checkpoint.rs`, which changed in `8e7a0eb`):

```powershell
git checkout a896ac7 -- <original path>
```

A restored module also needs its `mod` line and the seams above. To run a whole
archived subsystem, check out `8e7a0eb` in a separate worktree rather than
re-wiring it here.
