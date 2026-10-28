# Initial scaffold validation receipt

This is the earlier FP32 scaffold sweep, preserved as historical evidence.
The later ternary-pulse integrated check passed **41 Rust and 22 Python tests**;
see the archived [PULSE-RESULTS.md](../archive/docs/PULSE-RESULTS.md) for its logs
and bounded observations.
The counts and "final" wording below refer only to the original scaffold pass.

Date: 2026-10-23. Final integrated sweep completed with process exit 0 in the
isolated scaffold, using the existing Rust dependency cache and `--offline`.
No model weights, datasets or Python packages were downloaded. No live benchmark
endpoint or full-size numerical model was used.

Evidence: [final validation logs](../work/validation/1409ae660c4449458434c0ca6ec5e58d).
Logs and build products are ignored local artifacts, not intended Git contents.

## Final checks

| Stage | Result |
| --- | --- |
| Source structure, Python parsing and local documentation links | Passed: 22 source files, 11 documents |
| Rust formatting | Passed |
| Clippy, all targets, warnings denied | Passed |
| Rust unit/integration tests | 20 passed, 0 failed, 0 ignored |
| Python benchmark tests | 12 passed, 0 failed, no skips |
| Tiny, 81M and 287M allocation-free plans | All passed |
| Four-step CPU numeric fixture, seed 7 | Passed with finite results |

The Rust total comprises 2 unit, 4 CLI, 6 numerical and 8 planner tests.
Numerical coverage includes prefix causality with zero/nonzero history, route
credit direction, adversarial RNG seeds, finite state, gradient connectivity and
changes to router, pulse and curve parameters. Benchmark tests use mocks and
synthetic artifact files; they do not establish server compatibility or quality.

## Bounded fixture observation

| Quantity | Observed |
| --- | ---: |
| Actual parameters, including auxiliary weights | 22,591 |
| Optimizer updates | 4 |
| Synthetic target evaluations | 32 |
| Tile visits | 12 |
| Sampled learned halts | 0 |
| Initial reset-history evaluation loss | 5.52117538 |
| Final reset-history evaluation loss | 5.51572752 |
| Router / pulse / curve L1 parameter changes | 0.06846864 / 0.06377759 / 0.04069704 |
| In-process fixture seconds | 0.032747 |

The loss reduction is one synthetic observation, not a convergence finding.
Zero sampled halts in this run does not establish useful early-exit learning.
The elapsed time excludes build/startup and must not be extrapolated to a large
LLM or compared with a pretrained model. No durable task exposure or checkpoint
was committed; the ephemeral fixture model exited.

## Earlier attempts, retained rather than hidden

The first invocation was blocked before execution by Windows' default script
policy. An approved process-only policy override was used; machine policy was
not changed. Two harness issues were then corrected: selecting the first Python
application on PATH, and reading/resetting PowerShell's native exit code in its
global scope. A focused harness check captured exit codes 0 and 7 correctly.

The first compilation passed type checking but strict Clippy rejected a manual
divisibility expression. That single equivalent API-style fix was independently
reviewed. No Rust or Python test suite ran before the final successful sweep.
Earlier partial logs remain beside the successful receipt.

Windows PowerShell 5 may wrap ordinary native stderr (Cargo progress or unittest
output) in `NativeCommandError` formatting inside logs. The runner's actual native
exit codes and the test summaries determine success, not that wrapper label.

## Scope of approval

Separate reviewers inspected authored numerical, foundation, benchmark and
integration changes and their fixes. See [REVIEWS.md](REVIEWS.md). Software-check
success admits the scaffold only: full F parity, reduced residency, tensor fanout,
production supervision and full-model training remain unimplemented gates.
