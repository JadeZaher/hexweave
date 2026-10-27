# T2: observe task and route credit interference in U

Predeclared 2026-10-24 (America/Denver), before implementing this diagnostic or
observing its development results. T1's six-run budget is consumed and its
held-out outcomes must not be tuned on. This is a new bounded diagnostic, not
an optimizer change, a replay of T1 or permission to scale.

## Fixed question and data

Does routing credit change threshold-one U updates against the task correction,
and does route-to-U detachment retain its quality advantage on a fresh authored
development document? A is `coupled`; B is `route-to-u-detached`. Initialization,
forward geometry, parameter count, optimizer, actual gradients, route RNG, task
exposure and update order remain unchanged. The executable default stays coupled.

Training uses the unchanged `datasets/tiny/train.txt`. T2 evaluation must use only
the new authored `datasets/tiny/development-t2.txt`, never `valid.txt`. Its first
72 ASCII bytes are frozen as:

```text
Copper robots sort quiet notes while lanterns mark a narrow garden path.
```

Evaluation consumes 64 contexts of eight bytes and their next-byte targets,
therefore exactly bytes [0,72). The full document and that prefix receive SHA256
identities before execution, and the numerical receipt identifies the consumed
prefix, offset, byte count and example count. Authoring this tiny development
document does not supply a representative corpus or a language-quality benchmark.

Frozen identities (278 document bytes, 72 consumed prefix bytes):

- Development SHA256: `e3ae44c3a53ae682cbf42ed18a40659b29d1a3b307afb25ecb1bfd5882117032`.
- Prefix SHA256: `40698f4235a3ab2316155ac9b8762b103d0a49bca5b479485b3b75119260e2c2`.
- Document FNV-1a64: `dcae5afcc74a9d72`; prefix FNV-1a64: `1ec64f408ae9ca39`.
- Unchanged train SHA256: `b58add621357506a1eded9867080a76f7015789067ed089fc47818e424230f8a`.

The explicit profile is `t2-credit`, with `--credit-diagnostics on`. Diagnostic
off remains available only for observational equivalence checks on the same
profile/data; it cannot silently become T1 or evaluate the old held-out file.
Protocol and inspection-snapshot schema advance to v4 to require these identities.
Tile/shared schema 2 and non-resumability remain unchanged.

## Observational computation and denominators

At each anchor, gather all corrections against frozen versions as before.
For each unique selected tile, aggregate repeated visits in their original order.
Compute the actual gradient exactly as before, then independently shadow the
same `Gradients::add` calls into task-only (route-to-U detached), coupled, and
route-only (zero task correction, original advantage) gradients. Shadows never
replace actual arithmetic, load another tile, draw RNG, reorder commits or mutate
weights/evidence. All measurements occur just before the original `Node::apply`.

Each unique tile commit has 512 U-coordinate opportunities. Record per tile and
run: frame count, unique tile commit count, repeated-visit commit count, coordinate
opportunities, task/route/both nonzero and opposite-sign counts, task-route dot
product, task and route squared norms. Raw float accumulations are f64. A zero
denominator produces a null ratio, not an invented zero. Record the maximum
absolute `coupled - task - route` residual; finite checks are mandatory. The
residual is descriptive floating-point decomposition error, not a new gradient.

For each frozen U trit w, predict the exact reset-evidence threshold-one delta:
`delta(g,w) = clamp(w - sign(g), -1, 1) - w`, with sign(0)=0. Record task,
coupled and actual transition counts, different task/coupled results,
task-opposing coupled changes (`g_task * delta_coupled > 0`), suppressed task
changes (`delta_task != 0` and `delta_coupled == 0`), and new changes where
`g_task == 0`. Record these raw f64 sums:

```text
task_descent = -sum(g_task * delta_task)
coupled_projection = sum(g_task * delta_coupled)
deficit = sum(g_task * (delta_coupled - delta_task))
```

Count anchors with any different task/coupled result once per anchor, regardless
of tile count. Per-tile counts are separate. Reconcile predicted actual U changes
with `Node::apply`'s returned U transition count after every commit and in the
receipt. Preserve all existing route/exposure/I/O/predictor/residency diagnostics.
The diagnostic observes anchor credit only; local updates remain in the existing
transition totals and are not silently included in these anchor denominators.

## Fixed campaign and decision

Six sequential runs maximum: seed 7 A/B, seed 17 B/A, seed 29 A/B. Each has 256
opportunities, interval 4, pulse mode, 64 targets, backend auto, diagnostic on,
the same frozen train/development inputs and a 30-second whole-run deadline.
The campaign allowance is 240 seconds, including owner work. Every child uses
the existing supervised Windows Job/STOP path. No retry, auto-resume, optimizer
tuning, extra seed, external data fetch or follow-on training is authorized by
this contract. Failure, identity/finite/accounting mismatch and STOP are terminal.
Late persistence follows the existing single corrective failure-write rule;
external process wall is recorded separately from self-reported write boundaries.

Quality improvement is initial minus final development NLL. B advantage is B's
improvement minus A's improvement. Decide only after all six valid receipts:

- **GO, provisional:** every seed has B improvement >= 0.01 nats/byte and B
  advantage >= 0.01; every A run has at least 16 of 64 anchors with differing
  task/coupled results and `deficit / task_descent >= 0.10`; all integrity checks
  pass and task_descent is positive. This supports only considering the detached
  candidate for the next T3 design gate; it does not change the executable default.
- **REVISE:** all B improvements are <= 0, or all A advantages are >= 0.01.
  This diagnostic supports no candidate selection; any separate T3 experiment
  remains unsupported by this result.
- **INCONCLUSIVE:** every other valid complete outcome. Failure has no scientific
  decision. T3 implementation may still proceed with an explicitly experimental
  detached candidate, without calling it validated; this does not authorize a
  larger run or another tiny optimizer loop. No automatic retry or tuning follows
  any outcome. The 30-second bound stays fixed.

Recorded training and child wall times include diagnostics. All six children
enable diagnostics, so their arm ratios do not estimate incremental instrumentation
overhead. No separate timing experiment or extra numerical run is implied.
Structural gradient payload accounting distinguishes the largest actual-update
bundle (6,656 bytes), enabled shadows (three bundles, 19,968 bytes) and concurrent
bundle total (26,624 bytes). Off has zero shadows. Separate evidence and excluded
scratch/container/runtime costs mean these counts are not complete heap or RSS.

Two independent review cycles and one integrated software sweep precede this new
budget. Authored fixtures must cover on/off exact state/routes/exposure in both
arms with repeated visits, cancellation/saturation/zero-denominator metrics, and
fail-closed profile/data/protocol/metric receipts and campaign late STOP/deadlines.
No result or viable-assistant training-time estimate exists at declaration time.

After independent review and the final integrated check, the explicit entry point
is `python -B -m scripts.text_campaign --profile t2-credit`. An omitted or different
campaign profile fails before launch. Single-run fixtures retain their existing
default profile, coupled updates and diagnostic-off behavior; that is not a T2 run.
