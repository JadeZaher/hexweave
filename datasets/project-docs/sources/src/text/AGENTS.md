# Disk-local ternary byte experiment: text-pulse-v3

`text::run` is the bounded public entry point. `runner.rs` coordinates data,
evaluation, local visits, anchors, STOP and final inspection snapshots.
`storage.rs` is the only tile owner: it loads exactly one decoded `Node` inside
a synchronous closure, then drops it before the next load. `shared.rs` owns the
always-resident embedding, vocabulary readout and correction predictor.
`training_pulse.rs` owns the per-tile joint context trunk and stacked heads;
`gradients.rs` joins their task and routing credit before one pulse update.
`node.rs` implements transforms, continuous curves and learned hex routing;
`execution.rs` walks paths; `credit.rs` returns frozen-path corrections;
`history.rs` owns bounded version-checked demonstration history.

## What this preserves, and what it does not

All learned coefficients are packed ternary from initialization, including
embeddings, readout, transform, three activation knots, router, correction
predictor, and both joint-pulse matrices.
There are no float master parameters. Activations and conditional derivatives
use floating point. Threshold-one signed evidence moves each trit one adjacent
step; this research optimizer has no established language-model convergence.

This is a 256-byte vocabulary, width-16 experiment with an eight-byte causal
context. Input is a fixed position-weighted sum of embeddings, so changing a
future byte cannot affect current prediction. There is **no attention**. It is
not the planned 81M/287M attention architecture or an exact F port.

Each of seven hex tiles owns a square transform, three learned ternary knot
offsets per channel, one router row per immediate neighbor plus halt, and a
joint training pulse with its own ternary context and stacked-head matrices.
The center runs first, hard learned routing may revisit tiles, and the third
visit forces exit. There is no monotonic layer schedule, fanout, concurrent
wave, GPU placement or automatic resume.

The continuous piecewise-linear curve uses fixed
abscissae `[-2, 0, 2]` and ordinates `[-1, 0, 1] + ternary_offsets / 4`.
Endpoints extrapolate linearly. Knots therefore train directly through their
continuous relaxation; they are not the source F experiment's sampled actions.
The learned router reads `[y; history]` plus the joint pulse's route bias;
hard sampled choices receive
`(task_loss - previous_training_baseline) * grad(log probability)` at anchors.
Routing features are detached, not a claim of full joint path-policy gradient.

## Joint training pulse

The forward entry point is `TrainingPulse::forward` in `training_pulse.rs`.
One per-tile context trunk produces a shared latent, and one stacked projection
produces three control heads. `Node::forward` applies the transform and curve
heads; `Node::route` applies the routing head from that same cached pulse.
With hidden width `d = 16`, legal route count `r` (neighbors plus halt), incoming
state `x`, and detached version-checked history summary `h`:

```text
f = [x; h]                                      length 2d
a = tanh(U f / sqrt(2d))                         U: d by 2d
p = V a / sqrt(d)                               V: (2d + r) by d
gain = 1 + 0.125 tanh(p[0:d])                    range (0.875, 1.125)
shift = 0.125 tanh(p[d:2d])                      range (-0.125, 0.125)
z = (W x / sqrt(d)) * gain + h / 8
y = x + 0.25 * (curve(z) + shift)
route_logits = R [y; h] / sqrt(2d) + 0.25 p[2d:]
```

`U` and `V` are ternary from initialization. The activation controls are
continuous functions of their shared latent; this is neither a binary gate nor
an activation-selector implementation. Route logits are finite but not clipped
by a tanh head. Existing normalized kernels handle the matrix operations through
the selected SIMD/scalar dispatcher; no new custom kernel is claimed.

**The joint training operation is `Gradients::add` in `gradients.rs`.** It adds
task credit from gain and curve shift to the same head-correction vector that
receives sampled routing score credit. `TrainingPulse::add_gradients` then
backpropagates that combined vector through `V` and the shared tanh trunk `U`.
The default coupled arm trains shared parameters from both signals.
`Node::apply` commits the transform, curve,
router, pulse context and pulse heads under one node version.

Returned task corrections include the direct residual, transform/gain path,
and pulse-context-to-input path. Routing score updates train `R`, the route rows
of `V`, and `U`, while treating `[y; h]` and pulse input `[x; h]` as detached
observations for policy credit. Thus policy gradients do not return into earlier
tiles or through `y` into local transform/curve parameters. This explicit
conditional policy estimator is not a full joint score/path derivative.

On local steps, the predicted task correction trains `U` and the gain/shift
rows of `V`; it does not fabricate a route advantage or update route-head rows
directly. Nevertheless changing shared `U` changes future route-head outputs.
Likewise a route-only update of `U` can change future gain and curve controls.
This is the intended coupling, not proof of improved learning or stability.

### T1 route-to-U credit ablation

`Options::route_credit` records `coupled` (A) or `route-to-u-detached` (B).
`Gradients::add` computes exactly the same task and policy head credit for both.
`TrainingPulse::add_gradients` always trains all V rows, then B zeroes only the
route rows of the correction used by `context_derivative` to train U. R updates
and returned task corrections are unchanged. Local visits have no route advantage,
so their task-only update uses the same computation in both arms. Initialization,
parameter count, forward geometry, and tile serialization are unchanged from v2.
See [the predeclared experiment contract](../../docs/T1-CREDIT-EXPERIMENT.md).

The explicit predictor-error diagnostic divides the accumulated per-update MSE
by `predictor_updates`, including replay. Route counts cover training paths only;
I/O counts cover initialization, training and evaluation, with initial writes
separately identified. These scopes must not be mixed when interpreting costs.

The existing globally shared correction predictor is **separate** from this
per-tile conditioning pulse. It still maps input, pulse-conditioned output,
history and position to a predicted output correction. It does not become a
fourth head or a resource critic. Counterfactual exits, learned prefetch,
decentralized synchronization and GPU execution remain unimplemented.

## Locality and two correction phases

Non-anchor pulse visits predict a local output correction from
`[incoming; outgoing; compatible_history; axial_q; axial_r]`, commit that tile,
release it, and continue using its already-computed output. Later tile weights
are not loaded to produce that update. No target is read for local updates and
no local path tape is retained: only visited-ID metadata and current state.
The predictor is trained only against real anchor corrections, never itself.

Anchors freeze the realized path and shared weights. Their tape contains at
most three activation frames and version numbers, never tile weights. The
reverse phase reloads one tile at a time and returns analytic conditional-path
derivatives. Only after **all** corrections exist are updates committed; repeat
visits to one tile are summed before a single commit. The shared embedding and
readout then update. This is a surrogate continuous derivative evaluated at
ternary parameters, not an exact derivative of a discrete transition.

Frozen anchor frames now also retain pulse features, latent/head activations and
the unmodulated transform output. They never retain `U`, `V`, or any other tile
weights. One node version covers all five matrices, so existing history and
frozen-adjoint checks also cover pulse parameter changes.

History holds at most eight completed anchor demonstrations. Every record
includes all seven tile versions, shared version, features, correction and step.
Records expire after eight step indices or more than one version of lag in any
dependency. A single compatible record may train the predictor on a local step.
These lagged examples are approximate supervision, not exact current gradients.

The anchor-only mode skips non-anchor steps altogether, leaves the predictor
unchanged and uses zero history. Both arms share deterministic anchor contexts,
initialization and per-step sampling seeds. This is a periodic-anchor ablation,
not conventional every-example dense backpropagation.

## Storage, resources and artifact meaning

Runtime directories must be fresh children of the current workspace's `work/`.
Every tile commit writes a schema-2 versioned JSON file capped at 8 KiB, validates its
schema/identity/shape/trits/checksum on reload, and publishes using a synced
temporary file plus rename. At most `7 + 3 * completed_steps` tile files exist.
Old versions are retained for inspection, not silently deleted. FNV-1a detects
accidental corruption, not malicious edits or cryptographic authenticity.
The checksum includes both pulse matrices. Schema-1 tile files are rejected;
there is no checkpoint migration. Schema-3 final snapshots identify
`text-pulse-v3` and record the route-credit option; tile/shared schemas remain 2.
Snapshots remain inspection-only and cannot resume.

The added pulse matrices contribute 7,664 ternary coefficients across seven
tiles, for 19,776 total coefficients. Shared state remains 8,992 coefficients
(2,248 packed bytes). The largest tile has 1,664 coefficients (416 packed
bytes), plus its bounded temporary gradients/evidence. The transition array
order is transform, curve, router, pulse context, pulse heads; named report
fields repeat that mapping. Initialization reserves five distinct matrix seeds
per tile, so v2 is a new numerical experiment, not v1 output-equivalent.

`peak_live_tiles = 1` means one owned decoded tile, not one allocation: load/save
temporarily hold serialized bytes and packed copies of that same tile. Shared
weights and evidence, small data files, activation tape, gradients and history
remain in memory and are explicitly distinguished. OS file caching may retain
inactive file pages; this process neither pins nor evicts those pages. Payload
counts are not physical RSS measurements or a production memory admission.

STOP is checked at each step opportunity. A stopped run skips final evaluation,
writes an inspection snapshot and exits; it does not retry. Errors return to the
supervisor and partial artifacts remain for diagnosis. Final shared/snapshot
files are **not resumable checkpoints**: replay/history/cursor restoration and
whole-update transactional recovery are unimplemented. Training duration includes
disk I/O and progress records; initialization and evaluations are separately
timed. Disk counters include all phases, with initialization bytes identified.

The owned supervisor must enforce physical memory, deadline and process cleanup.
This library itself only bounds dimensions, steps, file size and ownership.
Never infer production readiness or speedup from these payload bounds.

## Data and evaluation

Use only the small newly authored `datasets/tiny` documents for the initial run.
Files are capped at 64 KiB, UTF-8 validated, document-separated and identified by
byte count plus noncryptographic FNV-1a. Evaluation reads at most 64 successive
held-out contexts, resets history on **every** example, never updates, and checks
shared-weight identity and all tile versions before/after. Do not tune on this
diagnostic or claim useful language quality from it.

Keep source files under 300 lines where feasible, separate functions with blank
lines, and put rationale here rather than verbose inline comments. Add all fixes
before the final integrated checks; obtain independent review before any run.
