# Ternary pulse experiment

This directory implements `ternary-pulse-v1`, a bounded numerical experiment,
not a full F model, language-model pretraining backend or decentralized trainer.
The user's intention is to learn sparse local surfaces from useful shared
correction history, with ternary parameters from the start and explicit total
cost accounting. The hypothesis is not that a smaller working set automatically
provides PEFT-like total pretraining cost.

## Entry points and file map

`mod.rs` validates the public `Options` and calls `runner.rs`. `execution.rs`
executes one path, `node.rs` defines its local transformation and router,
`credit.rs` performs the anchor correction wave, and `state.rs` owns shared
ternary matrices. `history.rs` bounds historical correction records;
`residency.rs` bounds integer optimizer evidence. `report.rs` declares accounting
and limitations. All functions are synchronous and the process exits after the
declared finite step count.

## Model and parameter representation

The fixture has seven radius-one hex tiles, width eight, vocabulary sixteen and
at most three visits per example. Inputs are individual synthetic token IDs,
not sequences with attention. All learnable matrices are packed ternary values:
input embedding, output readout, tile transformation, tile activation control,
tile router and the shared correction predictor. There is no floating-point
parameter master, trainable scale, bias or hidden optimizer momentum.

Floating-point activations, accumulators, conditional derivatives and bounded
historical correction records are deliberately permitted. Fixed normalization
and residual coefficients are hyperparameters, not learned weights. The packed
matrix primitive uses scalar or runtime AVX2 add/subtract/skip with float
activations; this is not bitwise-only training. See `docs/PERFORMANCE-LOOPS.md`
for fixed-fixture measurements rather than a full-model speed claim.

For width `d = 8`, a tile computes:

```text
x = 0.5 * E[:, token]
z = W x / sqrt(d)
c = 1 + 0.25 a                 a is a ternary activation-control vector
y = x + 0.25 c * tanh(z)
route_logits = R [y; h] / 4
task_logits = O y / sqrt(d)
```

Each router allocates exactly one row per legal neighbor plus one halt row.
There are no unreachable global-destination rows. Routing is sampled during
training and greedy in evaluation, which always resets history. The center tile
executes first, revisits are allowed, and the third visit forces exit. There is
no numerical fanout, learned long-range edge or activation-function selector;
the activation control learns a continuous tanh amplitude through a ternary
coefficient.

## Two kinds of learning event

The first step and every `anchor_interval` thereafter are **task anchors**. An
anchor builds a frozen path, evaluates real next-token cross entropy, and
returns analytic corrections through that realized path. All output corrections
are computed before any parameter changes. Repeated visits to one tile are
summed before its update. One tile's gradient and resident optimizer evidence
are admitted at a time; shared input/output updates use the same frozen trace.

These are exact conditional-path derivatives of the continuous relaxation at
the current ternary values, not derivatives of discrete trit transitions. Route
features are detached for the policy update. A sampled action contributes
`(loss - baseline) * (one_hot(action) - probabilities) * features / 4` to its
router gradient. The baseline starts at `ln(16)` and thereafter is an exponential
average of earlier **training anchor** losses. Evaluation never enters it.
This is a score-function routing estimator, not causal evidence that the next
tile alone caused the outcome, and not the full joint gradient through router
features into upstream transformations.

Other steps in **pulse** mode stop derivative propagation at each visited tile.
The predictor uses `[incoming x; outgoing y; history h; axial q; axial r]` to
predict `d(loss)/d(y)`:

```text
g_hat = 0.5 * tanh(P features / sqrt(26))
predictor_objective = mean((g_hat - clip_norm(real_returned_g, 1))^2)
```

The tile receives its predicted correction and commits immediately, before the
next tile runs. Its forward output is carried onward unchanged. No task target
is read on those steps, and no full local-step path tape is retained. Router,
embedding and readout receive real-anchor updates only. The predictor learns
from every fresh anchor record and, when available, one compatible previous
record on each local step. It never trains against its own predictions.

Updates use signed integer evidence and adjacent ternary transitions, not SGD
into floating-point weights. This toy protocol explicitly sets threshold **one**:
each nonzero gradient sign proposes an immediate adjacent transition. That keeps
learning observable with one resident evidence slot; it does not establish a
stable optimizer. Higher hysteresis thresholds with evidence discarded on
eviction can prevent learning and require a separately measured design.
At threshold one, every nonzero update clears its evidence; resident count
changes allocation and reset accounting, not preserved optimizer hysteresis.

## Historical signal, versions and causality

The shared history is a FIFO of at most eight completed anchor records. Each
stores source node, source/shared weight versions, a conservative version stamp
for **all seven tiles**, step, predictor features and returned correction. Its
summary is the mean returned correction, not a target-free latent state or a GAN.
Predictor inputs never include the current example's target or returned error.

Records are accepted for at most eight step indices and at most one version of
lag for every tile and the shared embedding/readout. One-version-old records
are intentionally approximate, not current exact gradients. Version checks run
before replay and again before each immediate local tile update. Recording all
tile dependencies also invalidates on unrelated changes; this is conservative.
The predictor itself changes without invalidating historical supervised pairs.
The records are examples to learn from, not gradients blindly reapplied to a new
weight version. Unknown/nonfinite arithmetic fails the bounded run.

## Residency and cost

All packed weights remain in host memory. Only per-tile integer optimizer
evidence is limited to one through three residents. LRU eviction discards that
evidence, and the report counts resets. Shared embedding/readout/predictor
evidence remains resident in both modes; even an unused anchor-only predictor
still incurs its explicitly reported state allocation. No disk offload or GPU
placement claim follows from this fixture.

`state_bytes` records logical payloads and declared bounds, not RSS. Container
capacity, allocator overhead, stack and temporary arithmetic copies are excluded
and named. `counts` includes tile visits, returned frames, route calls, predictor
forward/training/replay operations, target exposure, exits and coverage. Matrix
dimensions are reported rather than invented FLOP-equivalent speed estimates.

**anchor-only** is a periodic-anchor ablation: it skips non-anchor execution
entirely, and does not train the predictor or carry history. It is not an
ordinary every-example full-backprop pretraining baseline. Both modes receive
the same initialization, deterministic anchor inputs and per-step random route
seed, independent of the number of random draws on preceding local steps.

Anchor input IDs cycle through 0..11 once per twelve anchors. Local inputs use a
deterministic phase offset within that cycle; this avoids cadence/data aliasing.
Evaluation uses only IDs 12..15, with target `(token + 1) % 16`, four examples
before and four after training. Those input columns receive no direct training
updates: this is a weak synthetic heldout diagnostic, not language capability
or a corpus generalization benchmark. Declared steps, targets, visits and
elapsed time must remain distinct; skipped anchor-only steps do no training.

## Review and experiment discipline

Do not tune from heldout results, claim efficiency from a small fixture or start
a larger model from this command. Test finite-difference local derivatives,
stale dependency rejection, evaluation immutability, direct trit updates,
bounded residents and correct target counts. Any eventual performance claim
needs the same useful quality threshold and data budget, charging predictor,
anchor, replay, communication and residency costs together.

Related ideas, not equivalence claims: [synthetic gradients](https://arxiv.org/abs/1608.05343)
and [binary optimizer evidence](https://arxiv.org/abs/1906.02107). The latter is
not a proof that this threshold-one ternary update trains useful language models.
