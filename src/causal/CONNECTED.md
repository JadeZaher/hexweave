# Connected local training

This describes `causal-connected-v5` (FP32 forward: `causal-connected-fp32-v5`).
It differs from v4 by the
[label-conditioned credit predictor and gain-scaled decay](#label-conditioned-credit-predictor-and-gain-scaled-decay-v5)
below; v4 differed from v3 by the
[residual normalization and readout prior](#residual-normalization-and-readout-prior-v4),
v3 differed from v2 by the
[scale control, saturating curve, readout bias and per-anchor local steps](#scale-control-saturating-curve-readout-bias-and-per-anchor-local-steps-v3),
and v2 differed from v1 only by the
[calibrated predictor](#calibrated-predictor-and-trust-scaled-local-updates-v2).
v4, v3, v2 and v1 runs remain reproducible only through their pinned binaries. Read
`docs/STATUS.md` for verification and run evidence. The earlier anchor-only
`causal-latent` training remains a reference. It does not test the
predictor-driven local mechanism described here.

## Learning loop

The model retains the seven-tile, width-128 byte-level candidate geometry.
Eight logical lanes share one memory region; at most four CPU workers can own
distinct decoded tile/Adam bundles. A lane starts at the center, visits one tile
at a time, and ends at halt or four visits. Colliding tile requests queue.
Hex geometry constrains routes; it does not supply eight independent neighbors.

An anchor predicts a byte before observing its target. A frozen reverse pass
provides real correction vectors, embedding credit and route-performance credit.
These observations fit a small FP32 correction predictor and the router.
Actual anchor credit does not train the tile's main transformation matrices.
After the embedding update, the visible prefix is embedded again. Local lanes
then train on that same already-observed prefix using predicted corrections.
The local function computes no loss and performs no full-route reverse pass. From
v5 its predictor reads the embedding feature of the anchor's already-observed
target byte ([v5](#label-conditioned-credit-predictor-and-gain-scaled-decay-v5)).
Its predictor and memory contain information learned from observed targets; this
is supervised learning with predicted local credit.

Each worker reads a fixed memory/predictor snapshot for its dispatch group,
forwards one tile, predicts correction and selects a legal next action from that
forward result. It then differentiates the tile locally and returns
the gradient without updating the tile (v3, see
[per-anchor local steps](#scale-control-saturating-curve-readout-bias-and-per-anchor-local-steps-v3)).
Complete records publish in lane order within each worker group. Subsequent
groups read them during the same local phase. No lane observes a partially
written record. After all lanes finish, each visited tile with nonzero trust
takes one update, the cursor advances and the state may checkpoint.

Eight consecutive future byte targets must not share unrestricted memory in
parallel: later prefixes contain earlier targets. The first protocol confines
concurrent lanes to the same observed-data boundary. Failure mid-step poisons
the state; it cannot create a resumable partial learning step.

## Full-precision learning and quantization comparisons

Owner direction on November 4 starts a fresh FP32-forward reference before
reducing weight precision. Select `connected.precision: "fp32"` for
`causal-connected-fp32-v5`; omitted or explicit `"ternary"` selects
`causal-connected-v5`. The precision belongs to run/checkpoint identity. This
is FP32 stored weights and activations with the existing FP64 Adam
intermediates, not an FP64-everywhere implementation. The predictor, shared memory, learned curves,
local correction updates, routing policy, gains and ownership rules remain.
Evaluation rows report `forward` as the store's precision. Executables pinned
before November 5, including the stopped `connected-fp32-full-s7-20261104` run,
label every connected evaluation `"ternary"`; read `protocol`/`precision` there.
Anchor-only `causal-latent-v2` uses `"float"` for its own FP32 forward, so
compare that field only within one protocol.

Continuous master weights are initialized before the first forward computation.
Explicit views supply every matrix product, transpose, attention row block,
curve coefficient and tied embedding/readout. They do not use the older
thread-local float-forward fallback. Continuous updates refresh the views and
advance numerical versions even when no stored trit changes. Checkpoint restore,
evaluation replay and completion reconstruct views from verified canonical
master weights; the active FP32 function is covered by checkpoint/state hashes.

Each resident bundle may hold one extra FP32 coefficient view, and shared E holds
one global view. The four-bundle limit still counts cached and leased tiles
together. For four distinct candidate tiles including the center, the extra
tile arrays contain 4,504,064 bytes (4.30 MiB); shared E adds 131,072 bytes.
Completion retains the source state and one replay state, so their shared-E
views together use 262,144 bytes; each sequential comparison reuses that bound.
These are array counts, not a process-memory guarantee. No all-tile continuous
weight map is retained. The pilot must measure actual process memory.

`complete --quantize-audit` on an FP32 checkpoint compares three frozen weight
views: FP32, symmetric INT8 and scaled ternary. The INT8 scale is each matrix's
maximum absolute master weight divided by 127; ternary uses its mean absolute
master weight. Values are rounded to the nearest level and clamped to the
corresponding range, then multiplied by the scale; all-zero matrices stay zero.
Tile matrices and the shared embedding/readout are included; the correction
predictor, the v3 readout bias and activation arithmetic remain FP32. Each view
uses identical prompt warmup policy and fresh scratch memory. The checkpoint,
master weights, Adam, predictor and gains remain unchanged.

These comparisons execute FP32 arithmetic on rounded weights. They are neither
native FP8 kernels nor compressed checkpoint exports, and do not perform
quantized fine-tuning. A 120-second request budget is divided among the views;
truncated variants are labeled and cannot establish a completed comparison.
Trace output and source fingerprints permit independent checking. Compare
prediction sensitivity and free generation as well as losses before choosing a
lower-precision training or deployment stage; no automatic promotion is defined.

Implementation/validation and launch outcomes are recorded in `docs/STATUS.md`;
this design section alone is not evidence that the new study has run.

## Shared signal and learned consumers

The one memory region holds at most 32 records by default, each with input,
output, historical input, correction, tile identity, runtime revision, embedding
revision, event epoch, lane identity and evidence type. The types distinguish
observed corrections, predicted corrections and generated-byte observations.
Only observed actual corrections may fit the predictor or calibrate gains.

Retrieval rejects future records, event age over 64, source-tile revision lag
over eight, or embedding Adam revision lag over `connected.memory_lag` (default one;
see [memory reach](#memory-reach-and-visit-normalization-switches)). Packed-trit versions alone
are insufficient because latent weights can change without changing a trit.
For eligible records its weight is

`exp(3*(0.75*cos(query,input)+0.25*cos(query,output))) * provenance * locality / (1+age)`.

Provenance weights are 1 for observed, 0.25 for predicted and 0.125 for generated
evidence; locality is 1 for the queried tile and 0.75 for other tiles. The
weighted correction mean is unit-clipped and detached. This is an explicit
initial retrieval rule, not a claim of optimal memory addressing.

The retrieved vector conditions the learned pulse and routing matrices. In
connected mode, the first three learned pulse heads additionally produce
`c[k] = 0.125*tanh(head[k])`, added to the corresponding three activation-knot
heights across FFN channels. Knot-control adjoints add to pulse-head credit and
continue through P and U. Per-channel knot weights remain trainable. This
changes curve shape, beyond the existing pulse's final FFN gain and shift.

The FP32 predictor takes normalized tile input/output, detached retrieved memory,
the observed target byte's embedding feature (from v5) and hex coordinates. It
learns clipped returned corrections at a per-tile
normalized scale and emits the calibrated correction defined below. Prefit
error, zero-predictor error and cosine compare that calibrated correction with
the target before current-target fitting. Learned local credit follows the idea of
[synthetic-gradient interfaces](https://proceedings.mlr.press/v70/jaderberg17a.html);
that precedent does not validate this particular architecture.

## Lambert W and update direction

Each tile starts with gain one. From prefit observed metrics define
`q = 0.5*cosine + 0.5*clip(1-MSE/(zero_MSE+1e-12), -1, 1)`.
Update `ema = 0.75*ema + 0.25*q`, then `u = 6*(0.75*ema + 0.25*q)`.
The saved gain is `clip(exp(sign(u)*W0(abs(u))), gain_min, gain_max)`, with
default bounds 0.25 and 4. This permits damping and bounded amplification.
Positive-domain principal Lambert W uses safeguarded Newton iteration; see
[NIST DLMF §4.13](https://dlmf.nist.gov/4.13).
Reference mean-square energy at or below `1e-12` supplies no directional gain
evidence: it neither increases the gain nor refreshes the previous observation.

The log-gain decays by `1+age/16`, returns to neutral after 64 events or source
revision lag over eight, and applies only to later work. It is an evidence-based
controller, not a fitted gain network. Predicted evidence never calibrates it.

Adam first computes the proposed continuous tile update. One positive gain and
one uniform total-norm cap scale the entire selected update vector. Coordinates
are not individually clipped. This preserves the continuous proposal's direction;
ternary projection can still change the pattern of discrete transitions. A
single SIMD broadcast applies the group gain where supported, with a scalar
fallback and numerical parity checks. Matrix products reuse existing packed
scalar/AVX2 kernels. Neither sharing a signal nor using SIMD establishes speedup.

Actual route updates touch ROUTER and P's route rows only. They have independent
Adam clocks; P route rows use an auxiliary optimizer slot. Local transformation
updates exclude those route parameters. This avoids unrequested movement from
old Adam momentum in a supposedly frozen component.

## Calibrated predictor and trust-scaled local updates (v2)

Protocol v2 makes the predictor learnable as one behavior, with no option. The
v1 network trained by Adam at a fixed rate directly on raw corrections. Real
corrections shrank to norm about 0.01, but Adam's step is scale-free, so each fit
moved the output by about the same amount whatever the target size. Across both
v1 FP32 studies the prefit error was 12-189 times the zero-prediction error at
cosine 0.002-0.04, and predicted memory records were about 100 times larger than
observed ones.

`connected_calibration` keeps per-tile statistics over evidence-bearing observed
fits: mean target energy `E|t|^2`, alignment `E[p.t_hat]`, prediction energy
`E[p.p]` and a count. Each is a running mean for the first 1,000 observations,
then an EMA with weight 0.001 (0.01 in v3). The network output
`p = unit_clip(tanh(W f / sqrt(F)))` trains on the unit-scale target
`t_hat = t / nu_k`, where `nu_k = sqrt(E|t|^2)` comes from earlier observations
(the first uses its own norm); v3 clips `t_hat` to norm 4.
Trust `clamp(E[p.t_hat] / max(E[p.p], 1e-12), 0, 1)` is the least-squares
shrinkage slope of `t_hat` on `p`, zero before a tile's first observation.
Lanes, published predicted records and prefit metrics all use the calibrated
correction `nu_k * trust_k * p`, so the Lambert gain sees calibrated evidence.
A target at or below the skip threshold (the gain's `1e-12` mean square in v2;
effectively zero, mean square at most `1e-30`, in v3) still counts in `fits` (and the counter
invariants) but changes no weight, Adam moment or statistic; the predictor's
Adam bias correction counts evidence-bearing fits.

The local update multiplier is the Lambert `value_at` gain times `trust_k`. Zero
trust skips local differentiation and the update while the job still forwards,
routes and publishes its record, so state validation requires only
`local_updates <= predicted_records`. In v3 the multiplier scales one update per
visited tile per anchor, not one per visit. Route updates from actual anchor
credit keep the plain Lambert gain. Logs and reports show that gain as `gains`
and the multiplier separately as per-tile `trust`; `local_boosted`/`local_damped`
compare the applied product with one.

Because prefit metrics now score the shrunk calibrated correction, its error
rarely exceeds the zero-correction error, so the Lambert gain rarely falls below
one. Route updates (`connected_step`) use that gain undamped, whereas v1 route
updates were damped by poor raw predictor metrics; account for this when
comparing v2 results with v1 runs. While a tile's trust is zero its predicted
records carry zero corrections: retrieved direction then comes only from
observed records, but those zero records still occupy FIFO memory capacity and
retrieval weight, shrinking retrieved magnitude and curve conditioning, and
`cross_lane_reads` counts only nonzero records.

Without the target byte (v1-v4), the predictable part of a correction is its
conditional expectation given the other features, so low single-sample cosine is
expected even from a good predictor; calibration targets magnitude. v5 conditions
the predictor on the target byte, removing the label from that limit
([v5](#label-conditioned-credit-predictor-and-gain-scaled-decay-v5)). Clamping
also removes anti-aligned output: on a sign-alternating target the network lags
one step and its alignment is negative.
The v2 loader rejects v1 manifests by protocol and run identity. These are
design statements, not evidence that calibrated credit improves training.

## Scale control, saturating curve, readout bias and per-anchor local steps (v3)

Protocol v3 bounds master-weight scale. Before Adam's delta is added, every
coefficient in each FP32 master region that an update touches is multiplied by
`1 - weight_decay * rate` (from v5 a tile step uses its rate times the multiplier it
received, and from v5.1 local tile steps use `local_rate` in place of `rate`,
scheduled by `local_decay` from v5.2; see
[v5](#label-conditioned-credit-predictor-and-gain-scaled-decay-v5)): decoupled weight
decay, as in
[Loshchilov & Hutter](https://arxiv.org/abs/1711.05101). It covers tiles and E;
route-only updates decay their route regions only. After the commit, a touched
master region (each matrix as a whole; P's local and route rows separately)
whose RMS exceeds the hard cap `connected.master_rms_cap` is scaled uniformly to
the cap and its trits are re-derived. The uniform update-norm cap
and the gain act on Adam's proposal as before. `connected.weight_decay` defaults
to 0.01 (range (0, 1]) and `connected.master_rms_cap` to 1.5 (range [1, 8]); both
enter run identity. The v4 configs set `weight_decay` to 0.001; the defaults are
unchanged (see [v4](#residual-normalization-and-readout-prior-v4)).

The activation curve clamps its input to the knot range [-2, 2] under the
connected protocol, so the slope is zero outside the knots and an FFN
preactivation beyond them no longer extrapolates linearly. Numerics outside the
connected protocol are unchanged.

A learned per-byte readout bias adds 256 FP32 offsets to the tied-readout logits;
its gradient is the softmax minus the one-hot target. It has its own Adam slot and
is decayed like E, but not RMS-capped. It is persisted in `connected.json`,
covered by the state hashes and kept FP32 under every quantized view. It starts
at zero in v3; v4 starts a fresh run from the
[corpus byte log-frequencies](#residual-normalization-and-readout-prior-v4).

Local lanes no longer apply an update per visit. Each job forwards, routes,
differentiates and publishes its record as before, but returns its
predicted-credit gradient instead of updating the tile; a job with a zero
multiplier returns none. After all lanes finish, each visited tile receives one
Adam step on the mean of that tile's lane-visit gradients, scaled by the Lambert `value_at`
gain times `trust_k` evaluated at that point. Tiles are therefore fixed for every
lane within a local phase, and `local_updates` counts at most one update per tile
per anchor. Record publication order and `local_visits` are unchanged.

The predictor skips a fit only for an effectively zero target (mean square at
most `1e-30`, norm about 1e-14); the Lambert gain keeps its `1e-12` evidence threshold. The normalized
target `t_hat` is clipped to norm 4 before the alignment statistic and Adam see
it, which bounds it while the calibration statistics lag the live target. After
the 1,000-observation warmup the calibration EMA weight is 0.01.

Log rows and the report add `readout_rms` (RMS of the routed output before
readout normalization) and `knot_overflow` (fraction of FFN preactivations beyond
the knots), both measured on the anchor's observed route, interval means in rows
and lifetime means in the report; per-tile `master_rms`, the run-long maximum
(never decreasing, persisted as `mechanism.master_rms_max`) of any updated master
region's RMS after projection, so a tile that once reached the cap reads the cap
thereafter and current per-matrix values come from the checkpoint; `embedding_rms`; and
`embedding_spectral_ratio`, the ratio σ1/σ2 of E's two largest singular values by
[power iteration](https://en.wikipedia.org/wiki/Power_iteration) with one
deflation. A training step fails if the routed output RMS exceeds 1e3 or a tile's
or E's master RMS exceeds the cap (beyond float rounding), and state validation
requires the readout bias to equal its Adam master, which steps once per anchor.
In v4 the routed output is normalized, so `readout_rms` reads about 1 by
construction and the 1e3 bound applies instead to each observed visit's raw RMS
before normalization (see v4); the master-RMS bounds are unchanged.

These changes follow the v2 FP32 study (`docs/STATUS.md`, "Calibrated predictor
v2 FP32 study"). The RMS-normalized readout hides the output scale from the loss
but returns the real correction multiplied by `1/RMS(output)`. With unbounded
masters, an offline probe of the anchor-86,016 checkpoint found one center-tile
visit mapping an RMS-1 prefix state to RMS 1.1e5 and the returned correction at
1.4e-5 (0.42 in the frozen ablation). The logged counters date the onset before
anchor 2,000: all eight lanes started from one prefix state and applied about ten
coherent updates per anchor to the center tile under Adam, whose step is
scale-free.

The v3 loader rejects v2 and v1 manifests by protocol and run identity. These are
design statements, not evidence that v3 trains better.

## Residual normalization and readout prior (v4)

Protocol v4 adds two behaviors, neither with a configuration option (the normalization later
became switchable through `connected.visit_norm`; see
[memory reach and visit normalization](#memory-reach-and-visit-normalization-switches)). Under the
connected protocol each tile's output, the residual sum of its attention and FFN
branches, is RMS-normalized before it leaves the visit (post-visit RMS
normalization of the residual stream). This is the existing `norm::rms`, an
[RMS normalization](https://arxiv.org/abs/1910.07467) without a learned gain; its
1e-5 epsilon on the mean square leaves the result at RMS 1 up to that epsilon. The
state that enters the next visit and the readout therefore has RMS 1. The adjoint
passes through the normalization before it reaches the residual branches, in the
real reverse pass and in the predicted-credit local backward alike. Numerics outside
the connected protocol are unchanged. No visit can then be bypassed by magnitude:
an earlier output can no longer outgrow, and so drown, what a later visit adds. The
readout still returns the real correction multiplied by the inverse RMS of its input
(see the v3 motivation above), but that input now has RMS 1, so the correction no
longer scales with the stream's magnitude at the readout, and `readout_rms` reads
about 1 by construction, which only confirms that normalization is on. Credit along
the residual path through a visit is still damped by that visit's raw 1/RMS, because
the normalization follows the residual sum (the Post-LN property;
[Xiong et al.](https://arxiv.org/abs/2002.04745)), so alignment growth now shows as
raw visit scale. Rows therefore carry `visit_rms`, the interval mean of each anchor's
mean raw visit RMS (the lifetime mean in the report), and `visit_rms_max`, the run
maximum, and a step fails if any observed visit's raw RMS exceeds 1e3.

On a fresh start, before the first step, the readout bias is initialized to the
public training corpus' byte log-frequencies,
`b[x] = ln((count_x + 0.5) / (total + 128))`: the add-half smoothed unigram over the
256 byte values ([Krichevsky and Trofimov](https://doi.org/10.1109/TIT.1981.1056331)),
whose probabilities sum to one. The pass that verifies the configured train file
against its manifest hash (already in the run identity) also counts its bytes, so the
counts are those of the verified bytes, with no second read. The route baseline
starts at the prior's entropy, `-Σ_x p_x ln p_x` (the expected loss of predicting the
prior), instead of ln 256, so early route advantages are not offset by the gap
between the uniform and unigram losses. The report and the first log row carry
`"readout_prior": "corpus-log-unigram"`. The first bias update creates its Adam
master from this value as in v3; until then state validation requires only a finite
bias, where v3 required zeros. Resume keeps the restored bias and baseline; the
counts go unused. The intent is that E need not encode the unigram in a rank-one
direction and that the mean readout gradient (softmax minus the one-hot target)
starts near zero.

The v4 pilot and full-run configs (`work/core-runs/connected-fp32-v4-*.json`) set
`connected.weight_decay` to 0.001; its default of 0.01, its range and
`master_rms_cap` are unchanged. In the v3 pilot, decay 0.01 shrank untouched
outer-tile masters by about 20% in 8.6k anchors (`docs/STATUS.md`, "Protocol v3
implemented and piloted; 24-hour launch held"), a rate that compounds over a 24-hour
run. The RMS cap remains the hard bound on growth.

That STATUS section also records the motivation. Master RMS stayed at or below 1.02,
so the cap never bound, yet over 8,580 anchors `readout_rms` rose from 1.45 to 12.7,
E's σ1/σ2 from 1.5 to 9.7, mean trust fell from 0.40 to 0.03 and interval `zero_mse`
from 2.7e-3 to 1.5e-5. An offline probe of the final checkpoint put the center
tile's attention output at RMS 25 on an RMS-1 input and the routed state at cosine
0.69 with E's top singular direction. The reading recorded there is an alignment
attractor that isotropic decay and an RMS cap cannot touch: the cap bounds a
matrix's Frobenius norm rather than its spectral norm. v4 targets the two channels
that reading names, the stream's growth along a common direction (normalization)
and the rank-one drive on E from the unigram (the prior), rather than bounding
scale again.

The v4 loader rejects v3, v2 and v1 manifests by protocol and run identity. These
are design statements, not evidence that v4 trains better.

## Label-conditioned credit predictor and gain-scaled decay (v5)

Protocol v5 makes two changes, neither with a configuration option. Both follow the
v4 full run `connected-fp32-v4-full-s7-20261107`, stopped at 81,920 anchors (its
record belongs in `docs/STATUS.md`). Held-out loss was best at 3.636 bits/byte at
57,344 anchors, while mean predictor trust was 0.108 over the first tenth of log
rows and 0.0065 over the last tenth, and the ratio of applied to base local step
fell from 0.17 in the first thousand anchors to about 0.01 after 60k. Prefit cosine
rose to 0.30-0.52 within the first 128 anchors, was about 0.13 near 3k and averaged
about 0.014 after 15k. Tiles were effectively frozen after about 15k anchors, so the
held-out gains most likely came from E, the readout bias, routing and memory; no
v4-protocol frozen-tile ablation was run to confirm that.

The structural cause was the predictor's input,
`[rms(input); rms(output); history; q; r]`, with no target. At a route's final
visit the real correction is `P_h E (softmax - onehot(y)) / sqrt(d)`, which depends
on the target byte `y`. Without `y` the best possible prediction is the conditional
expectation `P_h E (p - p*) / sqrt(d)`, where `p*` is the true next-byte
distribution, and that vanishes exactly as the readout fits. The prefit cosine's
rise and fall above fits that account: the expectation was large and learnable
while the readout was poor, then fitted away. An offline probe of the final
checkpoint (`work/inference/version-retro-20261107/credit-probe/`) fitted a
closed-form ridge map to the real corrections. From the v4 predictor inputs it
explained held-out R² -0.009, -0.014, -0.004 and +0.022 at visits 4, 3, 2 and 1;
adding `rms(E[:, y])` raised that to 0.968, 0.845, 0.779 and 0.751. v5 conditions the
predictor on the label: the conditional variant (cDNI) of the
[decoupled neural interfaces](https://proceedings.mlr.press/v70/jaderberg17a.html)
cited above (Jaderberg et al. 2017). The label feature `Shared::target_feature(y)`
is E's column for `y`, RMS-normalized, with no positional term. It reads the active
embedding view: the FP32 view under FP32 precision (the rounded view under
`--quantize-audit`) and the trits otherwise. The connected predictor's
features become `[rms(input); rms(output); history; target; q; r]`, `4d + 2`
values from a connected-local count. `Geometry::features` (`3d + 2`) still sizes
the anchor-only core predictor, history records and codecs, whose numerics are
unchanged. The connected predictor plus its Adam moments now hold
`d * (4d + 2) * 3 * 4` bytes, 789,504 at width 128. Memory records and checkpoints
carry no target.

Training computes the observed label feature right after the anchor's reverse
pass, before the E and bias updates, because the observed corrections were computed
with that E, and fits every observed record of the anchor with it. After the
embedding update the lane phase receives the byte and computes its feature once
from the updated E, matching the re-embedded prefix it trains on. Evaluation replay
passes the scored byte to its frozen lanes; generation passes the selected byte. A
lanes-level contract checks that trusted lane predictions change with the byte and
repeat exactly for the same byte.

The leak invariant: a label reaches the predictor only after its anchor's forward
pass has been read out, in training after `anchor_target`, in evaluation inside
`score` after the forward pass, and in generation after selection, so every score is
read from target-free logits. In frozen evaluation, records conditioned on `y_t`
reach later anchors only through memory, exactly as observed corrections already
do; in training `y_t` also reaches later anchors through the updated tiles, E and
predictor, as in any online learning. The evaluation contracts check the invariant
exactly: after the same warm-up, every byte's score equals, bit for bit, that byte's
loss under the readout of an identically warmed replay that observed nothing.

The second change decays at the realized step multiplier. Through v4 a touched tile
master kept `1 - weight_decay * rate` per update, while the step itself is scaled by
the gain (the Lambert value times trust for local updates, the plain Lambert value
for route updates) and then by the uniform norm cap. Tiles with trust near zero
therefore decayed at the full rate while barely stepping: in the v4 run tile 3's
masters, at RMS about 0.82 initially, read 0.73-0.84 after 61,004 Adam steps (Q 0.74,
U 0.73, K 0.76, O 0.75, V and A 0.80, G 0.82, DOWN 0.84). v5 keeps
`1 - weight_decay * rate * m` per touched tile master, `m` being the multiplier the
step actually received: the gain after the uniform norm cap, never above the gain.
The cap scales the post-Adam step directly, an effective learning-rate cut, and
AdamW's decoupled decay follows the realized learning rate
([Loshchilov & Hutter](https://arxiv.org/abs/1711.05101)). E and the readout bias
decay at the plain rate. Route updates thus decay at the rate times their realized
Lambert multiplier, up to `gain_max`; a working predictor raises the Lambert gains,
so route steps and route decay grow together. Admission refuses
`weight_decay * rate * gain_max >= 1`, so no realized multiplier can push a kept
fraction below zero mid-run.

v5.1 adds `connected.local_rate` (default 3e-4, range (0, 0.1], part of run
identity): the Adam rate of predicted-credit (`Mode::Local`) tile steps, whose decay
follows it at `local_rate` times the realized multiplier. E, the readout bias and
route updates keep `rate`, the predictor keeps `predictor_rate`, and the admission
bound above uses `max(rate, local_rate)`. The v5 pilot (`connected-fp32-v5-pilot-s7-20261107`, 8,750
anchors) passed every mechanism gate by wide margins (trust 0.78 and flat, prefit
cosine 0.82, relative MSE improvement 0.62) but failed held-out: 4.306 bits/byte at
8,192 anchors against v4's 4.066. Every tile's master RMS climbed from 0.82 to the
1.5 cap by about 3,800 anchors and stayed pinned, knot overflow rose to 0.29, and
prequential loss beat v4 early, then fell behind. With real-quality credit the local
step ran at about 0.003 per tile update at the measured realized multiplier
(applied/base 1.04 over the last third), roughly 30 times the anchor-only reference
path's 0.003/32 ≈ 9.4e-5 per anchor. Protocol strings stay v5; with `local_rate` equal to
`rate` the numerics equal the pilot's, while the run identity still differs because
the option is serialized.

v5.2 adds `connected.local_decay` (default `constant`, part of run identity), a
schedule on `local_rate` by anchor index: anchor `t`'s local tile steps, and so
their weight decay, run at `local_rate x factor(t)`. The new `inverse_sqrt` kind has
factor `1 / sqrt(1 + t / anchors)`, inverse square-root decay after
[Vaswani et al. 2017](https://arxiv.org/abs/1706.03762), whose schedule is the warmup
form `min(step^-0.5, step * warmup^-1.5)`; this is the shifted form without warmup.
The anchor-only path's `decay` accepts it too. Both v5.1 arms (`local_rate` 1e-4 and 3e-4) failed held-out
with excellent credit (prefit cosine 0.97), and the 1e-4 arm's prequential loss sat
at 6.0-6.4 nats per byte at 95-150 anchors, where v4 was at 4.85. v4 had an
accidental schedule: its effective local rate, `0.003 x trust x Lambert`, started
near 2e-3 and decayed roughly as `t^-0.6` while trust fell (7.6e-4 at 400 anchors,
2.5e-4 at 3k, 1.25e-4 at 10k, 3.7e-5 at 75k), so fast early tile learning escaped the
poor initial forward and the decay then kept tiles stable. v5's constant trust lost
that schedule. With `local_rate` 2e-3 and `anchors` 50, `inverse_sqrt` tracks v4's
effective rate within about 1.15x up to 10k anchors and 1.4x at 75k (ratios 0.88 at
400, 1.02 at 3k, 1.13 at 10k, 1.40 at 75k), now with accurate credit. That compares
the base rate with v4's realized rate, which already includes trust x Lambert; v5.2
realizes about 0.8 of its base rate through trust and Lambert (as in the v5.1 arms).
E, the readout bias and routes keep a constant `rate` and the predictor its
`predictor_rate`: in v4, whose schedule this restores, only the predicted-credit tile
steps decayed (through trust), while those learners ran at a constant rate on
observed credit. Admission applies the shared schedule rules (positive anchors, a
cosine floor in [0, 1]) and also refuses a `local_decay` whose factor can reach zero
(cosine with floor 0), which would hand Adam a zero rate mid-run.

The v5 loader rejects v4, v3, v2 and v1 manifests by protocol and run identity.
Manifests of pre-v5.2 v5 runs are refused by run identity: the serialized connected
options have since gained `local_rate` and `local_decay`.

v5.4 adds two optional fields to `connected` (`skip_serializing_if` default, so
default configs retain their pre-v5.4 run identity while non-default values enter
run identity):

- `connected.credit_mode`: selects how tile transformation matrices (`Mode::Local`)
  receive credit at each anchor:
  - `"lanes"` (default): runs the 8 post-anchor local lanes on the re-embedded
    prefix using predicted corrections, scaled by `Lambert(gain) * trust_k`.
  - `"anchor_exact"`: skips the 8 local lanes (`predicted_records = 0`) and applies
    one local Adam step per visited tile using the mean of the exact per-frame
    `TileGradients` gathered on the anchor's scored forward route, with unit
    gain (`1.0`).
  - `"anchor_predicted"`: skips the 8 local lanes (`predicted_records = 0`) and
    evaluates the calibrated label-conditioned predictor on the anchor's scored
    route frames, applying one local Adam step per visited tile scaled by
    `Lambert(gain) * trust_k`.
  - `"hybrid"`: accumulates both the anchor route's exact `TileGradients` and the
    8 local lanes' trusted predicted-credit gradients into one `Accumulator`,
    applying one step per visited tile (scaled by `Lambert(gain)` on tiles that
    received exact anchor credit and by `Lambert(gain) * trust_k` on tiles visited
    only by local lanes).
  When `credit_mode` disables lanes (`"anchor_exact"` or `"anchor_predicted"`),
  evaluation replay also skips frozen local-lane memory publication, and state
  validation bounds `local_updates` by `observed_records + predicted_records`
  instead of `predicted_records` alone.
- `connected.slice_mode`: selects how a tile's 384-channel FFN bank (`G`, `A`,
  `CURVE`, `DOWN`) is indexed per visit from a deterministic 64-bit SplitMix64 seed
  (`tuning::hash`) over `tile_id` and 64-bit packed 32-element trinary deadband
  signatures (`±0.05`) of `(memory_tail ^ (rows << 48), input, history)`:
  - `"full"` (default): evaluates all $M = 384$ channels with unit mask and
    `down_scale = 1 / sqrt(M)`.
  - `"windowed"`: selects an active window length $m_{\text{active}} \in \{M/2,
    3M/4, 7M/8\}$ (`{192, 288, 336}` at $M = 384$) from `(visit_seed >> 32) % 3`
    and circular start offset `(visit_seed as usize) % M`, zeroing inactive
    channels and scaling `DOWN` by $1 / \sqrt{m_{\text{active}}}$ so forward
    variance and backward adjoints match the active slice while on-disk tile shapes
    and the 4-bundle residency bound remain unchanged.
  - `"hybrid"`: selects the same active window length $m_{\text{active}} \in \{M/2,
    3M/4, 7M/8\}$ at weight `1.0`, sets background channels outside the window to
    alternating weights `0.5` and `0.25` shifted by `tile_id + (visit_seed >> 48)`,
    and scales `DOWN` by $1 / \sqrt{\sum_j \text{mask}[j]^2}$.

These are design statements, not evidence that v5 or v5.4 trains better.

## Write-behind paging, visit-frequency eviction and resident cap

Three store controls trade RAM for SSD writes. `write_behind_tiles` and
`resident_tiles` are optional `connected` fields that are never serialized: they
are outside run identity and the checkpointed state, and a resume takes them from
the resuming config (`Options::with_paging`), so they may change between segments.
None changes numerics:
a bundle decoded from disk, from RAM bytes or kept cached is bit-identical
(`connected_paging_contracts` compares losses and every tile's hash/version/steps/
revision after a checkpoint).

- `write_behind_tiles` (default 0, admitted 0..=64): on eviction, a dirty bundle is
  encoded with the unchanged codec into RAM (`connected_store_paging::Pending`)
  instead of written. Its slot hash stays blank (still dirty). A later checkout
  decodes and hash-checks those bytes (a cache miss and `pending_hits`, not a disk
  read) and drops them; the lease is again the authoritative copy. When the buffer
  holds `write_behind_tiles` entries, the least recently evicted one is published
  through the existing partial-file, fsync, content-addressed rename path.
  `Store::flush` drains every pending entry first, so checkpoint save (copy, hash
  and manifest), evaluation's flushed-state snapshot, `fingerprint` and `set_view`
  see exactly today's on-disk invariants. A failed publication keeps the entry
  pending and poisons the store, as an eviction write failure always did. Pending
  bytes are session RAM, not checkpointed: a crash loses them exactly as it loses
  dirty cached bundles; resume always starts from a published checkpoint. Bound:
  `write_behind_tiles x ~3.7 MB` (64 is about 238 MB at the current candidate,
  inside the 512 MiB Job commit cap only with little else; size it to the owner's
  measured headroom). Stats: `writes_deferred`, `pending_hits`, `pending_peak`,
  `pending_bytes_peak`; `writes` still counts actual disk writes.
- Eviction (always on): the idle cached bundle with the least visit heat is
  evicted, heat being an exponentially decayed checkout count (half-life about 16
  checkouts), ties to the oldest cached entry. It replaces oldest-first and changes
  only paging statistics.
- `resident_tiles` (default absent, meaning `workers`; admitted `workers..=7`):
  the decoded bundle cap (leased plus idle cached). `workers` keeps its meaning
  (concurrent leases, 1..=4). This is an explicit memory-for-writes trade: each
  extra slot holds another ~3.7 MB decoded bundle plus view, and at 7 nothing is
  ever evicted, so paging is not exercised at all. `peak_resident` reports against
  `resident_cap` in the paging stats.

Each runtime publication still fsyncs its own partial before the rename. Runtime
blobs are not a recovery source (restore always copies from a checkpoint, which is
written with its own fsyncs), and a torn blob reused by name is rejected by its
hash, so batching those syncs would be safe; it is not done yet.

## History ablation and prioritized replay

`connected.history` (default true, serialized only when false) is the connected
forward-input ablation; the older top-level `history` knob stays refused by
connected admission. When false, every tile visit's retrieved-memory history input
is replaced by zeros of the same width at every reader: the training anchor,
replayed anchors, local lanes, and frozen evaluation/generation/completion replay
(`Options::history_input`). Memory is still queried and records (with that zero
history) are still published and fitted, so counters and logs stay comparable.

`connected.replay` = `{capacity 1..=4096, per_anchor 1..=8}` (absent is off;
admitted only with `credit_mode = anchor_exact`) keeps a session-only buffer of past
training positions (`connected_replay::Buffer`): the corpus target index, re-read
through `TrainingSource::replay_cursor` (a train-file cursor for `Corpus`; other
sources refuse), so held-out spans are never replayed. Priority is the position's
most recent loss. Right after the live anchor's forward and gather, the
`per_anchor` highest-priority entries (ties: oldest position) are re-run under the
same weights: forward with route sampling seeded by (seed, anchor, position), then
`connected_anchor::gather`. Its memory reads use `Memory::query_before` with the
replayed position as both the shared revision and the cutoff, so a replay sees only
records published by anchors strictly before it. Records from that anchor onward
carry corrections derived from its own or later targets; reading them would leak
the label. With the default eligibility (shared lag at most one, `memory_lag` 1) and capacity, a
replay therefore sees at most the previous anchor's surviving records and usually
none: replayed anchors generally run with zero history, a distribution mismatch
against live anchors that read the newest memory. Their per-frame tile gradients join the live anchor's
exact gradients in one `Accumulator`, so each touched tile takes a single
`local_rate` step on the mean over live plus replayed visits; a tile only a replay
visited is also stepped. Replays do not step E or the readout bias (their Adam
clocks stay equal to `anchors`), publish memory, fit the predictor, update routes,
advance the cursor or anchors, or enter `loss_sum`. Each replayed entry's priority
becomes its new loss; the live anchor is then inserted with its loss, and over
capacity the lowest-priority entry (ties: oldest) is forgotten. State validation
bounds `local_updates` by `7 x anchors` when replay is configured, and the
`anchor_exact` apply path refuses (non-persisted, per anchor) any phase that takes
more mean steps than tiles holding a gradient sum. Log rows and the
report carry `replay` session counters (`replay_anchors`, `replay_visits`, loss sum
and mean, `buffer_size`). Limits: the buffer is not checkpointed, so a resumed run
starts it empty and its replay schedule differs from an uninterrupted run; replay
costs `per_anchor` extra forward-plus-backward passes per anchor. Prioritization is
greedy on the last loss, after prioritized experience replay
([Schaul et al. 2015](https://arxiv.org/abs/1511.05952)) but without its stochastic
sampling, importance-sampling correction or any recency term, so replayed credit is
biased toward high-loss positions and can fixate on irreducibly hard ones (rare
bytes, story boundaries) whose loss never falls.

## Warm restarts and per-tile rate floor

Motivation: in the 24-hour inverse-sqrt run the local tile rate (2e-3, 500 anchors)
had decayed to about 3.5e-5 when held-out loss plateaued, while E and the readout
bias kept stepping every anchor at the constant anchor `rate`. A tile's inputs kept
moving while the tile itself was effectively frozen.

`connected.restart` = `{floor, threshold?, adapt}` (absent is off; part of run
identity when present, omitted from it when absent, so every earlier run and
checkpoint is unchanged). Admission: `floor` in [0, 1]; `threshold` positive and
finite when present; `adapt` in [0, 1) (default 0) and nonzero only with a
threshold. `connected_restart::factor` is the single local-rate factor of every
tile step (`connected_lane_apply`, all credit modes and replay):
`max(floor, local_decay.factor(anchor - origin[tile]))`; without `restart` it is the
plain `local_decay.factor(anchor)`, bit for bit. Route steps keep `rate`.

Change: each anchor yields one scalar, the sum of E's applied Adam step norm
relative to E's master norm after commit, plus, for every local tile step that
anchor, its applied norm relative to the L2 norm of the master regions it updated
(after decay, step and RMS-cap projection; `UpdateMetrics::weight_norm`, derived
from the RMS `commit` already computes, so no extra pass over the weights). The
readout bias and route steps are excluded. On the fixture (anchor-exact, `rate`
0.003, `local_rate` 3e-4) one anchor's change is about 2.3e-3, of which about
2.1e-3 is E: the signal mostly measures how far a tile's inputs moved.
`Counters::change_sum` accumulates it always (restart or not); logs report its
interval mean as `change_per_anchor` (the report: lifetime mean), so thresholds can
be chosen from a run's data. `connected.json` `counters` and the report `mechanism`
therefore gain `change_sum`: binaries built before this change refuse checkpoints
written after it (`deny_unknown_fields`), and a checkpoint written before it resumes
with `change_sum` 0, so that run's lifetime `change_per_anchor` understates.

Restart: a tile counts as visited on an anchor when its live anchor visits or local
visits advanced or it took a local step (replay-only steps included). Each visited
tile adds the anchor's change to its `drift`, and its loss to the cycle's loss sum.
After the whole anchor commits (`connected_step::step`, before the ready
validation), every visited tile whose `drift >= bar` restarts: `origin` becomes the
completed-anchor count, so its next step runs at the full `local_rate` and decays
again toward the floor; `drift` returns to 0; its records leave the shared memory
FIFO (`Memory::forget`, publication counters unchanged) and a configured replay
buffer is emptied (session counters kept, `buffer_clears` counted), so stale
corrections are neither retrieved nor replayed. Any one tile's restart clears the
whole replay buffer, so a threshold low enough to restart some tile most anchors
effectively disables replay. With `adapt > 0`, from the second restart on, the finished
cycle's mean anchor loss is compared with the previous cycle's: lower multiplies
`bar` by `1 - adapt` (restart sooner), otherwise by `1 + adapt`, clamped to
`[threshold / clamp, clamp x threshold]` (`clamp` default 8). Adam moments, E/bias clocks and tile step counts are
never reset; per-tile gains, predictor trust and tile revisions also carry across
a restart. This is warm restarts after SGDR
([Loshchilov & Hutter 2017](https://arxiv.org/abs/1608.03983)), triggered by
accumulated change instead of a fixed period.

Per-tile state (`TileRestart`: origin, drift, bar, restarts, cycle loss and count,
previous cycle mean) lives in `State.restarts` exactly when `restart` is configured,
is checkpointed in `connected.json` (omitted when absent) and is validated at every
boundary: origin and cycle within the anchor count, finite nonnegative accounting,
restarts zero exactly when origin is zero and no cycle mean exists, `bar` equal to
`threshold` without adaptation and inside the clamp with it, `drift < bar`. Log rows
and the report carry `restart`: per-tile `restarts`, `bar`, `drift`,
`forgotten_records` and the `rate_factor` the next anchor would use. `factor`
refuses options and per-tile state that disagree rather than falling back. The
relative change divides by the master norm floored at `sqrt(f64::MIN_POSITIVE)`.
Limits: change is a relative norm of
optimizer steps, not a measure of functional change in a tile's inputs; a tile is
credited every visited anchor's whole change, including other tiles' steps; the
adaptation rule compares noisy single-cycle means, and a cycle's mean loss depends
on which anchors visited the tile. No convergence or quality effect is established.

### Per-tile training state

`TileRestart` (one per tile, `State.restarts`, in `connected.json`) is the tile's
training-side artifact. It is kept apart from the model: never inside the paged
tile/Adam bundles, so it is read and updated without a checkout. Inference never
uses it: `connected_completion` only deserializes and validates it as part of `State`
and hashes it inside the immutability fingerprint, and `connected_store*` has no
reference to it. Its three roles:

1. Fast operations: restart, stagger, restart-replay and Lambert decisions read only
   this struct (`connected_restart::finish`, `connected_restart_due`), never a tile
   bundle.
2. Tile training memory surface: the restart clock (`origin`, `drift`, `bar`,
   `restarts`, cycle loss, `due_since`), the `TileSignal` (`connected_tile_signal`:
   input EMA, reference, gradient-norm EMA, loss EMA), the restart-replay accounting
   and the session-only ring of recent positions.
3. Plasticity dial: `plasticity` is multiplied into the single tile-rate helper
   `connected_restart::factor`: `local_rate x plasticity x max(floor, decay(anchor -
   origin))`. Under discrete restarts it equals its configured value (the optional
   `restart.plasticity: [f64; 7]`, each in (0, 4], else 1.0) and validation enforces
   that; only the Lambert law below moves it. At 1.0 the product is exact, so numerics
   are bit-identical. `validate_run` uses `local_rate x max(1, largest plasticity)` (4
   under Lambert) in its decay-safety bound. A zero entry is refused: it would hand
   Adam a zero rate.

The existing fields moved nowhere: new fields were added beside them, each
`#[serde(default)]` and skipped while unused (plasticity 1, zero sums, no `due_since`,
an unused signal), so earlier `connected.json` records load (a contract deserializes
a literal eight-key record with restarts and validates it) and a fresh record
serializes the earlier keys exactly. Once a restart run takes a visit its record
gains `signal`, so a binary built before this change refuses checkpoints written
after it.

`TileSignal` definition (live anchor visits only, every credit mode; replayed and
lane visits are not observed). Each visit's input vector is the tile's forward
`input` (width d) concatenated with its retrieved `history` input (width d; zeros
under `history: false`); the prefix window is excluded. Every EMA uses weight
`max(1/n, 1/256)` for its n-th sample, a running mean for the first 256 samples
(no zero-start bias) and a 256-sample horizon after. Warm-up: the first reference is
taken at the tile's 256th visit (`WARMUP`, one horizon), so it is a many-sample mean;
before that the reference is empty, the distance is 0, a `signal: tile` drift cannot
reach a positive bar, and restarts (possible under the shared signal) do not
snapshot. After warm-up every restart re-snapshots it. Distance =
`||ema - reference|| / max(||reference||, sqrt(f64::MIN_POSITIVE))`; it can fall as
well as rise. Two f64 vectors of width 2d per tile (2 x 7 x 2d values) are
checkpointed. The gradient-norm EMA folds the L2 norm of each tile's mean local
gradient before rate and gain (computed in `Accumulator::apply_with_trust`, only
when `restart` is configured); the loss EMA folds the anchor loss of each visited
anchor. Both are reported, not used. Logs and the report add per tile `distance`,
`gradient_norm_ema`, `loss_ema`, `plasticity`, `plasticity_mean`,
`restart_replays`, `restart_replay_steps`, `restart_replay_loss_mean`, `due_since`;
a rate-factor error fails the log line or report instead of dropping the block.

Fixture magnitudes (anchor-exact FP32, `abacabad` corpus, 2,400 anchors, release
build): only the start tile (visited every anchor) warms at anchor 256; the others
took about 1,550-2,100 anchors, and four tiles never reached 256 visits. Without
restarts the start tile's distance grew by a mean 1.2e-4 per visit with per-visit
standard deviation 2.7e-4 (largest single step 2.9e-3), reaching 0.26 after about
2,150 warm visits; two later-warming tiles grew 0.7e-4 and 2.4e-4 per visit with
standard deviation 2.0e-4 to 2.7e-4. With `threshold` 0.01 the start tile restarted
27 times (about every 80 visits). These are a repetitive fixture's numbers; real
text will be noisier. Per-anchor time was about 2.0 ms with or without the signal.

### Restart options

All under `connected.restart`, each omitted from serialization and run identity at
its default, so `{"floor":0.25,"threshold":2.0,"adapt":0.25}` keeps its identity.
`forget`, `signal: "tile"`, `stagger` and `replay` act only on restarts and are
refused without a `threshold`.

- `forget` (default true): false skips `Memory::forget` and the replay-buffer clear on
  a restart.
- `signal` (`"shared"` default, `"tile"`): with `"tile"` a visited tile's `drift` is
  its `TileSignal` distance (a level, not a sum); a restart snapshots the reference,
  so drift is exactly 0 after it. The distance is reported under either signal.
- `stagger` (default 0 = off): at least `stagger` anchors between any two restarts
  (a restart exactly `stagger` anchors after the last is allowed). The last restart's
  anchor is the newest `origin` of any restarted tile (already persisted). Once the
  window has passed, one due tile restarts: the one due longest (`due_since`, the
  anchor it was first left waiting, persisted and omitted when unset), ties to the
  larger `drift / bar`, then the lowest id. Ranking by wait prevents starvation under
  `signal: tile`, where a waiting tile's drift can stay just above its bar while
  others score higher ratios. Validation admits `drift >= bar` under stagger and
  requires `due_since` to be set exactly while a tile is due. A due tile need not be
  visited on the anchor it restarts.
- `clamp` (default 8, admitted [1, 1024]): the adaptive bar range.
- `replay` (default 0, admitted 0..=6, `anchor_exact` only): replay on restart.
  Each tile keeps a session-only ring of its last `replay` live-visit positions
  (empty after resume). After the anchor's own commits and the restart (and forget),
  each ring position, oldest first, is re-run through `connected_replay::replay_salted`
  (memory read with `Memory::query_before`, so records published at or after the
  position are never retrieved; route sampling uses its own salt, distinct from
  prioritized replay's), and the restarted tile alone takes one unit-gain Adam step at
  its fresh post-restart rate on that replay's mean gradient. A replay whose sampled
  route skips the tile takes no step. E, bias, routes, cursor, anchors, the predictor,
  the gains' stored state, memory contents and the prioritized buffer are untouched.
  Each step does commit the tile and so advances its store revision, which ages the
  tile's newest memory records and its gain: both expire past a revision lag of 8
  (`connected_signal::eligible`, `Gain::value_at`). A visited tile already takes two
  commits per anchor after those were recorded (its route step and its one local
  step), so at most 8 - 2 = 6 replay steps keep them usable at the next anchor; that
  is the admission cap. Steps count in `local_updates` (checked per replay) and
  `change_sum` (not in any tile's drift), and per tile in `replay_steps`, which
  extends the validated `local_updates` bound. Cost: `replay` forward-plus-backward
  passes per restart. Independent of `connected.replay`.

Untested: replaying more than one ring position per restart in a flow test (the
contract uses `replay: 1`), `signal: tile` under `lanes`/`hybrid` credit, stagger
with `signal: tile` in a flow test (the ranking is table-tested), and any quality
effect of any option.

### Greedy training routes

`connected.route_training` is `"sampled"` (default, omitted from serialization and
identity) or `"greedy"`. Training passes otherwise sample each route action from
the router's softmax, while evaluation and completion take the argmax, so the
routes a sampled run learns on are not the ones it is scored on; the reference
path trains greedy. With `"greedy"`, every training-time route draw passes no
generator and takes the argmax, except that with probability
`connected.route_explore` (in [0, 1), default 0, admitted only with greedy) a pass
samples as before. The draw is `tuning::hash(seed, key + [EXPL])`: the live anchor
and its lanes share key `[anchors]`; each prioritized or restart replay pass uses
`[anchors, position, salt]`. The route update, baseline and everything else are
unchanged, and credit goes to the action actually taken. Every credit mode routes
only through those draws (anchor_predicted reuses the anchor's trace), so all four
modes honour the option. Under greedy routes the score-function update scores the
argmax action rather than a sampled one, so it is a biased policy gradient, not an
unbiased REINFORCE estimate (exploration passes excepted). Implementation:
`connected_route.rs`.

### Anchor waves

`connected.anchor_batch` (u32, admitted 1..=32, default 1, omitted from serialization
and identity at 1, run identity above 1) is the number of anchors whose tile credit
is averaged into one tile update; it targets difference 1 of
`.omc/plans/connected-v58-update-quality-loop-2026-11-18.md` (the reference path
averages 32 positions per optimizer step). At 1 the training loop calls
`connected_step::step` unchanged. Above 1 it requires `credit_mode: anchor_exact` and
refuses `connected.replay` and `restart.replay` (both assume one update per tile per
anchor), and run admission requires `anchors`, `checkpoint_every` and `eval_every` to
be multiples of it. Implementation: `connected_wave.rs`.

Waves are aligned: wave k covers anchors `[k b, (k + 1) b)`, so the numerics are a
function of the anchor index alone and do not depend on `checkpoint_every`,
`eval_every` or `connected_eval`, which are not identity. STOP, the wall-clock
deadline, log rows, checkpoints and evaluations are checked only between waves, so
they always see a ready, validated state at a wave end, and any wave-end checkpoint
(including a STOP or deadline one) resumes bit-exactly. Retention probes due inside
a wave run at its end (a few anchors later; still post-exposure).

Within a wave the tiles (transformations and routes) are frozen: one revision
snapshot serves every anchor's memory reads. Per anchor, as at batch 1: forward with
the route draw keyed by `[anchors]` (`route_training` honoured), gather, the route
gradient at that anchor's advantage `loss - baseline`, E and readout-bias Adam steps
(their clocks stay equal to `anchors`), predictor fit, gain observation,
`observe_visit`, memory publication, visit counters, loss and scale counters, the
cursor and the baseline. Tile credit is summed in two `Accumulator`s outside the
tile bundles. At the boundary, before the last anchor advances (so the local rate's
anchor index is that anchor's, as at batch 1): each tile with a route sum takes one
`RouteOnly` step on its mean (divided by the number of anchors in which it routed) at
`rate` and the route gain recorded at its last routing anchor of the wave, evaluated
as batch 1 evaluates it (before that anchor's observations). The gain at the boundary
itself would be wrong in long waves: `epoch` advances once per published frame, so a
tile last observed early in the wave would be past the 64-epoch gain expiry and step
at a neutral 1.0. Then each visited tile takes one
unit-gain local step on its mean exact gradient at
`local_rate x restart::factor(...)`, both checked out one tile at a time, so the
resident cap holds however many tiles the wave touched. The two accumulators live
outside that cap: up to `2 x TILES` full tile-gradient sets during a wave (route and
local; §Memory scaling). Means divide by visit count
(a tile visited twice in one anchor counts twice), as at batch 1; the reference path
instead divides its batch sum by the batch size. The per-anchor guard becomes a
per-wave guard: at most one local step per touched tile per wave, so a tile commits
at most twice per wave (route, local) and memory/gain revision lags are unaffected.

Per-tile training state: the loss EMA and cycle loss fold every anchor that visited
the tile; the gradient-norm EMA folds once per wave (it is observed by the one local
step); drift, the Lambert dial and chase, due restarts and forgetting settle once at
the wave end (`restart::settle`). Under the shared change signal a touched tile adds
`change x (anchors that visited it in the wave) / (wave length)`, where `change` is
the whole wave's summed change, so a tile visited on one anchor of eight accrues an
eighth of it (the factor is exactly 1.0 at batch 1). This is an approximation: it
spreads the wave's change evenly over its anchors, whereas batch 1 credits each
visited anchor its own change, and the boundary's local tile steps are part of the
wave's sum.
`plasticity_sum` adds the dial times the anchors that visited the tile, so
`plasticity_mean` stays a per-visited-anchor mean; the reference chase runs once per
wave rather than once per anchor, so under the Lambert law the realized chase per
anchor is smaller than at batch 1.

Errors: the wave clears `ready` first and sets it only after the boundary and a
passing `State::validate`, so a failed anchor leaves the state poisoned rather than
half applied. `Counters::waves` (`completed`, `tiles_stepped`; omitted while unused,
validated against `anchors`) is in `connected.json` and the report `mechanism`; log
rows and the report carry `anchor_batch` and, above 1, `waves` and `tiles_per_wave`
(interval in rows, lifetime in the report). Untested: a wave error in flight, the
retention-probe shift and any quality effect.

### Lambert plasticity (experimental)

EXPERIMENT, may be removed or changed; a hypothesis with no measurements.
`restart.law: "lambert"` (default `"restart"`, `connected_plasticity`) replaces
discrete restarts with a continuous dial. It requires `signal: "tile"`, a
`threshold` (the bar), `floor > 0` and a constant `local_decay` (which it ignores),
and refuses `adapt`, `stagger`, a non-default `clamp`, `replay` and `forget: false`
(an explicit default value is indistinguishable from absent and is accepted). On
each visited anchor, after its commits: `drift` = the tile distance d; before the
signal's warm-up the dial holds at 1 (full rate), after it `plasticity = W(e d / bar)`
with the principal [Lambert W](https://dlmf.nist.gov/4.13)
(`connected_gain::lambert_w`), so d = 0 gives 0, d = bar gives 1 and larger d grows
logarithmically; then the reference chases the EMA,
`reference += min(1, plasticity / 64) (ema - reference)`, so drift raises
plasticity, which pulls the reference in and lowers drift. The tile rate is
`local_rate x max(floor, plasticity)`; `origin` is not used. The dial caps at 4
(W(4 e^4) = 4, reached at d >= 4 e^3 bar, about 80 bars); inputs past the cap skip
`lambert_w`, so its 1e6 input bound is never reached. `plasticity_sum` (reported as
`plasticity_mean`) gives the realized average dial, including warm-up anchors at 1.
Fixture, 2,400 anchors, warmed tiles only (end-of-run distance in brackets): bar
0.05 gave dials 0.24-0.44 (d 0.006-0.013), bar 0.2 gave 0.13-0.19 (d 0.010-0.017; a tile warmed 40 anchors before the end read 0.015)
and bar 1 gave 0.04-0.07 (d 0.015-0.027); lifetime means including warm-up were
0.50-0.83, 0.32-0.75 and 0.21-0.78. The loop settles where the chase balances input
drift, at distances well below the bar, so these dials sit below 1.

### Visit balance

EXPERIMENT (test F of `.omc/plans/connected-v58-update-quality-loop-2026-11-18.md`, extended):
"learn faster where visited less", applied to the tile rate, training route choice and
shared-memory retention. `connected.balance` = `{horizon, rate, route, memory}` (absent is
today's behaviour and omitted from identity; present is run identity, with defaults inside the
block omitted). Admission: `horizon` 64..=65536 (default 1024), `rate` [0, 1] (default 0),
`route` [0, 2] (default 0), `memory` (default false); a block enabling none of the three is
refused, and `rate > 0` or `route > 0` requires `credit_mode: anchor_exact` (the estimate counts
live-route visits, which equal the local steps only there; under lanes the steps follow lane
routes). `memory` is admitted in every mode. Implementation: `connected_balance.rs` (validation and report in
`connected_balance_check.rs`).

Estimate (`State.balance`, `connected_balance::Estimate`, in `connected.json`, omitted when
absent; training-side only, never in tile bundles, never read by evaluation or completion).
Each completed anchor folds, in `connected_step::advance` (per anchor under waves), the
indicator "the live anchor route visited tile k" into `share[k]` with weight
`max(1/n, 1/horizon)` for the n-th anchor: a running mean for the first `horizon` anchors, then
an EMA of that horizon (the `TileSignal` warm start, so there is no zero-start bias). Replayed
and lane routes are not observed. Every route enters at `START_TILE`, so its share is pinned at
1 and says nothing about whether the router chose it; it is excluded from the ratio. With `mean`
the average share of the other `TILES - 1` tiles, `r_k = mean / max(share_k, 1e-6)` for those
tiles and `r = 1` for the start tile, so its rate multiplier is exactly 1 and a route to it gets
bonus exactly 0; its share and advantage statistics are still tracked and reported. Every ratio
is 1 (multiplier exactly 1.0, bonus exactly 0.0) until the estimate has seen `horizon` anchors,
which removes the clamp-sized swings of the first anchors' running mean, and while the outer mean
share is at most 1e-6 (for example when the first anchors all halted at the start tile), which
would otherwise push every neighbour to the +2 clamp. An all-zero bonus takes the untilted route
path bit for bit.

- `rate`: every local tile step multiplies its rate by `clamp(r_k ^ rate, 1/4, 4)`
  (exactly 1 at `rate` 0). `connected_balance::rate_factor` = `restart::factor` (floor, decay,
  plasticity dial, Lambert law) times this multiplier, and is the one helper every tile step
  uses (`connected_lane_apply`, restart replay, the restart report's `rate_factor`), evaluated
  when the step is applied, so it composes with waves. The restart report's `rate_factor`
  therefore includes the balance multiplier. When the uniform `max_update_norm` cap binds, the
  realized step and its decay are rate-independent, so the multiplier (like any rate factor)
  has no effect on that step. `validate_run`'s decay-safety bound uses
  `local_rate x max(1, plasticity) x 4` whenever `rate > 0`.
- `route`: every training-time route draw built by `connected_route::draw` (live anchor,
  prioritized and restart replay) adds
  `clamp(route x ln r_j, -2, 2)` to the logit of each neighbour action whose destination is
  tile j; halt gets 0. Implemented as `p_a e^{b_a}` renormalized (equal to the softmax of the
  adjusted logits), then sampled with the pass's generator or, under greedy training routes,
  argmax. The route update credits the action taken but its score-function gradient uses the
  model's own unadjusted policy (`routing::policy` recomputes it), so the update is off-policy:
  a biased gradient whose bias is bounded by the ±2 logit clamp (at most a factor e^4 between
  two actions' sampling odds) and not corrected by importance weights. Lane draws
  (`connected_route::lane`) are never tilted, by construction: evaluation and completion replay
  reuse the lane path, so a relaxed admission rule still could not tilt them; evaluation's own
  anchor forward passes `Draw::GREEDY`.
- `memory`: when the FIFO is full, `Memory::publish` evicts the oldest record of the tile
  holding the most records (ties: the globally oldest of those) instead of the globally oldest
  (`Memory.balanced`, serialized only when true). Removal from the middle keeps the vector in
  publication order, so the epoch-monotonicity checks, `query`/`query_before` eligibility (a
  per-record test), `forget(tile)` and the publication counters are unchanged. Evaluation and
  completion build their isolated memory through the same `connected_balance::memory`, so they
  use balanced eviction exactly when training does.

Limits of `memory` (at the default `memory_lag` of 1; a larger lag lets eviction change what
anchor-exact training reads, see
[memory reach](#memory-reach-and-visit-normalization-switches)): a training read accepts only
records whose shared revision lags by at most one anchor (`eligible`) and that are at most 64
epochs old. With `anchor_exact` credit, four
visits and capacity 32, eviction never reaches the current or previous anchor's at most eight
records, so the policy changes which stale records are kept but not what training retrieves; the
flow test asserts the same routes and losses with different held records. It can matter in training only when one or two anchors
publish more than the capacity (lanes or hybrid credit), and in evaluation and completion, where
the shared revision is frozen and every record within 64 epochs is eligible. So a `memory: true`
arm mostly changes how it is scored, not how it trains. Balanced eviction counts every held
record, eligible or not, so under lanes it can evict a still-eligible record of a tile that holds
many stale ones and reduce usable retrieval. Under `anchor_exact` it does not affect
training at all, only evaluation scoring, so it must not be used in a pilot whose held-out scores
are compared with runs without it.

Reporting (log rows and the report, `balance`): per tile `visit_share`, `rate_multiplier`,
`route_bonus` (the current per-destination bonus, null without `route`), lifetime
`route_decisions` (live-anchor decisions under a nonzero bonus) and `route_changed` (those whose
adjusted argmax differs from the unadjusted one: for greedy routes, choices the bonus changed; in
sampled mode a proxy) and `route_bonus_mean` (mean absolute neighbour bonus per decision), and `memory_records` held.
The estimate also keeps, with the same horizon and warm start, an average of the anchor
advantage `loss - baseline` (the baseline before `advance` updates it, as the route update uses)
over anchors that visited the tile (`advantage_visited`, 0 until a visit) and over those that did
not (`advantage_unvisited`); `worth` = unvisited minus visited (positive: anchors through the
tile do better), null until both sides have a sample (always for the start tile, which is never
unvisited). This is a correlational estimate, not an
ablation: the model chooses its routes, so visited and unvisited anchors differ in content.

Fixture magnitudes (`the cat sat on the mat.`, 80 anchors, horizon 64, so 16 live anchors after
warm-up, debug build): `rate` 1 gave outer-tile multipliers 0.83-1.35 (outer shares 0.18-0.30) and
exactly 1 on the start tile; it changed E and the tiles, though the summed fixture loss over those
16 anchors matched the run without balance to every printed digit. `route` 2 under greedy routes
gave outer bonuses −0.51 to 0.89 (0 towards the centre), a mean absolute bonus of 0.33-0.80 per
tilted decision, changed 23 of 45 tilted live-anchor decisions and lowered the largest outer
tile's share of outer visits from 0.32 to 0.21. Untested: `memory` under lanes, hybrid and
anchor-predicted credit (the flow test runs in both the 7- and 19-tile builds),
`route` under prioritized or restart replay, and any quality effect.

### Memory reach and visit normalization switches

EXPERIMENT (`.omc/plans/connected-v58-update-quality-loop-2026-11-18.md`, differences 5 and 6):
two switches for the last untested differences from the reference path. Both are omitted from
identity and serialization at their defaults, so existing configs, checkpoints and the golden
trajectories are unchanged; a non-default value is run identity.

`connected.memory_lag` (1..=64, default 1) is the largest shared-revision lag a memory read
admits (`connected_retrieval::eligible`, `Memory.lag`, serialized only when not 1 and checked
against the options by state validation). Every reader goes through the one `Memory`: the live
anchor (`connected_reach::anchor_read`), lanes, prioritized and restart replay (`query_before`)
and frozen evaluation/completion replay, whose memory `connected_balance::memory` builds from
the checkpointed options.

Causality does not depend on the lag. A record's shared revision is fixed at publication: an
observed record carries the index of the anchor whose label produced it; a lane (predicted)
record carries E's Adam step after that anchor's E step, its index plus one. A record's shared
revision is therefore never below the index of the label it derives from. A replay of position
`p` admits only shared revisions strictly below `p` (`query_before`'s cutoff), so only labels of
earlier anchors; the lag only lowers the floor to `p - memory_lag`. A live anchor reads during
its forward pass, before its own gather and publication, so memory then holds only records of
earlier anchors (its lag-0 records are the previous anchor's lane records). Evaluation and
completion freeze the shared revision, so every record there has lag 0 and `memory_lag` does
not change them: they already read every record within 64 epochs, the train/evaluation read
mismatch that a larger lag narrows.

Effective reach. The lag is one of four limits, and the other three are unchanged:
- Event age 64 epochs: one epoch per published record, so about 16 anchors at four visits per
  anchor (fewer under lanes, which also publish predicted records).
- Capacity (default 32, at most 64): FIFO keeps about 8 anchors at four records per anchor, 16
  at capacity 64. At the default capacity a lag above about 8 has no effect.
- Source-tile revision lag 8: a tile that both routes and takes a local step commits twice per
  anchor, so the always-visited start tile's records expire after 4 anchors whatever the lag; a
  rarely visited tile's records are bounded by the other limits instead.
- `balance.memory`: at lag 1 it changes only which stale records are held; with a larger lag
  those records become readable, so balanced eviction changes anchor-exact training too.

Logs carry `memory_reach` per row: `memory_lag`, `reads` (live training-anchor reads in the
interval), `records_per_read`, `mean_lag` (mean shared-revision lag of the averaged records) and
`session_lag_max`; the report carries session totals. The tally (`State.reach`) is session-only,
so a resumed run restarts it; lane, replay and evaluation reads are not counted.

`connected.visit_norm` (default true) gates v4's post-visit RMS normalization. It is a `Knobs`
field (serialized only when false) set from the options by `Config::knobs`, and by
`tuning::set_connected` at every training step, wave and frozen replay, so evaluation and
completion use the checkpoint's setting whatever the thread held. Off, `tile_cache` keeps no
`output_norm`: the visible output is the raw residual sum, `raw_adjoint` is the identity and
`raw_rms` is measured, in the one cache every forward, gather, local backward and lane uses, so
forward and reverse stay consistent (the central-difference row in `connected_curve_contracts`).
The readout, routing and predictor features RMS-normalize their own inputs and are unchanged;
the returned correction scales with the inverse RMS at the readout, as before v4. The 1e3 visit
bound still applies, now to the stream itself: a run whose stream outgrows it stops with an
error rather than being stabilized. Tile bundles and the paged store hold no setting.

Fixture magnitudes (debug build, two visits, anchor-exact FP32, 48 anchors): without
normalization raw visit RMS stays near 1 (interval mean 1.008, maximum 1.10, against 1.007 and
1.12 with it), losses differ only in the third decimal and neither arm's loss falls, so the
fixture shows neither divergence nor benefit. Untested: whether the stream grows over a long run
without normalization (the v3 pilot's alignment growth is the risk), any quality effect of either
switch, and `memory_lag` under lanes, hybrid or waves.

### Shared sections

EXPERIMENT, a hypothesis with no measurements. `connected.shared` =
`{sections, wave, queue, teach, credit}` (absent is today's behaviour and omitted from identity;
present is run identity, defaults inside the block omitted). Admission: `sections` 1..=4
(required), `wave` 1..=64 (default 8) and a multiple of `anchor_batch`, `queue` (default true),
`teach` in [0, 0.5] (default 0), `credit` `"mix"` (default) or `"entry"`; `credit_mode:
anchor_exact` only; `connected.replay` and `restart.replay` are refused (replays do not read the
sections). Run admission requires `anchors`, `checkpoint_every` and `eval_every` to be multiples
of `wave`. Implementation: `connected_sections.rs` (storage), `connected_sections_config.rs`
(options), `connected_sections_check.rs` (validation, report), `connected_sections_read.rs` (read,
write credit, teaching), `connected_sections_write.rs` (wave loop, queue, write).

Storage. Each section is a `Tile` of the centre tile's shape (`Tile::new(geometry, START_TILE,
seed')`, so `tile.id` is `START_TILE` and slice-mode hashing treats it like the centre) with its
own Adam `Latents`, initialized exactly as a tile (FP32 masters from the trits under FP32). Its
seed block is the one a tile with id `TILES + k` would get, disjoint from every tile's and from
E's. Sections live in `State.sections`, not in the paged `Store`: the store's slots, heat, lease
flags, versions arrays and `quiescent` check are all sized `TILES`, and `geometry.shapes` refuses
ids >= `TILES`, so extending it would touch every paging path for four never-evicted bundles.
They are pinned (never paged or evicted), outside `resident_tiles`, and checkpointed as common
payloads `section-<k>.bin` in the unchanged tile codec (direction history all zero: sections
never track trit reversals), bound by the manifest and hash-checked on load; their hash,
version and steps sit in `connected.json` under `shared.bundles`. The report's `resident_bytes`
counts packed trits, masters, moments and the FP32 view. `causal plan`'s ledger has no
connected options and is unchanged.

Read (every anchor, training, evaluation and completion alike). With x the prefix's last row (the
state that entered the route before), r = mean over sections of `section.forward(x, prefix,
zeros)` (zero history: sections read no memory), and the route starts at `N(x + r)`, N the
post-visit RMS normalization when `visit_norm` is on (the identity otherwise), the convention a
visit's output gets. A section is an ordinary residual tile: its output is `s_k = N(x + delta_k)`,
close to x because x already has RMS 1 and delta_k is small, so r is close to x and the entry is
close to `N(x + mean(delta_k) / 2)`. The read therefore mostly re-adds x; what the sections
contribute is about half their mean residual change, which `read_delta` measures. The read is
detached in the reverse pass: the adjoint reaching x (and so E's last prefix row) is N's backward
of the route's input adjoint; no adjoint flows into the sections or through their attention over
the prefix. The full derivative is `d entry / dx = Nback (I + dr/dx)` with dr/dx close to the
projection orthogonal to x, so detaching roughly halves E's direct last-row gradient compared
with the true one. Sections commit only at shared-wave ends, so every anchor of a wave reads the
same frozen weights.

Write credit (per anchor, after `gather`). `credit: "mix"`: w_i = W(e x_i) / Σ_j W(e x_j) over
the route's visits, x_i = the visited tile's `TileSignal` distance / its `bar` (the Lambert law's
input), W the plasticity dial `connected_plasticity::dial` (principal Lambert W, capped at 4, so
x_i past about 80 bars ties). Equal weights 1/n when any visited tile's signal is unwarmed, its
bar is 0 (no `threshold`), `restart` is absent, or every W is 0; `lambert_share` reports how often
the Lambert weights applied. "Mix of local credit" is read as the mix of the visited tiles'
output corrections c_i (the unit-clipped adjoints at their outputs, the vectors memory records
and the predictor already treat as local credit): u = Σ w_i c_i is taken as the section's output
adjoint, and the section's gradient is its own backward from u at (x, prefix, zero history).
Mixing the visited tiles' parameter gradients instead is ill-posed (route rows differ in shape by
tile). `credit: "entry"`: the section's output adjoint is `unit_clip(g_r / K)`, g_r the exact
adjoint at the read (`Credit::read_adjoint`, the vector `gather` adds to the last prefix row) and
K the section count, since r averages the K outputs: the section's true gradient through the
read, up to the clip. In both modes `credit_alignment` is the lifetime mean of cos(u, g_r) (0 when
either is zero): a mix that does not point where the loss gradient points reads near 0. Each
anchor queues (x, prefix memory, the chosen adjoint); the queue is session-only and empty at
every wave end, which is where checkpoints, evaluations, STOP and the deadline fall
(`connected_wave::step` runs a whole shared wave).

Write (wave end, `connected_sections_write::finish`). m = the wave's mean x and c the queue's
`centre`, an EMA of earlier waves' m (weight `max(1/n, 1/16)` for the n-th wave, folded in after
the match; absent before the first wave, when no section has a key). Wave means share a large
common component (position `window - 1`, normalized rows), so uncentred cosines are all near 1.
match_k = `clamp((cos(m - c, key_k - c) + 1) / 2, 0.05, 1)`, 1 without a key, 0.5 when either
centred vector is zero; each section's last `match` is reported. With `queue`: every priority
gains its match, the highest priority wins (ties within relative 1e-9 go to the section written
longest ago, never-written first, then the lowest index), the winner pays `Σ_k match_k x r`,
`r = clamp(size / EMA(size), 0.5, 2)` (1 on its first write), size the L2 norm of its applied
Adam steps this wave, and then the mean priority is subtracted from every section, which keeps
them centred without changing any selection. Without `queue` the highest match wins, same ties,
priorities stay exactly 0. The winner takes one unit-gain `Mode::Local` Adam step per anchor of
the wave, each gradient computed against a clone of its wave-start weights, at `local_rate x
local_decay.factor(last anchor)` (no per-tile restart, plasticity or balance factor: sections
have none). Then the winner's key (stored uncentred) and size EMA fold m and size with weight
`max(1/writes, 1/16)`. `update_rms` is size / sqrt(locally updated coefficients). Queue state
lives in `connected.json` (`shared.queue`), never in a bundle; resume is bit-exact.

Teach (`teach` = tau > 0). A visited tile's exact gradient G(g) comes from its raw output adjoint
g: `gather` seeds the tile's backward with the unclipped upstream (`connected_anchor::gather`);
the stored correction is only its unit-clipped copy. The pull direction is `p = (y - s) |g| /
max(|y - s|, 1e-12)`, y the visit's output and s the anchor's frozen read r: `y - s` rescaled to
the visit's own adjoint norm (`Credit::adjoint_norms`), so tau is a true share of the update
instead of a raw `y - s` about 100 times the real credit. The local target `y - g` is blended to
`(1 - tau)(y - g) + tau (y - p)`, whose adjoint is `(1 - tau) g + tau p`; by linearity of the
backward in its seed the tile's credit becomes `(1 - tau) G(g) + tau G(p)`, one extra
parameter-only backward per visit. The adjoint propagated to earlier tiles and E is unchanged.
At tau 0 the blend is skipped, so updates are bit-identical to `teach` absent.

Monitors (`shared` in log rows and the report; lifetime values in both): per section `writes`,
`priority`, `match`, `last_write_wave`, `update_rms` of its last write and `weight_rms` (RMS over
its tile-matrix masters, null before a ternary section's first write); `mean_tile_weight` per
tile (mean w_i over its visits; collapse onto one tile shows as one tile near 1); `read_delta`
(mean RMS(r - x) / RMS(x), how far the read moves from the state it is added to; RMS(r) / RMS(x)
would read about 1 by the residual identity); `credit_alignment`; `lambert_share`;
`resident_bytes`. Each full evaluation adds `eval.shared`: `heldout_without_shared`, the same
held-out probes scored with the read skipped (route entry x, the no-shared path), with its own
`heldout_without_shared_partial`, and `tile_similarity`, the mean pairwise cosine of every tile's
residual change (output - input; the outputs themselves all sit close to the input) on the same 8
post-read route-entry states (round-robin first scored positions of the held-out probes, zero
history; `tile_similarity_positions`). The ordinary held-out score keeps its full budget share;
the read-free score runs after it under the same cumulative deadline and gets what is left (its
time counts against the later retention and sample shares too).

Cost per anchor: `sections` extra tile forwards (the read); amortized over a wave, one section
backward and one Adam step over one section (the write); with teach, one extra tile backward per
visit. Evaluation scores the held-out probes twice and runs 8 x `TILES` tile forwards. Memory:
4 pinned sections at width 128 under FP32 are about 4 x 4.6 MB (masters, two moments, view,
trits); the wave queue holds `wave` prefix copies (L x d x 4 bytes, 256 KiB at a 512-byte
window, so up to 16 MiB at wave 64); the write holds one cloned section and one gradient.

Limits: sections read no memory and receive no route-dependent input; under `mix` the write
credit is a proxy (other tiles' adjoints), not the section's own gradient; the read is detached,
so E learns nothing about it; the quantize audit rounds sections with the tiles. Tested: an
`entry` write on a fixed fixture input lowers that input's loss; Lambert weights on hand-warmed
signals (the flow fixture never warms one). Untested: any quality effect.

## Persistence and measurements

The store retains slot metadata and at most `resident_tiles` (default `workers`,
at most four) decoded tile/optimizer bundles, counting idle cache entries and
uniquely owned worker leases together.
Read-only access exposes immutable references. Mutable lease access captures the
pre-update trits for transition tracking and prevents return as a read-only lease.
Runtime revisions advance even when trits do not change. Dirty bundles remain
authoritative in memory until eviction or an explicit flush writes, hashes and
syncs them. Their slot hash is blank until that durable publication succeeds;
a failed writeback poisons the store rather than exposing an older disk state.
Completed
checkpoints copy tile payloads without decoding the whole model, and include E,
its optimizer, the v3 readout bias and its optimizer, predictor and optimizer,
shared memory, gains, cursor and counters.
Hashed, fsynced staging becomes a checkpoint only after complete publication.
Checkpoint save flushes dirty cached bundles before copying payloads.
Restore validates one tile at a time; evaluation also pages. Superseded runtime
blobs can be retired after successful durable writeback because checkpoints are complete
independent copies. Checkpoint-backed source files are never retired as scratch.

Each tile also persists the last nonzero transition direction for every trit.
Runtime counters distinguish changed coefficients from transitions that reverse
that saved direction, including reversals across a restored checkpoint. These
counters reset per runtime session; the direction history persists.

Evaluation freezes all model/predictor parameters and uses its own scratch
memory. It flushes the store before recording its immutable-state fingerprint.
Score each public held-out byte before observing it. Free generation
distinguishes emitted-byte feedback from true target observations. Evaluation
must not train the predictor, alter gains, or mutate training memory or weights.

The owned run uses the existing STOP, resource admission, Job cap, pinned binary,
receipts and cleanup. Initial weights and every learned or adaptive state are
fresh; old `causal-latent-v2` checkpoint contents are not imported. Mechanism
counters, predictor quality, cross-lane memory reads, route visits, gains, trit
transitions, paging I/O, decoded residency, physical memory and generated
samples accompany held-out loss. Source and fixture checks establish the curve connection; the first run
does not separately measure curve-response quality. One bounded run can establish execution and measured behavior;
it cannot establish convergence, better routing or long-term language ability.

## Memory scaling

At fixed tile dimensions, context, worker count and global-state sizes, adding
more tiles is intended to keep the dominant resident tile/optimizer working set
approximately fixed. Cached entries and worker leases share the same resident
cap. Total parameter count is not the same quantity as resident parameter count.
The tile count is a compile-time constant (§Tile count): seven by default,
nineteen under the `tiles19` feature. A growing tile bank has not been
implemented, and no 19-tile run has been measured.

A useful accounting model is `resident_bundles * tile_bundle_bytes + bounded
worker_scratch + shared_state + per_tile_metadata`. The metadata term grows with
tile count, while the complete on-disk tile/optimizer bank and checkpoint copies
grow with all parameters. Wider tiles, larger FFNs, more simultaneous workers,
longer contexts, or larger global embeddings/predictors/memory also increase RAM.
For width d, FFN width m and r route actions, a tile has
`8*d*d + 3*d*m + 3*m + 3*r*d` coefficients; widening the current m=3d shape has a
dominant quadratic cost even when only four tiles are resident.

The anchor-186164 checkpoint contains 281,216 coefficients per outer tile and
282,368 in the center. An outer encoded bundle is 3,732,852 bytes (3.56 MiB):
70,304 bytes of packed trits, 3,380,736 bytes of FP32 shadow weights and Adam
moments including route slots, 281,216 bytes of trit-direction history, and
headers. The center bundle is 3,752,724 bytes (3.58 MiB). Training state, rather
than packed weights alone, dominates these bundles.

Four decoded bundles (about 14.26 MiB of array payload) are read-only during
lane jobs in v3; four job gradients add about 4.30 MiB and the per-tile gradient
sums held until all lanes finish add up to seven times 1.07 MiB (7.51 MiB); anchor
waves (§Anchor waves) hold a route and a local sum per touched tile outside the
resident cap, at most 2 x 7.51 MiB (15.03 MiB) at 7 tiles and 2 x 20.42 MiB (40.84
MiB) at 19 (`4 x` the tile coefficients of §Tile count, per set); Adam
proposals and one before-update trit snapshot exist only for the single tile
being updated. This arithmetic excludes other allocations and is not an RSS
guarantee. The main always-resident arrays at width 128 are embedding plus Adam
(401,408 bytes), predictor plus Adam (789,504 bytes from v5; 592,896 before), and
the vectors in 32 shared memory records (65,536 bytes). One 512-byte prefix uses
264,704 bytes. Training data is streamed; the current public held-out bytes occupy
1,098,000 bytes.

Tile checkpoint copying uses a bounded streaming buffer and restoration loads
tiles sequentially. Common state is cloned and JSON-serialized, so checkpointing
temporarily duplicates some global state. OS file caches and allocator overhead
are outside the explicit bundle count. The earlier 52.48 MiB measured peak
working set and host-memory stop do not establish a parameter-capacity limit:
that stop observes system-wide available memory, independently of the trainer's
512 MiB Job commit cap.

## Tile count

`geometry::RADIUS` is 1 by default and 2 under the Cargo feature `tiles19`;
`TILES = 1 + 3 * RADIUS * (RADIUS + 1)` (7 or 19). It stays a constant because
about 66 fixed-size per-tile arrays live in state, reports and serialized records;
a run-time tile count is the growth spec's Phase 2. One process-wide `Mesh`
(`geometry::mesh()`) holds axial coordinates in increasing `q`, then `r`, so the
order is point-symmetric and the centre `(0, 0)` is the middle id,
`START_TILE = TILES / 2` (3 or 9). Radius 2 ids: `q = -2`: 0-2; `q = -1`: 3-6;
`q = 0`: 7-11 (`r = -2..2`, centre 9); `q = 1`: 12-15; `q = 2`: 16-18. A tile's
route actions are its neighbours in ascending id order plus halt: 7 for the
centre and inner ring, 5 for an outer-edge tile, 4 for an outer corner (radius 1:
7 at the centre, 4 elsewhere). Tile seeds stay `seed + 16 + 16 * id`; eleven
matrices fit in each 16-wide block, so no tile's seeds collide at any id.

At width 128 (m 384) a tile has `279,680 + 384 * r` coefficients: 19 tiles hold
5,353,472, plus 82,176 shared (E and the predictor), 5,435,648 in all (7 tiles:
2,051,840). `causal plan` prints `tiles`, `radius` and `start_tile`, and its ledger
uses the centre tile as the largest.

Identity: a run identity gains `"tiles": TILES` only when the count is not seven,
so every seven-tile identity and checkpoint is unchanged and a 19-tile run with an
otherwise equal config has a different identity. A run config may state its build
with a top-level `"tiles": 19` (absent means 7); train and the `complete` command
refuse a config whose count differs from the binary's, naming the build to use.
The field is checked, never identity, so existing configs keep their bytes and
identities. Checkpoint loading compares the manifest's tile count first: another
build's count (7 or 19) is refused as a foreign checkpoint that names the needed
build, before any array is decoded, and is left in place, never quarantined.
Completion runs the same check on the source manifest before copying anything.
Any other length is plain corruption and takes the integrity path (quarantined,
then fallback to an older checkpoint). Changing the feature changes the binary,
so resumes run under the pinned build that started the run (`scripts/AGENTS.md`). `resident_tiles` admits `workers..=TILES`, and
`restart.plasticity` takes exactly `TILES` values. The core v2 path also builds at
19 tiles but has no 19-tile goldens; only the connected path is a Phase 1 target.

## Read-only checkpoint completion

`hexweave complete --config <run.json> --checkpoint <aNNNNNNNNNNNN> --prompt
<text> --max-new-bytes <count>` supports this connected protocol, including beside
a live owner (`owner_lock_present` reports whether the run lock file existed).
Completion first copies the exact named checkpoint's hashed payload set into a
unique owned `work/inference/` directory with bounded, hashing reads, then loads
that snapshot through the existing verifying loader, which restores tiles into
its own scratch store. The live source is thus read once, the reported hash
covers exactly the loaded bytes, and later pruning cannot fail the request;
pruning during the copy fails it without affecting training. Inference opens
source files only for reading and never takes the run lock. Host memory admission
is the caller's job: inference (about 40 MB peak when measured) runs outside the
owner's Job, and the owner requests STOP after ten one-second samples below
256 MiB available, consuming one bounded recovery restart. Check available
memory first, as the `work/inference` scripts do with `owner_resources`.
It never calls training resume and rejects identity or corruption failures without
quarantining or falling back. It opens the verified corpus only for the public
held-out text that fills the generation window (below); it reads no evaluation files.
Config seed, connected options and the checkpoint cursor's corpus manifest/window
must agree. The run's STOP and runtime directories remain untouched.

Generation reuses frozen evaluation replay: prompt observations and generated
bytes update isolated scratch memory with their existing provenance, while the
source training memory, weights, predictor, optimizer and gains stay frozen.
Replay starts with empty scratch memory and event epoch zero; it does not carry
the checkpoint's training-memory records into generation. Warmup observes at
most the last two prompt bytes and runs predicted local replay. The full recent
prompt still enters the forward prefix. Thus trained parameters persist while
the dynamic signal-memory context is rebuilt for the request.
The corrected `connected-story-window-frozen-local-lanes-v2` generation policy
uses the recent bytes of the explicit `[END] + prompt + generated` stream. It
does not prepend another START marker. Warmup observes only the last one or two
real prompt bytes, never the synthetic leading END. Earlier connected generation
added that extra START for short prefixes and observed END for a one-byte prompt.
This correction affects generation; corpus scoring retains its actual stream's
START/window convention, and the running pinned training executable is unchanged.
The prompt plus leading marker must fit model context; the output cap is 1,023
bytes. A 120-second deadline starts before loading and is checked between bytes.
Policy v3 (2026-11-11, `connected-full-window-frozen-local-lanes-v3`): the short
`[END] + prompt` opening put the newest byte far from position `window - 1`, where
training always puts it, so earlier connected samples and next-byte probes were out
of distribution. Completion and evaluation samples now open with held-out text
ending at a story boundary, filling the window (`decoding::story_opening`, shared
with the reference path, `AGENTS.md` §Generation). `complete --top-p` selects
seeded nucleus sampling; evaluation samples use temperature 0.8, top-p 0.9, seed =
prompt index. An empty context (tests only) keeps the v2 opening.
Hash verification and cleanup may complete after that cooperative deadline.
An existing training STOP does not terminate this separate inference request.

Fingerprints compare the snapshot (whose hash is that of the source bytes read)
and complete loaded training state before and after generation. Tile fingerprints stream actual cached packed
weights, Adam weights/moments/clocks and trit-direction history through the
existing codec. Cache order and read counters are excluded. Loading and hashing
retain at most four decoded bundles. RAII owns scratch cleanup on success and
error; successful JSON reports cleanup, residency, deadline and fingerprint
evidence. The training protocol, checkpoint encoding and pinned run executable
are unchanged by this inference command.

`--trace-bytes N` records at most 16 generation readouts, including the full 256
finite logits, selected byte, probability, runner-up margin, ties, space rank,
visible prefix, route tiles and memory evidence counts. A scalar tied-embedding
transpose provides an independent readout-index check. The finite greedy selector
chooses the lowest byte index on a tie, including signed zero. Trace zero is the
default and performs no diagnostic readout work.

`--feedback prompt-only` requires a nonzero trace and labels the result as a
diagnostic ablation. It preserves prompt warmup and freezes subsequent memory
evolution: generated observations, their local replay and event aging are all
skipped. The default `--feedback generated` retains that post-prompt evolution.
Comparing these modes tests dependence on generated feedback; it is not a change
to training or an isolated test of a single evidence weight. Repetition alone
does not identify a decoder bug: the reported logits must establish whether the
selected byte is the model's actual maximum.

### Context reach

`hexweave reach --config <run.json> --checkpoint <aNNNNNNNNNNNN> [--spans 32]` is a read-only
diagnostic on the same snapshot path as completion (`connected_completion_reach`,
`connected_eval_reach`). For each retained length k in 1, 2, 3, 4, 8, 16, 32, 64, 128, 256 below
the window, then the full window, it rescores the standard held-out probes with only the newest k
bytes of every window left true; each older position holds a byte of another held-out span
(Khandelwal et al. 2018, arXiv:1805.04623, perturb distant context). The window stays full
length, so the newest byte keeps its trained position. Rows report bits per byte overall and by
target class: the first, second, third and later bytes of a run of ASCII letters, and every other
byte. The full-window row must equal the standard held-out score or the command fails.

Scope: only the prefix rows are perturbed. The replay's event memory still carries records of the
preceding scored positions, so a row measures what the prefix beyond k bytes adds, not the
model's total reach; small-k rows are therefore a lower bound on the loss at truly k bytes. The
replacement text comes from a span at least half the probes away that fills a whole window, so
it is never cycled when one exists.

### Contextual rows

EXPERIMENT, a hypothesis with no measurements (`.omc/plans/connected-v6-growth-gpu-2026-11-16.md`,
"Context plan" step 1: contextual rows with no writer credit). Context reach showed trained
checkpoints take 94% of their benefit from the newest 4 bytes, because a window row is only a
byte plus its position (`Shared::prefix`) and only the newest position is routed.
`connected.context_rows` (bool, default false) lets older rows carry what the routed stream
computed there. Off is omitted from serialization and identity, so existing configs,
checkpoints and golden trajectories are unchanged; `true` is run identity. Admission:
`credit_mode: anchor_exact` and `anchor_batch` 1; refused with `connected.shared` (keeps the
first pilot attributable) and with `connected.replay` or a nonzero `restart.replay` (a replayed
anchor would read its rows without the states it trained with). Implementation:
`connected_context.rs` (buffer, rows, fill, refill, admission, monitors), `Shared::prefix_with`.

Definition. Position k of a stream is the prefix whose newest byte is the k-th visible byte:
training anchor a (keyed by its anchor index; its newest byte is train byte a-1, or `[START]` at
an epoch's first anchor), evaluation index t of `window_before(bytes, t, window)`, generation
index t of `recent_window(text[..t], window)`. Its state s_k is that position's routed final
output (`Trace::output`, the vector the readout consumes, width d). For a prefix of L bytes whose
newest byte is position k, row i (i < L-1) is position `k - (L-1-i)`, and
`row_i = RMS((E[:, b_i] + p_i) + s_{k-(L-1-i)})` when that state is buffered; the newest row
(no state yet) and rows without a buffered state are built exactly as before. s is a detached
constant: no adjoint reaches it or the tiles that produced it (no writer credit). E's gradient
is unchanged in form: `add_prefix_gradient` differentiates the RMS at the stored row values and
inverse, which is the exact derivative with respect to the pre-normalization sum, and that sum's
derivative with respect to `E[:, b_i]` is the identity whatever constant s adds (pinned by a
finite-difference oracle in `connected_context_contracts`).

Buffer (`Context`, session-only, never checkpointed). States of consecutive positions, capacity
`geometry.context`; a pushed key that does not follow the newest restarts it. Training pushes
the live anchor's output right after its forward pass (`connected_step::forward`); nothing reads
the buffer between that push and the anchor's end, since lanes, sections, waves and both replays
are refused. Stream restart: at an epoch's first anchor (cursor offset 0, prefix `[START]`) the
buffer is emptied before the push. Rows never reach across the wrap anyway (the prefix clips at
`[START]`, so row keys stay at or after that anchor); the clear makes the rule explicit.

Warm-up (frozen replays: held-out and retention probes, context reach, evaluation samples,
`hexweave complete`). With the option on, a replay starts observing a full window earlier:
`Replay::observed_from` moves the first observed position from `start` (two before the first
scored byte, or the prompt's last two bytes in generation) to `start - (L-1)`, L the length of
`start`'s prefix. Every warm-up position runs over true context bytes exactly like a scored
one (forward, push its state, observe its true next byte and publish records into the replay's
scratch memory) but is not counted in the loss or in `prompt_observations`; the two existing
observed positions are part of this range, never observed twice. When the first counted byte is
scored, buffer and scratch memory look like a replay that had been running for a window. The
earliest warm-up positions still see partly empty buffers and memory (a span far from its
stream start is never reproduced exactly). Generation then pushes every generated position
(also under `--feedback prompt-only`, which freezes memory, not this buffer). Off, the replay
observes from `start` exactly as before.

Remaining train/evaluation difference. Replays use greedy routes; training states used the
training routes, `route_training: sampled` by default, which visit other tiles than the argmax.
Replays also read their own scratch memory with every record at lag 0, while training reads
under `memory_lag`. Training states were therefore produced by a noisier route distribution
than the greedy states evaluation adds; the likely direction is that evaluation rows are
slightly cleaner than the rows the model was trained on, so the measured benefit can be
mildly optimistic for sampled-route runs and exact in kind for `route_training: greedy`.

Context reach. Warm-up positions read true bytes (no splice); from `start` on, rows older than
the newest k are raw foreign rows (their state dropped) and each spliced pass pushes its own
state. Kept rows carry states that were computed from true older context during the warm-up and
through the chain of earlier kept rows, so with the option on a small-k row no longer isolates
raw-byte reach: read it as "loss when the distant rows' bytes are foreign but the newest k rows
keep whatever their states learned from further back", an upper bound on what the raw newest k
bytes alone could give, and compare k rows across checkpoints rather than against an
option-off run. The full-window row still equals the held-out score (no splice, same warm-up).

Tile similarity. The shared-section `tile_similarity` probe (`connected_eval_probe::similarity`)
builds raw rows; it only runs with `connected.shared`, which this option refuses.

Resume (`connected_context::refill`, called by `connected_train::run` after loading, through
the forward-only `connected_context::fill`). The buffer is refilled for the positions the next
anchor's rows read, under the restored weights, each with its own anchor's training route draw
(`connected_route::anchor`, the live anchor's salted draw) and memory cut off before its label
(`query_before`). Refilled states see only the restored memory, i.e. the newest records the
checkpoint kept, so the oldest refilled rows are built on less retrieval than training had
when it computed them; and training had computed them under their own anchors' weights. A
resume under `context_rows` is therefore not bit-exact against an uninterrupted run; every run
with the option off keeps its bit-exact resume contract.

Monitors: each `log.jsonl` line carries `context` = {`rows` (older prefix rows of live training
anchors in the interval), `filled_share` (share of them with a state), `state_to_row_rms` (mean
RMS(s) / RMS(E[:, b] + p) over filled rows)}; fills are not counted. Staleness (distance between
a cached state and one recomputed under current weights) is deferred: it needs recomputing old
prefixes.

Cost. Training: one d-vector clone per anchor and two RMS reductions per filled row (the prefix
build was already L x d). Buffer: up to `geometry.context` x d x 4 bytes (512 KiB at width 128).
Evaluation: a held-out probe processes up to `window - 1` warm-up positions before its 130
previously observed ones (each a forward, a gather and a memory publication): 641 instead of 130
at a 512-byte window, about 4.9x the probe's scoring time (estimated, not measured), so more
`partial` scores inside the same budget. Retention probes (window + 2 bytes of left context)
and evaluation samples add the same warm-up; context reach repeats it for every k row.
Untested: any quality effect, whether
states grow stale enough to matter between their push and their last read (up to `window - 1`
anchors), and the resume discontinuity's effect on loss.

## Full-run measurements

The optional `connected_eval` object configures assessment time, retention probe
count and generated sample length. It is outside the training numerical identity.
Public held-out probes have fixed offsets; each reports actual target counts and
whether scoring completed. Separate fixed training-span probes become eligible
only after every scored target has already been observed by training. They use
the same frozen model and isolated memory as held-out assessment.

The first complete score for a previously observed span is a measurement-session
reference. Later complete scores over identical targets report their change;
partial scores do not produce a comparison. Resume begins a new reference
session. These measurements describe changes on previously observed text, not
verified mastery or semantic understanding. Interval predictor statistics report
predictions before individual fits between logged anchor boundaries, separately
from cumulative statistics.

The full study started from a fresh current-size model with a 24-hour or one-pass
budget. Its authorized continuation uses the verified checkpoint and bounded,
opt-in owner recovery after clean memory stops; see `scripts/AGENTS.md` at
repository root. Dynamic tile growth and training from corpus-verified generated
attempts remain later stages. The throughput change preserves the existing
training arithmetic and evidence provenance.
