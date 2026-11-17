# Connected local training: implementation and first owned study

Status: complete and independently verified, 2026-11-03 America/Denver.
Authorization: user requests complete connection, planning, execution, launch and monitoring; explicitly requires training from scratch.

## Required outcome

Implement and run `causal-connected-v1` through `hexweave train --config` and the existing capped core owner. Fresh seeded tiles, tied embedding, predictor, optimizer, shared memory and modifiers; a new output directory with resume false. Do not load old weights or checkpoint state. Existing public prepared train/dev corpus may be used in place. No private evaluation material, downloads, cloud, servers or larger geometry.

The architecture must actually use predicted local corrections for tile training, one shared structured memory for routing and activation-curve conditioning, concurrent bounded tile work, positive direction-preserving Lambert W update modulation, and paged tile plus optimizer residency. An anchor-only run or all-resident substitute does not satisfy this objective.

## Execution and causality

1. One true anchor is scored under a frozen paged model before its target is observed. Reverse its realized route for reference corrections and tied-E credit. Fit a resident correction predictor and update routing from actual outcome evidence. Main tile transformations learn through predicted local corrections rather than falling back to direct anchor tile training.
2. Publish observed correction records with source tile/version/epoch. Anchor supervision remains explicit, distinct from predicted evidence.
3. Eight logical local lanes operate on that same already-observed visible prefix; each starts at the center and ends at halt or four visits. Up to four CPU workers hold distinct tiles. Same-tile collisions queue. This avoids later corpus prefixes leaking targets into earlier examples.
4. Each job reads a coherent shared-memory snapshot, forwards one tile, predicts correction, differentiates that local tile, scales the complete Adam update vector by one positive saved gain, and emits a new memory contribution, next legal route and modifier evidence. Local jobs receive no target byte and perform no full-route reverse pass.
5. Publish results in declared lane order within bounded worker groups. Later groups read the updated shared memory during the same local phase. This deliberately couples lanes; it is not eight frozen independent anchors.
6. Complete all lanes before advancing the target cursor or checkpointing. Failures poison the in-memory step and cannot publish a complete checkpoint.

## Numerical contract

- Shared memory is a bounded collection of structured observed/predicted records, with contextual keys, tile identity, age and revision. Retrieval produces one unit-bounded detached vector; workers share one authoritative memory owner.
- A compact seeded FP32 correction predictor consumes normalized tile input/output, memory and hex coordinates, fits clipped actual corrections, and supplies the local upstream vector. Predictor fitting and its optimizer are persistent and measured separately.
- Existing learned pulse projection conditions both routing and curve shape. Three bounded controls from learned pulse heads perturb knot heights; their adjoints flow through the pulse projection and its input weights. This goes beyond output gain modulation.
- Lambert W is evaluated on the principal nonnegative branch. A signed evidence control yields bounded positive gains allowing damping, amplification and brief spikes. The gain policy is evidence-driven, not claimed learned unless fitted. One scalar governs the entire tile update; cap total step norm uniformly. Do not independently clip coordinates and claim unchanged direction.
- Tile updates use continuous Adam state, followed by ternary projection. SIMD applies to bulk update scaling and existing packed matrix products; no speedup claim without measurements.
- Actual route outcome credit and predicted local correction remain separate. Routing probabilities also select training-lane successors under explicit seeded sampling; inference can use greedy decisions.

## State and storage

Dedicated connected state/identity avoids weakening v2 fixture contracts. Metadata-only paged store checks out at most four uniquely owned tile+Adam bundles. No all-tile decoded restore or evaluation. Lease freshness uses runtime revision/hash, since optimizer state changes without trit changes.

Checkpoints quiesce lanes, stream complete tile blobs plus shared E/optimizer, predictor/optimizer, structured memory, gains, cursor and counters into staging, hash/fsync and publish a complete directory atomically. Resume validates identity and one tile at a time. Partial publication and corrupt checkpoints must never become mixed resumable states. Fresh launch refuses prior checkpoints.

## Configuration and rollout

Optional `connected` config selects this protocol with lanes 8, workers 4, visits 4, memory capacity 32, explicit predictor rate and gain bounds, plus update norm bound. Connected mode requires corpus data, ternary forward, history on, batch 1, fixed base rate for the first study. Unsupported combinations fail explicitly. Existing v2 is a named reference, not the architecture under test.

First owned run: fresh seed 7, base rate 0.003, prepared TinyStories corpus (manifest `2710d21277ed6e1817477ae6a61444ace674b7dc371a1ec3ea304a9e93575af8`), window 512, 900-second loop deadline, 16,384 anchor cap, checkpoint every 1,024, evaluation every 2,048. Evaluation uses the same eight-lane memory replay with frozen parameters and scratch memory. Because this adds significant paged work, the first study uses one 128-target public held-out span and one at-most-64-byte generated sample, rather than the earlier proposed eight spans. Per-evaluation time/STOP bounds report actual counts and any partial scoring explicitly; no quality comparison to the old larger evaluation is implied. Initial finalization overhead is reported separately. Existing 512 MiB Job cap, memory floors, STOP grace and cleanup remain enforced.

## Review resolutions

Observed actual records alone fit the predictor and calibrate gain, using predictions made before current-target fitting. Predicted/self-generated records retain provenance and lower retrieval weights. Local prefixes are rebuilt after E updates. Evaluation freezes every parameter and gain, uses isolated scratch memory, scores before observation, and tags emitted-byte observations distinctly. Dispatch-group inputs are immutable; publication happens only between complete groups. Route-only Adam has a separate P route-row slot and cannot move main parameters or moments. Exact retrieval, gain formula, ageing and direction scope are documented in `src/causal/CONNECTED.md`.

## Evidence before launch

Independent implementation review precedes one integrated test/lint sweep. Tests must cover actual predictor influence on local updates, shared-memory cross-lane influence, learned curve finite differences, positive whole-tile gain direction, bounded decoded tile/Adam residency including evaluation/restore, collision serialization and deterministic publication, exact continuation, corruption/partial checkpoint rejection, and causal target boundary. Reuse existing fixtures; no duplicate generic harnesses.

Monitoring records predicted applies and predictor fit error/agreement, observed/predicted memory publications and cross-lane reads, routing/curve responses, gains and bounds, trit changes/reversals, paging bytes/time and decoded high water, physical owner memory, public held-out predictive quality and samples. Verify real progress via live owner/process handles, then terminal receipt, complete checkpoint and cleanup. A low loss alone is insufficient.

## Completion audit

The goal is complete only after all connections are implemented, reviewed, integrated checks pass, the fresh owned bounded run is launched and monitored through verified termination, and a report links mechanism counters, resource evidence, checkpoint integrity and quality limitations. No convergence or smarter-routing claim follows from one bounded run.

## Prelaunch evidence

Independent engine/numerical/checkpoint review approved the final source after
fixing canonical optimizer-slot ordering, zero-evidence gain calibration and
second-moment validation. The integrated sweep first found storage-split import
and lint issues, fixed together before proceeding. The completed sweep is
`work/validation/4b54829c8a11477a93b9c7708d67c8c4`: structure, formatting,
clippy with warnings denied, Rust and Python tests, all three allocation-free
planning profiles and the smoke fixture passed. New contracts exercise actual
predictor influence on latent weights, memory/curve/router connections, exact
threaded checkpoint continuation, direction-preserving gains, SIMD parity,
paging bounds, corruption and resumed trit reversals.

Fresh launch config: `work/core-runs/connected-v1-s7-20261103.json`.
Build-source hash manifest: `work/core-runs/connected-v1-s7-20261103-source.json`.
The output directory was absent during independent launch review. The existing
prepared public corpus manifest was verified; no checkpoint is an input.

## Measured first run

The capped owner ran from 15:07:01 to 15:22:59 UTC and exited 0 at the configured
deadline. Initial state was anchor 0, no predictor fits or local updates, empty
training memory and neutral gains. It ended at 1,201 anchors with 28,411 predicted
local updates, 3,564 observed fits, 28,409 cross-lane reads and peak 4 decoded leases.
Peak process working set was 49.574 MiB. Both 128-target frozen dev probes completed:
9.87813 to 4.85555 bits/byte, with repetitive 64-byte `e` samples in both.

The predictor's cumulative prefit MSE was 2.460 times the zero reference and mean
cosine 0.13126. Tile reversal fraction was 83.75%; requested gain amplification
occurred 563 times and damping 16,750 times, with net applied/base norm sums 0.7232.
Paging cost 165.534 GiB of application reads and 108.363 GiB writes including
evaluation. These results establish connected execution, not useful predictor
quality, convergence, speedup or larger-model feasibility.

Final checkpoint `a000000001201`, report, assessment and owner receipts are under
`work/core-runs/connected-v1-s7-20261103`. The owner session returned 0, OWNER.lock
was removed and training PID 108248 no longer exists. Both the first 1,024-anchor
checkpoint and the final checkpoint passed independent full payload/hash/state
inspection. Final checkpoint counters, report, log and owner receipt agree; no
partial publication or runtime error remains. The fresh production run has not
been resumed.
