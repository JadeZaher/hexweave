# Hexweave status

Updated 2026-10-28 America/Denver. **The core training path is implemented and
is the build; no run is in progress. No useful-assistant result or finite
training ETA is established.**

## Implemented

| Area | State |
| --- | --- |
| Core training | `hexweave train --config <json>` (`src/causal/core_train.rs`): the v3 forward model on the 2,051,840-coefficient candidate with latent-weight Adam, greedy eval-matched routes and an anchor-only schedule; all tiles resident |
| Owner | `scripts/core_owner.py`: `work/core-runs/<name>` rule, single-owner lock, pinned binary, 512 MiB Job commit cap, sampled 256 MiB memory floor, graceful STOP with a 180-second grace, start and end receipts |
| Checkpoints | Hash-manifested and fsynced, newest three kept. Resume identity-checks every readable manifest first, is bit exact, quarantines integrity failures and refuses to restart from anchor 0 when every checkpoint fails. Checkpoints from the `8e7a0eb` binary (earlier identity formula, earlier resume logic) resume only with their pinned `out/bin` copy |
| Data | `science`: pinned QA dataset, development-split evaluation. `drill` v2: the assay curriculum re-dealt to the harness `Shuffled` stream, with key-swap and attention probes |
| Drill v2 harness parity | `core_contracts`: 12 production-drill anchors equal the validated tuning harness bit for bit, and the drill stream equals `Shuffled` across cycle boundaries |
| Evaluation and telemetry | Held-out, teacher-forced science and generation replay history like training (`replay.rs`); drill evaluation stays harness-identical. Each log line carries per-phase wall time and per-tile route counts (`telemetry.rs`) |
| Software fixtures | FP32 `smoke`, allocation-free `plan` (tiny, 81M, 287M), `causal plan`, ternary scalar/AVX2 kernels with parity tests |

Design and invariants: [core path](../src/causal/AGENTS.md) and
[core owner](../scripts/AGENTS.md).

## Archived

The v3 owned-study chain (owner requests, launch receipts, DiskStore
persistence, release/seal/retention, prompt evaluation), the six-arm
retained-text study scripts, the `pulse` and `text` experiments with their
owners and campaigns, and the closed predictor and comparison diagnostics. They
no longer build; see [the archive guide](../archive/README.md).

## Planning only

The 81M and 287M profiles stay allocation-free plans until independent
numerical, resource, ownership and checkpoint reviews pass. No full-size
training fit has been measured.

## Current runs

- `work/core-runs/drill-v2` (seed 7, rate 0.003, 128,000 anchors in two launches,
  peak working set 101 MiB): bit-identical to the harness seed-7 evaluation at
  anchor 1,600; 8-16/32 key-swap argmax flips from 80k through 121.6k anchors
  (interval train loss 0.121-0.169, below the 0.139 key-blind floor at 8 of
  those 14 evaluations), then 0/32 at 124.8k and 128k. Paraphrase stays near
  chance. Not yet stable: no learning-rate decay or best-checkpoint retention exists.
- `work/core-runs/science-overnight` (seed 7, rate 0.003, 72 train rows; stopped
  by the operator at 578,547 anchors, epoch 86): dev NLL bottomed at 2.430 per
  byte at 80k anchors (epoch 12) and rose to 3.089 by the stop while train loss
  kept falling; 0/24 dev cases valid JSON, exact answer or exact citation at
  every evaluation. Dev generations loop and cite train source T06 although each
  dev prompt names its own D01-D06 SOURCE: the citations were memorized.
- `work/core-runs/science-redeal-s7` and `-s17` (same settings plus
  `redeal_citations`, 6-hour deadline, 1,045,861 and 930,466 anchors, peak
  working set 93 MiB): best dev NLL 2.355 (100k anchors, epoch 15) and 2.366
  (140k, epoch 21), both below the 2.430 without re-deal; teacher-forced dev
  citation NLL fell to 2.402 and 2.499 (3.069 at best without re-deal) and kept
  improving while total dev NLL overfit to ~3.1 by the deadline. Greedy dev
  generations stayed degenerate: at most 1/24 valid JSON at any evaluation, no
  exact answer or citation, and a `"citation":"` field in at most 2 of 18
  answerable cases. Re-dealing removed citation memorization; a copy circuit
  for the variable-offset citation did not form within ~1M anchors. Best states
  are in each run's `out/best/`.
- `work/core-runs/tiny-ternary-s7` and `tiny-float-s7` (TinyStories corpus,
  window 512, seed 7, rate 0.003, 8-hour deadline; 1,409,322 and 1,829,605
  anchors, one pass, peak 92 MiB each): best held-out 2.623 bits/byte
  (ternary, 1.16M anchors) and 2.612 (f32, 1.52M); final 2.761 and 2.669.
  At equal bytes the two curves agree within evaluation noise (1M: 2.756 vs
  2.787); f32 trained ~30% faster (66 vs 50 anchors/s, scalar trit decoding).
  The f32 model ternarized afterwards scored 2.681 (+0.01); the ternary model's
  latents read as f32 scored 3.015 (+0.25). A byte 5-gram trained on the same
  bytes scores 1.68 (1 MB) and 1.59 (2 MB). Greedy story continuations loop on
  single letters. Training (prequential, no byte seen twice) logged 2.19-2.25
  bits/byte versus 2.67-2.76 held-out. These held-out numbers passed zero
  history, as did every evaluation logged before history replay.
- `work/core-runs/rescore-float` and `rescore-ternary` (the two best states
  above, evaluated once with history replay): 2.6166 bits/byte (f32; 2.612
  zero-history) and 2.6226 (ternary; 2.623). Alternate modes at these same
  best states, zero-history: 2.6536 (f32 read as trits) and 2.9733 (ternary
  latents read as f32) -- not the 2.681/3.015 final-state numbers below;
  replay moved them to 2.660 (+0.0060) and 2.972 (-0.0015). History (zero vs
  up to two anchors) is therefore at most a 0.005 nats/byte effect; training's
  real window is 1-2 anchors (now logged as `history_len`), not Replay's
  fixed two (see `src/causal/AGENTS.md` §core). Frozen weights score 2.791
  (f32) and 2.941 (ternary) bits/byte on four fixed 128-byte train spans
  (`train_frozen_*`, `rescore-*-3`): those spans were mostly unseen
  train-file text (offsets beyond the ~1.2-1.5 MB the best states had
  reached), and the frozen scores there are no better than held-out
  (2.617/2.623), so the held-out spans were not an unusually hard sample.
  The byte-matched `train_ahead_frozen_*`/`train_ahead_prequential_*` pair
  now measures the online-adaptation gap directly on bytes the run has
  actually reached; under v2 (fixed weights within a step) the logged
  training loss is expected to rise toward the frozen numbers with no
  quality change. Greedy continuations still loop.
- `work/core-runs/profile-telemetry` (TinyStories, window 512, ternary, 3,000
  anchors, peak 96 MB): ~21 ms per anchor at 48 anchors/s. Latent Adam apply
  27%, per-tile backward 21%, route backward (`gather`) 19%, predictor and
  history 13%, forward 10%, window embedding 7%, everything else ~3%;
  evaluation 27-31 s each. Routing shows an early transient lock-in: 48
  distinct routes across anchors 1-1,000 (tile 5 touched on 107 of them), down
  to 3 distinct routes by 2,000-3,000 (`[3,1,4]` on 986 of the last 1,000).
  Over the full tiny-* runs every tile was updated (`latent_steps`): minor
  tiles received 10-20% of the start tile's update count, and every tile
  reached at least ~9.7% of anchors. The two arms favored different tile
  sets, so usage is run-specific, not a fixed collapse. Exploration (e.g.
  `no_halt`) is a measured experiment to try next, not a fix for a proven
  capacity problem.
- `work/core-runs/drill-acceptance` (drill v1, 96,000 anchors): key-blind
  throughout (0/32 flips), superseded by v2's harness curriculum.

## Closed history

The authorized retained-text study (2026-10-25) failed physical admission
before worker creation: 3,078.4289 ms external wall, exit 1, zero training
opportunities or observations
([outcome](../../nonmonotonic-training-lab/work/f-retained-assay-drive/RUNTIME-OUTCOME.md),
[terminal verification](../../nonmonotonic-training-lab/work/f-retained-assay-drive/TERMINAL-VERIFICATION.md)).
Earlier, E3 failed the public quality gate, E4 ended at a resource stop without
final evaluation, and the bounded anchor-only/current comparison had 0/12 exact
final answers
([E3](../../nonmonotonic-training-lab/work/f-disk-reservation-drive/E3-OUTCOME.md),
[v3 verification](../../nonmonotonic-training-lab/work/f-joint-production-drive/FINAL-VERIFICATION.md),
[E4](../../nonmonotonic-training-lab/work/f-joint-science-drive/E4-OUTCOME.md),
[bounded comparison](../../nonmonotonic-training-lab/work/f-anchor-comparison-drive/DIAGNOSTIC-INTERPRETATION.md)).
The full record is in the [training handoff](TRAINING-HANDOFF.md),
[tasks](TASKS.md) and [feature tree](FEATURE-TREE.md).
