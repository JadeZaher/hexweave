# Hexweave status

Updated 2026-10-27 America/Denver. **The core training path is implemented and
is the build; two runs are in progress. No useful-assistant result or finite
training ETA is established.**

## Implemented

| Area | State |
| --- | --- |
| Core training | `hexweave train --config <json>` (`src/causal/core_train.rs`): the v3 forward model on the 2,051,840-coefficient candidate with latent-weight Adam, greedy eval-matched routes and an anchor-only schedule; all tiles resident |
| Owner | `scripts/core_owner.py`: `work/core-runs/<name>` rule, single-owner lock, pinned binary, 512 MiB Job commit cap, sampled 256 MiB memory floor, graceful STOP with a 180-second grace, start and end receipts |
| Checkpoints | Hash-manifested and fsynced, newest three kept. Resume identity-checks every readable manifest first, is bit exact, quarantines integrity failures and refuses to restart from anchor 0 when every checkpoint fails. Checkpoints from the `8e7a0eb` binary (earlier identity formula, earlier resume logic) resume only with their pinned `out/bin` copy |
| Data | `science`: pinned QA dataset, development-split evaluation. `drill` v2: the assay curriculum re-dealt to the harness `Shuffled` stream, with key-swap and attention probes |
| Drill v2 harness parity | `core_contracts`: 12 production-drill anchors equal the validated tuning harness bit for bit, and the drill stream equals `Shuffled` across cycle boundaries |
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
- `work/core-runs/science-overnight`: running (8-hour wall budget); no result is
  claimed here yet.
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
