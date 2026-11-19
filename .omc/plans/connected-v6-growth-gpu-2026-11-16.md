# Connected v6: tile count, growth by splitting, and batched GPU waves (spec)

Status: owner decisions recorded 2026-11-16 (see "Owner decisions"). Nothing in this spec is implemented.
Order of work: this spec starts only after the two items in "Prerequisites" are finished.

## Goal

Test whether the connected (paged, bounded-residency) path keeps its quality when the model is
larger than what is resident, then let the model grow on demand, then give a GPU a batch of tiles
to work on at once. Each phase has a gate. A failed gate stops the sequence and is a result.

The three claims, in order:

1. **Paging holds at scale.** With most tiles on disk, held-out loss per anchor is no worse than
   the 7-tile model. Nothing so far tests this: every run has had 7 tiles with 6 resident.
2. **Growth works.** A tile that is overloaded can be split into two without losing what the model
   has learned, and the larger model then improves past where the smaller one stalled.
3. **Batching pays.** With many tiles, a wave of anchors lands on mostly different tiles, so a GPU
   can run them as one operation without the stale-update penalty seen at 7 tiles.

## Evidence this builds on

| Run | Result | Source |
| :--- | :--- | :--- |
| 24-hour run, global rate decay | plateau near 3.35 bits/byte, 3.70 TB written | `work/core-runs/connected-fp32-v54-anchor-exact-window-full-s7-20261111/` |
| Long run, dynamic restarts, RMS cap 3 | mean 2.65 at 1.0–1.2M anchors, best 2.551, 8.5 GB written for 1.3M anchors, about 50 MB peak | `work/core-runs/connected-fp32-v56-adapt-cap3-full-s7-20261113/` |
| GPU wave of 8 at 7 tiles | 17.4 anchors/s and 4.383 at 4,936 anchors, against 27.9/s on CPU | sibling repo `hexweave-gpu`, pilot `connected-fp32-v54-gpu-wave8-window-s7-20261111` |

Yardsticks on this corpus: unigram 4.414, bigram 3.31, trigram about 2.57, reference
(non-connected) path 2.216 at 2.09M anchors.

Known limits of that evidence: one seed (7); held-out scores come from 1,024 bytes and move by
about ±0.1 between adjacent evaluations; throughput on this machine varied from 20 to 29 anchors/s
with identical work, so comparisons are made at matched anchors, not matched time.

## Prerequisites (not part of this spec, must finish first)

1. **Current training:** the resumed long run finishes or stops; record final curve and samples.
2. **Current work and its training:** the per-tile training state, restart options and the
   experimental Lambert law pass independent review and one full sweep; a calibration run sets the
   tile-signal threshold; the pilot round runs (three arms that separate rate, jitter and
   forgetting, plus one arm per new option); the winning recipe gets a long run.

This spec takes one input from that work: the **base recipe** (the winning restart configuration)
and the **per-tile signal** that will trigger splits. Until then both are placeholders.

## Phase 1: fixed tile count (the premise test)

**Question:** does quality hold when most tiles are on disk?

- **Change:** allow a radius-2 hex (19 tiles) as a second fixed geometry at the current tile size.
  No growth, no GPU. Parameters go from about 2.05M to about 5.6M.
- **Arms**, each a fresh run with the base recipe and seed 7:

| Arm | Tiles | Resident | Purpose |
| :--- | ---: | ---: | :--- |
| A | 7 | 6 | Control (rerun, same binary) |
| B | 19 | 19 | Size effect with no paging pressure |
| C | 19 | 6 | The claim: same model as B, mostly on disk |

- **Evaluation:** 4,096 held-out bytes (`eval_spans` 32) for every arm in this spec, so the 0.05
  gates are decidable. Scores from 1,024-byte evaluations are not compared against these.
- **Length:** 2 hours each first; the best of B and C continues to 24 hours only if the gate passes.
- **Measures:** held-out at matched anchors; visits per tile and how many tiles receive under 1% of
  visits; cache misses, reads and bytes written per anchor; peak memory.
- **Gate:** C within 0.05 bits/byte of B at matched anchors (mean of the last four evaluations), and
  bytes written per anchor within 3× of arm A. If B is no better than A, routing is not using the
  extra tiles and Phase 2 must fix routing spread before splitting means anything.
- **Risk to watch:** exposure skew. At 7 tiles the busiest tile already gets about 16× the visits of
  the quietest. At 19 the outer ring may starve.

## Phase 2: dynamic tile count by cloning

**Question:** can the model add capacity where it is needed without a relearning dip?

- **Dynamic tile count.** The number of tiles is run state, not a constant. Tiles are addressed by
  an id that can be added at run time, and routes can gain a branch.
- **Split by cloning** (function-preserving, after Net2Net, Chen et al. 2016,
  https://arxiv.org/abs/1511.05641). The parent's weights and optimizer state are copied into two
  children of the same shape; each child gets fresh per-tile training state; the router starts at an
  even division between them. Every tile keeps one shape, which Phase 3 needs. Halving a tile's
  width is out of scope.
- **Padding for clone operations.** The run declares a tile capacity. A reserve of free slots
  (state arrays, route entries, store slots and disk space for a full bundle) is kept allocated
  ahead of need, so a clone never waits on allocation and never happens mid-anchor. A clone
  consumes one padded slot at a quiescent point; the pad is refilled afterwards while capacity
  remains. With no padded slot free, no tile splits.
- **Smart threshold.** A tile splits when its overload signal, read from the per-tile training
  state without loading the tile, stays above a threshold for a sustained window. The threshold is
  relative, not fixed:
  - it is measured against the other tiles (for example a multiple of the median signal), so it
    tracks the model as it trains;
  - it rises as the padding runs down, so the last free slots go to the most overloaded tiles;
  - a minimum gap between splits lets the router and both children settle before the next one.
  The exact signal comes from the prerequisite pilots (candidates: plasticity that stays high,
  gradient size that does not settle, visit share). All three are already logged per tile.
- **Every tile earns its place (owner principle, 2026-11-18).** Tiles need not be used equally,
  but each must have a reason to be there. The counterpart of splitting is retiring: a tile whose
  measured worth stays at or below zero for a sustained window, despite the balance mechanisms
  (visit-compensated rate, route bonus, memory retention) having had time to act, is a candidate
  to be removed and its slot returned to the padding. Worth is first estimated from the per-tile
  advantage statistics logged by `connected.balance`; because that estimate is correlational, a
  retire decision must be confirmed by an ablation (held-out loss with the tile's routes masked)
  before weights are dropped. Retiring is specified here and not yet designed in detail.
- **Arms:** start at 7 tiles with capacity 19, against Phase 1 arm C (19 from the start) and arm A
  (7 fixed). A capacity-37 arm follows only if this gate passes.
- **Measures:** held-out immediately before and after each split; anchors until the two children's
  outputs differ measurably; tile count over time; padded slots free over time; final held-out
  against fixed-19.
- **Gate:** no split raises held-out by more than 0.05 at the next evaluation, and the grown model
  ends within 0.05 of fixed-19 at matched anchors while having trained fewer tile-anchors.

## Phase 3: batched GPU waves

**Question:** with many tiles, does a wave of anchors run faster on the GPU without costing quality?

- **GPU code lives in `hexweave-gpu`** (owner decision). That repo is brought up to date first: it
  has none of the paging, restart, per-tile-state, fixed-19 or cloning work, and it holds
  uncommitted fixes to its wave loop and test. The update is its own reviewed step: bring the
  current connected sources across, keep the GPU session and wave code, and re-run its validation
  before any GPU pilot. This repo stays CPU-only.
- **Batching.** Resident tiles are stacked so one wave is one operation. Anchors in a wave that
  touch the same tile are the stale-update case; measure how often that happens.
- **Arms** at 19 tiles (or the grown model): CPU single-anchor, GPU wave of 8, GPU wave of 32.
- **Measures:** anchors per second; held-out at matched anchors; share of wave anchors that collide
  on a tile; device memory.
- **Gate:** a wave is within 0.05 bits/byte of single-anchor at matched anchors and at least 2× the
  CPU anchors per second. Otherwise the GPU stays parked.

## Constraints (from AGENTS.md and standing owner decisions)

- Every training run goes through `scripts/core_owner.py` on an explicit owner request, on public
  train/dev data only. Private evaluation material stays unread.
- A changed geometry is a new run identity. No checkpoint migration: existing checkpoints are not
  converted.
- The 81M and 287M profiles stay planning-only. Phase 1 stays at the current tile size on purpose.
  Dynamic geometry, the split operation and any GPU numerical path each need independent numerical,
  resource, ownership and checkpoint review before a bounded pilot.
- Admission requirements stay: four CPU threads, 1 GiB device reserve, 12 GiB owned-tree ceiling.
- No silent model downloads, servers or cloud. Reap every process started.
- Authoring and review are separate. All review fixes land before one integrated sweep.
- No performance, convergence or equivalence claim without the measurement behind it.

## Implementation outline

| Phase | Main code areas | Review needed |
| :--- | :--- | :--- |
| 1 | `src/causal/geometry.rs` (second fixed geometry), route tables, state arrays sized by tile count, validation bounds | numerical, checkpoint |
| 2 | tile ids and growable routes, split operation in the store, per-tile state creation, split log in the report | numerical, checkpoint, ownership |
| 3 | stacked tile tensors, wave scheduler, collision accounting, device residency cap | numerical, resource |

Tests follow the project's lean rule: one flow test per distinct risk (fixed-19 trains and resumes
bit-exactly; a split preserves the forward output exactly at the split anchor; a GPU wave matches
the host wave within tolerance), plus table rows for config admission.

## Owner decisions (2026-11-16)

| # | Decision | Outcome |
| :--- | :--- | :--- |
| 1 | How a tile splits | Cloning at a fixed tile size. |
| 2 | Tile count | Dynamic, driven by a smart (relative) threshold, with padding reserved for clone operations. Fixed-19 remains the Phase 1 premise test and the comparison for growth. |
| 3 | Second seed | Granted: one 2-hour run of the current best recipe with a different seed, before Phase 1. |
| 4 | Gate tolerance | 0.05 bits/byte on a four-evaluation mean, with the evaluation widened to 4,096 bytes. |
| 5 | GPU code | Lives in `hexweave-gpu`, which is updated as part of Phase 3. |

## Assumptions

- The base recipe from the prerequisite work transfers to 19 tiles without retuning. If Phase 1 arm
  B diverges, retune the restart threshold before judging the gate.
- Bytes written scale with tile swaps, not with tile count. Phase 1 measures this.
- 24-hour runs remain acceptable machine use. Each phase needs at most one.
