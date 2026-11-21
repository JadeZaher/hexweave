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

### Phase 1 result (2026-11-20): gate failed on writes; extra tiles unused

Three 2-hour runs, Lambert recipe, seed 7, 4,096-byte evaluations
(`work/core-runs/connected-fp32-v60-{a-7t-r6,b-19t-r19,c-19t-r6}-s7-20261120/`).

| Arm | Anchors | Held-out, last four evaluations up to 134,761 | Written per anchor | Tiles under 1% of visits |
| :--- | ---: | ---: | ---: | ---: |
| A (7 tiles, 6 resident) | 134,761 | 3.012 | 7.6 KB | 0 of 7 |
| B (19 tiles, 19 resident) | 179,015 | 3.017 | 19.6 KB | 9 of 19 |
| C (19 tiles, 6 resident) | 154,911 | 3.017 | 741 KB | 9 of 19 |

- **Quality under paging:** C equals B exactly at all 24 shared evaluations. Paging is lossless, so
  this half of the gate holds by construction and was not a real test.
- **Writes:** C wrote 114.9 GB in 2 hours (30,697 bundle writes of about 3.7 MB), 98× arm A per
  anchor against a limit of 3×. Gate failed. Each visit to a non-resident tile costs one full-bundle
  read and one full-bundle write; a write-behind of 3 does not absorb 13 non-resident tiles. No
  24-hour run at this setting.
- **Size effect:** B is level with A at matched anchors. Tiles 9 (start), 5 and 10 take 80% of
  visits (40.5%, 28.3%, 11.2%); nine tiles are each under 1%. Routing does not use the extra tiles,
  so by the rule above this must be fixed before Phase 2.
- **Not done:** sample-quality scores for these checkpoints. Arm C's weights were deleted (the
  model is the same as B's up to its last anchor); reports and logs are kept.

Follow-up chosen by the owner (2026-11-20): shared tile-sized sections with a priority-queue write
and a teaching path, applied on the wave pattern. Its first pilots run at 19 tiles with all tiles
resident, to keep disk traffic out of the comparison.

### Shared-section pilots (2026-11-20): Lambert-mix write with the queue kept

Follow-up to the Phase 1 result: `connected.shared` (CONNECTED.md §Shared sections), 2 sections,
wave 8, at 19 tiles all resident, seed 7, compared with arm B at matched positions (mean of the
last four evaluations, 4,096 bytes each). One seed per arm.

| Arm | `shared` | Positions | Held-out | Arm B, same positions | Score with the read off | Positions/s |
| :--- | :--- | ---: | ---: | ---: | :--- | ---: |
| Mix | `{sections 2, wave 8}` | 173,224 | 2.766 | 2.949 | 0.10 to 0.20 worse | 18.5 to 19.8 |
| Entry | `credit: entry` | 128,560 | 3.023 | 3.017 | about 0.03 worse | 20.9 |
| No queue | `queue: false` | 122,880 | 2.969 | 3.024 | no change | 20.1 |
| Teaching | `teach: 0.05` | 94,232 | 3.026 | 3.131 | 0.6 to 1.6 worse | 15.2 |

- Mix was ahead of arm B at every evaluation; the lead was 0.11 at 122,000 positions and 0.18 at
  173,000. Mix at the positions of the last three rows: 2.930 (no queue), 3.019 (teaching).
- No queue is 0.04 behind mix at the end and was 0.1 to 0.27 behind before 90,000 positions. Writes
  stayed even without the queue (7,668 / 7,692), so the queue's effect is not write balance.
- Teaching equals mix per position, is 25% slower, and makes the tiles depend on the read.
- Risk monitors (mix, 173,000): writes 10,824 / 10,829; tile similarity 0.05; centre-tile mean
  plasticity 1.11 (lowest tile); 6 of 19 tiles under 1% of visits (arm B: 9); `read_delta` rose
  0.69 to 1.00 without levelling off; `credit_alignment` 0.78.
- Sampled text (`work/inference/shared-pilots-20261120/`, nucleus T0.8 p0.9, real-word share): arm B
  0.735 at 179,015; mix 0.697 at 173,224 (0.573 at 122,616); no queue 0.568; teaching 0.592;
  entry 0.555. No arm improves sampled text over arm B yet.
- Kept: mix (checkpoints) and teaching (checkpoints). Entry and no-queue weights deleted.
- Open: a second seed for mix; whether `read_delta` needs a cap in a longer run; entry credit
  without the 1/K scaling, to separate step size from direction.

#### Six-hour extension (2026-11-21): mix and arm B, 6 hours of wall time each

Same configs and seed, resumed from their checkpoints. Arm B was stopped once near mix's final
position for a matched text score (it stopped at 315,220, 5% short of 331,096).

| Comparison | Mix | Arm B | Difference |
| :--- | ---: | ---: | :--- |
| Positions after 6 hours | 331,096 | 457,261 | arm B 38% more |
| Held-out, last four evaluations, mix's positions (311k to 331k) | 2.599 | 2.669 | mix 0.07 better |
| Held-out, last four evaluations at 6 hours | 2.599 | 2.539 | arm B 0.06 better |
| Best single evaluation | 2.548 | 2.498 | arm B |
| Real-word share, nucleus, matched positions (331,096 / 315,220) | 0.768 | 0.760 | level |
| Real-word share, nucleus, 6 hours | 0.768 | 0.859 | arm B |
| Greedy samples that loop, matched positions | 0 of 4 | 4 of 4 | mix |

- Per position mix stays ahead at every evaluation, but the lead shrank: four-evaluation means
  0.19 (147k to 172k), 0.17, 0.08, 0.10, 0.07, 0.10 (295k to 319k).
- Per hour arm B wins: mix ran 16.8 positions/s in its last leg against 21.7 for arm B.
- Sampled text at matched positions is level; the loss lead did not become better text.
- Risk monitors (mix, 331,096): writes 20,698 / 20,689; tile similarity 0.055; score with the read
  off 0.08 to 0.16 worse; `read_delta` 1.00 at 177k, 1.05 at 241k, 1.09 at 331k (slowing, not
  flat); `credit_alignment` 0.78 throughout; centre-tile mean plasticity 1.15; 6 of 19 tiles under
  1% of visits (arm B: 8); top tile 28.6% of visits (arm B 39.7%); no restarts in either run.
- Decision input: the shared read passes the 0.05 keep rule per position and fails it per hour.
  It is worth keeping only if its cost per position falls (it adds a 2-section forward at every
  route entry and a write every 8 positions) or if the gain grows with more sections.

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
