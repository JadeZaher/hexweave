# Connected v5.8: closing the gap to the reference path (limits, tests, loop)

Status: owner-directed 2026-11-18. Loop not yet started. This document is the loop's contract:
the target, the suspected limits, the tests, and the rules for keeping a change, stopping and
reassessing. Results are appended at the end as rounds finish.

## Target

| Measure | Target | Why |
| :--- | :--- | :--- |
| Held-out bits/byte, 7 tiles, windowed mean | 2.2 | The reference path reached 2.216 at 2.09M anchors with the same parameter count. Matching it means bounded residency costs nothing in quality at this size. |
| Inference quality | Reported every round; bar set after the first scoring pass | `scripts/sample_quality.py` (real-word share, distinct 4-grams, loop share, the completions). The reference checkpoint's score from the same script is the bar. |

Where the connected path stands (seed 7, 1,024-byte evaluations):

| Recipe | Result |
| :--- | :--- |
| Dynamic restarts, 24 h | last 2.563, best 2.476, mean 2.557 over 1.6–1.69M anchors |
| Constant rate 0.8× base, 6 h | mean 2.904 at 400–500k anchors |
| Lambert plasticity, 6 h | mean 2.870 at 300–400k anchors (continuation to about 1.5M in progress) |

Runtime rule: a run earns more time only while its windowed mean is still falling by at least
0.03 per 200k anchors.

## Verified differences between the two paths

From a code comparison on 2026-11-18. File references are to `src/causal/`.

**Not a difference: data order.** Both paths read the train file strictly in order through the
same cursor (`core_corpus.rs:164-177`, `advance_anchor` at `:273-277`). The reference path's batch
of 32 is 32 adjacent bytes with overlapping windows (`core_batch.rs:68-112`). An earlier suspicion
that the connected path suffers from a correlated stream while the reference path does not was
wrong: the reference path has the same stream and reaches 2.2.

| # | Difference | Reference path | Connected path | Evidence |
| :--- | :--- | :--- | :--- | :--- |
| 1 | Positions per optimizer step | 32, gradients averaged, all computed under the same frozen weights | 1 | `core_gradients.rs:50-66`; `connected_step.rs:58,137` |
| 2 | Route choice in training | Greedy (argmax), same as evaluation | Sampled from the softmax; evaluation is greedy, so it scores routes training never fixed | `core_route.rs:33`, `routing.rs:51-58`; `connected_step.rs:61-67`, `connected_eval_replay.rs:62-67` |
| 3 | Update shaping | None | Per-step norm cap `max_update_norm` 1.0, decoupled `weight_decay` 0.001, per-matrix RMS cap 3.0 | `connected_update.rs:78,83-90,251` |
| 4 | Feed-forward channels | All | `slice_mode: windowed`: half to seven-eighths active per visit | `ffn.rs:115-146` |
| 5 | Forward numerics | Ternary forward with latent weights | FP32 masters; RMS normalization after each visit | `ternary.rs:225-233`; `tile.rs:265-269` |
| 6 | Context input | Mean of recent correction records | Similarity-weighted retrieval over 32 records | `core_batch.rs:81-85`; `connected_signal.rs:229-237` |

Both paths use exact backpropagation through the realized route and the shared embedding, Adam
with the same betas, a score-function route update with a moving baseline, and window 512.

Not determined: the reference path's embedding optimizer internals; whether difference 5 makes the
reference path's 2.2 a fair target for an FP32 model with post-visit normalization (it is a
different function class, so 2.2 is a target, not a proven ceiling or floor).

## A separate limit seen in the logs: uneven tile use

Share of visits per tile (tile 3 is the centre):

| Anchors | Lambert | Constant rate |
| :--- | :--- | :--- |
| ~25k | 19, 15, 19, 31, 2, 12, 2 | 4, 2, 42, 42, 1, 7, 3 |
| ~400k | 13, 11, 22, 29, 7, 11, 7 | 8, 1, 21, 42, 10, 5, 14 |

Under constant rate one tile stayed at 1–2% of visits for the whole run. Under Lambert the least
visited tiles carried the highest plasticity (1.7–1.8 early, against about 1.0 for the centre), so
in practice the law acts as "learn faster where visited less". Lambert beat rate-matched constant
runs by about 0.1 at 200–400k anchors. Whether a plain visit-compensated rate does the same is
test F.

## Tests

Each is one change against the current best recipe. Base recipe at loop start: constant tile rate
0.8× base (`local_rate` 0.0016, constant schedule), RMS cap 3, 6 resident tiles, 3-entry
write-behind buffer, exact anchor-route credit, windowed slicing, seed 7. The base moves when a
change is kept. If the Lambert continuation holds its lead it becomes the base and the tests are
re-pointed at it.

| Test | Change | Targets difference | Cost |
| :--- | :--- | :--- | :--- |
| A | Several positions per update (CPU wave: accumulate tile and route gradients over N anchors, apply once; embedding and bias still step per anchor). Try N = 4, 8, 32. | 1 | Port from `hexweave-gpu/src/causal/connected_wave.rs`; review |
| B | Greedy routes in training, matching evaluation (with a small exploration rate as a second arm) | 2 | Small code change behind a config key; review |
| C | Looser update shaping: `max_update_norm` 8 and `weight_decay` 1e-4, separately | 3 | Config only |
| D | `slice_mode: full` | 4 | Config only. One earlier pilot pair favoured windowed from 24k anchors. |
| E | Jump around the corpus (strided cursor) | none verified | Code change in `core_corpus.rs`; retention probes must be disabled or remapped. Lowest priority: the reference path does not do this. |
| F | Visit-compensated tile rate without the Lambert law (rate scaled by inverse visit share, bounded) | uneven tile use | Small code change using the per-tile plasticity dial |

Order: C and D first (no code), then B, then A, then F, then E.

## Loop rules

1. **One change per round.** A round is a 30-minute pilot against the base at matched anchors.
2. **Promote** to a 6-hour run if the pilot's five-point mean beats the base by more than 0.05, or
   if it is within 0.05 and there is a stated reason to expect an effect later in training (for
   example a mechanism that only binds after the rate or the weights have moved). A pilot within
   0.05 with no such reason is recorded as neutral and not promoted. (Amended 2026-11-18: the
   first wording promoted every neutral change.)
3. **Keep** a change if its 6-hour windowed means beat the base by more than 0.03 in at least two
   consecutive 100k-anchor windows. A kept change becomes part of the base.
4. **Code changes** get independent review and one full validation sweep before any run that
   uses them. Config-only rounds need neither.
5. **Report each round** here: the curve, the inference-quality score of the final checkpoint,
   bytes written, and keep or drop with the reason.
6. **Stop and reassess** when any of these holds:
   - the target is reached;
   - two consecutive rounds keep nothing;
   - six rounds have run;
   - a run fails for a reason that is not understood.
7. **Statistical caution.** Every comparison is one run per arm on seed 7 unless stated. A kept
   change is confirmed on seed 11 before it is relied on for the growth work.
8. **Machine use.** Runs go through `scripts/core_owner.py`, one at a time for 6-hour runs. Weights
   of dropped arms are deleted; reports are kept.

## Relation to the growth spec

`connected-v6-growth-gpu-2026-11-16.md` Phase 1 (19 tiles) tests whether the model can grow. This
loop tests whether each update is as good as it can be. They are independent, and the recipe this
loop ends with is the base recipe for Phase 1. Uneven tile use is the one item that bears on both.

## Results

Held-out bits/byte at 8,192 / 16,384 / 24,576 / 32,768 / 40,960 anchors and the five-point mean.
Base (constant rate 0.8×): 3.900, 3.756, 3.570, 3.674, 3.459 | 3.672.

### Round 1 (2026-11-18): config-only tests C and D

| Pilot | Readings | Mean | Outcome |
| :--- | :--- | ---: | :--- |
| `weight_decay` 1e-4 | 3.949, 3.723, 3.604, 3.674, 3.574 | 3.705 | Neutral, not promoted |
| `slice_mode: full` | 3.933, 3.779, 3.580, 3.623, 3.529 | 3.689 | Neutral, not promoted |
| `max_update_norm` 8 | did not run: admission caps the value at 4; rerun at 4 in round 2 | | |

Neither weight decay nor windowed slicing limits the connected path over the first 41k anchors.

### Round 2 (2026-11-18): step-size cap, greedy routes (test B), waves (test A)

Ran beside the Lambert long run, on the build from commit 0d1da33 (7 tiles).

| Pilot | 8,192 | 16,384 | 24,576 | 32,768 | 40,960 | Anchors/s | Outcome |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| Base | 3.900 | 3.756 | 3.570 | 3.674 | 3.459 | 26.1 | |
| `max_update_norm` 4 | 3.900 | 3.755 | 3.570 | 3.674 | 3.459 | 25.1 | Identical to base: the cap never binds. Dropped. |
| `route_training: greedy` | 3.883 | 3.709 | 3.734 | 3.688 | – | 21.6 | Neutral. Not promoted. |
| greedy, `route_explore` 0.05 | 3.860 | 3.742 | 3.624 | 3.716 | – | 23.6 | Neutral. Not promoted. |
| `anchor_batch` 4 | 4.268 | 3.897 | 3.617 | 3.567 | 3.455 | 45.2 | Behind early, level from 32k. Promoted. |
| `anchor_batch` 8 | 4.336 | 4.033 | 3.782 | 3.728 | 3.566 | 53.3 | Behind; closing by 74k. |
| `anchor_batch` 32 | 4.372 | 4.226 | 4.010 | 4.037 | 3.885 | 52.5 | Well behind early; 0.05 behind by 82k. |

Past 41k anchors, against the 6-hour constant-rate run at the same anchors (base / wave 4 / wave 8 /
wave 32): 49,152: 3.512 / 3.508 / 3.567 / 3.762; 57,344: 3.318 / 3.347 / 3.426 / 3.616; 65,536:
3.446 / 3.385 / 3.487 / 3.589; 73,728: 3.347 / 3.308 / 3.382 / 3.469; 81,920: 3.383 / – / 3.468 /
3.437.

Reading: waves do not improve quality per anchor at this horizon, so single-sample updates are not
shown to be a quality limit. They roughly double throughput (fewer tile commits), and a wave of 4
is level with the base per anchor from 32k anchors, so it reaches a given loss in about half the
time. Promoted on that basis, and because fewer, averaged updates may matter more later:
`connected-fp32-v58-wave4-6h-s7-20261118` (6 hours). Larger waves take proportionally fewer tile
steps at the same step size; a wave-32 arm with a raised tile rate is a candidate for a later
round.

**6-hour wave-of-4 run (876,780 anchors, 44.6 anchors/s, 5.51 GB written): dropped.** Windowed
means by 100k anchors, constant rate / wave 4: 0–100k 3.496 / 3.553; 100–200k 3.188 / 3.269;
200–300k 3.062 / 3.144; 300–400k 2.995 / 3.034; 400–500k 2.904 / 3.008; then wave 4 alone 2.971,
2.943, 2.943, 2.898 to 900k (the restart run was at 2.77–2.79 over 600–900k). It fell behind per
anchor after the pilot horizon and flattened. At equal wall time (6 hours) it ended at 2.90 against
the constant-rate run's 2.87, so the doubled throughput bought nothing. Its tile use collapsed:
visit shares 25, 6, 31, 38, 0, 0, 0 per cent, with three tiles untrained (master RMS 0.85–0.87).

**Lambert continuation (in progress at 1.06M anchors):** windowed means 2.799 (400–500k), 2.792,
2.701, 2.709, 2.688, 2.736 (900k–1.0M), 2.627 (1.0–1.06M, 7 evaluations); best single 2.566 at
688k. Against the restart run at the same windows it leads by 0.05–0.13 through 900k.

### Stop rule reached after round 2

Rounds 1 and 2 kept nothing, which is the "two consecutive rounds keep nothing" condition.
Reassessment: five of the six suspected differences from the reference path are cleared as limits
at the horizons tested (weight decay, slicing, step-size cap, sampled routes, single-anchor
updates). Every piece of positive evidence points at uneven tile use: Lambert's lead comes with a
flatter visit distribution; the history-off pilot and the wave-4 run both lost tiles; under a
constant rate one tile stays at 1–2% of visits. The owner directed on 2026-11-18 that the visit
balance mechanisms (rate, training-time route bonus, memory retention) and a per-tile worth
estimate be built and tested together. That is round 3 and the loop continues on that direction,
with the stop rule reset: if round 3 keeps nothing, stop and reassess with the owner before any
further rounds.

(Later rounds are appended here.)
