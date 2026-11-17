# Connected protocol v5: label-conditioned credit predictor, gain-scaled decay (plan)

Status: owner-approved 2026-11-07 ("stop v4 ... start and run v5"). v4 stopped at 81,920 anchors.

## Evidence (v4 full run `connected-fp32-v4-full-s7-20261107`)

- Held-out best 3.636 bits/byte at 57,344 anchors, 3.690 at 73,728 (ablation 4.594, v2 4.015 there).
- Predictor trust 0.075 at 3k anchors, 0.003 at 75k; per-interval applied/base local step ratio
  0.17 → 0.01; prefit cosine ~0.02; relative MSE improvement over predicting zero ~0. Tiles were
  effectively frozen after ~15k anchors: the gains came from E, bias, routing and memory.
- Structural cause: `connected_predictor.rs::features` is `[rms(input); rms(output); history; q; r]`,
  no target. At the final visit the real correction is `P_h E (softmax - onehot(y)) / sqrt(d)`; without
  `y` the best predictor is `E[c | features] = P_h E (p - p*) / sqrt(d)`, which vanishes exactly as the
  readout fits. Conditioning on the label is the conditional-DNI variant of the paper CONNECTED.md
  already cites (Jaderberg et al. 2017, https://proceedings.mlr.press/v70/jaderberg17a.html).
- Offline probe of the final checkpoint (`work/inference/version-retro-20261107/credit-probe/`,
  3,000 public train positions, closed-form ridge, held-out 20%): R2 over predicting zero for the
  v4 inputs is -0.009 / -0.014 / -0.004 / +0.022 at visits 4 / 3 / 2 / 1; adding rms(E[:, y]) gives
  0.968 / 0.845 / 0.779 / 0.751; rms(E[:, y]) alone 0.837 / 0.732 / 0.671 / 0.625. Replaying the
  checkpoint's own predictor reproduces the logged near-zero alignment.
- Secondary: `connected_update.rs::apply` decays at the full `rate` while the step is scaled by
  `gain = Lambert x trust`, so untrusted tiles erode (tile-3 masters 0.82 → 0.74 by 61k updates).

## Changes (one behavior each; no option toggles)

A. Target feature: `Shared::target_feature(byte) -> Result<Vec<f32>>` in `shared.rs` = RMS-normalized
   embedding column `E[:, byte]` from `embedding_column` (no positional term; FP32 view or ternary).
B. Predictor features (connected only): `[rms(input); rms(output); history; target; q; r]`, count
   `4 * width + 2` from a connected-local function. Do NOT change `Geometry::features()` (it sizes the
   non-connected predictor, history and codecs). Signatures: `features(input, output, history, target,
   tile_id)`, `fit(&mut self, record, target, rate)`. Validate the target (length `width`, finite).
C. Target plumbing (records stay unchanged; memory and checkpoints carry no target):
   - `connected_step.rs`: `observed_target = state.shared.target_feature(target.byte)` right after
     `gather` (before the E/bias updates; the correction was computed with that E) and pass it to
     `observe` → `fit`. After `local_prefix`, `lane_target` from the updated E → `connected_lanes::run`.
   - `connected_lanes.rs` `run`/`replay`/`drive` and `connected_lane_job.rs` `compute`/`execute` take
     `target: &[f32]` and pass it to `predictor.features`.
   - `connected_eval_replay.rs::observe`: target feature of the scored byte (evaluation) or the
     selected byte (generation feedback), passed to `replay`.
   Invariant to keep and document: the target reaches the predictor only after the anchor's forward
   was read out (training after `anchor_target`, evaluation in `score` after `forward`, generation
   after selection). Records conditioned on `y_t` reach only later anchors through memory, exactly as
   observed corrections already do.
D. Gain-scaled decoupled decay: `connected_update::apply` decays at `rate * factor`, where
   `factor = bound_scale(gain, base_norm, max_norm)` is the multiplier the step actually received
   (AdamW decays at the effective learning rate; here the norm cap scales the post-Adam step, so it
   is a learning-rate cut). E and bias keep gain 1. Admission refuses
   `weight_decay * rate * gain_max >= 1` (review S2/S3; the first draft used the pre-cap gain).
E. Protocol `causal-connected-v5` / `causal-connected-fp32-v5`; the loader accepts only v5;
   `scripts/core_owner_checkpoint.py` appends v5 (keep v1-v4 admission); Python tests extended.
F. Docs: CONNECTED.md header and a v5 section (evidence, cDNI, the leak invariant, decay rule,
   predictor+Adam bytes now `width * (4 * width + 2) * 3 * 4` = 789,504 at width 128); amend the
   "receives no target byte" sentence, the predictor-features paragraph and the
   conditional-expectation paragraph. Protocol pointers in `src/causal/AGENTS.md`, `README.md`,
   `scripts/AGENTS.md`. `docs/STATUS.md` is the orchestrator's.

## Tests (lean; extend existing files; every file stays under 300 lines)

1. The predictor learns a target-dependent correction only through the target feature
   (`connected_predictor_contracts.rs`): observed stream with correction `-k * target` plus small
   noise over random targets; table {with target: trust >= 0.5 and cosine >= 0.8; target zeroed:
   trust <= 0.1}.
2. Gain-scaled decay: coordinates with zero Adam delta shrink by exactly `1 - weight_decay*rate*g`
   for g in {1, 0.01} (`connected_numeric_contracts.rs` is at 298 lines; use another file).
3. Evaluation scores come from one target-independent distribution: scoring one prefix with every
   target, each from a fresh replay, gives `sum_y exp(-loss_y) = 1` within 1e-4 (guards C's invariant).
4. Update signatures and protocol expectations in the existing contracts and Python tests.

## Pilot and acceptance (set before the pilot; evidence at >= 8,000 anchors)

30-minute FP32 pilot, seed 7, v4 full config plus A-F. Pass: interval relative MSE improvement
>= 0.10 and prefit cosine >= 0.30 over the last third; mean trust >= 0.15 and not falling over the
last third; applied/base local ratio >= 0.10; `visit_rms_max` <= 4; masters <= cap; E sigma1/sigma2
< 5; held-out at 8,192 no worse than v4's 4.066 + 0.10. Then the 24 h run through the capped owner.
Stop criterion during the run: mean trust < 0.05 for 10,000 consecutive anchors.

## Pilot verdict (2026-11-08 UTC, `connected-fp32-v5-pilot-s7-20261107`, 8,750 anchors): FAIL 7/8

Mechanism passed by wide margins: relative MSE improvement 0.615, prefit cosine 0.824, mean trust
0.78 and flat (0.791 → 0.773), applied/base 1.04, visit RMS max 3.90, E σ1/σ2 2.17, 5.24 anchors/s.
Held-out failed: 4.306 at 8,192 (v4 4.066) and 4.363 at 8,750. Diagnosis: every tile's master RMS
climbed from 0.82 to the 1.5 cap by about 3,800 anchors and stayed pinned; knot overflow rose
0.02 → 0.29; prequential train loss beat v4 at equal anchors by 0.06–0.42 nats until 8,000, then
fell behind. Tiles now get real-quality credit at full rate: the realized local multiplier
(applied/base, Lambert x trust after the cap) was 1.04, so tile steps ran at about 0.003 per update,
about 15x v4's trust-throttled ~0.0002 and about 30x the reference path's per-anchor movement (rate
0.003 per batch of 32). Fast adaptation to the current story helps prequential loss and hurts
frozen held-out evaluation.

Inference on the pilot's final checkpoint (`work/inference/version-retro-20261107/v5-pilot/`,
weights then removed): greedy output collapses to repeated letters, not spaces (p(space) after
"Once upon a time" 0.017 vs unigram 0.194); mean p(repeat last byte) 0.217 vs 0.046 unigram;
next-byte distributions peaked (KL vs unigram 1.07, highest of all runs); context beyond the last
byte 0.50 bits at k=3 and k=24 (v4 at most 0.16). Hypothesis for v5.2 if v5.1 still drifts: the
online predictor captures the label term (-E_y) better than the model-expectation term (+E p(h)),
so local updates keep pulling toward likely-target embeddings even when the model is right (the
true correction's expectation over y vanishes at p = p*, the label-only part does not). Candidate:
give the predictor the visit's own early-readout direction E(p_k - e_y)/sqrt(d) (exact at the final
visit) instead of rms(E[:, y]). v5.1 watch items: master RMS reaching the cap, knot overflow, and
on the selected arm's checkpoint, p(repeat last byte) and p(space).

## v5.1: separate learning rate for predicted-credit tile updates

`connected.local_rate` sets the Adam rate of `Mode::Local` tile updates (and their decay, at
`local_rate x realized multiplier`); E, bias and route updates keep `rate`. Part of run identity;
range (0, 0.1]; admission bound uses `max(rate, local_rate)`. Two parallel 30-minute pilot arms,
`local_rate` 1e-4 (the reference path's per-anchor movement) and 3e-4, same gates. Selection rule,
fixed now: the arm passing all gates; if both, the lower held-out at the end; if neither passes the
held-out gate, report and do not launch the 24-hour run.

## v5.1 verdict (2026-11-08 UTC): both arms FAIL 7/8 (held-out); no 24-hour launch

| Arm | Anchors | Held-out (8,192 / end) | Cosine | Rel. MSE | Trust | Master max | Knot |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| local_rate 3e-4 | 12,171 | 4.419 / 4.298 | 0.968 | 0.768 | 0.82 | 1.19 | 0.27 |
| local_rate 1e-4 | 8,871 | 4.371 / 4.331 | 0.969 | 0.772 | 0.85 | 0.88 | 0.05 |

Prequential loss, nats per byte, 0-1k anchors: v4 3.653, v5 3.532, 3e-4 4.373, 1e-4 4.662; the 1e-4
arm sat at 6.0-6.4 at 95-150 anchors where v4 was at 4.85 and v5 at 3.82. Diagnosis: v4 had an
accidental schedule. Its effective local rate, 0.003 x trust x Lambert, was about 2e-3 at the start
and decayed roughly as t^-0.6 as trust fell: 7.6e-4 at 400 anchors, 2.5e-4 at 3k, 1.25e-4 at 10k,
3.7e-5 at 75k. Fast early tile learning escapes the poor initial forward; the decay then keeps
tiles stable. v5 has constant trust, so it either starts fast and drifts (3e-3) or starts too slowly
(1e-4, 3e-4).

## v5.2: decaying local rate

Add `Decay::InverseSqrt { anchors }` (factor `1 / sqrt(1 + t / anchors)`) to the existing schedule
enum and a connected option `local_decay` (default `constant`) applied to `local_rate` at each
anchor. With `local_rate` 2e-3 and `anchors` 50 the base rate tracks v4's effective schedule within
about 1.15x up to 10k anchors and 1.4x at 75k (v5.2 realizes about 0.8 of the base through trust x
Lambert), now with accurate credit. Review: approved; resume-with-schedule test and doc fixes
applied in parallel with the pilots (no production change). Arms: `anchors` 50 (v4-matched) and 500 (slower
decay), both `local_rate` 2e-3, same gates. Selection rule, fixed now: the arm passing all gates;
if both, the lower held-out at the end; if neither passes the held-out gate, report and stop
iterating on rates (next suspects: shared-memory history, Lambert route gains).

## v5.2 verdict (2026-11-08 UTC): both arms FAIL 7/8 (held-out); stop iterating on rates

| Arm | Anchors | Held-out | Cosine | Trust | Master max |
| --- | ---: | --- | ---: | ---: | ---: |
| inverse_sqrt 500 | 12,811 | 4.332 at 8,192; 4.198 at 12,811 | 0.89 | 0.81 | 1.5 (cap) |
| inverse_sqrt 50 | 7,184 | 4.348 at 7,184 (v4 pilot 4.206 at 7,049) | 0.97 | 0.87 | 0.87 |

The fast start returned (prequential 0-1k: 3.631 for decay 500 vs v4 3.653), but every v5 variant
(five pilots, local step rates 1e-4 to 3e-3, constant and decaying) learns as fast as v4 or faster
up to about 5-7k anchors and then falls behind it (7-9k: v4 2.814 vs 2.86-3.23; 9-12k: v4 2.607 vs
2.83-2.96). That is where v4's tiles froze. Conclusion: accurate predicted local credit (cosine up to
0.97) trains tiles, but continued tile learning makes the whole system worse than freezing tiles,
at every schedule tested. The bottleneck moved from credit quality to how lane-based tile updates
interact with the rest of the system. Per the selection rule, no 24-hour run.

Next experiments, in order of information per hour (none started):
1. Exact anchor-route credit (no predictor, no lanes) with the same v4/v5 machinery: if tiles trained
   from exact credit also fall behind frozen tiles, the connected machinery (memory, routing,
   per-anchor single-sample steps) is the bottleneck, not credit.
2. Predicted credit applied only on the anchor route's visits (no lane updates): isolates whether
   lanes' imagined routes produce misdirected updates.
3. Shared-memory history ablation (history zeroed in the forward) and Lambert route gain fixed at 1
   (config: gain_min = gain_max = 1).

## Review record (2026-11-07, independent opus review)

Approve-with-fixes; no label leak on any scored path (training, held-out/dev/retention probes,
generation). Fixes: S1 per-target exact leak oracle (the sum-to-one check passed under an injected
leak), S2 admission bound, S3 decay at the realized multiplier, S4 label-plumbing assertions, doc
evidence corrected (N1-N3, N7, N8). Pilot watch item from N4: a working predictor raises Lambert
gains, so route steps and route decay grow together; watch per-tile anchor visits for route collapse.

## Deferred

- Depth one-hot in the predictor features; per-depth calibration.
- Project predictions onto h-perp at the final visit (the real correction is exactly orthogonal to
  h there; +1.5 points of probe R2). Exact final-visit lane credit from the readout instead of a
  prediction (the probe shows it is closed-form); revisit if visits 1-3 lag the final visit.
- Spurious trust on label-varying streams: a NumPy mirror of `fit` exceeded trust 0.1 in 25% of
  seeds on an alternating-sign stream with a fresh label per fit (prediction norm <= 0.0012).
  Trust sizes the scale-free local step; revisit if pilot trust rises while prefit cosine stays low.
- Memory crowding by predicted records; E Adam-epsilon floor; tied-embedding copy logit at init.
