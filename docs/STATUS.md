# Hexweave status

Updated 2026-11-11 UTC. **Generated text was being measured out of distribution,
and the reference path is training again.** Every recorded sample, connected or
reference, showed the model only the prompt, with positions counted from its first
byte, while training always fills a 512-byte window. Given a full window of
preceding held-out text, the reference path's 2.216 bits/byte checkpoint writes
words instead of `aaaa…rrrr…`; with the new seeded nucleus decoder its samples
stop looping and read as TinyStories-like word salad. A 23.5-hour continuation of
that run started 2026-11-11 12:57 UTC (see
[reference path](#reference-path-full-window-generation-sampling-and-continuation)).

Updated 2026-11-07 America/Denver (2026-11-08 UTC). **Connected protocol v5 solved
the credit problem but not the learning problem. Its label-conditioned predictor
matches the real correction (prefit cosine up to 0.97, trust about 0.8, where v4's
trust collapsed to 0.007), yet all five bounded pilots (local step rates 1e-4 to
3e-3, constant and decaying) trailed the v4 run's held-out: tiles learned as fast
as v4 or faster for 5,000–7,000 anchors, then fell behind v4, whose tiles had
frozen. Per the pre-registered rule no 24-hour v5 run was launched. No connected
version has reached a byte bigram trained on the same data (3.313 at 81,920
bytes); their samples repeat, but came from the out-of-distribution short context
above. The non-connected reference path, which trains the same candidate from real
anchor credit, scored 2.216 at 2.09M anchors. Weights of all connected runs and pilots were removed after
their inference records; results, [retrospectives](CONNECTED-RETROS.md) and a
[feasibility and utility assessment](CONNECTED-FEASIBILITY.md) are kept. Useful
language completions and larger-model resource feasibility remain unproven.**

## Reference path: full-window generation, sampling and continuation

**Finding (2026-11-11).** `Shared::prefix` adds each byte's position counted from
the first visible byte, and training predicts every byte from a full window, so the
newest byte always sits at position `window − 1` (511 here). Story generation fed
`END + prompt` alone, 12–35 bytes, putting the newest byte at positions training
almost never produced. A read-only probe of the reference run's best checkpoint
(`a000002086528`, 2.216 bits/byte) gave, for "Once upon a time": bare prompt
`  aaaaaaaaaaaaaaarrrrrrrrr…`; the same prompt after 496 bytes of public held-out
text ending at a story boundary `, there was a little boy named Tim was a big
becar and the big and the big…`. The model was better than its samples; the
single-letter collapse came from the input layout, the phrase loop from greedy
decoding. Connected generation (`connected_eval_replay.rs`) builds the same short
opening, so the connected inference records are out of distribution too; see the
corrections in the [retrospectives](CONNECTED-RETROS.md) and
[feasibility assessment](CONNECTED-FEASIBILITY.md).

**Change.** Story generation policy `story-window-v2`
([decoding.rs](../src/causal/decoding.rs)): the opening is held-out text ending at
an evenly spaced story boundary, then the prompt, filling one training window. Each
byte is chosen by a `Decoder`: greedy (lowest byte on ties, as before) or seeded
nucleus sampling (Holtzman et al. 2019, arXiv:1904.09751), a pure function of the
logits, temperature, top-p, seed and byte position. Training evaluations record a
greedy and a sampled continuation (temperature 1.0, top-p 0.9, seed = prompt index)
for each story prompt. `hexweave complete` opens the verified corpus for the context
and takes `--top-p P [--temperature T] [--seed N]`; connected checkpoints refuse
sampling. Training numerics and run identity are unchanged.

**Measured (checkpoint `a000002086528`, 4 prompts, 200 bytes,
`work/inference/reference-decoding-20261111/`).** Greedy: 1 of 4 continuations
loop by the retro definition, distinct 4-grams 0.34, 99% of words in the training
vocabulary. Nucleus p 0.9 (3 seeds each): no loops; temperature 1.0 gives distinct
4-grams 0.89 and 79% known words, temperature 0.8 gives 0.80 and 88%. Integrated
sweep passed (structure, format, strict Clippy, Rust and Python tests, plan
profiles, smoke): `work/validation/0d72ef2c88924a9c93938cb1edaaa3dc`. The
continuation was stopped by the owner at 2,282,080 anchors (2.222 bits/byte) to
refocus on the connected path. The same openings and decoder were then ported to
connected generation; integrated sweep passed:
`work/validation/c1ce13650b9a424e95ab4aeb74151406`. First connected samples under
it: pilot `connected-fp32-v53-window-pilot-s7-20261111` (v5.2 decay-500 settings,
30 minutes, 15,750 anchors, 53 MB peak) reached held-out 4.332 at 8,192 and 4.180
at 15,750, matching the earlier v5.2 pilot. Its nucleus samples are letter soup
built around "the" (`tttthe ther o attthease`), as a model just under the unigram
floor predicts: the window fix removed the collapse, but quality is limited by
learning, not decoding.

**Run.** `tiny-v2-b32-s7-continue2-20261111.json` resumes
`tiny-v2-b32-s7-scan-r003-20261102` from `a000002086528` under its pinned November 2
binary: B32, seed 7, constant rate 0.003, 64 held-out spans, an 84,600-second loop
budget. The owner started it at 2026-11-11 12:57:07 UTC. At the measured ~100
anchors/s it reaches the 10,000,000-anchor target after about 22 hours, before the
loop budget; `STOP` in the run directory ends it earlier with a checkpoint. The
decoder change was not independently reviewed (owner stopped the review to save
usage); review it before relying on sampled-sample statistics.
Its pinned binary predates this change, so its own logged samples still use the
bare-prompt `story-window-v1` policy; judge its generation with `hexweave complete`
on its checkpoints instead.

## Connected study

The `connected` config object selects predicted local corrections, shared memory
between eight logical lanes, learned activation-curve conditioning, whole-tile
Lambert W update gains, and paged tile/Adam storage with at most four decoded
bundles. The current source implements protocol v5 (`causal-connected-v5`,
`causal-connected-fp32-v5`): v4's post-visit normalization and corpus readout
prior, v3's scale control and one mean local step per tile per anchor, a credit
predictor conditioned on the target byte's embedding, and decay at the realized
step multiplier. Runs recorded below as v1–v4 can be repeated from scratch only
with their pinned executables; their weights were removed on November 7 (receipt
`weights-removed-20261107.json` in each run directory). See
[design and numerical contract](../src/causal/CONNECTED.md).

### Protocol v5.4: anchor-route credit and `(input, tile_id)` windowed FFN slicing break the 4.18 barrier (3.573 bits/byte)

Implemented 2026-11-11 from `.omc/plans/connected-v54-anchor-slice-2026-11-11.md`
after `connected-fp32-v53-window-pilot-s7-20261111` plateaued at `4.1872` held-out
bits/byte (`16,593` anchors, `25.5%` FFN knot overflow, Tile 3 pinned at `1.500`
master RMS with `51,384` local updates against `1,200` on Tile 6). v5.4 adds two
orthogonal, checkpoint-verified options in `connected`:
- `connected.credit_mode`: `"lanes"` (default), `"anchor_exact"` (apply exact
  task-anchored tile gradients from the realized anchor trajectory at unit gain `1.0`,
  skipping the 8 post-anchor local lanes), `"anchor_predicted"` (query the local
  correction predictor on each realized anchor visit using the pre-update anchor
  prefix and observed target feature, scaled by `Lambert(gain) * trust`, skipping
  post-anchor lanes), and `"hybrid"` (seed the per-anchor tile accumulator with exact
  anchor-visit gradients scaled by `Lambert(gain)` and merge the 8 local lanes scaled
  by `Lambert(gain) * trust`).
- `connected.slice_mode`: `"full"` (default), `"windowed"` (deterministic active FFN
  channel window of size `M/2`, `3M/4`, or `7M/8` at offset `seed % M` hashed from the
  last 4 prefix rows, routed state signs, history, and `tile_id`, with energy-matched
  `DOWN` scale `1 / sqrt(sum(mask_j^2))`), and `"hybrid"` (`1.0` inside the active
  window and `{0.5, 0.25}` parity weights outside).

Independent review found and fixed three numerical bugs before validation: `Hybrid`
credit mode previously passed `use_trust = false` to `Accumulator::apply_with_trust`,
giving unit gain to lane-only tiles; `AnchorPredicted` previously passed the
post-update `local_prefix` to `run_anchor_predicted`; and `ffn::seed` used `% 62` on
a 32-element loop (`2 * i <= 62`). Integrated validation passed all 9 stages
(`work/validation/b46b9583270a4cd999d2a1debc117527`: structure, format, strict Clippy,
151 Rust tests with 3 ignored, 26 Python tests, three plan profiles, smoke).

Four sequential 30-minute pilots (`wall_seconds: 1800`, seed 7, `rate: 0.003`,
`local_rate: 0.002` with 500-anchor `inverse_sqrt` decay) ran one at a time under
`scripts.core_owner --resume-on-memory`:

| Pilot | `credit_mode` | `slice_mode` | Anchors (30m) | Anchors/s | Held-out @ 8,192 | Held-out @ 16,384 | Held-out @ 24,576 | Held-out @ 40,960 | Final Held-out | Knot overflow |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v5.3 baseline | `lanes` | `windowed` | `16,593` | `9.79` | `4.2059` | `4.1872` | — | — | `4.1872` | `25.50%` |
| Pilot 1a (`anchor-exact`) | `anchor_exact` | `full` | `51,841` | `30.45` | `4.1488` | **`3.8212`** | `3.7593` | `3.7673` | `3.5864` | `6.15%` |
| Pilot 1b (`anchor-exact-window`) | `anchor_exact` | `windowed` | `48,398` | `27.94` | `4.1634` | `3.8642` | **`3.7149`** | **`3.6786`** | **`3.5728`** | `7.48%` |
| Pilot 2 (`anchor-pred-window`) | `anchor_predicted` | `windowed` | `57,568` | `33.55` | `4.1566` | `4.0137` | `3.8929` | `4.0052` | `3.8351` | `25.86%` |
| Pilot 3 (`hybrid`) | `hybrid` | `hybrid` | `8,865` | `5.43` | **`4.1137`** | — | — | — | `4.1273` *(8,865)* | `12.98%` |

Findings from the four v5.4 pilots:
1. **All four v5.4 arms beat both the v5.3 `lanes` baseline (`4.2059`) and the
   frozen-tile baseline (`4.191`) at 8,192 anchors** (`4.1137` in Pilot 3 `hybrid`,
   `4.1488` in Pilot 1a, `4.1566` in Pilot 2, `4.1634` in Pilot 1b).
2. **Even pure local-predictor credit (`Pilot 2: anchor_predicted`) broke through the
   4.18 plateau (`3.8351` bits/byte at 57,568 anchors)** once the 8 unscored
   post-anchor local lanes were removed, proving that the v5–v5.3 plateau was caused
   by the 8 post-anchor lanes over-driving the center tile rather than by predictor
   inaccuracy.
3. **`(input, tile_id)` windowed FFN slicing (`Pilot 1b`) beat full FFN matrices
   (`Pilot 1a`) from 24,576 anchors onward** (`3.7149` vs `3.7593` at 24,576;
   `3.6786` vs `3.7673` at 40,960; **`3.5728`** at 48,398 vs `3.6936` at 49,152 and
   `3.5864` at 51,841), achieving the best held-out score of any connected run while
   keeping 4-tile decoded residency (`4.50 MiB` view bytes) and beating the entire
   24-hour v4 full run (`3.636` best at 57,344 anchors) in 30 minutes.
4. **Per-anchor sample efficiency (`Pilot 3: hybrid`) was highest at 8,192 anchors
   (`4.1137` bits/byte)**, while on CPU its 8-lane work ran at `5.43 anchors/s`
   versus `27.9–33.5 anchors/s` for the anchor-only credit modes.

### Protocol v5: credit fixed; five pilots trail frozen tiles; no 24-hour run

Implemented 2026-11-07 from `.omc/plans/connected-v5-label-credit-2026-11-07.md`.
The credit predictor's features gain the RMS-normalized embedding column of the
anchor's target byte, the conditional variant of synthetic gradients. The tile
forward, routing and every scored readout still never see the target; it reaches
the predictor only after the anchor's forward has been read out (training after
`anchor_target`, evaluation after scoring, generation after selection). Decay now
runs at the multiplier each tile step actually received, so untrusted tiles stop
eroding. Evidence: on v4's final checkpoint a closed-form linear fit from v4's
predictor inputs explained about 0% of the real correction at every visit, while
adding the target embedding explained 97% at the final visit and 75–85% at the
three earlier ones (`work/inference/version-retro-20261107/credit-probe/`). An
independent review approved with fixes and found no label leak on any scored path;
the fixes (an exact per-target leak oracle, an admission bound on decay, label
plumbing assertions, corrected evidence wording) are applied before one integrated
sweep. Pre-registered pilot gates over its last third: interval relative MSE
improvement at least 0.10, prefit cosine at least 0.30, mean trust at least 0.15
and not falling, applied/base step ratio at least 0.10, raw visit RMS maximum at
most 4, E σ1/σ2 below 5, and held-out at 8,192 anchors no worse than 4.17.

Integrated sweep after the review fixes: structure, format, strict Clippy, 145
Rust tests (three ignored), 26 Python tests, three plan profiles and smoke passed
(`work/validation/5e7bf8b8b4f445aaa408566bb3493223`); release executable
`00acb3b61071e02b…43ae`. Pilot `connected-fp32-v5-pilot-s7-20261107`: 8,750
anchors in 30 minutes at 5.24 anchors per training second, clean deadline, 57 MB
peak. Verdict FAIL, 7 of 8 gates. The credit mechanism passed by wide margins:
relative MSE improvement 0.615, prefit cosine 0.824, mean trust 0.78 and flat,
applied/base 1.04 (v4's pilot on the same gates: 0.007, 0.10, 0.079 falling,
0.09). Held-out failed: 4.306 at 8,192 (v4 4.066) and 4.363 at 8,750. Every tile's
master RMS climbed from 0.82 to the 1.5 cap by about 3,800 anchors and stayed
pinned, and knot overflow rose from 0.02 to 0.29. Prequential training loss beat
v4 at equal anchors by 0.06–0.42 nats until 8,000, then fell behind. With credit
now informative, the realized local step multiplier was 1.04, so tile steps ran at
about 0.003 per update, roughly 30 times the reference path's per-anchor movement
(0.003 per batch of 32). v5.1 adds `connected.local_rate` for
predicted-credit tile updates only; two pilot arms (1e-4 and 3e-4) follow the same
gates, with the selection rule fixed in the plan before they run.

v5.1 result (sweep 146 Rust, three ignored, and 26 Python tests passed; executable
`0c0b7122fe1368c6…735d`; both arms concurrent, 4.7–7.2 anchors per second): both
arms FAIL 7 of 8 on held-out, so no 24-hour run was launched. Credit stayed
excellent (prefit cosine 0.97, relative MSE improvement 0.77, trust 0.82–0.85) and
scale stayed healthy (masters at most 1.19 and 0.88), but held-out at 8,192 was
4.419 (3e-4) and 4.371 (1e-4) against v4's 4.066. Prequential loss over the first
thousand anchors was 4.37 and 4.66 nats against v4's 3.65: at a low constant rate
the tiles leave their poor initial state too slowly. v4 had an accidental schedule:
its effective local rate started near 2e-3 and decayed roughly as t^-0.6 while its
trust collapsed. v5.2 restores that shape on purpose with an inverse-square-root
decay of `local_rate` (two arms, decay constants 50 and 500 anchors, peak 2e-3).
Pilot weights were removed after each verdict; logs, reports and receipts stay.

v5.2 result (independent review approved; executable `ea78e049bb76800f…4858`):
both arms FAIL 7 of 8 on held-out again. The fast start returned (prequential
loss over the first thousand anchors 3.631 nats for the 500-anchor decay against
v4's 3.653), and credit stayed excellent (cosine 0.89–0.97, trust 0.81–0.87), but
held-out was 4.332 at 8,192 and 4.198 at 12,811 (decay 500) and 4.348 at 7,184
(decay 50). Across all five v5 pilots, prequential loss matched or beat v4 up to
about 5,000–7,000 anchors and then fell behind it (9,000–12,000 anchors: v4 2.607
nats against 2.832–2.956), which is where v4's tiles froze as its trust
collapsed. Conclusion: accurate predicted local credit trains the tiles, but
continued lane-based tile learning makes the whole system worse than frozen tiles
at every schedule tried. The bottleneck has moved from credit quality to how tile
updates interact with the rest of the system. Next experiments, ranked in the plan,
are not started: exact anchor-route credit without lanes, predicted credit on the
anchor route only, and shared-memory and route-gain ablations. Final integrated
sweep after the v5.2 review fixes (resume with a decaying schedule, schedule
admission tables, documentation): structure, format, strict Clippy, 148 Rust tests
(three ignored), 26 Python tests, three plan profiles and smoke passed
(`work/validation/2f1214720eb347eaac1609c9ca50011a`). Tests now remove their
scratch directories through one shared guard (`src/causal/scratch_dir.rs`).

For scale: the non-connected v2 reference path, which trains the same
2,051,840-coefficient candidate from real anchor credit, scored 2.896 bits/byte at
392,640 anchors and 2.216 at 2.09M (64 spans; see Current runs). That passes the
matched-exposure trigram (2.443 at 2M bytes); no connected version has passed the
bigram.

### Protocol v4 FP32 study: stopped at 81,920 anchors, weights removed

Implemented 2026-11-07 after the v3 pilot below: post-visit RMS normalization of
each tile's output under the connected protocol (the adjoint passes through it;
non-connected numerics unchanged), a readout bias initialized at fresh start to the
public training corpus' byte log-frequencies (`readout_prior:
"corpus-log-unigram"`, counted in the same verified pass that hashes the corpus;
the route baseline starts at that prior's entropy), and `causal-connected-v4` /
`causal-connected-fp32-v4` identity; the v4 configs use `weight_decay` 0.001. An
adversarial review found the arithmetic sound and one real blind spot: with the
stream normalized, the readout's `output_rms` is about 1 by construction, so the
1e3 step bound and the planned pilot gate on it could never fail while the 1/RMS
credit factor had moved inside each visit. Rows and the report now carry the raw
per-visit RMS (`visit_rms` interval mean, `visit_rms_max` run maximum), the bound
applies to that maximum, and the production prior path is tested through the
extracted `fresh_state` seam. See
[the v4 section](../src/causal/CONNECTED.md#residual-normalization-and-readout-prior-v4).
Integrated sweep passed (142 Rust tests, three ignored, 26 Python, strict Clippy,
plan profiles, smoke); executable `e40f2e0d7530aa06…6ee8`.

Pilot `connected-fp32-v4-pilot-s7-20261107` (seed 7, 30 minutes) started
2026-11-07T16:44Z. Its first rows confirm the mechanics (`readout_prior` on the
anchor-0 row, `readout_rms` 1.000, `visit_rms` 1.01–1.04, E σ1/σ2 1.03, masters at
init) but the host was saturated by an unrelated four-process workload at normal
priority while the owner runs the trainer below normal, so the trainer received
about 13% of one core and 0.2 anchors per training second (5.1 on the idle host
this morning). The 8,000-anchor acceptance gate is therefore not reachable in this
pilot; its verdict is pending a rerun on a free host. One early observation worth
recording: the anchor-0 held-out was 9.57 bits/byte against 10.32 without the
prior, far from the 4.4-bit unigram, because the tied-embedding readout at
initialization carries a large "repeat the last byte" logit (the residual stream
starts at that byte's own embedding row) that dominates any prior; this is the
same start every connected run has had and is not a defect of the prior. The
starved pilot ended at its deadline after 387 anchors (0.25 anchors per training
second) with held-out 4.730 bits/byte at that point — earlier runs needed about
8,000 anchors to approach 4.5 — `visit_rms` 1.33 (maximum 2.21), E σ1/σ2 1.40,
masters at init and trust 0.18–0.24 per tile. Too early for a verdict.

Rerun `connected-fp32-v4-pilot2-s7-20261107` on the free host (same binary and
config, fresh directory): 7,049 anchors in 30 minutes at 4.1 anchors per training
second, clean deadline, peak working set 59 MB, held-out 4.206 bits/byte at the
end (v3 4.185 at 8,580; the frozen-tile ablation 4.48 near 8k). Over the window
from 6,000 anchors: raw per-visit RMS mean 1.19–1.37 after peaking at 2.11 near
4,000, run maximum 3.21 reached at 4,000 and never exceeded; E σ1/σ2 3.15 with
effective rank 66.7 of 128 (v3 pilot 6.8, ablation 3.7, v2 2.2); interval
`zero_mse` 0.9–1.6e-3, flat over the run (v3 fell to 1.5e-5); mean trust by
quarter 0.056, 0.064, 0.065, 0.071 (v3 0.40 → 0.03); masters at most 0.95 against
the 1.5 cap; 23,520 predictor fits, none skipped; cumulative prefit cosine 0.124
and error 1.3% below the zero prediction. An offline probe of checkpoint
`a000000007049`: center-visit attention output RMS 2.7 (v3 pilot 25), FFN 3.4
(9.3), spectral norms 1.2–2.5 times random (2.6–3.8), knot overflow 18% (76%),
cosine between the routed state and E's top singular direction 0.29 (0.69).
Verdict: the planned gate passed on every item except the raw-visit maximum,
which exceeded the 3.0 guess by 7% while its mean was already falling; accepted.
These are pilot measurements, not evidence of language ability or convergence.

The 24-hour run `work/core-runs/connected-fp32-v4-full-s7-20261107.json` (seed 7,
rate 0.003, `weight_decay` 0.001, `master_rms_cap` 1.5, same data as every FP32
run) started 2026-11-07T18:25:24Z under the capped owner with bounded memory
recovery (`--resume-on-memory`), pinned executable `e40f2e0d7530aa06…6ee8`,
provenance receipt `connected-fp32-v4-full-s7-20261107-source.json`. First rows:
protocol `causal-connected-fp32-v4`, `readout_prior` on the anchor-0 row, 5.0
anchors per training second, `visit_rms` 1.10. Compare its held-out at equal
anchors with the frozen-tile ablation (400–600k mean 4.009, best 3.940) before
claiming tile learning; the collapse signatures to watch are a rising
`visit_rms_max`, E σ1/σ2 and falling trust and `zero_mse`.

Outcome. Held-out reached 3.636 bits/byte at 57,344 anchors and 3.690 at 73,728,
ahead of the frozen-tile ablation (4.530) and v2 (4.140) at 57,344, but behind a
Witten-Bell byte bigram trained on the same 81,920 bytes (3.313 on the same spans;
trigram 2.570; `work/inference/version-retro-20261107/ngram/`). Mean predictor
trust fell from 0.108 over the first tenth of log rows to 0.0065 over the last,
the applied-to-base local step ratio from 0.17 in the first thousand anchors to
about 0.01, and prefit cosine from 0.3–0.5 in the first anchors to about 0.01 after
15,000. Tiles effectively stopped learning while full-rate decay eroded their
untrusted masters (tile 3 at RMS 0.73–0.84 after 61,004 Adam steps against an
initial 0.82). The cause is structural: without the target byte the predictor can
learn only the expected correction given the state, which vanishes as the readout
fits. Raw visit RMS stayed near 1.1 and E kept an effective rank of about 60 of 128,
so v4's numerical fixes held. The read-only inference record found next-byte
distributions that depend almost only on the last byte and three of four greedy
continuations looping on one byte. Operator STOP at 2026-11-07T22:41Z; the 81,920
evaluation was cut short by it, and its two scored probes matched the previous
evaluation. Weights removed; see the [retrospectives](CONNECTED-RETROS.md).

### Protocol v3 implemented and piloted; 24-hour launch held

Implemented 2026-11-07 from `.omc/plans/connected-v3-scale-control-2026-11-07.md`
after the v2 diagnosis below: decoupled weight decay with a hard master RMS cap on
tiles and E, a saturating activation curve, a learned readout bias, one mean local
Adam step per visited tile per anchor, an effectively-zero-only predictor skip with
a clipped normalized target, scale metrics in every log row and report, and
`causal-connected-v3` / `causal-connected-fp32-v3` identity; see
[the v3 section](../src/causal/CONNECTED.md#scale-control-saturating-curve-readout-bias-and-per-anchor-local-steps-v3).
Three concurrent authoring lanes worked against a written seam contract; two
independent reviews followed. The Rust review found one defect before the sweep:
the new per-tile maximum was an `f32` counter that `connected.json` and
`report.json` printed differently, which would have made automatic memory
recovery refuse every v3 run; it is now `f64` with a serialization-parity
regression. The docs review found wording only. The integrated sweep passed (142
Rust tests, three ignored as before, 26 Python, strict Clippy, plan profiles,
smoke); executable `3860648602b21d11…a24b`.

Pilot `connected-fp32-v3-pilot-s7-20261107` (seed 7, 30 minutes, the full-run
config with `weight_decay` 0.01 and `master_rms_cap` 1.5) ran under the capped
owner to its deadline: 8,580 anchors at 5.1 anchors per training second (v2 2.6,
frozen-tile ablation 20.7), peak working set 58 MB, held-out 4.290 bits/byte at
8,192 anchors and 4.185 at 8,580 (v2 4.63 and the ablation 4.48 near 8k; v3 route
updates share v2's undamped-gain caveat). Scale behavior over the run: master RMS
stayed at or below 1.02 so the cap never bound, while decay shrank untouched
outer-tile masters to 0.66 (0.8x init); readout output RMS rose 1.45 → 12.7
(an offline probe of checkpoint `a000000008580` measures 18.7 after four visits);
FFN knot overflow 3% → 34%; E σ1/σ2 1.5 → 9.7 with effective rank 6.8 (v2 2.2,
ablation 3.7); interval `zero_mse` 2.7e-3 → 1.5e-5; mean trust 0.40 → 0.03; the
predictor never skipped a fit (28,240). The probe shows where the growth comes
from: with masters at init scale, the center tile's attention output has RMS 25 on
an RMS-1 input, its V, G and DOWN masters sit at 3.0–3.8 times the random
spectral norm at unchanged RMS, and the routed state's cosine with E's top
singular direction is 0.69. Reading: the v2 explosion (RMS 1e5) is fixed, but an
alignment attractor remains that isotropic decay and an RMS cap cannot touch —
E tends to rank one, every prefix row becomes nearly the same vector, attention
amplifies that common direction, the stream grows along it, and the real
correction both shrinks as 1/RMS and loses its component along the state. The
collapse is about five times slower than v2's. The 24-hour run is held; the
candidate follow-up (per-visit residual normalization, readout bias initialized
from the public training corpus' byte log-frequencies, weaker decay) is planning,
not implemented. Pilot artifacts are preserved.

### Calibrated predictor v2 FP32 study: stopped and preserved (collapse diagnosed)

Owner direction on November 6: make the predictor learnable and launch fresh. The
predictor now trains on per-tile normalized targets and publishes `nu * trust *
p_hat`, where trust is an online least-squares shrinkage slope; local steps use the
Lambert gain times trust. See
[calibrated predictor](../src/causal/CONNECTED.md#calibrated-predictor-and-trust-scaled-local-updates-v2).
Independent review approved after fixes; the integrated sweep passed (134 Rust
tests, three ignored, 26 Python, strict Clippy, profiles, smoke); executable
`7b8b7daf…23b7`. The 640-anchor pilot completed cleanly (9/9 hashes): predictor
error 0.979 times the zero-correction error (v1: 2–189 times), trust 0.15–0.25 per
tile, 99.3% of local visits trained, held-out 4.640 bits/byte on 128 targets, 2.6
anchors per training second. The full run
`work/core-runs/connected-fp32-v2-full-s7-20261107.json` (seed 7, 24 hours, same
data and optimizer settings as the earlier FP32 runs) started 2026-11-07T05:30Z.
Because calibrated metrics keep the Lambert gain near or above one, route updates
are less damped than in v1; compare with that in mind.

Interim diagnosis, 2026-11-07 ~11:00Z (read-only numpy probe over checkpoint
`a000000086016`, replicating the tile forward and `shared.rs::task`; scratch
script, not part of the tree): the real corrections have collapsed because the
residual stream exploded, not because the predictor failed. Center-tile FP32
masters sit at RMS 32.9 (`A`), 30.4 (`CURVE`) and 6.5 (`DOWN`) against the 0.82
ternary init; one center visit maps an RMS-1 prefix state to RMS 1.1e5, with every
FFN preactivation beyond the outer knots where the curve extrapolates linearly.
The RMS-normalized readout hides this from the loss (held-out 4.09 at 90k anchors,
ahead of the frozen ablation at equal anchors) but returns the correction scaled
by `1/RMS(output)`: probe 1.4e-5 versus 0.42 in the ablation, matching the logged
interval `zero_mse` of about 1e-12. Later visits are bypassed (`out ≈ x`). Log
counters date the onset before anchor 2,000: all eight lanes start from one prefix
state and queue on the center tile, so about ten near-identical local updates per
anchor accumulate under scale-free Adam (cumulative applied norm 3.0e3 by anchor
1,007); the 640-anchor pilot ended before this showed. E's top singular direction
correlates |ρ| = 0.96 with the log byte unigram in both runs (σ1/σ2 9.7 here, 2.7
in the ablation). Calibration lags the live target about 1000x and the 1e-12
no-evidence threshold is now being crossed. Remediation is planned, not
implemented: `.omc/plans/connected-v3-scale-control-2026-11-07.md`.

Owner decision 2026-11-07 ~11:20Z: the run was stopped by operator `STOP` once the
diagnosis showed it could not recover (nothing in v2 shrinks masters at RMS 33).
It ended cleanly at 92,281 anchors (`terminal: stopped`, return code 0, checkpoint
`a000000092281` verified by the owner); best held-out 4.015 bits/byte at 73,728
anchors, last 4.094 at 90,112, trust 0.005–0.074 per tile. Artifacts are preserved
as the measured record of the collapse; protocol v3 is being implemented.

### Ablation: predicted local updates off, completed

Started 2026-11-07T04:41:21Z for 8 hours (`connected-fp32-ablation-off-s7-20261106`):
the fail-closed study's FP32 config with every predicted local tile update skipped,
so tiles keep their initial weights while embedding, routing, predictor fitting and
memory continue. Executable `f2f21e1a…5caf`, built from a temporary `gate: "off"`
value that the predictor-calibration change removes; the run's pinned copy keeps
it reproducible. It ended at its deadline (`terminal: deadline`, return code 0,
625,059 anchors, checkpoint `a000000625059`). Held-out bits/byte, mean (minimum)
by anchor window: 0–200k 4.775 (4.368), 200–400k 4.133 (4.001), 400–600k 4.009
(3.940), 600k–625k 3.987 over four points; last-eight mean 3.978, best 3.940 at
589,824. Against the fail-closed study below (0–200k 4.192, 200–400k 4.068,
400–600k 4.120, 600–800k 4.100): the frozen-tile run starts worse and is at least
as good from 400k on. Conclusion: the fail-closed study's sub-floor result came
from embedding, routing and memory without tile learning; its 0.2% of trusted
tile updates bought only an earlier descent. A connected FP32 model with
untrained tiles reaches about 3.94–4.0 bits/byte on this held-out set; that is
the baseline any tile-learning protocol must beat at equal anchors.

### Fail-closed gate FP32 study: completed, artifacts and code removed

A 24-hour fresh FP32 run (Nov. 5–6) skipped predicted local updates until a tile's
observed predictor quality was positive. It ended cleanly at its deadline after
1,990,781 anchors. Held-out bits/byte, mean (minimum) by anchor window: 0–200k
4.192 (3.862), 200–400k 4.068, 400–600k 4.120, 600–800k 4.100, 800k–1M 3.890,
1–1.5M 3.872 (3.756), 1.5–2M 4.063 (best 3.735). It was the first connected run
below the 4.414 unigram floor, but greedy samples remained repetitive. The
predictor never became informative (cumulative cosine 0.040, error 12.3 times
zero), so the gate skipped 99.8% of 55.5M local visits and tiles stayed near
their early state. Evaluation uses 1,024 targets; single readings vary 0.1–0.2.
By owner direction on November 6 its run artifacts and gate option were removed;
the predictor itself is the next target. Corpus byte-unigram entropy is 4.414.

### Fresh connected FP32 study: stopped and preserved

Stopped by operator STOP at 164,764 anchors on 2026-11-05 after a clean
completion of its first attempt; all nine final checkpoint payload hashes
verified. Held-out stayed at 4.44–4.54
bits/byte from 8,192 anchors on, at the 4.414 unigram floor. A live-checkpoint
probe at 139,264 anchors found next-byte distributions within 0.06 bits of the
unigram for every prompt and 0.004 bits apart from each other. Predictor cosine
to real corrections fell from 0.17 to 0.002 while about 72% of 3.3M predicted
updates ran at gain one on expired evidence. Hypothesis, not established:
uninformative predicted credit, the tiles' only teacher, prevented contextual
learning. Its eval rows label `forward` as `"ternary"`; read `precision`.

Owner direction is to start with full precision and quantize down. The new
`connected.precision: "fp32"` option selects `causal-connected-fp32-v1`, retaining
predictive local updates, shared signal memory, learned activation curves,
routing and Lambert W gains. Explicit continuous views cover the full forward
and adjoint path; FP32 master state and versioning remain checkpointed. See
[precision design](../src/causal/CONNECTED.md#full-precision-learning-and-quantization-comparisons).

Prepared fresh configs are `work/core-runs/connected-fp32-pilot-s7-20261104.json`
(640 targets or 900 seconds) and `connected-fp32-full-s7-20261104.json`
(24 hours or one 21,006,506-byte public corpus pass). Both retain seed 7, rate
0.003, eight lanes, four workers and 512-byte context. Neither imports trained
state from the ternary study. Launch remains contingent on actual validation,
pilot and immutable quantization-view verification results below.

`complete --quantize-audit` compares FP32, scaled INT8 and scaled ternary
tile/embedding views, leaving the correction predictor and activation arithmetic
FP32. It preserves the source checkpoint and reports completed versus truncated
variants. This is quantization emulation, not a compressed checkpoint export or
native low-bit kernel benchmark, and no quality-based promotion has occurred.

### Earlier full connected ternary study: stopped and preserved

`work/core-runs/connected-full-s7-20261103` launched through the existing
`scripts.core_owner` at 2026-11-04 03:28:27 UTC (November 3, 20:28:27 MST).
The adjacent JSON config specifies fresh initialization, seed 7, rate 0.003,
512-byte context, eight logical lanes, four workers and the current 2,051,840
coefficient model. It uses the pinned public TinyStories train/dev corpus.
The stop budget is 86,400 seconds or 21,006,506 targets (one corpus pass),
whichever comes first; this does not promise a completed pass. The owner retains
its 512 MiB child commit cap, resource floor and graceful STOP behavior.
The initial segment imported no earlier trained state; the continuation restores
its verified checkpoint. Capacity growth and self-training are future experiments;
this run measures broader exposure and earlier-span retention.

The pinned executable SHA-256 is
`3bc1a95146e11bedfd6dbf0be4befce2f4382f5db777c3175be1d658aadb3725`.
The launch receipt records owner PID 91620. A separate startup inspection found
the live trainer at PID 109720 using that run's pinned executable. Initial
evaluation completed all 1,024 public held-out targets at anchor zero. The first
retention reference completed all 128 targets at anchor 128. At the startup
observation of anchor 624, the run had 14,653 predicted local updates, 1,835
predictor fits and peak decoded residency of four bundles; both error logs were
empty. These are startup observations, not a completed-run result.

The new bounded cache avoids repeated decoding and durable writes until dirty
eviction or checkpoint/evaluation flush. A matched fresh seed-7 comparison over
128 anchors produced identical hashes for all nine checkpoint payloads, identical
mechanism counters, held-out scores and generated bytes:

| Measurement | Earlier eager storage | Bounded cache |
| --- | ---: | ---: |
| Training-step seconds | 96.735 | 31.185 |
| Overall elapsed seconds | 165.416 | 47.423 |
| Application bytes read | 40,537,685,964 | 10,261,851,292 |
| Application bytes written | 12,693,827,476 | 3,746,386,792 |

This is 3.10 times faster training and 3.49 times faster overall, with 74.7%
fewer application bytes read and 70.5% fewer written in that short-prefix test.
I/O remains substantial; these are not physical SSD traffic or sustained
512-byte-context/day-long throughput measurements. Comparison evidence:
`work/core-runs/connected-cache-comparison-20261103.json`.

Independent review approved cache ownership, persistence, exact continuation,
measurement causality and launch configuration. Validation in
`work/validation/509a6ed575be4da1a57068f2595fde3b` passed 126 distinct Rust tests
and 19 Python tests, strict clippy, formatting, structure, all three planning
profiles and smoke. Three existing Rust tests remain ignored. One new evaluation
fixture initially exceeded its tiny context; after correcting only that fixture,
its affected tests and the remaining checks passed.

Checkpoints are scheduled every 4,096 anchors and frozen evaluation every 8,192.
Evaluation records per-span target counts, truncation, generation samples and
changes against complete references for fixed, already-observed training spans.
Recent predictor error/cosine metrics are logged separately from cumulative
values. Generated samples are assessment only. Live evidence is in `log.jsonl`
and the owner receipts; graceful stop uses the run directory's `STOP` file.
The first segment ended at 2026-11-04 15:54:48 UTC (08:54:48 MST), with owner
reason `owner_memory_stop` and trainer exit code zero. Ten consecutive host
memory samples below 256 MiB requested the graceful stop. The receipt does not
identify what caused host memory pressure. Trainer peak working set was
55,033,856 bytes (52.48 MiB), distinct from the Job commit cap.

It reached 186,164 targets (0.886% of one corpus pass), 5,633,482 predicted local
updates and 704,273 predictor fits in 44,781.094 owner seconds. The final
checkpoint is `checkpoints/a000000186164`; all nine payload hashes were verified
before any continuation, and a copy plus original receipts was preserved under
`work/core-runs/connected-full-s7-20261103-preserved`. Final evaluation honored
STOP and scored zero targets, so it supplies no final-checkpoint quality score.

The last complete evaluation before the memory stop, at 180,224 targets, scored
all 1,024 fixed held-out targets at 4.47742 bits/byte, versus 10.32309 initially. Both greedy continuations
were 64 spaces. Five complete 128-target previously observed-span probes had
changes of -1.41445, -0.08534, +0.23430, +0.11428 and +0.00629 bits/byte from
their first-exposure references (lower is better). These mixed small-probe
results do not establish stable retention or useful language generation.
Cumulative prefit predictor MSE was 18.65 times the zero-prediction reference;
mean cosine agreement was 0.05363. These online aggregates do not isolate the
predictor's causal benefit or evaluate the final frozen predictor.

The November 4 owner request authorized graceful memory recovery, inference
testing and continuation of this checkpoint. The new opt-in owner mode retains
the lock through memory recovery, requires 30 stable samples above the original
start floor, verifies the final checkpoint again after waiting, and accepts only
clean memory-stop exits. It deducts attempt time, allows at most three automatic
restarts and six cumulative waiting hours, and respects operator STOP. See
[the owner contract](../scripts/AGENTS.md) and
`.omc/plans/connected-resume-2026-11-04.md` at repository root.

The integrated sweep in `work/validation/0ff3e9865b62465e9021a2957cea50c8` passed
128 Rust tests, 25 Python tests, formatting, strict Clippy, structure, all three
planning profiles and smoke. Three existing Rust tests remain ignored. Both
inference and recovery received independent source review before validation.

Actual read-only inference from the final checkpoint completed three prompts,
each returning 128 spaces. Before/after checks preserved all 45 original run
files and the loaded training state, tile/Adam payloads and checkpoint hashes.
Peak decoded residency was four; peak process working set across the samples
was 37,789,696 bytes (36.04 MiB). Scratch was removed after each completion.
Evidence is `work/inference/connected-20261104/results.json`; its separate
inference binary SHA-256 is
`1e5f5119668ed9134ee07dc5b2c99717f466583ab20d42563e72c6f062e3ec0a`.
This verifies execution and immutability, not useful language generation.

The continuation owner PID 114648 started at 2026-11-05 00:27:01 UTC (November 4,
17:27 MST). After 30 stable samples it started trainer PID 110336 at 00:27:32
UTC using the original `3bc1a951...` executable. The initial restored log row
had anchor 186,164, 704,273 predictor fits, 32 training-memory records and
mechanism counters identical to the previous final report. Its full frozen
evaluation completed all 1,024 held-out targets at 4.42689 bits/byte before
any new training. Startup inspection then observed progress to 186,360 targets,
with four-bundle residency and empty trainer/owner error logs.

The continuation retains the original total-target cap and has 41,618 whole
seconds from the original 24-hour budget. Without another recovery wait, that
budget ends near November 5, 05:01 MST, followed by finalization. The resume
config and launch/source/startup receipts are adjacent to the run and prefixed
`connected-full-s7-20261103-resume-20261104`. The live recovery receipt is
`owner-recovery-20261105T002701Z-3dbccd53.json` inside the run. The study remains
active at that startup observation. No reboot restart is added.

The FP32 direction subsequently requested a graceful operator STOP. The owner
exited zero at 2026-11-05 03:58:20 UTC (November 4, 20:58:20 MST), with reason
`stopped` and checkpoint `a000000236228`. The final state has 889,489 predictor
fits and 7,116,866 local updates. Independent verification confirmed all nine
payload hashes, report/counter agreement, pinned executable/config hashes,
empty stderr, both processes exited and owner lock removal. The checkpoint and
receipts are preserved under
`work/core-runs/connected-full-s7-20261103-preserved/a000000236228`; its
`termination-evidence.json` and the adjacent FP32 transition receipt record the
copy hashes and verification. This is an operator transition, not a new claim of
convergence or a forced termination.

### Decoder audit: corrected prefixes, repetition remains

The November 4 audit found two connected-generation bugs: using the corpus
window helper added an extra START marker to short explicit story prefixes, and
one-byte prompt warmup incorrectly observed the synthetic END marker. The new
`connected-story-window-frozen-local-lanes-v2` policy fixes both and uses the
finite lowest-byte greedy selector, including signed-zero ties. This is a
generation policy version, not a training-protocol change. The running study
retains its pinned executable and earlier generation policy.

A hash-verified copy of checkpoint `a000000186164` was tested with six prompts:
three story prefixes, `X`, and 511-/513-byte window-boundary inputs. Each received
16 generated bytes under the earlier decoder, corrected decoder and diagnostic
prompt-only feedback mode: 18 completed requests. Five prompts still emitted
only spaces. For `X`, the earlier R/dash sequence changed to sixteen J bytes.
Corrected and prompt-only outputs were identical for every tested prompt.

Independent arithmetic checks passed all 192 corrected/ablation trace rows:
visible windows, finite logits, greedy selection, softmax probability and byte
emission agreed. The separately indexed scalar tied readout had zero maximum
absolute error against the packed kernel. Checkpoint source files and loaded training state
were unchanged, decoded residency stayed within four, and each scratch directory
was removed. These checks cover the final readout and selection, not all upstream
forward or learning mathematics.

For the three story prompts, the second generated space had probability
0.2105-0.2125 and a logit margin of 1.0340-1.0411 over the runner-up. A first
space after a word is reasonable; repeatedly selecting another is the observed
failure. Prompt-only mode retains prompt-derived memory and freezes all later
memory evolution, including local replay and aging. It did not remove repetition
in this bounded comparison. This does not establish global collapse or identify
which training mechanism caused the scores.

Evidence: `work/inference/decoder-audit-20261104/analysis.json` and
`comparison.json`; corrected executable SHA-256
`52e04813c77dc570dfc483fa6e8b2407f0e528334031b139876faca6e6b3fb91`.
Independent review and validation logs in
`work/validation/ea495053cfda433b9fb0b701dc6e6c24` confirm 130 passing Rust tests,
25 passing Python tests, formatting, strict Clippy, structure, all three planning
profiles and smoke; three existing Rust tests remain ignored. See
[memory scaling](../src/causal/CONNECTED.md#memory-scaling) for the distinction
between bounded resident tile state and total parameter storage.

### Precision follow-up: redundant projections in the saved checkpoint

Further November 4 inspection of the same anchor-186164 checkpoint found a
specific representation problem upstream of byte selection. Tile 3's DOWN
matrix (128 by 384) has four distinct rows and three row families up to sign;
126 rows belong to one family. Tile 6's A matrix (384 by 128) has five distinct
rows and three sign families; 379 rows belong to one family. Each of these
individual linear matrices therefore has rank at most three. Residual,
attention and nonlinear paths remain; this is not a rank claim about the whole
tile or model. Both tiles occur on the frequently observed inference route
`[3,6,3,6]`.

Their FP32 shadow weights are far outside the fixed +/-0.5 ternary thresholds:
median absolute values are 62.40 for tile 3 DOWN and 72.50 for tile 6 A, with
98.55% and 99.30%, respectively, above magnitude ten. Independent decoding
found zero mismatches between shadow quantization and saved trits. These are
observed redundant forward projections and shadow drift, not proof that all
ternary training fails or that higher precision alone repairs local credit.

The tied embedding has 64 distinct columns out of 256, but its largest group
of 191 identical columns contains 182 unseen byte types and nine seen types
accounting for only 113 of the first 186,164 training targets. Common letters
and space retain distinct columns. That rare-byte finding does not explain
the common-text space loop by itself.

Twelve additional completions supplied true prefixes from the first public
training story, at word ends, after the next space, and after the first letter
of the next word. All produced sixteen spaces; each traced first readout used
route `[3,6,3,6]`, with space probability 20.77%-21.27%. These requests use completion's fresh
two-byte warmup, not the original training-stream memory. Observed training
space frequency was 19.47%, consistent with a frequency-like bias hypothesis,
but this comparison does not establish an exact unigram model.

The checkpoint/statistics inspection and runtime results received independent
verification. Evidence is under `work/inference/decoder-audit-20261104/`:
`inspect_quantization.py`, `quantization-inspection.json` and
`context-response/summary.json`. Saved checkpoint hashes and loaded state stayed
unchanged; all twelve requests kept residency at four and removed scratch.
No training code or running executable was changed for this follow-up.

A fresh connected FP32-forward reference is authorized in
`.omc/plans/connected-precision-reference-2026-11-04.md` at repository root.
Its implementation/launch status is tracked above. The earlier connected training uses FP32
master weights, Adam, predictor and activations, but ternary forward matrices.
The old `float_forward` flag is not sufficient for explicit paged connected
state. Earlier anchor-only float runs also repeated, so those runs neither prove
ternary is the sole cause nor settle precision for connected local learning.

### First connected study: completed

The first run used the existing public TinyStories train/dev corpus, seed 7,
base rate 0.003, a 900-second deadline and `resume:false`. It launched at
2026-11-03 15:07:01 UTC through the existing capped owner into
`work/core-runs/connected-v1-s7-20261103`. Initial evaluation records anchor
zero, zero predictor fits, empty training memory and neutral gains. The pinned
binary SHA-256 is `76d26e267816309181622cad05d3f9ff3847d313d421148650dc49e2474bd4f6`.

Independent source review approved the connected engine, numerical and
checkpoint paths. The integrated sweep at
`work/validation/4b54829c8a11477a93b9c7708d67c8c4` passed: 119 Rust tests,
19 Python tests, strict clippy, formatting, structure, three planning profiles
and smoke. Three existing Rust tests remain ignored; they were not executed.
The v2 measurements below describe the earlier anchor-only protocol and are
not measurements of connected learning.

The connected run stopped at its configured deadline and exited with code 0 at
15:22:59 UTC. Owner elapsed time was 957.797 seconds, including 57.036 seconds
for final evaluation; measured training-step time was 886.708 seconds. No old
weights, predictor, optimizer, memory or gain state was loaded.

| Measurement | First fresh connected run |
| --- | ---: |
| Anchors / predicted local updates | 1,201 / 28,411 |
| Observed correction records / predictor fits | 3,564 / 3,564 |
| Cross-lane memory reads | 28,409 |
| Tiles receiving local work / peak decoded leases | 7 / 4 |
| Owner peak process working set / Job commit cap | 49.574 MiB / 512 MiB |
| Requested gains above / below one | 563 / 16,750 |
| Cumulative applied / base update-norm sums | 0.7232 |
| Tile trit transitions / reversals | 1,966,259 / 1,646,826 |
| Cumulative prefit predictor MSE / zero-predictor MSE | 0.00065516 / 0.00026633 |
| Cumulative mean prefit predictor cosine agreement | 0.13126 |
| Initial / final frozen dev bits per byte | 9.87813 / 4.85555 |
| Application paging read / written bytes | 165.534 GiB / 108.363 GiB |

Both frozen evaluations completed the same 128 public held-out targets without
truncation. Both generated samples were 64 repeated `e` bytes. The lower dev
score does not establish useful generation or isolate the predictor's benefit:
cumulative prefit predictor MSE was 2.460 times the zero reference, and 83.75% of tile trit
transitions reversed an earlier nonzero transition. Requested gains are measured
before the uniform norm cap; their counts do not imply net amplification.
Application paging totals include evaluation and are not physical SSD traffic.
The mechanism and low residency are demonstrated; efficient training and useful
learned corrections are not. This is one short run, one seed and a small probe.
Predictor metrics summarize predictions made before each fit across the run;
they are not a separate frozen evaluation of the final predictor. Peak working
set and the Job commit cap measure different resource quantities.

Final checkpoint: `checkpoints/a000000001201` under the run directory. Runtime
artifacts there include `report.json`, `assessment.json`, `log.jsonl`, and
`owner-end-20261103T150701Z.json`. The owner exited, its lock was removed, and
the training process is no longer live. Exact continuation was tested on the
bounded fixture. Independent read-only inspection of the first and final
production checkpoints verified every payload hash, optimizer shape/moment
invariant and accounting field. The final checkpoint was not resumed.

## Implemented

| Area | State |
| --- | --- |
| Connected training | `causal-connected-v5` / `causal-connected-fp32-v5`: predicted local corrections from a calibrated predictor conditioned on the target byte's embedding, scaled by least-squares trust and applied as one mean Adam step per visited tile per anchor; post-visit RMS normalization; corpus log-unigram readout prior; decoupled weight decay at the realized step multiplier with a hard master RMS cap on tiles and E; saturating activation curve; eight shared-memory lanes, curve conditioning, bounded Lambert W gains and at most four decoded tile/Adam bundles across cache and leases; v1–v4 runs need their pinned executables |
| V2 reference training | `causal-latent-v2`: the 2,051,840-coefficient candidate with fixed-weight batches of 1–32 positions, ordered mean-gradient latent Adam, optional cosine decay and deterministic exploration; all tiles resident. Float training requires batch 1 |
| Story generation | `story-window-v2`: a full training window of held-out text ending at a story boundary, then the prompt; greedy or seeded nucleus decoding (`decoding.rs`); evaluations record one of each per prompt; `hexweave complete --top-p/--temperature/--seed`. Connected generation uses the same openings and decoder (policy `connected-full-window-frozen-local-lanes-v3`); its evaluation samples are nucleus (T 0.8, p 0.9) |
| Owner | `scripts/core_owner.py`: `work/core-runs/<name>` rule, single-owner lock, pinned binary, 512 MiB Job commit cap, sampled 256 MiB memory floor, graceful STOP with a 180-second grace, start and end receipts |
| Checkpoints | Hash-manifested and fsynced, newest three kept. Resume identity-checks every readable manifest first, is bit exact, quarantines integrity failures and refuses to restart from anchor 0 when every checkpoint fails. Checkpoints from the `8e7a0eb` binary (earlier identity formula, earlier resume logic) resume only with their pinned `out/bin` copy |
| Data | `science`: pinned QA dataset, development-split evaluation. `drill` v2: the assay curriculum re-dealt to the harness `Shuffled` stream, with key-swap and attention probes |
| Drill v2 harness parity | `core_contracts`: 12 production-drill anchors equal the validated tuning harness bit for bit, and the drill stream equals `Shuffled` across cycle boundaries |
| Evaluation and telemetry | Held-out, teacher-forced science and generation replay history like training (`replay.rs`); drill evaluation stays harness-identical. Each log line carries per-phase wall time and per-tile route counts (`telemetry.rs`) |
| Software fixtures | FP32 `smoke`, allocation-free `plan` (tiny, 81M, 287M), `causal plan`, ternary scalar/AVX2 kernels with parity tests |

Design and invariants: [core path](../src/causal/AGENTS.md) and
[core owner](../scripts/AGENTS.md). For a guided source tour and the measured
bottlenecks, start with [the architecture study guide](../src/causal/STUDY.md).

In the v2 reference, Stage 3 step 2 adds at most four scoped gradient workers
while preserving the sequential forward/correction history dependency. V2 predictor fitting and
checkpoint payloads are removed; legacy numerical fixtures remain test-only.
Owner decision 2026-11-02: retain the original variable history policy (age at
most eight ticks and version lag at most one), deferring the proposed explicit
one-anchor window. That proposal conflicted with exact v1 startup parity:
before trits change, v1 can retain two anchors.

The v2 reference bounds retained position gradients to four results plus one
accumulator and streams checkpoint payload encoding/decoding. These changes do
not page the v2 model: all tiles and decoded Adam state remain resident.
The test-only single-tile paging fixture covers trajectory parity and selected
failure paths; it does not establish physical RSS, throughput or crash recovery.
The completed scan and selected continuation use the earlier pinned executable
SHA-256 `0be1563b6d3cee8524e82e862c2bb198b88b8ed595ed36920d10a87c232be309`.
Their measured timings cannot be attributed to these later source changes.

Independent source review approved the v2 implementation. Batch-one parity
passes over twelve anchors in all history/float combinations; threaded batch
four matches sequential execution and checkpoint continuation under non-default
knobs. The unchanged v1 harness and span golden passed. Re-scoring the saved
100k history-on checkpoint reproduced 3.3469990231317888 bits/byte and alternate
3.4310734260737847 exactly. The completed v2 scan below adds frozen held-out
measurements, while its greedy samples remain repetitive. Software checks alone
establish neither speedup nor improved learning.

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

- `profile-v2-b{1,8,32}-20261102` completed through the capped owner, 3,008
  anchors each, seed 7, rate 0.003, original history policy, exploration off.
  Training throughput was 51.71 / 92.35 / 89.76 anchors/s; owner peak working
  sets were 91.6 / 94.1 / 120.1 MiB; optimizer steps were 3,008 / 376 / 94.
  B8 and B32 measured 1.79x and 1.74x B1 in these single short runs. B8's
  2.9% lead over B32 is not an established batch-size winner: trajectories and
  route lengths differ, and there are no repetitions. All seven tiles updated,
  unevenly. Mean history records rose from 3.43 at B1 to 6.28 at B8 and 6.59
  at B32 because fixed weights preserve older records longer. B32 tile-apply
  time was 1.07 s versus B1's 18.38 s; its remaining training time was mostly
  sequential gather (12.68 s), forward (6.24 s) and prefix construction
  (4.21 s), plus parameter backward (8.87 s). End-to-end times were 223.4 /
  202.4 / 186.4 s, dominated by evaluation; B32 evaluated four times while
  B1/B8 evaluated five, so these totals do not rank training throughput.
  Detailed receipts and aggregation: `work/core-runs/v2-profile-summary.json`.
- `tiny-v2-b32-s7-scan-r003-20261102` and `-r017-20261102` completed
  sequentially at their 5,400-second training-loop deadlines: B32, seed 7,
  original history policy, exploration off, 64 held-out spans (8,192 targets).
  Rate 0.003 reached 392,640 anchors / 12,270 optimizer steps and scored
  **2.89609 bits/byte**; rate 0.017 reached 381,504 anchors / 11,922 steps and
  scored **2.97685 bits/byte**. The 0.08076 difference exceeds the predeclared
  0.02 tie margin, selecting 0.003 without a tie-break rescore. This compares
  equal configured time budgets, not equal training examples, and has no
  repeated seeds. Owner elapsed times were 5,537.765 and 5,590.469 seconds;
  final evaluation added 136.902 and 188.988 seconds respectively. Both arms'
  saved greedy continuations still loop on letters. Verified final snapshots,
  reports and receipts are preserved under `work/core-runs/scan-final-preserved/`.
  The selected continuation launched through `scripts.core_owner` with
  `tiny-v2-b32-s7-selected-continue-20261102.json`, resuming the 0.003 arm with
  a 23,026-second loop budget. It ended at that deadline on 2026-11-03 at
  2,086,528 anchors (23,178 seconds, about 73 anchors/s over the launch) with
  **2.21573 bits/byte**, its best reading (`best/selection.json`). Held-out fell
  from 2.896 at 392,640 anchors to 2.220 at 1,849,248, with interval gains
  shrinking from 0.15 to 0.04, then read 2.250 and 2.216 at the last two
  evaluations. Its float-forward score on the same weights was 2.543. A second
  continuation is running; see
  [reference path](#reference-path-full-window-generation-sampling-and-continuation).
  The completed 0.003 arm spent 1,990.16 seconds gathering corrections,
  1,329.37 on tile-parameter backward, 983.35 on forward and 727.56 on prefix
  construction, versus 161.97 on tile updates. Route `[3,1,4]` accounted for
  251,273 / 392,640 anchors (64.0%); all seven tiles received updates, unevenly.
  These measurements identify costs and usage, not the cause of repetition.
  Its final logged prequential window starts at anchor 0 and therefore cannot
  establish late-training adaptation. Aggregation:
  `work/core-runs/tiny-v2-b32-s7-scan-r003-20261102-routing-summary.json`.
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
  latents read as f32) -- not the 2.681/3.015 final-state numbers above;
  replay moved them to 2.660 (+0.0060) and 2.972 (-0.0015). History (zero vs
  up to two anchors) is therefore at most a 0.005 bits/byte effect; training's
  real window, measured on `work/core-runs/profile-ahead` (`history_len`
  equals the mean route length there), is one anchor's records in steady
  state, not Replay's fixed two (see `src/causal/AGENTS.md` §core). Frozen weights score 2.791
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
  distinct routes across anchors 1-1,000 (tile 5 touched on 107 of those
  1,000 anchors), down to 3 distinct routes by 2,000-3,000 (`[3,1,4]` on 986
  of the last 1,000). Over the full tiny-* runs every tile was updated
  (`latent_steps`): minor tiles received 10-20% of the start tile's update
  count, and every tile was stepped on in at least ~9.7% of anchors. The two arms favored different tile
  sets, so usage is run-specific, not a fixed collapse. Exploration (e.g.
  `no_halt`) is a measured experiment to try next, not a fix for a proven
  capacity problem.
- `work/core-runs/ablation-history-on` and `-off` (TinyStories, window 512,
  ternary, seed 7, rate 0.003, 100k anchors each, identical byte stream; the
  off arm sets `history: false`). The pre-registered rule (off within 0.02
  bits/byte of on at 100k held-out) failed as written: 3.4422 vs 3.3470
  (+0.095). The 2,048-target held-out score cannot resolve 0.02: it swings
  0.03-0.15 between consecutive evaluations of one arm, and the arms cross
  (off better by 0.145 at 40k and 0.110 at 60k, worse at 80k and 100k). The
  paired prequential loss (20k unseen bytes per interval, same bytes in both
  arms) is lower with history off in all four intervals, by 0.025-0.13. Off
  runs 17% faster (54 vs 46 anchors/s) and needs no backward pass at
  inference. The decision rests on a tie-breaker fixed before it runs:
  re-score both arms' 80k and 100k checkpoints on 16,384 held-out targets;
  drop history in v2 if the mean (off - on) is at most +0.02, otherwise keep
  an explicit one-anchor window. `history_len` stayed at 3.06-3.18, the route
  length, confirming the one-anchor steady state. The train-ahead pairs at
  60k and 80k show no consistent adaptation effect at this scale. Minor tiles
  received 9-22% of the start tile's updates in both arms. Tie-breaker
  (`rescore-abl-*`, `eval_spans: 128`, 16,384 targets each): on 3.1689 and
  3.1902, off 3.2288 and 3.2177 at 80k and 100k; mean (off - on) +0.0437,
  above +0.02, so v2 keeps history as an explicit one-anchor window. A
  16-span re-score of the on-arm 100k checkpoint reproduced its logged
  3.3469990231317888 exactly.
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
