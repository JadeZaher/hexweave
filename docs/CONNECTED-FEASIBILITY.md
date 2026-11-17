# Connected architecture: feasibility and utility

Assessment of 2026-11-07 (America/Denver), written when v4 stopped, updated on 2026-11-08 UTC
after five v5 pilots, and corrected on 2026-11-11 for out-of-distribution generation probes. Per-version detail is in the [retrospectives](CONNECTED-RETROS.md); live state
is in [STATUS.md](STATUS.md).

## Short answer

- **Utility today: none as a language model.** No connected version has reached the quality of a
  byte bigram trained on the same number of bytes. Their recorded samples repeat, but they were
  generated from a short context that training never used, so they say little either way (see
  [Context use and generation](#context-use-and-generation)).
- **Feasibility: not demonstrated, and the blocker has moved.** v5 solved the problem it was built
  for: a credit predictor that also sees the target byte's embedding matches the real correction
  (prefit cosine up to 0.97, trust about 0.8). Yet all five v5 pilots, across local step rates and
  schedules, fell behind the frozen-tile v4 run after 5,000–7,000 anchors. Tile learning from
  accurate local credit helps early and then hurts. Until a tile-learning protocol beats frozen
  tiles at equal anchors, the design's central mechanism has no demonstrated benefit.
- **Comparator inside this repository.** The non-connected reference path trains the same
  2,051,840-coefficient candidate from real anchor credit, with ternary forward weights. It passes
  the matched trigram and runs 10–17 times faster per anchor. Today it is the working baseline; the connected path is a research
  bet that must first show tile learning, then match that baseline's sample efficiency.

## Quality against simple baselines

Held-out bits per byte, lower is better. Connected and n-gram figures use the connected
evaluation's eight 128-byte public held-out spans. The reference path used 64 spans, so compare it
by margin, not by the second decimal. N-grams are interpolated Witten-Bell byte models trained on
the first N training bytes (`work/inference/version-retro-20261107/ngram/`).

| Model | Training bytes | Held-out |
| --- | ---: | ---: |
| Byte unigram | 81,920 | 4.378 |
| Connected v4, best (tiles nearly frozen after 15k) | 57,344 | 3.636 |
| Byte bigram | 81,920 | 3.313 |
| Byte trigram | 81,920 | 2.570 |
| Connected, tiles frozen (v1 FP32 ablation), best | 589,824 | 3.940 |
| Byte bigram | 625,059 | 3.306 |
| Reference path, real anchor credit | 392,640 | 2.896 |
| Byte trigram | 625,059 | 2.471 |
| Reference path, real anchor credit | 2,086,528 | 2.216 |
| Byte trigram / 5-gram | 1,990,000 | 2.443 / 1.547 |

The reference path learns context beyond a trigram. The connected path has stayed between the
unigram and the bigram.

## Context use and generation

A read-only inference record of every connected version's final checkpoint
(`work/inference/version-retro-20261107/`) found:

- v4 assigns the right next byte about twice the unigram probability on the probe prompts, but it
  does so almost only through the last byte. Swapping the 96 bytes before a fixed suffix moved its
  next-byte distribution by at most 0.011 bits.
- The v1 runs were context-blind. v2 reacted to earlier text but put less probability on the right
  byte than the unigram.
- Every version repeats under greedy decoding: 12 of 28 continuations are 160 identical spaces.

**Correction, 2026-11-11.** All of these probes showed the model only the prompt, at most about 120
bytes, with positions counted from the first visible byte. Training always fills a 512-byte window,
so the newest byte always sits at position 511. The reference path makes the effect visible: from
the bare prompt its 2.216 bits/byte checkpoint writes `aaaa…rrrr…`, and after a full window of
preceding held-out text the same checkpoint writes `, there was a little boy named Tim…` before
falling into a greedy phrase loop. The connected figures above therefore describe an input layout
training never produced. With the connected weights removed they cannot be re-measured; held-out
bits per byte used full windows and stand.

## Why the tiles have not learned

Each version fixed a measured failure and exposed the next:

1. v1: predicted credit uncorrelated with real credit (prefit cosine 0.002–0.05).
2. v2: residual-stream explosion starved the real correction to 1e-5.
3. v3: an alignment attractor drove the embedding toward rank one.
4. v4: numerically healthy, but the predictor never sees the target byte. The best it can learn is
   the expected correction given the state, which vanishes as the readout fits. Trust fell to 0.007
   and tiles stopped learning.

The v4 diagnosis is measured, not inferred. On v4's final checkpoint, a closed-form linear fit from
the v4 predictor's inputs explained about 0% of the real correction at every visit. Adding the
target byte's embedding explained 97% at the final visit and 75–85% at the three earlier visits
(`work/inference/version-retro-20261107/credit-probe/`).

## Cost and throughput

| Path | Anchors per second | Target bytes per day | One pass over the 21 MB train file |
| --- | ---: | ---: | ---: |
| Connected v4, 4 workers | 5.4 | about 470,000 | about 45 days |
| Reference path, batch 32 | 52–92 | 4.5–7.9 million | 3–5 days |

The connected run's owner peak working set was 60 MB, so memory is not the constraint. Each
connected anchor runs one scored four-visit route plus eight unscored lanes of four visits that
exist only to generate local credit and memory records. The path pays for its credit mechanism in
throughput, and must earn that back through what local credit enables: asynchronous lanes, bounded
memory, and no full reverse pass.

## What v5 tests

v5 conditions the credit predictor on the target byte's embedding, the conditional variant of
synthetic gradients (Jaderberg et al. 2017, https://proceedings.mlr.press/v70/jaderberg17a.html). The
tile forward, routing and every scored readout still never see the target. It also decays each
tile at the multiplier its step actually received, so untrusted tiles stop eroding. Plan and
pre-registered gates: `.omc/plans/connected-v5-label-credit-2026-11-07.md`.

Decision rules, set before any v5 data (outcome: rule 1 passed in every pilot; rule 2 was never
reached because no pilot passed its held-out gate; rule 3 partly, see v5 results):

1. **Mechanism (30-minute pilot).** Trust must hold at 0.15 or more, the predictor must beat
   predicting zero by at least 10%, and local steps must apply at least 10% of their size. If these
   fail even with the label, predicted local credit is not working at this scale. The remaining
   levers are exact final-visit credit from the readout and a nonlinear per-tile predictor. If those
   fail too, the honest conclusion is to train tiles from real credit.
2. **Quality (24-hour run).** Beating the matched bigram (about 3.31) shows learned tiles add real
   context. Approaching the matched trigram (about 2.5) makes the path competitive in quality per
   byte. Its anchors cost 10–17 times more than the reference path's, so it must also close that gap.
3. **Context and generation.** Earlier-text sensitivity well above v4's 0.011 bits, and greedy
   continuations that stop collapsing to one byte.

## v5 results

Five 30-minute pilots, seed 7, same data and evaluation as v4, judged against gates set before
each ran. Every pilot passed the mechanism gates and failed the held-out gate, so per the
pre-registered rule no 24-hour run was launched. Pilot weights were removed after each verdict.

| Pilot | Local step rate | Prefit cosine | Trust | Held-out at about 8k (v4 4.066) |
| --- | --- | ---: | ---: | ---: |
| v5 | 3e-3, constant | 0.82 | 0.78 | 4.306 |
| v5.1 | 3e-4, constant | 0.97 | 0.82 | 4.419 |
| v5.1 | 1e-4, constant | 0.97 | 0.85 | 4.371 |
| v5.2 | 2e-3, inverse-square-root decay over 500 anchors | 0.89 | 0.81 | 4.332 |
| v5.2 | 2e-3, inverse-square-root decay over 50 anchors | 0.97 | 0.87 | 4.348 at 7,184 |

Prequential training loss, measured on each anchor before learning from it, shows the pattern. All
v5 variants matched or beat v4 up to about 5,000–7,000 anchors. After that v4 pulled ahead in every
case: over anchors 9,000–12,000 it scored 2.607 nats against 2.832–2.956. That is exactly where
v4's tiles froze as its trust collapsed. A constant high rate drove every master to the RMS cap; low
constant rates started too slowly; decaying rates fixed the start but not the later lag.

In the same short-context probe, the v5 pilot's next-byte distributions used context beyond the
last byte more than any v3 or v4 run (0.50 bits against at most 0.16), and greedy output collapsed
to repeated letters rather than spaces; the correction above applies.

What this means: credit quality is no longer the limit. The new limit is how lane-based tile
updates interact with the rest of the system, whether through the eight unscored lanes' imagined
routes, the shared memory that now carries large label-derived signals into every visit, or the
readout and routes having to chase moving tiles. The next experiments, ranked in the plan and not
started, separate these: exact anchor-route credit without lanes, predicted credit applied on the
anchor route only, and shared-memory and route-gain ablations.

## Recommendation

- For real language modelling soon, invest in the reference path. Its greedy output also loops,
  so a sampling decoder is the first missing piece. Update, 2026-11-11: that decoder now exists
  (seeded nucleus sampling, `hexweave complete --top-p`), together with the fix that mattered
  first: generation now opens with a full training window. On the 2.216 checkpoint, sampled
  continuations no longer loop and read as TinyStories-like word salad; greedy output still loops
  on phrases. A 23.5-hour continuation of that run is in progress ([STATUS](STATUS.md)).
- For the connected research question, run one decisive experiment before any further tuning:
  train the connected tiles from exact anchor-route credit. If they still fall behind frozen tiles,
  the connected machinery is the bottleneck and predicted credit cannot rescue it. If they beat
  frozen tiles, lanes are the problem and anchor-route predicted credit is the next protocol.

## Limits of this assessment

- One seed, one corpus (TinyStories), one 2M-coefficient model size, and no repeated runs. Single
  held-out readings vary by 0.1–0.2 bits.
- The reference path and the connected path differ in more than credit: batching, residency,
  evaluation span count and forward precision (the reference run used ternary forward weights
  trained through float latent copies; connected v4 and v5 ran FP32). The comparison bounds the
  cost of predicted credit; it does not isolate it.
- Context-use and generation figures for the connected versions come from short-context probes
  that training never produced (see the correction above); only held-out bits per byte are
  comparable across paths.
- Conditioning the predictor on the label makes it a learned model of the true gradient. Whether
  that still serves the project's local-learning goals is an owner decision, not a measurement.
