# Connected study retrospectives

Written 2026-11-07 (America/Denver), when the owner had the trained weights of every connected run
removed. Each run keeps its config (`work/core-runs/<run>.json`), `log.jsonl`, `report.json`, owner
receipts, stdout/stderr, STOP files and pinned executable under `work/core-runs/<run>/`. Checkpoints
and paged tile stores are deleted, so none of these runs can be resumed or probed again. Inference was
recorded first, in `work/inference/version-retro-20261107/`. Design history lives in
[CONNECTED.md](../src/causal/CONNECTED.md); run narratives live in [STATUS.md](STATUS.md).

## Yardsticks

All figures are held-out bits per byte on the connected evaluation's eight 128-byte public held-out
spans. The n-gram rows are interpolated Witten-Bell byte models trained on the first N training
bytes, with N matched to a run's exposure (one target byte per anchor). Script and results:
`work/inference/version-retro-20261107/ngram/`.

| Training bytes | Unigram | Bigram | Trigram | 4-gram | 5-gram |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 81,920 (v4 run) | 4.378 | 3.313 | 2.570 | 2.186 | 2.156 |
| 625,059 (frozen-tile ablation) | 4.377 | 3.306 | 2.471 | 1.867 | 1.639 |
| 1,990,000 (gate study) | 4.376 | 3.301 | 2.443 | 1.810 | 1.547 |
| 21,006,506 (whole train file) | 4.376 | 3.298 | 2.431 | 1.789 | 1.485 |

The corpus byte-unigram entropy is 4.414. No connected run has reached the bigram row for its
exposure. Single held-out readings vary by 0.1–0.2 bits.

## Results at a glance

| Version | Run (anchors) | Best held-out | Last complete | Tiles learned from predicted credit | Ended by |
| --- | --- | ---: | ---: | --- | --- |
| v1 ternary | `connected-full-s7-20261103` (236,228) | 4.416 at 98k | 4.457 | No: prefit cosine about 0.05 | operator STOP |
| v1 FP32 | `connected-fp32-full-s7-20261104` (164,764) | 4.412 at 147k | 4.450 | No: cosine 0.17 to 0.002 | operator STOP |
| Gate study B | removed 2026-11-06 (1,990,781) | 3.735 | 3.948 | No: gate skipped 99.8% | deadline |
| Tiles frozen | `connected-fp32-ablation-off-s7-20261106` (625,059) | 3.940 at 590k | 3.976 | Off by design | deadline |
| v2 | `connected-fp32-v2-full-s7-20261107` (92,281) | 4.015 at 74k | 4.094 | Destructively | operator STOP |
| v3 | `connected-fp32-v3-pilot-s7-20261107` (8,580) | 4.185 | 4.185 | Briefly: trust 0.23 to 0.04 | pilot deadline |
| v4 | `connected-fp32-v4-full-s7-20261107` (81,920) | 3.636 at 57k | 3.690 at 74k | Until about 15k: trust 0.11 to 0.007 | operator STOP |
| v5 (five pilots) | `connected-fp32-v5*-pilot-s7-20261107` (7–13k each) | 4.198 at 12,811 | n/a | Yes: trust 0.78–0.87 | held-out gate |

Trust figures in the table are means over the first and last tenth of each run's log rows. The
gate study and the tiles-frozen ablation ran the v1 FP32 protocol. Smaller runs (the first
1,201-anchor study, two 128-anchor profiles, the 640-anchor pilots and both v4 pilots) are summarized
in `work/inference/version-retro-20261107/run-metrics.json`.

## v1 ternary (November 3–5)

The first protocol trained ternary tiles only from predicted local credit, with shared lane memory,
learned activation curves and Lambert W update gains. Over 236,228 anchors in two segments the
held-out score stayed on the unigram floor, 4.42–4.46. Greedy samples repeated one byte; after two
decoding bugs were fixed, the audit still produced 16 spaces after "Once upon a time". The predictor
never agreed with the real corrections (prefit cosine about 0.05), while about 30 predicted updates
per anchor kept moving the tiles. Lesson: read every score against the unigram floor and probe
context sensitivity before treating a falling loss as learning.

## v1 FP32 (November 4–5)

Owner direction was to start at full precision and quantize down, which removed ternary weights as
a suspect. The run reached 164,764 anchors with held-out at 4.44–4.54. A probe at 139,264 anchors
found next-byte distributions within 0.06 bits of the unigram for every prompt and 0.004 bits apart
across prompts. Predictor cosine to the real corrections fell from 0.17 to 0.002 while predicted
updates outnumbered real fits about eight to one. Lesson: precision was not the bottleneck; credit
quality was.

## Gate study B (November 5–6)

This run skipped a tile's predicted updates until its predictor beat predicting zero. It was the
first run under the floor, with a best of 3.735 and a final 3.948 after 1.99M anchors. The gate
skipped 99.8% of 55.5M local visits because the predictor never became informative, so the gains came
from the embedding, routing and memory around nearly fixed tiles. Its artifacts and the gate option
were removed on November 6 by owner direction. Lesson: an improvement measured while the mechanism
under study is switched off needs an ablation before it counts.

## Tiles frozen (November 6–7)

That ablation turned predicted local updates off under the pinned v1 FP32 binary for 625,059 anchors.
Held-out window means were 4.775 (0–200k), 4.133 (200–400k) and 4.009 (400–600k), with a best of
3.940. Everything the gate study achieved was available without tile learning, and this became the
bar for any tile-learning protocol.

## v2: calibrated predictor (November 7)

v2 made the predictor learnable: normalized targets, least-squares trust, and local gain multiplied
by trust. Its pilot was the first to beat predicting zero (0.979 of the zero error). The full run was
stopped at 92,281 anchors with a best of 4.015. A checkpoint probe found a residual-stream explosion.
Scale-free Adam, with eight lanes replaying the same prefix state, pushed center-tile masters to RMS
33 against an initial 0.82. The unbounded activation curve extrapolated linearly, and one visit
mapped an RMS-1 input to RMS 1e5. Because the real correction scales as one over that RMS, it fell
to 1e-5 and starved the predictor. The RMS-normalized readout hid all of this from the loss. Lesson:
log the raw scale of every quantity a normalizer hides.

## v3: scale control (November 7, pilot only)

v3 added AdamW weight decay with a hard master RMS cap, a saturating activation curve, a learned
readout bias and one mean local step per tile per anchor. Its pilot (8,580 anchors, held-out 4.185)
removed the explosion but exposed an alignment attractor. With masters at initial scale, the center
tile's attention output had RMS 25 on an RMS-1 input, the embedding's top singular ratio climbed to
9.7, and trust fell from 0.40 early in the pilot to 0.03 at its end. Lesson: Frobenius caps and
isotropic decay cannot fix a
directional collapse. The 24-hour launch was held.

## v4: residual normalization and readout prior (November 7)

v4 RMS-normalized every visit output, initialized the readout bias to the corpus log-unigram, cut
decay to 1e-3 and logged the raw per-visit RMS. It produced the best held-out of any connected run,
3.636 at 57,344 anchors, where the frozen-tile ablation stood at 4.530 and v2 at 4.140. The
embedding's effective rank held at about 60 of 128 and raw visit RMS stayed near 1.1. Trust, however,
decayed from 0.11 to 0.007. The applied-to-base local step ratio fell from 0.17 to 0.01, and the
predictor explained essentially none of the correction variance. Tiles effectively froze after about
15,000 anchors, and full-rate decay eroded the untrusted masters. The cause is structural: the
predictor never sees the target byte, so the best it can learn is the expected correction given the
state, which vanishes exactly as the readout fits. An offline probe of the final checkpoint
confirmed this on 3,000 public training positions (`work/inference/version-retro-20261107/credit-probe/`).
A closed-form linear fit to the real correction from the v4 predictor's inputs explained about 0% of
its variance at every visit. Adding the target byte's embedding raised that to 97% at the final visit
and 75–85% at the earlier three. The owner stopped the run at 81,920 anchors for v5. Lesson: a
fail-safe that switches the mechanism off and keeps it off is a diagnosis, not a success.

## v5: label-conditioned credit (November 7–8, five pilots)

v5 gave the credit predictor the target byte's embedding, the conditional form of synthetic
gradients, and decayed each tile at the multiplier its step received. The credit problem was
solved: prefit cosine to the real correction reached 0.97 and trust held near 0.8. Five pilots then
varied the predicted-credit step rate: 3e-3, 3e-4 and 1e-4 constant, and 2e-3 with an
inverse-square-root decay over 50 or 500 anchors. Every pilot passed the mechanism gates and failed
the held-out gate (4.198–4.419 against v4's 4.066 at about 8k anchors), so no 24-hour run was
launched. Across all five, prequential loss matched or beat v4 up to 5,000–7,000 anchors, then fell
behind it, which is where v4's tiles froze. Lessons: v4's trust collapse had acted as an accidental
learning-rate schedule; and accurate local credit trains tiles but, in this system, continued tile
learning makes the whole worse than frozen tiles. Detail and next experiments:
[feasibility and utility](CONNECTED-FEASIBILITY.md).

## Inference before deletion

Read-only `complete` calls at each run's final checkpoint, through its pinned binary, with every run
directory hashed unchanged before and after. An independent recomputation from the raw call records
found no discrepancies. Records: `work/inference/version-retro-20261107/` (`summary.md`,
`results.json`, `calls/`, `context-control/`).

**Correction, 2026-11-11: these probes ran out of distribution.** Each call showed the model only
the prompt after one story marker, at most about 120 bytes, with positions counted from the first
visible byte. Every connected run trained on full 512-byte windows, where the newest byte always
sits at position 511. The same short-context generation turned the reference path's 2.216
bits/byte model into single-letter repeats (`aaaa…rrrr…`); after a full window of preceding
held-out text the same checkpoint wrote words, then phrase loops ([STATUS](STATUS.md)). The table
and bullets below therefore describe behaviour on an input layout training never produced, not
context use or repetition in the models' own regime. With the weights removed they cannot be
re-measured. Held-out bits per byte used full windows and are unaffected.

| Version | Mean p(next) on 8 probes | Max KL across prompts | Max KL, earlier text only | Greedy loops | Distinct 4-grams |
| --- | ---: | ---: | ---: | ---: | ---: |
| v4 full | 0.099 | 1.82 | 0.011 | 3 of 4 | 0.04 |
| v4 pilot (7k) | 0.062 | 0.51 | 0.05 | 3 of 4 | 0.08 |
| v3 pilot | 0.062 | 0.45 | 0.05 | 3 of 4 | 0.03 |
| v2 full | 0.043 | 3.86 | 2.80 | 2 of 4 | 0.08 |
| Tiles frozen | 0.073 | 0.67 | 0.0002 | 0 of 4 | 0.20 |
| v1 FP32 | 0.050 | 0.010 | 0.005 | 4 of 4 | 0.006 |
| v1 ternary | 0.047 | 0.002 | 0.0003 | 4 of 4 | 0.006 |

The unigram assigns the probed next bytes a mean probability of 0.049. KL is in bits between
next-byte distributions. "Earlier text only" holds the last 1–24 bytes fixed and swaps the 96 bytes
before them, so it isolates context use beyond the last byte. Its v4 maximum excludes one 0.157
outlier. A loop is a 64-byte tail with a period of at most 32 bytes. Hand-written English scores
0.85–0.97 on distinct 4-grams. The v1 ternary run's own binary predates the trace option, so its
calls used the earliest later build that reads its protocol; equal arithmetic is assumed.

- With short contexts, no version stopped repeating under greedy decoding, the only decoding policy.
  Twelve of the 28 continuations are 160 identical spaces. Every loop has period one. The
  tiles-frozen run's 0 of 4 is fragments such as `aa..a he he`, not text.
- With short contexts, v4 responded to the prompt almost only through its last byte. Its held-out
  score, measured with full windows, stays above the matched bigram (3.636 against 3.313).
- v2 was the only run whose short-context predictions moved with earlier text, and it put less
  probability on the right byte than the unigram does. The v1 runs were context-blind, consistent
  with their unigram-level held-out scores.
- Every run with three or more complete evaluations ended worse than its best reading.

## Lessons across versions

1. Compare every held-out figure with the matched-exposure bigram, then the trigram. The unigram
   floor is necessary and far from sufficient.
2. Probe context sensitivity before claiming learning: next-byte distributions that barely change
   across prompts are a context-free model, whatever the loss says. Lay probe inputs out as
   training does: a full window, newest byte last. Shorter inputs move every position.
3. Each version fixed a real failure and exposed the next one: credit quality (v1), scale (v2),
   direction (v3), then information (v4). In every predictor version, trust started between 0.1 and
   0.6 and decayed toward zero.
4. Normalizers hide failures from the loss. v2's readout and v4's visit normalization both needed
   their raw quantity logged before the failure became visible.
5. Tile learning has not yet been shown to help held-out quality. The best results so far came with
   tiles frozen or nearly frozen, including after v5 made the credit accurate.
6. A gate that silently switches a mechanism off can also be acting as a schedule. Before removing
   it, measure what its trajectory was doing.
