# First bounded disk-local text result

Run date: 2026-10-24 UTC (2026-10-23 local). Both native Windows supervised
runs completed with exit zero, confirmed child reaping, unchanged executable/data
SHA256 identities, and successful terminal OS memory queries. No Hexweave process
remained after the comparison. Nothing was scheduled to restart.

## Fixed experiment

12,112 learned ternary coefficients; seven hex tiles; width 16; eight-byte causal
context; 256-byte vocabulary; at most three visits per path. The context encoder
is a weighted embedding sum, **not attention**. Train/validation contain 671/279
newly authored bytes. Evaluation uses 64 held-out contexts before and after,
with history reset per example and no evaluation updates.

Both modes use seed 7, 256 step opportunities, anchor interval four, and
**64 actual target exposures**. The anchor-only control skips non-anchor
opportunities; it is not an ordinary every-example dense-training baseline.

| Measurement | Pulse | Anchor-only control |
| --- | ---: | ---: |
| Child wall time, including startup/monitor overhead | 2.109 s | 0.547 s |
| Numerical training phase, including tile I/O | 1.896 s | 0.465 s |
| Peak child working set | 8.68 MiB | 4.29 MiB |
| Actual target tokens / training second | 33.76 | 137.78 |
| Training tile visits | 642 | 164 |
| Local predicted updates | 477 | 0 |
| Returned anchor frames | 165 | 164 |
| Predictor training updates, including replay | 229 | 0 |
| All-phase tile loads / commits (includes initialization) | 1,268 / 616 | 792 / 155 |
| Tile bytes read / written | 628,738 / 301,877 | 395,674 / 75,981 |
| Initial held-out NLL, nats/byte (lower is better) | 5.52451 | 5.52451 |
| Final held-out NLL | **5.70083** | **5.91539** |

Pulse finished with lower loss than its control, but **both worsened from
initialization**. Pulse used about 4.08x the training-phase time and 3.86x child
wall time at equal target exposure. One seed and a tiny authored fixture cannot
establish convergence, statistical advantage, energy savings, or pretraining
efficiency. This result does not justify a larger training campaign.

## Locality and resources actually demonstrated

Each tile load is scoped to one owned decoded tile, and subsequent tile weights
are not loaded to compute the current local correction. The local pass retains
no full activation tape. This is application-level weight locality; the OS may
cache inactive files. Shared packed weights (2,248 bytes) and shared integer
evidence (17,984 bytes) remain resident. Payload counts exclude allocator/runtime
and serialization copies; the independently observed working set is above.

The native Job Object accepted suspended assignment and enforced one process and
512 MiB process/job committed-memory caps. Runs also used below-normal priority,
single-thread backend settings, CPU-only execution, 30-second deadlines, sampled
RSS/available-memory guards and a 5,128,175,001-byte adaptive host cushion on this
machine. The Python owner is outside the job; neither a hard RSS cap nor a CPU
quota is claimed. No stop or retry was needed. Final snapshots are for inspection,
not resumable checkpoints.

## Next hypotheses, not additional changes in this run

1. **Stabilize the discrete update rule before scaling.** Threshold one can move
   every nonzero-gradient trit on every update. Compare smaller accepted update
   budgets, confidence gating and accumulated evidence using training-side
   diagnostics and a newly reserved evaluation split. Tile evidence is currently
   fresh per application; merely raising its threshold would suppress updates.
   Accumulation requires versioned evidence persisted with the streamed tile.
2. **Measure predictor usefulness, not just calls.** Add returned-correction
   alignment, uncertainty/calibration and accepted-update accounting on anchors.
   History reached three records and rejected 165 stale records; conservative
   all-node version invalidation may limit transfer. Relaxing it is an algorithmic
   change requiring new provenance and bias tests, not a free speed optimization.
3. **Instrument storage before changing its protocol.** This run made hundreds
   of small synced JSON commits. I/O is a plausible major cost, but no exclusive
   I/O-time profile was recorded. Compact binary tile records or a small bounded
   resident cache should be compared with the same frozen-version semantics,
   actual RSS and equal target exposure. Do not call reduced durability a gain.

These remain proposed follow-ups after the requested three loops. There is no
fourth unbounded optimization cycle or background trainer.

## Receipts and validation

- [Pulse terminal report](../work/text/3b3000bcd0ad4fabbd2601d857bfc2e4/report.json)
- [Anchor-only terminal report](../work/text/949eb8f0a5244c85ba2e51532ea7b6ce/report.json)
- [Pulse ownership events](../work/text/3b3000bcd0ad4fabbd2601d857bfc2e4/events.jsonl)
- [Integrated Rust tests](../work/validation/a9a9b4cb5d5d4114a28fd94c3d905b33/rust-tests.log)
- [Integrated Python tests](../work/validation/a9a9b4cb5d5d4114a28fd94c3d905b33/python-tests.log)

Validation passed 58 Rust and 38 Python tests, structure checks, formatting,
Clippy with warnings denied, three planning profiles and the existing FP32 smoke
check. A separate reviewer examined numerical semantics, AVX2 safety, rejected
optimization handling, native ownership/cleanup, documentation and receipts.

Final release binary SHA256:
`48c5402d14b16b0f4169b7ba5e441a452d268e321c83b5cd271387cd42473bd6`.
It was built from the source-matched staging checkout using the local cached
dependencies, then hash-verified into this repo's owned runtime directory.
