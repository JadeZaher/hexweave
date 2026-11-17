# How Hexweave works, and where progress gets expensive

The owner-directed connected architecture is documented in
[CONNECTED.md](CONNECTED.md). It restores predictive local credit, shared
cross-lane memory, learned curve conditioning, saved Lambert W gains and bounded
paged tile ownership. The guide below describes the earlier anchor-only core
and its measured v2 runs; those measurements do not benchmark the new protocol.
For the current architecture, see [memory scaling](CONNECTED.md#memory-scaling)
and the [decoder audit](../../docs/STATUS.md#decoder-audit-corrected-prefixes-repetition-remains).

The remainder is a reading guide to the **earlier anchor-only causal core**. Start with
`core_batch::prepare`, then follow one byte through the files below. The central
performance result is that batching reduced optimizer cost, but left a large
sequential prefix/forward/correction phase. The central quality question remains
open: measured next-byte prediction has improved without establishing useful
free-running completions. More stored parameters alone solves neither problem.

## 1. Know which implementation you are reading

| Surface | Meaning |
| --- | --- |
| `hexweave train --config` | Production entry point, `core_train.rs`, `causal-latent-v2`, all seven tiles resident |
| Completed Nov. 2 rate scans | Pinned executable; measurements below come from these runs |
| Current working source | Includes later bounded gradient retention, streaming checkpoints and story generation changes; those changes are not measurements of the scan executable |
| `core_paged_experiment*.rs` | Test-only paging of one tile and its optimizer state; no production residency guarantee |
| [FP32 architecture](../../docs/ARCHITECTURE.md) | Older Candle `smoke` reference; its sequence computation and optimizer are different |
| 81M / 287M profiles | Planning arithmetic, not admitted causal training geometries |

Both completed rate scans used executable SHA-256
`0be1563b6d3cee8524e82e862c2bb198b88b8ed595ed36920d10a87c232be309`.
Their preserved receipts identify the actual program even though source-tree
receipt fields differ. Reading new code does not retroactively benchmark it.

The nominal candidate ledger is 2,051,840 coefficients, including a dormant
predictor allocation. V2 does not train or consult that predictor in its forward
path. The model consumes **256 possible bytes**, not subword tokens. Candidate
width `d=128`, FFN width `m=384`, attention heads `H=4`, tile count `7`, maximum
visits per byte `4`; geometry permits 1,024 prefix bytes, while these runs use
a 512-byte corpus window. See [geometry.rs](geometry.rs) and [core_config.rs](core_config.rs).

## 2. Reading route and vocabulary

| Order | Open here / search for | Question to answer |
| --- | --- | --- |
| 1 | [core_train.rs](core_train.rs) `drive`; [core_batch.rs](core_batch.rs) `prepare`, `commit` | What stays fixed across a batch, and what still changes? |
| 2 | [shared.rs](shared.rs) `prefix`, `task`; [attention.rs](attention.rs) `forward` | What is a prefix row, and how does the current state read it? |
| 3 | [tile.rs](tile.rs) `cache`; [ffn.rs](ffn.rs) `curve`; [pulse.rs](pulse.rs) `forward` | What are the learned transformation, activation and pulse? |
| 4 | [core_route.rs](core_route.rs) `forward`; [routing.rs](routing.rs) `policy` | How is the next tile selected and credited? |
| 5 | [credit.rs](credit.rs) `gather`; [core_gradients.rs](core_gradients.rs) `mean` | Why are there two backward phases? |
| 6 | [history.rs](history.rs) `compatible`, `mean`; [replay.rs](replay.rs) `observe` | What survives between anchors, including inference? |
| 7 | [generation.rs](generation.rs); [core_eval.rs](core_eval.rs) | Why can held-out loss improve while output still loops? |
| 8 | [tuning_update.rs](tuning_update.rs); [resident.rs](resident.rs); [store_api.rs](store_api.rs) | Where do weight and optimizer bytes actually live? |

An **anchor** is one next-byte training example: a visible prefix plus its next
target byte. A **frame** is one visit to a tile along that anchor's route. A
**batch** prepares B anchors under fixed parameters, then applies their mean
gradients. **Correction** means a conditional loss derivative, not a corrected
text string. An **adjoint** is the derivative passed backward into an intermediate
vector. **History** holds a small number of clipped correction records. Attention
`memory` means the prefix feature array; it is unrelated to paging weight storage.

FFN means feed-forward network. RMS normalization divides a vector by
`sqrt(mean(x^2) + 1e-5)` here, keeping its typical magnitude controlled. Adam's
two moments are moving estimates of gradients and squared gradients; the shadow
weights are the FP32 values being optimized before conversion to packed trits.

In the prepared Neovim session, F1 returns to the guide, F2/F3 move between
topics, and F4 opens the landmark index (Enter follows an entry). Standard
`gt` / `gT` switch tabs; `Ctrl-w w` switches panes. Search a function with
`/fn prepare`, for example. The layout uses an isolated local configuration.

## 3. Follow one next-byte prediction

```text
visible bytes b[0..L)             correction history from completed anchors
         |                                      |
E[:, byte] + position -> RMS rows M[L,d]          mean[d]
         |                         |               |
         +-> last row = x0[d]      |               |
                     |            |               |
                center tile 3 <---+---------------+
                     |   attention -> FFN + pulse -> next state
                     v
              greedy neighbor / halt -> repeat, at most 4 visits
                     |
              RMS(final state) -> tied E transpose -> 256 logits
                     |
                  next-byte loss against actual target during training
                     |
         reverse realized route -> prefix/E credit + frame corrections
                     |
         publish clipped corrections before preparing the next anchor
```

[Shared::prefix](shared.rs) constructs each row by RMS-normalizing the sum of its
byte embedding and sinusoidal position vector. Positions are relative to the visible
window. Every tile queries the same immutable rows; only the single current
128-dimensional state changes along the route. Prefix rows are not contextual
outputs of earlier tiles. Attention can still combine content and positions,
but a longer visible window does not add a deep computation for each past byte.

The embedding matrix `E` has shape `d x 256` and serves both byte lookup and final
readout. Its gradient adds two uses: the final prediction, and every occurrence
of a byte in the prefix. The final prefix row also starts the route, so its
gradient includes that direct path as well as all attention paths.

### Attention without a full per-token K/V bank

For one head, let `k=d/H`, `q_h` be that head's query slice, and `M_i` a prefix row.
Ignoring scales just for this identity:

```text
q_h dot (K_h M_i) = M_i dot (K_h^T q_h)
sum_i p_i (V_h M_i) = V_h (sum_i p_i M_i)
```

The code first forms a score vector `K_h^T q_h`, scans rows for scores, applies
softmax, forms one weighted prefix summary, then projects that summary with
`V_h`. Q/K/V/O projections include `1/sqrt(d)` scales; scores also use
`attention_scale/sqrt(k)`. Read the concrete ordering in `attention::forward`
before attempting a numerically exact replacement.

This factorization avoids storing all token K/V projections. Per tile visit,
the attention work is roughly `O(d^2 + H*L*d)`, with prefix storage `O(L*d)` and
`H*L` attention probabilities. There is one query, not an `L x L` attention map.
It still scans the visible prefix, and backward replay does that work again.
[attention_reverse.rs](attention_reverse.rs) handles both score and value credit.

### The learned curve and pulse are concrete, limited mechanisms

Within [Tile::cache](tile.rs), `u = input + attention_mix * attention_output`.
The FFN and pulse receive `z = RMS(u)`:

```text
gate        = sigmoid(G z / sqrt(d))
pre         = A z / sqrt(d)
ffn         = DOWN(gate * curve(pre)) / sqrt(m)
pulse_latent= tanh(U [z, history] / sqrt(2*d))
heads       = P pulse_latent / sqrt(d)
gain_j      = 1 + 0.125*tanh(heads[j])
shift_j     = 0.125*tanh(heads[d+j])
output_j    = u_j + 0.25*(gain_j*ffn_j + shift_j)
```

Each FFN channel has three learned knot offsets at inputs `-2, 0, 2`. Base knot
heights are `-1, 0, 1`, plus `0.25 * weight`; interpolation is linear and values
outside the knots extrapolate. This is a small piecewise-linear activation
family. The connected protocols change this forward: v3 clamps the curve input
to the knot range and adds a learned readout bias, and v4 RMS-normalizes
`output` after every visit; see `CONNECTED.md`. The equations above remain the
anchor-only core path. The pulse learns bounded gain and shift and supplies
extra route biases.
Its history input is detached: derivatives do not continue into earlier anchors.

The eleven matrices in a tile are Q, K, V, O (`d x d` each), G and A (`m x d`),
DOWN (`d x m`), CURVE (`3 x m`), ROUTER (`r x 2d`), U (`d x 2d`) and P
(`(2d+r) x d`). `r` counts legal neighbor actions plus halt: seven for the center,
four for an outer tile. [tile_gradients.rs](tile_gradients.rs) names the indices.

### Routing chooses a path; it does not blend every tile

Router scores use normalized tile output concatenated with history, plus the
last `r` pulse heads scaled by `0.25`. Core routing chooses the largest
probability, breaking ties by lowest action index. Optional training exploration
can override a later action toward a less-updated legal neighbor; these scans
used `explore=0`. Evaluation and generation use greedy routes. Visits happen
sequentially, can revisit a tile, and stop at halt or the four-visit limit.

Task derivatives follow the realized path with its discrete choices held fixed.
Separate route credit uses `advantage = loss - step_start_baseline`, multiplied
by `(one_hot(selected) - probabilities)`. With gradient descent, a below-baseline
loss reinforces the chosen action. This updates ROUTER and P's route rows;
that policy credit does not propagate through U or upstream representations.
This is a detached routing surrogate on greedy trajectories, not a gradient
through argmax or an unbiased estimate from sampling the stated policy.

## 4. Why batching leaves much of training sequential

`core_batch::prepare` runs prefix -> route -> `credit::gather` -> history insertion
in order for every position. Parameters stay fixed, but position n+1 consumes
position n's correction history. The loop therefore cannot simply become B
independent forward tasks while preserving the current behavior.

`gather` first differentiates the loss through the route to find each frame's
upstream correction and the shared prefix/E gradient. It requests **prefix
credit**, without accumulating tile parameter gradients. After all B positions
are prepared, `core_gradients` revisits each frame to accumulate **tile parameter
credit**, including the routing surrogate. This phase can run across positions.
Repeated visits sum within a position; reduction follows original position order,
and the denominator is all B positions, including zero contributions for tiles
an anchor never visited. Each visited tile gets one apply, then E gets one apply.

Each call to `Tile::backward` rebuilds `Tile::cache`. A normally trained visit
therefore computes the forward cache once initially, once for correction gather,
and once for parameter backward. The latter two also run derivatives. The frame
tape retains inputs and identities instead of the whole cache: less live
activation storage, more recomputation. Retaining selected caches might help,
but would consume more memory across batch positions and visits; that tradeoff
needs measurement, not an assumption that every cache should be retained.

Current source limits retained full tile gradients to four position results plus
one accumulator. It still retains B prefixes, traces and E gradients. It also
creates scoped threads per group of four; the throughput effect of this newer
retention change is not measured by the older scan executable.

### History is a short correction window, not long-term language memory

History holds at most eight **frame records**, with unit-clipped corrections.
Pruning checks age at most eight ticks and version lag at most one against E
and every tile. One anchor advances the clock by four ticks. Under unchanged
versions, records from two preceding anchors can survive: at clock 8, records
from clocks 0 and 4 are both age-eligible. Capacity and version changes can
remove them sooner. More visits also consume more of the eight record slots.

The user chose exact original v1 behavior on Nov. 2; the proposed mandatory
one-anchor window is deferred. Versions advance when packed trits change, not
on every latent Adam step. Startup can therefore retain two anchors even at B1.
`History::mean` weights surviving frame records equally, not anchors equally.
There is no gradient through this mean into previous anchors and no fitted
predictor in the v2 forward path. Read [history.rs](history.rs) beside [replay.rs](replay.rs).

## 5. The measured speed bottleneck

The completed rate-0.003 scan recorded 392,640 anchors at 75.03 training anchors/s.
Its [phase and routing summary](../../work/core-runs/tiny-v2-b32-s7-scan-r003-20261102-routing-summary.json)
contains the following breakdown of **5,232.88 seconds excluding evaluation**:

| Training phase | Seconds | Share | Main code |
| --- | ---: | ---: | --- |
| Gather corrections and E credit | 1,990.16 | 38.03% | `credit::gather` -> `Tile::backward` |
| Parameter-gradient backward | 1,329.37 | 25.40% | `core_gradients::mean` |
| Route forward | 983.35 | 18.79% | `core_route::forward` |
| Build prefix embeddings | 727.56 | 13.90% | `Shared::prefix` |
| Tile optimizer apply | 161.97 | 3.10% | `Tile::apply` -> latent Adam |
| All other training time | 40.47 | 0.77% | Data, records, E apply, checkpoint, unattributed |

The 5,400-second run deadline starts before opening state and the initial
evaluation. Training telemetry excludes evaluation, while the owner's total
also includes final evaluation and process overhead; these clocks differ.

Prefix, forward and gather together are **70.73%**. Merely increasing parallelism
in parameter backward cannot remove that sequential work. For intuition, if the
entire parameter-backward phase disappeared at unchanged workload and trajectory,
the arithmetic ceiling would be `1/(1-0.2540) = 1.34x`. Deleting all tile-apply
time gives only `1.03x`. These are hypothetical bounds, not forecasts.

Short [3,008-anchor profiles](../../work/core-runs/v2-profile-summary.json) measured
51.71 / 92.35 / 89.76 anchors/s at B1 / B8 / B32. Tile apply fell from 31.59%
at B1 to 3.20% at B32. Batching addressed that cost; it changed which work
dominates. B8's slight lead is not an established optimum: these are single
runs with different learning trajectories. Their evaluation counts also differ,
so total owner durations are unsuitable for comparing training throughput.

[telemetry.rs](telemetry.rs) times disjoint wall-clock phases. Threaded backward
is elapsed time for its scope, not the sum of workers' CPU times. Timed history
record construction/insertion was only 0.039% in the long scan; pruning and mean
formation are outside that timer. The large history-related costs are the
dependency and reverse computation, not the timed record insertion phase.

Other visible costs worth isolating are repeated prefix normalization/positional
construction, scalar packed row products in [head_rows.rs](../ternary/head_rows.rs),
prefix scans and allocations. The current telemetry does not rank these inner
operations. Packed ternary storage does not imply uniformly accelerated kernels;
the head-row operations are scalar even when full products select AVX2.

## 6. Why lower loss can coexist with repetitive output

The preserved [0.003 report](../../work/core-runs/scan-final-preserved/tiny-v2-b32-s7-scan-r003-20261102/report.json)
scores **2.89609 bits/byte** on 8,192 held-out targets after 12,270 optimizer steps.
The [0.017 report](../../work/core-runs/scan-final-preserved/tiny-v2-b32-s7-scan-r017-20261102/report.json)
scores **2.97685** after 11,922 steps. The difference, 0.08076, exceeds the declared
0.02 tie margin. This selected 0.003 under the declared time budget; it is neither
an equal-example comparison nor a replicated quality result. The 0.003 saved
greedy samples still degenerate into repeated characters. Useful completions
remain unproved.

Teacher-forced scoring asks for the next byte after a true prefix, then observes
the true next byte. Free generation repeatedly picks its own byte and feeds it
back. With history enabled, [Replay::observe](replay.rs) even computes correction
history using that generated byte as its target. The resulting prefixes and
corrections can differ from scoring. A loss improvement alone does not establish
robustness along these self-generated trajectories.

History-on generation also calls `credit::gather` after each byte. It freezes
weights but still pays backward work, currently even computing tied-E credit
that replay discards. [generation.rs](generation.rs) rebuilds the visible prefix
for every generated byte. This explains why inference is not just the forward
phase in the timing table. New story-window stopping/prefix policy in the working
source is separately versioned; it is not evidence that these saved samples improve.

Measured on 2026-11-11, the largest gap was the input layout, not the loop above.
[Shared::prefix](shared.rs) counts positions from the first visible byte, and
training always fills the window, so its newest byte sits at position 511. The
saved samples came from `END + prompt` alone, 12–35 bytes. On the 2.216 bits/byte
checkpoint that gave single-letter repeats; after a full window of held-out text
(`story-window-v2`, [decoding.rs](decoding.rs)) the same weights wrote words, then
greedy phrase loops, and nucleus sampling removed the loops.

The most frequent route `[3,1,4]` accounts for 64.0% of scan anchors. All seven
tiles are visited; outer minor tiles participate in roughly 8.2-10.0% of anchors,
and 97 different routes occur. This demonstrates uneven use, not total starvation
or a proven cause of repetition. B32 also makes one optimization step per 32
bytes: compare both examples and optimizer steps when interpreting learning.

Potential limits to investigate are the small active state, raw prefix rows,
at most four visits, short detached history, greedy routing credit and ternary
forward precision. Each mechanism is real; which one limits language quality
has not been isolated. More selectable tiles would not automatically increase
per-byte depth, contextualize past rows, or train seldom-used capacity.

## 7. Why sparse routing has not removed the memory bottleneck

Production [ResidentStore](resident.rs) owns every tile. Each initialized matrix
also retains FP32 shadow weights plus two FP32 Adam moments in
[tuning_update.rs](tuning_update.rs): **12 bytes per coefficient**, in addition to
roughly 0.25 byte of packed trits. An initialized N-coefficient region therefore
needs about `12.25*N` persistent payload bytes before gradients, activations,
metadata and allocator overhead. Ternary forward arithmetic still uses floating
point activations and accumulators. Weight compression is only one budget line.

| Memory term | What is bounded now? |
| --- | --- |
| Tile weights + optimizer | All initialized tiles remain resident in production; visiting few does not evict the rest |
| Full tile-gradient results | Working source caps at five, independent of B; this still scales with tile size |
| Batch prefix rows | `4*B*L*d` bytes of f32 payload: 8 MiB at B32/L512/d128 |
| Retained tied-E gradients | `4*B*256*d` bytes: another 4 MiB at B32/d128 |
| Attention/reverse scratch | Recomputed per visit; costs depend on width, heads, prefix and workers |
| Streaming checkpoint buffers | Fewer serialization copies; restore still reconstructs the whole resident model |

Those are payload calculations and implementation bounds, not physical RSS
measurements. The fixed seven-tile geometry, full version snapshots and shared E
also matter when proposing many more parameters; current admitted geometries
cannot simply be replaced by an 81M/287M planning profile.

[TileStore](store_api.rs) makes read/update access explicit. The
[paged fixture](core_paged_experiment.rs) exercises one loaded tile and its Adam
state, with shared E retained. Its parity and one-live-tile checks are useful
software evidence. It has no production batch-atomic checkpoint/recovery protocol:
publishing updated tiles separately can leave a mixed batch after interruption.
A production design must bound both weights and optimizer state, preserve update
ordering and recover a complete batch. It also needs measured physical memory
and I/O cost. Paging can exchange memory pressure for storage traffic without
fixing sequential compute or poor language learning.

## 8. How to turn a suspected limit into evidence

| Question | Useful controlled comparison | What would not settle it |
| --- | --- | --- |
| Is replay the main avoidable compute? | Profile inner cache/reverse work; retain selected caches under an explicit peak-memory budget | Calling all backward time a matrix-kernel problem |
| Is correction history worth its inference cost? | Matched history-on/off training and frozen scoring plus free completions; report speed as well | Dropping history only at generation and treating it as the same model |
| Is routing underexposing capacity? | Controlled exploration/coverage change with per-tile counts, held-out scores and completions | Dominant-route percentage alone |
| Is precision limiting learning? | Match architecture, data and suitable update/exposure controls across modes | Comparing float B1 and ternary B32 as a pure precision experiment |
| Is prefix representation limiting quality? | A separately versioned contextual-prefix/depth experiment with measured added compute | Increasing stored tile count and assuming richer token states |
| Does paging scale? | Parameter scaling with physical owned-tree memory, I/O and interruption/recovery checks | A logical single-tile fixture assertion alone |

These are experiment designs, not new results or launch instructions. Keep the
existing owned run isolated from competing benchmarks. Use the
[current status](../../docs/STATUS.md) and [core invariants](AGENTS.md) for execution
rules. The historical train-ahead number with `prequential_window_start=0` is an
early probe, not a late adaptation gap; historical scores with different held-out
sample sizes should not be presented as a new matched benchmark.

For a first study pass, trace one anchor through sections 3-4, locate the exact
phase timers in section 5, then follow generation and resident optimizer storage.
At each boundary ask: **what state survives, what is recomputed, what receives a
gradient, and which claim has actually been measured?**
