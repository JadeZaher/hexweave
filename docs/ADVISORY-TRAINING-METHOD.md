# External advisory: fixing the training method (2026-10-24)

Status: **external advice, not adopted scope.** This memo adopts no declaration,
authorizes no run, changes no contract and claims no result. It reads the
evidence already recorded in [STATUS.md](STATUS.md) and recommends what to
declare next. Every recommendation below is a hypothesis until its own gated
fixture says otherwise.

## What your own evidence already establishes

1. **The architecture is the conventional part.** A hex grid of FFN tiles with
   a router and a recurrent pulse is a mixture-of-experts with adjacency
   structure plus recurrence. That class is proven elsewhere; the open risk is
   concentrated in the update rule and the credit path, which is also where
   every recorded failure lives.
2. **Direct ternary coordinate descent takes adverse steps.** The four-commit
   trace: first-pair MSE rose 0.279286 → 0.987887 while the continuous
   gradient dotted with actual displacement was ≈ −3.88688
   ([STATUS.md](STATUS.md), four-commit section). Downhill direction, uphill
   outcome: a full ±1 trit move overshoots the gradient's region of validity.
3. **Sequential presentation interferes; joint updates fit.** online64 failed
   its every-pair gate; joint64 passed on the same four frozen targets
   ([STATUS.md](STATUS.md), E3/diagnostics summary). The rule is
   variance/interference-limited, not proven broken.
4. **The objective admits magnitude collapse.** ≈ 96.17% of the coordinate
   arm's raw-error decrease came from reduced prediction energy while final
   direction stayed negatively aligned with every target, near tanh
   saturation ([STATUS.md](STATUS.md), two-arm comparison).
5. **Simpler credit paths won in your ablations.** T1/T2: every detached arm
   improved held-out NLL, every coupled arm worsened it.
6. **Weight-space local learning already showed the retention tradeoff.**
   Locals increased old-target loss while lowering pending-byte loss
   (learning-gap fixture) — the stability/plasticity dilemma in miniature.

## Ranked recommendations

### R1. Latent accumulator with error feedback (highest leverage)

Replace direct trit coordinate moves with a small tile-local accumulator per
weight: updates accumulate in a quantized (8-bit) latent value with error
feedback (store the residual after each trit commit); a trit flips only when
its accumulator crosses a threshold. This is BinaryConnect / BitNet-style
latent-weight training (Courbariaux et al. 2015, arXiv:1511.00363; Ma et al.
2024, arXiv:2402.17764) with 1-bit-Adam-style error compensation (Tang et al.
2021, arXiv:2102.02888), sized to stay tile-resident.

It directly targets failure 2: small steps accumulate instead of being forced
to quantize to ±1. The accumulator is training state that **streams with the
tile**, so the memory architecture survives (see Invariant below).

Suggested fixture: same seed-7 predictor, same four frozen pairs, same
every-pair dot/error gate and bounds as joint64, so the result chains onto the
existing comparison. Declare finite accumulator precision, threshold and
transition caps up front.

### R2. Commit once per residency (batching, scaled from joint64)

While a tile is resident, accumulate contributions across many examples and
commit trit transitions once per residency. This scales the joint64 result
(failure 3) and is also the economic requirement of streamed training:
arithmetic intensity per byte streamed must be high for tile-swapping to beat
resident training. The algorithmic fix and the memory architecture want the
same change; one fixture can carry evidence for both, with the usual caveat
that equal row-move budgets are not equal compute.

### R3. Direction-aware objective; de-saturate the correction head

Add a cosine/direction term (or normalize targets) alongside MSE so shrinking
prediction energy stops being the cheapest descent path (failure 4), and
normalize before the tanh so corrections operate outside saturation. Keep the
existing raw-objective versus delivered-gate distinction; change the fitting
objective, not the gate.

### R4. Freeze routing before learning it

Routing gradients are the noisiest credit path in any MoE. Hash/fixed routing
performs close to learned routing (Roller et al. 2021, arXiv:2106.04426), and
the hex geometry provides a natural position-structured fixed route. For the
next text-level campaign, consider freezing routing and training tiles + pulse
only; reintroduce learned routing as its own later gated feature. Your T1/T2
detached-credit evidence (failure 5) already favors simpler credit paths.

### R5. Sever the locals bet; anchors + replay first

Target-free locals driven by the learned correction predictor are synthetic
gradients (Jaderberg et al. 2017, arXiv:1608.05343) — historically workable
small, degrading with depth/scale — and they produced the retention tradeoff
(failure 6). Anchors plus replay is the conventional, defensible path
(experience replay: Rolnick et al. 2019, arXiv:1811.11682); v3's newest-four
replay is already the right mechanism. Validate anchor-only learning to a
non-degenerate text endpoint first; re-admit locals as a separately gated
ablation that must beat the anchor-only baseline.

### R6. Spend no full-pass budget before predictor-level gates pass

E3's cost is measured and its endpoint was degenerate (all outputs closing
braces — mode collapse onto a high-frequency structural byte). Before any E4
launch consumes the two-hour ceiling, require: R1/R3 gates passed at fixture
scale, plus a declared mode-collapse countermeasure at the text head
(byte-frequency-aware loss weighting or an entropy term) and a data-mix check.
A full pass is the most expensive possible way to discover an optimizer bug
that a four-pair fixture can catch.

### R7. Memory elasticity: RAM is a cache size, never a semantic input

Owner-stated principle (2026-10-26): the architecture should run well on
whatever memory is made available. The strongest implementable form: **the
amount of available memory may change throughput, never results.** The
content-addressed immutable tile store makes this provable rather than
aspirational — if residency is a pure cache over disk-truth tiles, any cache
size yields bit-identical computation, only slower or faster. Three
consequences:

1. **Separate host protection from run sizing.** The adaptive cushion
   (max(2 GiB, 15% of RAM)) protects the host; the run's true floor is one
   resident tile + owner overhead + the child Job cap — tens of MiB enforced,
   512 MiB committed. The 2026-10-25 admission failure demanded ≈ 5.5 GiB for
   a workload whose measured v2 child peak was 4.41 MiB. Recommend: a fixed,
   explicitly declared host cushion (its own reviewed decision) plus a
   workload floor derived from the plan, replacing the conflated percentage.
2. **Elasticity at attempt boundaries, not mid-run.** Determinism forbids
   runtime-adaptive sizing. Instead: sample availability at admission,
   quantize to a declared memory tier (resident-tile budget, replay length,
   batch-per-residency), record the tier in the plan, run deterministically at
   that tier. ResourceStop checkpoints are already resumable
   (checkpoint_owner rejects only Failed/BudgetStop), so checkpoint → resume
   at a different tier is the elasticity mechanism — provided the
   attempt-slice BudgetStop defect (review finding M3) is fixed so resumable
   work is not stranded.
3. **Eviction must never order computation.** Tile scheduling must remain a
   function of the data and plan alone; residency only decides which loads hit
   cache. Guard this with a fixture asserting bit-identical results across two
   residency tiers.

This property is also the load-bearing one for the long-term deployment
thesis (commodity hosts with variable headroom); it deserves the same gated
protection as the numerical contracts.

## Invariant worth adopting as an admission requirement

**All training state must be tile-local and streamable.** Accumulators, error
residuals, replay cursors, any future optimizer state: owned by the tile,
loaded and evicted with it, counted in the tile's byte budget. The moment any
per-weight state must stay globally resident, the low-memory training thesis
is silently dead even if every fixture passes. Making this an explicit
admission line item protects the project's central claim from well-meaning
optimizer upgrades.

## Sequencing

R1 → R3 (same four-pair harness, comparable to joint64) → R2 (multi-example
residency fixture) → R4+R6 preconditions → next text campaign with anchors +
replay only (R5). Keep joint64 as the frozen comparison baseline; preserve
failed receipts as ever. Nothing above needs a big-bang rewrite, and each step
is small enough for the existing two-review-cycle cadence.

## Measured follow-up: retained-text tuning loop (2026-10-27)

Exploratory evidence from the test-only harness (`src/causal/tuning*.rs`,
`work/tuning/`), not adopted numerics or a passed gate. At pinned knobs the
harness reproduces production study 0a191c84 arm 0 byte for byte (8/8
observation SHA-256 names), so it measures the production model.

1. **The assay failure is the update rule, not the architecture.** Threshold-one
   with fresh `Evidence` flips ~2/3 of visited trits and ~64% of E per anchor;
   loss stays worse than uniform and every prompt yields one identical output.
2. **R1 is confirmed in direction.** Latent weights + Adam (BinaryConnect +
   Adam, TWN threshold) fit the retrieval-free answer bytes to ~0 nats and
   memorize the 16 foundation rows within the 20-pass course; memoryless
   stochastic rounding needs external annealing and plateaus.
3. **The assay data invites shortcuts.** The source word pair determines the
   answer in every group, so counterfactual success is not copying evidence;
   tuned models on the fixed data learned memorization or pair recognition, and
   a key-swap probe (flip the asked key) exposes both.
4. **Retrieval is learnable by this architecture.** With values and the asked
   key re-dealt every epoch, latent Adam (rate 0.003) and greedy (eval-matched)
   routes, three of three seeds learned key-conditioned retrieval after
   ~600-1,000 passes: train loss 0.11-0.125 vs a 0.139 key-blind floor, swap
   retrieval ~12.5/16, and a later visit attending to the asked-key byte (up to
   one-hot) after the first visit reads the value slots. Paraphrase (new
   template and positions) stays near chance.

Suggested declarations: a latent-accumulator fixture (R1) with tile-local
latent state; a training set whose answers require the asked key (re-dealt
values) plus a key-swap gate; eval-matched routing during training; and a
budget sized for a late two-hop phase change (hundreds of passes), not 20.
