# Core path: next optimizations and runs (execution plan, 2026-10-28)

> Cutover update 2026-11-02: the owner explicitly chose exact original v1
> variable history behavior and deferred the one-anchor change. That decision
> supersedes the four-tick window and asymmetry-removal claims below. Stage 3
> is committed as dc785bf; profiles are recorded in 2fe8bf3. Current Stage 4
> execution and active-owner details are in stage4-codex-execution.md and
> ../handoffs/codex-cutover.md. Inspect that state before launching training.

Executor: hexweave-61 (sole writer in the tree from now on). Planner and reviewer: the
TinyStories session. Base: commit 08b6d76 plus the uncommitted edits listed in Stage 0.
Report to the reviewer after every stage; it reviews the commit read-only and sends
findings, which land as a follow-up commit.

## Goal and evidence

User goal: pretraining applied, large-parameter models on consumer hardware (this box:
i5-12400F, 31.8 GB RAM, RTX 4060 8 GB) with energy- and memory-efficient inference.
Measured so far (`docs/STATUS.md` "Current runs"): ternary and f32 learn the same per byte
(2.623 vs 2.612 bits/byte best held-out on TinyStories); an f32 model ternarized afterwards
loses 0.01; both sit ~1 bit/byte above a byte 5-gram trained on the same bytes (1.59-1.68);
throughput is 50-66 anchors/s (~0.35 GFLOP/s achieved) and is the binding constraint;
training loss runs ~0.5 bits/byte below evaluation, most likely because evaluation and
generation passed zero `history` while training feeds the previous anchors' corrections.

## Rules for every stage

- Never read `work/causal-inputs/*/evaluation/`. Never touch `datasets/**` or
  `docs/*CONTRACT*.md`. No downloads, servers or cloud. Keep the tree compiling at all times.
- Files < 300 lines, functions < 90 lines (`scripts/check_structure.py`); one-line doc
  comments, rationale in the directory AGENTS.md.
- Edits first, then ONE `powershell -NoProfile -ExecutionPolicy Bypass -File
  scripts/validate.ps1 -Offline` per stage (work/target; nothing else building). Commit
  with explicit `git add <paths>`, never `-A`. Keep harness parity
  (`core_drill_reproduces_validated_harness_numerics`) green unless a stage says otherwise.
- Runs: one owner per `out`, configs under `work/core-runs/<name>.json`, message the
  reviewer before every launch and when it ends. Rebuilds are safe once the owner has pinned
  `out/bin`.
- Any new `Knobs` field or protocol change lands in ONE identity bump (Stage 3).

## Stage 0: history-consistent evaluation (on disk, uncommitted, compile-checked)

Already in the tree (`cargo check --all-targets` clean; `cargo fmt --check` flags three hunks
to fix: core_eval.rs begin/end tuple, credit.rs `records(...)` call, evaluation.rs import order):
- `src/causal/credit.rs`: `pub(super) fn records(state, trace, corrections, shared_version,
  opportunity) -> Result<Vec<Record>>` extracted from `train_records`, which now calls it.
  Pinned by the parity fingerprint (history is part of it).
- `src/causal/replay.rs` (new): `Replay { history, clock }`; `input()` = training's
  `prune` then `mean`; `observe(state, store, prefix, trace, target)` = `credit::gather` +
  `records` + `push`, clock += 4; weights untouched. This is training's exact schedule:
  records live for two anchors.
- `core_eval::heldout_nll` (2 unscored warm-up positions per span, then Replay),
  `evaluation::teacher` (fresh Replay per row), `generation::generate` (observes the
  emitted byte, so greedy output is a low-surprise history). `mod replay;` in mod.rs.
- Drill evaluation (`tuning_measure`) is deliberately unchanged: it must stay identical to
  the harness evaluation.

Remaining:
1. Contract `src/causal/replay_contracts.rs` (cfg(test) in mod.rs), through the public seam
   on a few-anchor-trained then frozen drill state: `input()` is all zeros before any
   observe; after observing positions 1..3 of one prompt, `input()` equals that of a Replay
   that observed only 2..3 (two-anchor window) and differs from one that observed only 3;
   `immutable::identity(state, store)` is unchanged by observe. Under 60 lines.
2. `tuning::steps() -> Vec<((usize, usize), u64)>` (per-key `Latent.step`, no weight clone).
3. Docs: `src/causal/AGENTS.md` §Core path: evaluation and generation replay history from
   the actual next byte (generation: the emitted byte), ~3x cost per evaluated byte, drill
   evaluation unchanged and why. `docs/STATUS.md`: all dev numbers logged before this
   change were zero-history.
4. Measure the gap on the saved best checkpoints, no retraining: for X in
   {float: `tiny-float-s7`, best `a000001520000`; ternary: `tiny-ternary-s7`, best
   `a000001160000`}: `mkdir work/core-runs/rescore-X/checkpoints`, copy the best checkpoint
   directory in, write `work/core-runs/rescore-X.json` = that run's config with `out` =
   `work/core-runs/rescore-X`, `resume: true`, `anchors` = the checkpoint's anchors (so
   `drive` evaluates once and exits "completed"); launch through `core_owner`. Identity is
   unchanged (no Knobs change), so the checkpoint loads. Report `dev_bits_per_byte` and
   `alternate_bits_per_byte` against the logged zero-history 2.612 / 2.623, and the four
   story continuations. If the history number is not clearly better, say so; that result
   decides whether history is worth its inference cost.
5. Sweep, commit, report.

## Stage 1: telemetry and profile (hexweave-61's in-flight work)

`telemetry.rs`, `START_TILE`, credit.rs timing wrappers, the core_train hook, §Telemetry,
then the profiling smoke (`work/core-runs/profile-telemetry.json`, 3,000 anchors).
Deliverable: per-anchor wall time by phase (forward, gather/backward, apply including Adam
and the E update, evaluation, checkpoint) and per-tile visit counts, plus a one-paragraph
reading of where the 50 anchors/s goes. Decision input for Stage 3: the fraction of step
time in apply/commit.

## Stage 2: single-anchor throughput

AVX2 block products for `src/ternary/head_rows.rs` (`forward_rows`, `transpose_rows`) in
the same reduction order as the scalar code, so outputs stay bit-identical (the existing
kernel tests require exact scalar/AVX2 equivalence; add a head-row parity test on random
inputs across row offsets and counts, including the `< 8` fallback). Then a second profile
with the same config; report anchors/s before and after. No identity change. If the apply
phase dominates in Stage 1, also look at `TernaryMatrix::apply_latent` (one pass over
weights and moments per matrix per anchor) but do not change its numerics.

## Stage 3 (gated): batched segment training, protocol `causal-latent-v2`

Gate: Stage 1 shows apply/commit at or above ~30% of step time, or the user confirms the
pivot (the reviewer relays). Design:
- A step processes `batch` consecutive positions under fixed weights: per position, forward
  (greedy route as now), `gather`, and history via `Replay` within the step, so Replay becomes
  the single history policy for training and evaluation. Gradients accumulate per visited
  tile matrix and for E; one latent-Adam `apply` per tile per step on the mean gradient
  (tile-major commit). History records keep the per-position clock.
- Threads over positions with `std::thread::scope` (no new dependencies): forward and gather
  only read tiles, so give `ResidentStore` a `&Tile` accessor for the batched path;
  commits stay single-threaded or tile-disjoint.
- Config: `batch` (default 1) and a learning-rate `decay` schedule; both enter the identity.
  `PROTOCOL` becomes `causal-latent-v2`. Checkpoint format otherwise unchanged. Raw-byte
  memory stays (contextual memory is a later protocol).
- Contracts: (a) `batch: 1` reproduces v1 bit for bit over at least 12 anchors (fingerprint
  including history), the key oracle; (b) `batch: 4` over 12 anchors equals a single-threaded
  reference that runs the four forwards sequentially under frozen weights, then one commit,
  which pins the threading; (c) resume is bit exact at `batch > 1`; (d) the identity table
  gains the two fields and the new protocol string, and the golden digest is re-pinned once.
- Measure anchors/s at batch 1, 8 and 32 on the profile config; expect commit amortization
  plus up to ~5x from six cores.

## Addenda (2026-10-28, after the Stage 1 profile and the Stage 0 re-scores)

Profile (3,000 anchors, 48 anchors/s, ~21 ms per anchor): tile_apply (latent Adam) 27%,
tile_backward 21%, gather 19%, records (predictor features + history) 13%, forward 10%,
prefix 7%. Route collapse: greedy routing locks onto tiles [3, 1, 4] for 986/1,000 anchors
by 2k anchors; tile 5 gets zero updates, tiles 0/2/6 about 1%. Effective capacity ~3/7.

Decisions:
- Gate passed; Stage 3 runs before Stage 2. Stage 2 only afterwards, and only if a finer
  profile (attention / ffn / pulse split of tile_backward) shows head-row blocks matter.
- Exploration knob joins the Stage 3 identity bump: off by default; draws are pure
  `tuning::hash(seed, anchor, depth)` (never the CALLS counter); explored hops excluded from
  the REINFORCE update; eval route stays greedy; drill contract: every tile receives updates
  with epsilon > 0, numerics unchanged with epsilon = 0. STATUS: tile-count scaling is moot
  until routing uses the tiles.
- Threading hazards: KNOBS, LATENTS and CALLS are thread_local. Workers must receive Knobs
  explicitly (or run a knobs-only install); reject `float_forward` with `batch > 1` in
  Config::validate until latents leave thread-locals; the batch:4 contract compares the full
  fingerprint so a worker on pinned knobs fails it. Raise eval_every/checkpoint_every in
  Stage 4 configs so evaluation stays under ~5% of wall time.

Re-scores (history replay at evaluation, frozen best checkpoints): float 2.612 -> 2.617,
ternary 2.623 -> 2.623 bits/byte. The trained models do not use the history input. So:
- Keep Replay as the evaluation default (train/eval consistency; it becomes the training
  history policy in v2 too), but treat the mechanism as a removal candidate: a history input
  the model ignores costs 13% of each training step (records) and would force a backward
  pass per byte at inference.
- History ablation (part of Stage 3, before the v2 design is frozen): `Knobs` field
  `history: bool` (default true; off = zero history input and no records/predictor work);
  two bounded corpus runs of 100k anchors each (on/off, same seed) compared on held-out
  bits/byte at 100k. If off matches within 0.02, v2 drops history and records entirely
  (no Replay in the step, no backward pass at inference). Enters the same identity bump.
- The remaining gap (train 2.19-2.33 vs held-out 2.62 bits/byte) has two candidate causes:
  (1) dynamic evaluation: training loss is scored on weights that just adapted to the
  current story (Krause et al. 2018, arXiv:1709.07432, 0.1-0.3 bits/char from exactly this);
  (2) the 16 held-out spans are harder than average. Diagnostic (Stage 0, cheap, permanent):
  add `train_frozen_bits_per_byte` to the corpus evaluation, 4 fixed 128-byte spans of
  train.txt read at open (same window_before logic), then re-run the two re-scores. ~2.6
  means (1), ~2.2 means (2). If (1): under v2 (fixed weights within a step) the logged
  training loss will rise toward the held-out number without any quality change; compare
  runs on frozen held-out numbers only, never on train loss across protocols.

## Follow-up to 8641c71 (independent review: REQUEST CHANGES; land before Stage 3)

Code verified numerics-neutral (timing wrappers, START_TILE, telemetry) and Replay mechanics
correct. Required fixes: (H1) Replay's two-anchor window is an upper bound on training's
1-2 anchors (version pruning after trit flips); log `history_len` in telemetry and fix the
docs; conclusion unchanged. (M1) replace the fixed-offset train spans (never trained on in
epoch 0) with a byte-matched probe: frozen score of the next 512 train bytes ahead of the
cursor plus the prequential loss over the same 512 anchors in the next interval; their
difference measures online adaptation directly. (M2) STATUS alternate-mode baselines must be
the best-state values (2.6536 / 2.9733), so replay moved them +0.0060 / -0.0015. (M3) remove
the stale "zero history" sentences in AGENTS.md. (M4) warm the generation Replay on the last
SPAN_WARMUP prompt bytes; document that science evaluation starts history fresh per row.
(M5) collapse was transient: over the full runs every tile stepped on >= 9.7% of anchors
(tile 5 18.6%); rewrite STATUS as uneven usage and drop "tile-count scaling is moot".
(LOW) `eval_version` stamp in best/selection.json; `ensure!(span.len() >= scored)`; golden
message mentions training numerics and platform libm; one-line doc comments; split
tuning.rs (exactly 300 lines) before adding the Stage 3 knobs.

## Follow-up to 70a4c51 (independent review: REQUEST CHANGES, small; land before Stage 3)

Verified: ahead alignment, slicing, accumulator, generation warm-up, EVAL_VERSION, no numerics
change. Required: (HIGH) the prequential probe is not null after a resume and can pair with
the wrong frozen score; disarm on resume and log `train_ahead_window_start`. (MEDIUM) reword
AGENTS.md: the online-adaptation claim is the hypothesis the ahead pair tests, not yet
measured on a state with a gap; the prequential value is the training loss, not score_spans.
(LOW) history_len counts records: steady-state training keeps exactly ONE anchor (profile
intervals 2-3 equal the mean route length); the older anchor is pruned on the second version
bump since its forward; document wrap context, tiny-corpus probe behavior, the EVAL_VERSION
rotation side effect and the science warm-up asymmetry; alignment and null-rule tests;
STATUS nits; one-line doc comments; SPAN_WARMUP to replay.rs; finalize_last argument struct;
split core_train.rs (299 lines) before Stage 3.

## Stage 3 step 1 landed (def511f: `Knobs.history`, tuning.rs split; approved with nits)

Carry into step 2: (1) with history off, `Replay::observe` must return
`state.shared.readout(&trace.output)?.loss(target)` and skip `credit::gather` (identical value,
no backward pass per evaluated byte), pinned bit for bit in history_contracts; (2) tighten the
history-off invariant: if `repeated_visits` means per-anchor revisits under anchor-only, assert
`anchor_tile_commits + repeated_visits == visits`, else `anchor_tile_commits <= visits`.
Ablation launched 2026-10-30: `work/core-runs/ablation-history-on` and `-off` (100k anchors,
seed 7, window 512, ternary); report per arm: held-out at 100k, the 80k/100k train_ahead
pairs, history_len, latent_steps, anchors/s (off expected ~13% faster).

## History ablation result and pre-registered tie-breaker (2026-10-30)

Result: the pre-registered rule ("off within 0.02 bits/byte of on at 100k held-out") FAILED as
written: off 3.4422 vs on 3.3470 (+0.095). But the instrument cannot resolve 0.02: the
2,048-target held-out score swings 0.03-0.15 between consecutive evaluations of one arm, and
the arms cross (off better by 0.145 at 40k and 0.110 at 60k, worse at 80k and 100k). The
better-powered paired measure (identical byte stream, prequential loss over 20k bytes per
interval, each byte unseen when scored) favors off in all four intervals by 0.025-0.13.
Off is 17% faster and needs no backward pass at inference. The 0.02 threshold was set below
the instrument's resolution; that was the planner's error, not a reason to move the goalposts.

Tie-breaker, fixed BEFORE it runs: re-score the frozen 80k and 100k checkpoints of both arms
on a held-out sample 8x larger (128 spans x 128 = 16,384 targets per checkpoint; same
`score_spans`, same warm-up, each arm under its own history mode). Decision: drop history in
v2 if mean over the two checkpoints of (off - on) <= +0.02 bits/byte; otherwise keep history
in v2 as an explicit one-anchor window and revisit after batching exists. Implementation:
an `eval_spans` config field (default 16, not part of the run identity; `heldout_targets`
already logs the sample size) and an EVAL_VERSION bump or `eval_spans` recorded in
`selection.json` so best selection never compares different sample sizes. Step 2 waits for
this result because the v2 step design differs with and without in-step history.

Also recorded: history_len 3.06-3.18 equals route length (one-anchor steady state confirmed);
train_ahead pairs at 60k/80k show no consistent adaptation effect at this scale (512-byte
windows are noisy; the long v2 run decides); minor tiles get 9-22% of the start tile's
updates in both arms (uneven, not collapsed).

Tie-breaker result (2026-10-30, 16,384 held-out targets per checkpoint): on 3.1689 / 3.1902,
off 3.2288 / 3.2177 at 80k / 100k; off - on = +0.0600 / +0.0274, mean +0.0437 > +0.02.
Outcome per the rule: v2 KEEPS history as an explicit one-anchor window (training and Replay
share the window parameter). Oracle: the 16-span re-score reproduced the ablation's 100k value
bit for bit. Noted without reinterpretation: the large held-out sample and the paired
prequential loss point in opposite directions; the on-arm evaluation replayed up to two
anchors while on-arm training ran on one, an asymmetry v2 removes.

Consequences for Stage 3:
- Keep the `history` knob so the ablation can be rerun under v2 without the asymmetry.
  Pre-registered v2 re-ablation (Stage 4): history on vs off under v2 at batch 1, same seed,
  100k anchors, 128-span held-out at 80k and 100k, same rule and threshold; the outcome
  decides whether inference keeps paying a backward pass per byte.
- The predictor is dead weight even with history on: the forward input is
  `History::mean`, which reads only each record's `correction`; `Record.features` and
  `Predictor::train_joint` feed nothing else under anchor-only. v2 drops the predictor fit and
  the `features` field (records carry correction, versions, clock, tile) and the predictor
  counters; the step-2 author verifies that no forward path reads `features` before relying
  on this. Parity contract (a) then compares tiles, E, latents, baseline, losses and the
  history correction vectors; predictor bytes and features are excluded by design, and the
  exclusion is stated in the contract's doc comment.

## Stage 3 knobs (agreed 2026-10-28; all in `Knobs`, one identity bump, PROTOCOL `causal-latent-v2`)

1. `history: bool` (default true). Off: zero history input; `records`, the history push AND the
   predictor update are skipped (the predictor is unused under anchor-only); `Replay::input`
   returns zeros. `State::validate_boundary`'s predictor-update invariant
   (state_counters.rs: predictor_anchor_updates == anchors) is gated on the knob: off requires
   predictor updates == 0 and an empty history. Knobs are installed before load on resume, so
   restore validates under the same rule. Document in AGENTS.md §State encoding. Goes first,
   for the ablation (2 x 100k anchors, same seed, under the v1 step as it really runs, i.e.
   on = the measured steady-state one-anchor window; off within 0.02 bits/byte held-out =>
   v2 drops history, records and the predictor). If history survives, v2 defines the window
   explicitly instead of inheriting it from version pruning: one anchor (age <= 4 ticks),
   matching v1's measured steady state, and Replay takes the same window parameter so
   evaluation matches training by construction rather than as an upper bound.
2. `batch: u32` (default 1). Positions per step under fixed weights: per position forward
   (greedy) and gather; gradients accumulate per visited (tile, matrix) and for E; one
   latent-Adam apply per tile per step on the MEAN gradient (tile-major). Baseline rule: every
   position uses the step-start baseline; after the commit, apply the B baseline updates
   sequentially in position order. `batch: 1` is bit-identical to v1 (contract a).
   `float_forward` with `batch > 1` is rejected in `Config::validate` (latents are
   thread-local). Workers receive `Knobs` explicitly, never via the thread-local.
3. `decay: Constant | Cosine { anchors, floor }` (default Constant): step size at anchor n is
   `rate * (floor + (1 - floor) * 0.5 * (1 + cos(pi * min(n, anchors) / anchors)))`,
   evaluated at the step's first anchor index (checkpointed, so resume is exact).
4. `explore: f64` in [0, 1) (default 0). At route depth >= 1 only, with probability
   `explore`, the greedy action is replaced by a hop (never halt) to the legal neighbor with
   the fewest cumulative `Latent.step`s (any matrix of the tile; no latent = 0; ties to the
   lowest id). The draw is `tuning::hash(seed, anchor, depth) < explore`, a pure function, so
   threads and resume cannot change it. Epsilon-greedy policy gradient: forced frames
   contribute their taken action's log-prob gradient (excluding them would leave router
   logits moving only on greedy frames, so an under-used tile could never become preferred).
   Fallback to exclusion only if telemetry shows routing destabilizing. Starts are not
   explored (every tile is one hop from START_TILE at radius 1). Eval and generation stay
   greedy. Interval telemetry is not persisted and must not drive exploration.

Stage 4 under v2: judge per hour, not per byte; include a rate scan (0.003 and ~0.017 at
B=32) because Adam takes B-fold fewer steps per byte; compare only frozen held-out numbers
across protocols (train loss rises under fixed-weight steps without any quality change).

## Stage 4: runs

After Stage 2: profile only. After Stage 3: `tiny-v2-b32-ternary-s7` on TinyStories for 8 h,
compared against the v1 ternary curve per byte AND per hour, and against the n-gram table
(`work/core-runs/ngram-curve.json`; `scripts/ngram_baseline.py` for other byte counts).
Add the f32 arm only if time allows. Report best and final held-out bits per byte, the
alternate-mode numbers, throughput and the story continuations; update `docs/STATUS.md`.
