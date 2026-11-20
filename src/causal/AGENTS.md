# Causal model: current core and numerical references

## Connected training

The in-session owner request on 2026-11-03 authorizes connecting the shared-memory
predictor architecture, implementing and independently reviewing it, and launching
and monitoring a bounded fresh run. All weights and adaptive state start from
scratch; no previous checkpoint is an initialization source. The connected
profile (now `causal-connected-v5`) is selected by the `connected` config object through
the existing `train` command and capped core owner. Its complete learning,
causality, memory, optimizer and checkpoint contract is [CONNECTED.md](CONNECTED.md).
Do not substitute the retained anchor-only v2 mechanism for that experiment.
Verification and runtime outcomes belong in `docs/STATUS.md` and run receipts;
the presence of source code is not validation evidence.

Owner direction on November 4 additionally authorizes fresh connected
FP32 forward training and immutable rounded-weight comparisons; see
[precision design](CONNECTED.md#full-precision-learning-and-quantization-comparisons).
The connected ternary protocol and its pinned checkpoints remain distinct from
the FP32 protocol (now `causal-connected-fp32-v5`).

Owner direction on November 6 removes the earlier gate option and makes the
predictor learnable as one behavior: per-tile normalized targets, least-squares
trust and trust-scaled local updates under protocol v2; see
[calibrated predictor](CONNECTED.md#calibrated-predictor-and-trust-scaled-local-updates-v2).
v1 checkpoints resume only under their pinned binaries.

Connected protocol v3 (2026-11-07; unrelated to the archived v3 owner chain)
follows the diagnosis of the v2 FP32 study in `docs/STATUS.md`. It adds
decoupled weight decay with a hard master RMS cap per updated region, a saturating
activation curve, a learned readout bias, one local Adam step per tile per
anchor and a narrower predictor skip with a clipped target; see
[the v3 section](CONNECTED.md#scale-control-saturating-curve-readout-bias-and-per-anchor-local-steps-v3).
v1 and v2 checkpoints resume only under their pinned binaries.

Connected protocol v4 (2026-11-07) follows the v3 pilot recorded in
`docs/STATUS.md`. It RMS-normalizes every tile's output after the residual sum
(the adjoint passes through the normalization) and, on a fresh start, initializes
the readout bias to the training corpus' byte log-frequencies; the v4 configs also
set `weight_decay` to 0.001. See
[the v4 section](CONNECTED.md#residual-normalization-and-readout-prior-v4).
v1, v2 and v3 checkpoints resume only under their pinned binaries.

Connected protocol v5 (2026-11-07) follows the v4 full run. The correction
predictor also reads the observed target byte's RMS-normalized embedding column
(conditional synthetic gradients), only after that anchor's readout, and each
tile update's decoupled decay uses the multiplier its step actually received (the
gain after the uniform norm cap); E and the readout bias decay at the plain rate.
v5.1 adds `connected.local_rate` (default 3e-4, in (0, 0.1], run identity), the
rate of those predicted-credit tile steps and their decay; E, bias and routes keep
`rate`, the predictor `predictor_rate`. Admission refuses
`weight_decay * max(rate, local_rate) * gain_max >= 1`. v5.2 adds
`connected.local_decay` (default `constant`, run identity), a schedule on that local
rate by anchor index; `inverse_sqrt` scales it by `1 / sqrt(1 + anchor / anchors)`,
and a schedule that can reach zero (cosine floor 0) is refused. v5.4 adds optional
`connected.credit_mode` (`lanes` default, `anchor_exact`, `anchor_predicted`,
`hybrid`) and `connected.slice_mode` (`full` default, `windowed`, `hybrid`),
omitted from serialized identity when default so existing v5.2/v5.3 configs keep
their run identity. Optional `connected.restart` (absent by default, run identity
when present) floors each tile's local-rate factor and restarts its schedule after
accumulated relative change; see
[warm restarts](CONNECTED.md#warm-restarts-and-per-tile-rate-floor). Its per-tile
training state (`TileRestart`, `TileSignal`: never in tile bundles, never read by
completion or paging), the `forget`/`signal`/`stagger`/`clamp`/`replay`/`plasticity`
options and the experimental Lambert law are in
[per-tile training state](CONNECTED.md#per-tile-training-state),
[restart options](CONNECTED.md#restart-options) and
[Lambert plasticity](CONNECTED.md#lambert-plasticity-experimental). See
[the v5 section](CONNECTED.md#label-conditioned-credit-predictor-and-gain-scaled-decay-v5).
v1-v4 checkpoints and the pre-v5.2 v5 pilots resume only under their pinned binaries.
`connected.anchor_batch` (default 1, identity above 1, `anchor_exact` only) freezes
tiles over aligned waves of that many anchors and steps each touched tile once at
the wave end; see [anchor waves](CONNECTED.md#anchor-waves).
Optional `connected.balance` (absent by default, identity when present; experimental) scales
tile rates, biases training routes and reshapes memory eviction by a per-tile visit-share
estimate kept in training state; see [visit balance](CONNECTED.md#visit-balance).
`connected.memory_lag` (default 1, identity above it) widens the shared-revision lag a memory
read admits, and `connected.visit_norm` (default true, identity when false) gates the v4
post-visit normalization; see
[memory reach](CONNECTED.md#memory-reach-and-visit-normalization-switches).
Optional `connected.shared` (absent by default, identity when present; experimental,
`anchor_exact` only) adds up to four pinned tile-shaped sections read at every route entry and
written once per shared wave (write credit `mix` of the visited tiles' corrections or `entry`,
the exact read adjoint; `teach` pulls at a tile's own adjoint scale); see
[shared sections](CONNECTED.md#shared-sections).

Connected checkpoint completion uses `connected_completion` and frozen evaluation
replay. It snapshots the exact named checkpoint into owned `work/inference/`
scratch and loads only that copy, so it may run beside a live owner; it preserves
the run's STOP and saved training state.
It opens no corpus. Source-file and actual tile/Adam fingerprints, bounded
residency, a cooperative inference deadline and RAII cleanup are part of the
contract; see [read-only completion](CONNECTED.md#read-only-checkpoint-completion).

Owner approval on 2026-11-03 authorizes the next full fresh study: bounded LRU
tile/Adam caching, expanded frozen learning/retention measurements, independent
review and a run of up to 24 hours or one public corpus pass. Count idle cached
bundles together with outstanding leases against the configured decoded resident
cap (`connected.resident_tiles`, default `workers`; CONNECTED.md §Write-behind paging).
Dirty entries must flush before checkpoint publication; eviction failures poison
the store. Keep update order and numerical identity unchanged. The full study
uses the existing core owner as a persistent supervisor; the plan is
`.omc/plans/connected-full-run-2026-11-03.md` at repository root.

Start with [the architecture study guide](STUDY.md) for the current call path,
equations, measured bottlenecks, memory costs and questions the evidence leaves
open. Keep detailed teaching notes there and source annotations short.

Archived v3 owner-chain, study, comparison and diagnostic sections:
`../../archive/src/causal/AGENTS-history.md`.

## Core v2

Configurations without `connected` select `causal-latent-v2`. The v1 description
below records the retained numerical reference; `execution` and `Knobs::core`
remain that reference for harness and span-scoring contracts. `Config::knobs`
selects v2: `fit_predictor: false`, `history_ticks: 8`, plus `batch`, `decay`
and `explore`. All enter one protocol/identity change; v1 checkpoints require
their original pinned binaries and cannot resume as v2.

`core_batch` builds B consecutive prefixes, routes and conditional corrections
under fixed weights. History makes this phase sequential: position n+1 needs
position n's returned correction. It retains the original age-at-most-eight-ticks and version-lag-at-most-one
policy used by Replay. Under fixed weights this can retain two anchors. The later parameter-gradient backward passes are independent;
`core_gradients` uses up to four scoped workers over positions, one tile at a
time. Workers receive and install the complete knobs explicitly. Position
gradients are produced in groups of at most four, then added to one accumulator
in original position order; at most five full tile gradients coexist. This
bounds gradient retention independent of batch size, without bounding tile,
optimizer, activation or checkpoint residency. Gradients reduce with repeated
frames in forward order; division is
by the full B, including positions that never visited a tile. Tiles commit in
ID order, then tied E. Batch one avoids extra reduction arithmetic and worker
threads, preserving its thread-local float latents. Float mode with B>1 is
rejected. Batch is bounded to 1..=32; no GPU or larger-model execution is added.
Workers are scoped per group, so larger batches create more threads per tile;
measure the throughput tradeoff before using this change in a training study.
Core routing, correction gathering and tile commits use `TileStore` access;
the production implementation remains `ResidentStore`, and latent optimizer
state and checkpoint buffers are not paged by this interface change.
Step-start exploration counts also come from the store; the resident store
still derives them from the thread-local optimizer state.
Checkpoint save streams and hashes tile payloads one tile at a time and latent
weights/moments by borrowed map entry, preserving the existing byte format.
Restore hashes while decoding individual tiles and latent vectors, and installs
optimizer state only after both payloads verify. It still reconstructs all
resident state; this does not establish bounded residency or a physical-memory
guarantee.

`core_paged_experiment` is test-only: a single decoded tile and its latent Adam
state are loaded from one hash-checked disk payload, while shared E remains in
the thread-local optimizer. Per-tile maximum step metadata stays available for
exploration after eviction. The fixture compares its training trajectory with
the resident store and asserts one live tile during steps; its final parity
snapshot intentionally reconstructs the whole fixture. It is not a production
store, physical RSS proof, checkpoint protocol, crash recovery mechanism, or
authorization for larger geometry. Read/write failure and interruption
handling require independent review before any promotion.

Every position uses the step-start baseline; after commit, baseline updates
and loss accounting run in position order. Cosine or inverse square-root
(`inverse_sqrt`) decay is evaluated at the first anchor of the step, with constant
decay the default. Exploration is a
pure seed/anchor/depth hash: at depths 1 and above it selects the legal
neighbor with the fewest cumulative optimizer steps (lowest-ID tie break),
never halt. Counts are frozen at step start and come from checkpointed Adam
state. Forced actions retain their route log-probability gradient. Evaluation
and generation continue using greedy `execution::forward_path` without this
exploration override.

The predictor is unused by the core forward path: `History::mean` reads only
corrections. V2 computes no predictor features or fits, emits no feature vectors
or predictor counters in checkpoints, and omits `predictor.bin`. The common
State still holds a dormant seeded predictor for compatibility with legacy
numerical fixtures; restoring v2 reconstructs that unchanged object. This is
not a claim that its small in-memory allocation has been removed. Legacy local
execution still uses predictor features and validates their full shape.

`optimizer_steps` counts completed batches; `anchor_tile_commits` counts real
tile commits. Per-anchor `repeated_visits` remains unchanged. At B=1, commits
plus revisits equal visits; at B>1, commits are bounded by completed batches
and unique per-anchor visits. Checkpoint versions cannot exceed optimizer
steps. An entered batch that fails becomes Poisoned and cannot checkpoint.
STOP and deadline checks occur between complete batches. Requested terminal
anchor counts must be divisible by B, so an extended run never begins after
an implicitly shortened optimizer step. Eval/checkpoint schedules fire on
crossed intervals without splitting batches; these scheduling fields remain
outside numerical identity. The train-ahead probe receives every committed
position's loss in order.

Owner decision 2026-11-02: preserve exact original v1 history behavior and
DEFER the explicit one-anchor change. Early latent updates do not change trits,
so v1 initially retains two anchors; the earlier handoff's one-anchor policy
and exact-from-initialization parity requirement were incompatible. Evaluation
version 2 and the legacy span-scoring golden remain unchanged. V2 quality still
requires new frozen evaluations because batched optimization changes training. Phase telemetry measures elapsed
wall time around the whole gradient scope, not summed worker CPU time.

The batch contracts compare B=1 to v1 over twelve anchors, excluding predictor
bytes/features while retaining correction history, E, tiles, latents, baseline
and loss. B=4 compares sequential and threaded execution with non-default
forward knobs, history on/off, decay and exploration, and checkpoint reload.
No throughput or quality improvement is implied by these software contracts.

## Core path: causal-latent-v1 (retained reference)

`hexweave train --config <json>` (`core_train`) is the production training path,
run under `scripts/core_owner.py`. It keeps the v3 forward model unchanged and
changes how it learns, per the 2026-10-27 tuning loop (`§tuning` below):
latent-weight Adam on every tile matrix and tied E (`tuning::Knobs::core`,
`TernaryMatrix::apply_latent`; BinaryConnect arXiv:1511.00363, Adam
arXiv:1412.6980, TWN threshold arXiv:1605.04711), greedy (eval-matched) routes,
and an anchor-only schedule (`State.anchor_only`, StateRecord schema 3, never
combined with the assay ledger) that reuses the assay clock arithmetic
(numerical clock = 4 x physical, history on multiples of 4) without the ledger.

The numerics profile is a thread-local set only by `tuning::install`, which only
`core_train::run` calls in production (`drive` re-checks it equals
`Knobs::core(rate, seed)`); every other production path (`plan`, `smoke`) stays
pinned, and tests set non-pinned knobs only under `scoped()` or `install`
(`core_contracts`, `core_resume_contracts`, the gradient oracles, the harness). All seven tiles and their latent state stay resident
(`resident::ResidentStore`, ~25 MB of latent f32 + Adam moments for 2.05M
coefficients); there is no per-update persistence. History corrections still
feed the forward pass; the predictor is trained and checkpointed (counter
invariants require it) but nothing reads it under anchor-only.

Checkpoints (`core_checkpoint`, payload codecs in `core_codec`):
`checkpoints/a<12-digit anchors>/` holds state, E, predictor, tiles, latents
and a SHA-256 manifest. Files are fsynced in `<name>.partial`, which is then
renamed; renames, reads and removals retry Windows sharing errors (os 5/32,
e.g. a scanner holding a just-closed file) with backoff, and pruning to the
newest three is never fatal. Resume first identity-checks every readable
manifest: a checkpoint of another run identity (protocol, geometry, data,
seed, rate and the serde-serialized `Knobs::core` profile; a golden digest in
`core_contracts` pins the formula) aborts with nothing touched. Checkpoints
written by the `8e7a0eb` binary use the earlier formula (geometry as the
literal `"candidate"`, no knob fields), so only their pinned `out/bin` copy, which the owner keeps on resume,
can resume them; that binary also has the earlier resume logic (it
quarantines on any load error and starts fresh once none verify). It then walks newest first: one
that fails integrity (hash, decode, missing file, name/anchor mismatch) is
renamed `.rejected-<pid>` and the next is tried, while any other I/O error
aborts with that checkpoint left in place; restore verifies every hash, the
directory name, then reinstalls latents and the draw counter, so resume is bit
exact. `resume: true` starts fresh only when no checkpoint (complete or
rejected) exists; if every checkpoint fails verification it refuses to restart
from anchor 0. A fresh run refuses existing checkpoints. Changing `anchors` or
`wall_seconds` extends a run. `out` must be `work/core-runs/<name>`.

Data: `corpus` (`core_corpus`) is next-byte pretraining over a pinned plain-text
corpus (`scripts/prepare_corpus.py`: stories END-terminated with 0x03, every
20th held out, SHA-256 manifest). Target n is train byte n mod length, predicted
from the last `window` (32-1023) bytes of `[START 0x02] + train`; the train file
is read by window from disk, so memory stays flat however large it grows. Its
evaluation is teacher-forced held-out bits per byte over `eval_spans` fixed spans
(a `Config` field, default 16, not part of the run identity) of 128 bytes in both
forward modes on the same weights (`dev_*` for the run's mode, `alternate_*` for
the other), plus a greedy and a nucleus-sampled continuation of four story
openings, each after a full window of held-out text (§Generation).
`train_ahead_frozen_*` frozen-scores the next `AHEAD_ANCHORS` (512) train bytes
ahead of the cursor the same way (`Corpus::ahead_span`, wrapping once at the
file end for byte selection; the prefix context before it clips at the file
start like `window_before`, never wrapping backward) -- null, cheaply, on a
corpus shorter than `AHEAD_ANCHORS` (`Corpus::is_short`). The wrapped bytes see
the previous epoch's tail as their context (the span is one flat byte array),
while training itself restarts context at `[START]` at the epoch boundary, so
the probe is an approximation near a wrap, not a reproduction of training's
own epoch-boundary context. `core_log::feed_ahead` separately accumulates
`train_ahead_prequential_bits_per_byte` in `core_ahead::Ahead`: the real
prequential anchor loss (`execution::Step::Committed`'s loss) over exactly the
first 512 anchors trained after each evaluation, logged on the next evaluation
line alongside `train_ahead_window_start` (the eval anchor its frozen probe
was logged at) -- pair the prequential value with the frozen probe logged at
that anchor; a null `train_ahead_window_start` means it is unpaired. Both are
null short of 512 anchors; `Ahead` is also disarmed for the one window
spanning a resume (its cursor's frozen probe was never logged, since resuming
skips the initial record line -- see §Core path below), and on a short corpus
every window is null. The gap between the two is the online-adaptation effect
on the same bytes; this replaces the old fixed-offset `train_frozen_*` probe,
which mostly sampled train-file text the run had not yet reached. The held-out
and train-ahead-frozen probes score byte-for-byte comparably to each other via
`score_spans`, which steps only the warm-up and scored positions of each slice
(the bytes before them are prefix context); `train_ahead_prequential` is not
scored this way at all -- it is the raw training loss on those bytes, logged
after the fact -- and even the two `score_spans` probes are only
approximately comparable to training's own loss, since `Replay`'s history
window differs from training's own real window (§Core path below) and the
wrap-context caveat above. `replay_contracts` pins its f64 result on a fixed
probe: if that moves, evaluation numerics changed, and logged numbers from
earlier commits must be re-scored before they are compared.

Forward mode: `float_forward: true` (a `Knobs` field, so part of the identity;
absent means ternary) makes every weight product read the matrix's f32 latent
instead of its trits (`weights.rs`: tile project/transpose, head row blocks,
activation-curve offsets, tied-E column/project/transpose); latent Adam then
trains a float model of the same architecture with exact gradients, and the
trits are its post-hoc ternarization. Off, every call reaches the unchanged
ternary kernels; the harness parity test and the gradient oracles pin both: the
parameter oracle, the input adjoint and the tied-E check (`float_contracts`)
each rerun on off-lattice latents and fail if float mode does not engage, and
float resume is bit exact. Reading the comparison: float latents start on the
ternary lattice and are clipped to [-1, 1] (not a conventional float init);
versions and `immutable::identity` still track trits, not the float function
(history staleness stays bounded by age); evaluation now replays training's
own up-to-two-anchor history window (`replay.rs`) instead of zero history,
which moves held-out NLL by at most 0.005 nats/byte. The hypothesis the
`train_ahead_*` pair tests (`core_eval::corpus`, `core_ahead::Ahead`) is that
most of the remaining train/held-out gap is online adaptation (prequential
scoring on weights that just updated on the same bytes), not history -- this
is untested on a run with an actual gap: the only logged pair so far
(`work/core-runs/profile-ahead`, a 3,000-anchor smoke run) shows the frozen
and prequential values nearly equal (4.4888 vs 4.4886 bits/byte), so the
hypothesis wants reading from a late interval of a long run, not this smoke
data. Corpus runs ignore `eval_cases`; best-checkpoint selection applies
to them as to science runs. `prepare_corpus.py` writes LF manifests and refuses
an existing corpus directory; the first TinyStories manifest (pinned by the
2026-10-28 runs) predates that and has CRLF line endings.

The `history` `Knobs` field (Stage 3 ablation, default true; serialized only when
false, the mirror image of `float_forward`'s serialize-only-when-true, so an
absent or `true` value keeps the existing golden identity and every old
checkpoint resumable) turns off the history input end to end: the anchor branch
of `execution::complete` passes a zero vector of `state.history.mean`'s width
instead of calling it; `credit::anchor` skips `train_records` entirely, so no
features are built, the predictor never fits and nothing is pushed to
`History` (the predictor is unused under anchor-only regardless); and
`replay::Replay::input` does the same for evaluation and generation. Off,
`Replay::observe` also skips `credit::gather`'s backward pass entirely (no
corrections, no records) and returns `Shared::readout(output)?.loss(target)`
directly -- the same value `gather` computes via `Shared::task`, so the loss
is unchanged but costs one fewer backward pass per evaluated byte. A `false`
run gets a different run identity (`core_contracts`'s
identity table pins both directions) and a different trained loss sequence
from the same seed, since the forward pass sees different input; see §State
encoding for the counter invariant this gates.

Data: `science` trains on a pinned QA Dataset's train split and evaluates the
development split with `evaluation::evaluate` (teacher-forced NLL, coverage,
greedy JSON answers, exact answer/citation) plus four train-row samples. Every
answerable row cites exactly its prompt's single `SOURCE` value, but with six
train labels the 72-row run memorized them (dev outputs cited train source T06,
while dev prompts name D01-D06). `redeal_citations: true` (`core_redeal`)
replaces that value with a fresh same-shape one per (row, epoch) in the prompt
and its citation, keeping every length, kind and cursor of the inner dataset,
so citing can only be learned as copying; dev is never re-dealt. Science runs
keep their lowest-dev-NLL state under `out/best/checkpoints` (up to three; the
newest is the selected one) with an identity-stamped `selection.json` published
atomically (`core_best`): another run's selection is ignored, a fresh run (and
the owner) refuses a non-empty `out/best`, a better repeat at an anchor that
already has a best checkpoint only updates the selection, and a failed best
save is logged, never fatal. `best_so_far` ignores a selection from an earlier
`EVAL_VERSION` (bumped when evaluation numerics change, e.g. history replay) or
whose stamped `eval_spans` differs from the run's (a selection predating the
field counts as 16), so held-out samples of different sizes are never compared:
resuming a run from before `70a4c51` therefore starts a fresh best selection
from infinity, and since only the newest three best checkpoints are kept, the
old run's best state can rotate out after three more saves at the new eval
version -- copy it aside first if it must be kept. The newest-three rotation
had pruned the 72-row run's best state (80k anchors) once the 140k checkpoint
landed.
`drill` (`core_drill`, v2) replays the assay schedule in 1,600-anchor cycles
(16 foundation passes, then 4 retention-stress passes) and re-deals values and
the asked key with `tuning_sources::redeal` under the harness's per-cycle salt,
so its stream is exactly the harness `Shuffled` stream that learned retrieval;
it is evaluated with the assay rows, key-swap and attention probes. A v1 drill
that interleaved all 32 templates per 160 anchors stayed key-blind for 96,000
anchors (0/32 swap flips; `work/core-runs/drill-acceptance`), while the three
harness seeds first reached at least 8/32 key-swap argmax flips at about 16-42k
anchors and train loss below the 0.139 key-blind floor at about 30-61k, so the
curriculum is part of the recipe. The v2 run (`work/core-runs/drill-v2`, seed 7)
matched the harness bit for bit through anchor 1,600, then flickered (up to
10/32 flips at 33-53k, earlier than harness seed 7's first >=8/32 at 41.3k, but
not held), held 8-16/32 flips from 80k through 121.6k (interval loss
0.121-0.169, below 0.139 at 8 of those 14 evaluations; first below the floor
at 89.6k versus 30-61k for the harness seeds) and showed 0/32 at 124.8k and
128k: retrieval is learnable on the production path but unstable at a constant rate.

`core_contracts` pins the transfer: 12 anchors of the production drill under
anchor-only match `Shuffled` under the assay ledger bit for bit (tiles, E,
predictor, latents, history, baseline, loss); the drill stream equals
`Shuffled` across cycle boundaries; a golden digest pins the run identity.
`core_resume_contracts`: resume past a corrupt newest checkpoint is bit exact,
leaves a stale `.partial` alone, stops on any foreign identity before
quarantining anything, quarantines only integrity failures and, once every
checkpoint is quarantined, refuses a fresh start; best-checkpoint selection
survives a repeated anchor and ignores other runs. `core_redeal_contracts`
walks the real cursor through an epoch boundary: only the SOURCE value and its
citation change, every citation byte copies the prompt, and values change per
epoch and per seed. Beyond the first cycle runs agree in
distribution, not bits: the harness rebuilds State (empty history) every 1,600
anchors and the core path does not. Logs go to `out/log.jsonl` (lines after the last checkpoint repeat
after a resume), the summary to `out/report.json`. End a run with `out/STOP`
(final checkpoint) or the wall deadline, not by killing the trainer.

Evaluation and generation replay history from the actual next byte (generation:
the byte it just emitted, after a warm-up that observes the prompt's own last
`SPAN_WARMUP` bytes so the first generated byte is not zero-history either)
through `replay::Replay`, matching training's own anchor schedule with frozen
weights (`replay_contracts`): the window holds up to two anchors, but training
prunes the older anchor once any visited tile's or E's version has advanced
twice since it was recorded, not once (`history::Record::compatible` tolerates
a lag of 1), so Replay is an upper bound on training's real window.
`telemetry`'s `history_len` logs the mean history RECORDS present per training
anchor right after prune -- one record per route visit, not one per anchor --
and measured on `work/core-runs/profile-ahead` it equals the interval's mean
route length (3.298 and 3.014 at intervals 2-3, matching `route_lengths`), so
steady-state training there kept exactly one anchor's records, not two. This
replaces the zero-history teacher
forcing used before; it costs one extra backward (gather) pass per evaluated
byte (about 3x evaluation cost). `evaluation::teacher` starts a fresh `Replay`
per development row (zero history at each row's first byte), while training
carries history continuously across anchors, so `science` evaluation still
does not reproduce training's own history shape even with replay.
Generation's own warm-up (below) compounds this for science: it calls
`Replay::observe` on the prompt's own last `SPAN_WARMUP` bytes as if they were
targets, something science training never does (training only ever credits
completion/citation bytes, never prompt bytes), so the history at the first
generated (answer) byte matches neither training's own history nor a
zero-history baseline exactly. Drill
evaluation is intentionally zero-history (`tuning_measure`/`assay_measure` run
their own zero-history forward loops and never call `generation::generate`),
so it stays identical to the validated harness evaluation it is pinned against.

## Telemetry

`telemetry.rs` keeps one thread-local interval of phase wall time and realized
training routes. `core_log::record` drains it into each `log.jsonl` line under
`telemetry` and resets it. Nothing is persisted, hashed or part of the run
identity, so it cannot change numerics, checkpoints or resume; the first line
after a resume covers only the resumed interval. Phases are disjoint call sites:
`data` (prefix bytes, anchor target), `prefix` (embedding the window),
`forward` (`forward_path`), `gather` (route backward), `tile_backward` and
`tile_apply` (per unique tile: parameter gradients, then the update rule, latent
Adam on the core path), `shared_apply` (tied E), `records` (predictor and
history) and `checkpoint` (rotation and best keep). `unattributed` is the
training wall minus their sum (counters, pruning, timer overhead). `evaluate`
runs outside that wall and is reported beside it. Never nest `time`, or
`unattributed` goes negative; the contract test checks it is non-negative.
Routes count training anchors only: visits per (depth, tile), anchors touching
each tile (that interval's tile commits), route lengths, distinct routes and
the eight most frequent. `execution::START_TILE` is the single entry tile for
training, evaluation and generation. Each log line also carries `latent_steps`
(`tuning::steps`): per-tile max latent Adam step plus tied E's, under the core
path's latent update rule; and `history_len` (`telemetry::history`, recorded
once per training anchor right after `prune`): the mean history RECORDS
present at forward (null if no anchors in the interval; one record per route
visit, not one per anchor) -- measured on `work/core-runs/profile-ahead` it
equals the interval's mean route length, i.e. steady-state training there
kept one anchor's records, not Replay's two-anchor ceiling, since training
prunes a stale anchor on the SECOND version bump of any visited tile or E,
not the first (see §Core path). `history: false` (the ablation knob, §Core
path) drives this to exactly 0 for every anchor, since `telemetry::history`
still runs but `state.history.records` never receives a push; the
`history_contracts` test pins this at 0.0 over a 12-anchor drill run.

## Separate owned retained-text assay

The closed experimental protocol and its exact byte-level update definition are
in `../../docs/CAUSAL-RETAINED-TEXT-ASSAY-V1-CONTRACT.md`. It is prepared for review;
no campaign or useful-model endpoint is established by this implementation.
`assay_update` implements only the tied-E comparison, retaining dense threshold
one as the control. Its counter-hash samples and scalar probability sum create
no resident per-weight optimizer state. `assay_source` accepts only the three
pinned synthetic artifacts and derives all cursor positions from physical work.
`assay_state` owns the compact rule, update and observation ledgers.

The common step uses a named numerical clock: physical work for v3, four times
physical work for assay anchors. The assay and core anchor-only schedules
bypass locals. Counter validation lives in `state_counters` with the same v3
field order and rules. Assay StateRecord schema 2 carries a required non-null
descriptor; v3 schema 1 omits it and retains its former serialized bytes and
numerical hash.

`assay_measure` polls at every teacher/greedy readout and separates content from
LF. `assay_observation` (test-only: the tuning harness and `core_contracts`)
measures every ordered case, checks unchanged complete numerical identity and
publishes one immutable bounded artifact. The owned assay runner, study binding
and observation restore are archived with the v3 owner chain.

## Joint predictor fitting

`predictor_joint.rs` owns the production batch-1?4 arithmetic, shared by anchor
and replay fitting. Borrow feature/correction pairs; validate the whole batch
and apply the second clip before proposing. Rank each
frozen row by the reviewed f64 raw mean, canonical pair/column/direction order,
and no-move ties. Guard with actual f32 predictions, merge canonical trits,
recheck each selected row, then commit once. Validation errors leave Predictor
unchanged; an execution error still poisons the enclosing opportunity. No dense
predictor gradient, Evidence or float master belongs to this rule. The existing
threshold-one evidence encoding still applies to tile/E updates and packed
constructors; predictor fitting has its own explicit configuration rule ID.

`credit::train_records` gathers all current-anchor records before one fit and
then pushes them in forward visit order, including repeated tiles. It does not
mix retained history into the fit. `History::replay` borrows the newest at most
four records in retained insertion order after the existing entry prune;
`execution::local` fits that suffix once before tile work. Later per-visit
pruning, capacity eight, FIFO, age/version eligibility and target isolation are
unchanged. No new controller, timer, telemetry or model-data path is introduced.

Counters distinguish batch calls (including stalls) from pair contributions.
Anchor batches equal anchors. Anchor pairs plus local tile updates equal visits;
replay batches are bounded by locals, and each nonempty batch contributes 1?4
pairs. Transitions are bounded by width times batches and predictor version by
batches. Each precommit pair MSE is reduced in f32, then accumulated into f64 in
pair order, retaining exact checkpoint bits. Its denominator is pair counts.

The new predictor-phase ledger counts current shared packed weights, two extra
packed predictor matrices, one i8 assembly, retained history, four new anchor
records and second-clipped targets, twelve prediction vectors, old row errors
and selection flags. Prefix, trace and original gathered corrections remain
live. The E gradient has already been explicitly dropped. Replay borrows records.
A separate 16*d scratch term covers feature/projection temporaries and 4096 bytes
cover bounded numerical metadata/descriptors/stack arrays, beyond the earlier
record/tape terms. These payload bounds exclude allocator overhead and external
State/Cursor/data/store/runtime allocations; native physical admission remains
required. No single-tile residency is inferred from the in-memory test Store.

## Model and correction boundaries

`shared.rs` owns tied E and constructs immutable normalized byte-plus-position
prefixes. Prefix vectors are not contextual word/sequence representations; that
may limit supplied-evidence QA capacity. `attention.rs` queries those vectors at
each visited tile. `ffn.rs`, `pulse.rs` and `routing.rs` implement the declared
gated curve, output modulation and detached route-to-U candidate. `norm.rs`
holds exact RMS derivatives and finite checked primitives.

`tile.rs` owns eleven packed matrices and exposes pure forward/replay calls.
Forward output/latent/heads are temporary. Execution tapes must retain only
input plus ID/version/action/depth metadata, not Forward, caches or weights.
`backward` independently accepts prefix and parameter sinks: phase one gathers
input/prefix adjoints while all weights freeze; phase two recomputes and sums
parameter gradients per unique tile. Do not add prefix adjoints twice. Policy
credit is detached and never modifies returned input/prefix corrections.

`Shared::task` returns the tied readout gradient. Execution adds the returned x0
correction into the final prefix row, then `add_prefix_gradient` scatters every
prefix occurrence into that same gradient. Apply E once after complete anchor
credit gathering. On this core path local predicted corrections never update E or
read labels; connected v5 local predictions do read the already-observed target
byte's label feature ([CONNECTED.md](CONNECTED.md#label-conditioned-credit-predictor-and-gain-scaled-decay-v5)).
`predictor.rs` trains only on real clipped returned corrections, independently
versioned from E; feature/target observations are detached.

## State encoding

Tile matrix order is Q,K,V,O,G,A,D,CURVE,ROUTER,U,P. Shared E and predictor are
separate packed payloads. `from_packed` validates every shape, canonical trit
encoding and the exact evidence rule `threshold-one-canonical-zero-v1`.

The inherited threshold-one algorithm starts every evidence entry at zero and
resets on every nonzero sign, including saturated transitions. Zero gradient
preserves zero. Therefore the complete persistent evidence state at every update
boundary is canonically all-zero and can be encoded by this checked rule rather
than a redundant payload. A different threshold cannot reuse this encoding.
Fresh temporary Evidence is allocated per matrix during apply; under this rule
no float master or omitted nonzero optimizer state exists (the core path's
latent-Adam shadow weights are separate `tuning` state, checkpointed as
`latent.bin`). A version advances once when weights change. Checkpoint, cursor
and history state belong to `checkpoint_state` and `core_checkpoint`, not these
numerical constructors.

All gradients are shape/finite-validated before tile apply mutates any matrix.
Invalid version increment fails first. A failed numerical calculation never
justifies advancing the committed checkpoint/cursor.

`state_counters::Counters::validate_schedule` gates its predictor-anchor
invariant on `tuning::current().history`: on, `predictor_anchor_updates ==
anchors` as always; off, it instead requires `predictor_anchor_updates == 0`,
`predictor_anchor_pairs == 0` and an empty `History` (the knob's effect on
`credit::anchor`, §Core path), while the surrounding tile-commit and
predictor-replay checks are unconditional either way. Knobs are installed
before a checkpoint loads (`core_train::run`), so `resume` validates a
restored state under the same rule the run was trained under.

`state_codec` writes each tile, E and the predictor as one binary object: magic
`HWCPACK1`, geometry (width, FFN, heads, context), object kind, tile ID, weight
version, evidence tag 1 (`threshold-one-canonical-zero-v1`) and matrix count,
then per matrix its rows, columns and byte length followed by the packed trits
in registration order. Decode requires the exact total length, every header
field and shape, and no trailing bytes; `from_packed` then checks canonical
trits and padding. Objects are capped at 128 KiB (`storage_paths::FILE_CAP`).
`core_codec` length-prefixes the seven tile objects and serializes latent Adam
state; `core_checkpoint` binds every payload file by SHA-256 in its manifest.
(Folded from the archived `PERSISTENCE.md`, whose DiskStore publication, owner
release and retention rules no longer apply.)

## Allocation ledger

Geometry's ledger follows the adopted payload formulas and intentionally remains
conservative for evidence: it reserves 2P although only one matrix's zero evidence
buffer is live during apply. Every numerical heap vector is one of:

- Packed matrix storage; temporary constructor vector of eleven matrix headers.
- Prefix bytes, L inverse RMS values and L*d memory, with one embedding-column
  and normalized row construction scratch; positions are computed on demand.
- One tile cache: factored score vectors/summaries (2Hd), probabilities (HL), normalized x/query/
  attended/O output (4d), normalized u (d), FFN gate/z/curve/product (4m), FFN
  output (d), pulse features/latent/heads (5d+r), final output (d).
- Replay scratch: score adjoints (HL); three FFN derivative vectors (3m);
  bounded current-state/pulse/normalizer/head-summary adjoints. These fit the reserved
  64d/8r vector sections with temporary kernel-return vectors included. Prefix
  adjoints are accumulated per head; no additional L*d arrays.
- The eleven dense gradient vectors (4P_tile), E gradient (4Vd), predictor
  gradient (4dF), or the caller-owned summed prefix adjoint/correction tape.
- Two readout vocabulary vectors at most; predictor feature/prediction/target/
  derivative vectors; route features, probabilities and policy scores.

Current helpers allocate named bounded vectors rather than a monolithic arena.
Vector/container metadata, allocator capacity/rounding, stacks, code, serialized
payload overlap, checkpoint/data/owner buffers and OS file cache require separate
accounting/measurement. The ledger is not measured RSS or physical admission.
Execution must release obsolete Forward results and its actual input/output
copies according to the integrated lifetime ledger before claiming that bound.

## Authored numerical fixtures

`numerical_tests.rs` covers epsilon-aware RMS adjoints, current-state and prefix
conditional derivatives, two-phase sink equivalence, detached policy credit,
tied E readout plus repeated-prefix derivatives, and fail-closed versioned update
and state decoding. Register it only under cfg(test) in the integrating module.

Tests that write files take a `scratch_dir::ScratchDir`: a unique
`work/validation/<prefix>-<pid>-<nanos>-<seq>` directory that its guard removes on
drop, also while a failing test unwinds, so sweeps leave no per-run directories
behind (owner request: clean as we go). Declare the guard before the stores and
corpora that live in it, so it drops last.

`attention.rs` and `attention_reverse.rs` implement the v2 single-query
factorization. `attention_contracts` compares every Q/K/V/O, x and M derivative
with an independent legacy f64 forward and checks all optional sink combinations.
Its float weights exist only inside the test reference. Cache payload replaces
2Ld with 2Hd; Geometry reserves 83,696 scratch bytes at candidate context cap.
Row-block products use fixed scalar packed kernels; full Q/O dispatch is unchanged.
Real-arithmetic equivalence does not imply equal f32 signs, routes or trajectories.
Old protocol/config/checkpoint identities are rejected before continuation.
These numerical fixtures' authoring-time pending status was superseded by the
retained v2 validation (see `../../archive/src/causal/AGENTS-history.md`),
scoped to that receipt's source identity.

## Integration API inventory

| Type | Entry points |
| --- | --- |
| Geometry | fixture/candidate constructors, validate, routes, shapes, features, ledger |
| Tile | new(geometry,id,seed); from_packed(geometry,id,version,eleven byte vectors,evidence rule); packed() borrowed fixed array; forward(input,memory,h); route(Forward,h,optional RNG); backward(input,memory,h,b_y,optional prefix sink,optional TileGradients,optional action-index/advantage); apply(TileGradients) |
| Forward | temporary output[d], latent[d], heads[2d+r]; never persist in path tape |
| TileGradients | zeros(geometry,id), validate; values in canonical eleven-matrix order |
| Shared | new/from_packed/packed; prefix(bytes); readout(output); task(output,target byte); add_prefix_gradient(Prefix,b_M,existing gradient); apply(combined gradient) |
| Prefix | bytes, normalized memory, inverse scalars, E version; initial(geometry) borrows last row of a constructor-produced prefix |
| Readout / Task | Readout has normalized final state/logits and loss(target), avoiding E-gradient allocation during evaluation; Task adds input_correction and embedding_gradient |
| Predictor | separate new/from_packed/packed/version; features(input,output,h,q,r), predict(raw tanh), correction(unit-clipped), train_joint(feature/correction pairs) returning per-pair MSE and transitions |

Matrix/evidence helpers remain in the crate ternary module. Route actions are
legal indices mapped through sorted Mesh neighbors then halt; forced fourth
visits skip routing. Replay validates versions and retains no Forward cache.

## Execution and complete boundaries

`execution::step` takes State, a TileStore, a TrainingSource and the caller's
current STOP observation. Production TrainingSource implementations (Dataset,
the assay Source and `core_drill::Drill`) delegate only to pinned data and the
Cursor API. They supply no target to local numerical updates; an anchor
requests its target after the frozen forward trace. Cursor advances only after
complete anchor numerical/history/counter work.
`core_train` polls `out/STOP` between complete opportunities and checkpoints
before its stopped exit. A mid-opportunity error sets Poisoned; that in-memory
state cannot continue or checkpoint. Recovery loads a previous complete immutable
checkpoint in a fresh process (`core_checkpoint::resume`). There is no rollback
claim.

`credit.rs` gathers returned corrections and all prefix contributions before
commits. Phase two groups each tile's repeated visits in forward order and
commits that tile once; it never adds prefix adjoints again or draws route RNG.
E's readout and normalized prefix gradients share one update. Predictor records
use frozen inputs, outputs (next frame's input or final output), captured h and
dependency versions. Predictor training happens after tile/E commits.

State exposes geometry, seed, global opportunity, baseline, Shared, Predictor,
all tile versions, History, Cursor, Counters and Boundary. History and counters
are serializable value records; packed model constructors own weight decoding.
Ready validation checks exposure schedule, cursor masks, visits versus updates,
finite aggregates, bounds and version plausibility. Dataset validation checks
the stronger cursor identity/order/epoch relation. Stale history at a boundary is
legal; the next opportunity prunes it. Future dependencies are invalid.

## Execution allocation additions

Trace owns at most four input vectors, one captured h and final y. Credit owns
four correction vectors and one E gradient; phase-one prefix adjoints are freed
before parameter aggregation. Only one tile gradient exists at a time. The E
gradient is freed before predictor replay/training. During predictor training,
Trace and correction vectors remain live, so Geometry's predictor-phase bound
adds their payload and reserves caller correction plus predictor's target,
prediction and derivative vectors. This is an integration correction to the
initial proposed phase formula; it is still not measured heap or RSS.

Execution reads the process-wide Mesh (`geometry::mesh`, CONNECTED.md §Tile count), allocates one temporary <=6-neighbor
vector, four Frame headers, four correction Vec headers and eight Record
headers. On 64-bit targets Frame is expected to require 64 bytes (including its
24-byte Vec header and 16-byte Option<usize>); the earlier 32-byte tape metadata
term alone does not cover all headers. The integrator must count actual
size_of values, State/Cursor strings and headers, allocator capacity and bounded
data/store/checkpoint overlaps separately. Source prefix bytes overlap their
copy during Prefix construction, then the source copy is released. The local
path retains one temporary Forward while replaying its gradient; this is within
the numerical scratch reservation, not a retained tape cache.

`execution_contracts` uses `execution_fixture` only under cfg(test). Its bounded
seven-tile memory store is a test implementation, not the production residency
policy. Fixtures cover local target-read isolation, STOP at a ready boundary,
frozen repeated-visit aggregation, one tied-E commit, stale-frame rejection,
poisoned failure and exact split continuation of weights/history/cursor/metrics.
`parameter_contracts` uses `continuous_fixture` only under cfg(test): an independent
f64 continuous forward checks parameter derivatives in every task matrix and
detached policy R/P rows. Float reference weights exist only in this fixture.
Register numerical_tests, execution_contracts and parameter_contracts as test
modules; their helper files are nested by explicit path.

## Read-only evaluation and generation

`causal plan` returns configuration and payload arithmetic without constructing
weights, opening data, selecting a numerical kernel or starting a process.

### Generation

Generation accepts nonempty UTF8 prompts and rejects requests exceeding context.
It warms `replay::Replay` up from the prompt's own last `SPAN_WARMUP` bytes,
then replays each generated byte's history like training (not zero history),
uses greedy legal routes, and returns raw bytes without UTF8/JSON repair. Each
byte comes from a `decoding::Decoder`: greedy (lowest byte on ties) or nucleus
sampling (Holtzman et al. 2019, arXiv:1904.09751), which draws from the smallest
most-probable set holding top-p of the tempered mass with
`assay_update::draw(seed, position)`, so a sample is a pure function of logits,
settings, seed and byte position. General generation stays greedy and stops on
newline or byte budget. Corpus story generation instead preserves newlines and
stops before the END marker or at the byte budget; its samples record stop
reason and UTF8 validity. Each prediction uses the most recent corpus-window
bytes, with positions reset at the beginning of that slice.

Story openings (`generation_policy: story-window-v2`) end a full training window
with the prompt: `decoding::story_opening` prepends public held-out text ending
at an evenly spaced story boundary. Positions count from the first visible byte
and training predicts every byte from a full window, so the newest byte always
sits at position `window - 1`. The v1 opening, `END + prompt`, put it at the
prompt's length, a layout training almost never produced; on the reference
run's 2.216 bits/byte checkpoint it turned "Once upon a time" into
single-letter repeats, while a full window gave words (`docs/STATUS.md`
§Reference path). Evaluations record a greedy and a nucleus continuation
(temperature 1.0, top-p 0.9, seed = prompt index) per story prompt, prompt k
using boundary k of four. Connected generation uses the same openings and
decoder (CONNECTED.md, policy v3).

`hexweave complete` loads an explicitly named corpus checkpoint, checks its
run identity, hashes and cursor corpus pin, opens the verified corpus for the
held-out opening (the median story boundary), and emits a bounded read-only
continuation without launching a worker. `--top-p P [--temperature T]
[--seed N]` selects nucleus sampling (defaults 1.0 and 0); connected
checkpoints accept it too. It targets the
currently supported candidate geometry and isolates optimizer state in a fresh
thread, including for library callers. It
rejects an over-context request before loading the checkpoint. The reported
v2 immutability fingerprint covers packed model and continuation state, not the
thread-local latent optimizer map. Connected completion additionally fingerprints
tile/Adam state and follows the connected contract above. The prompt and leading marker must fit the
model context; the byte limit is independently bounded at 1,023.
The frozen teacher-forced scoring and best-checkpoint selection are unchanged.
Evaluation
uses only the development Dataset, exact pinned prompts and answer-only teacher
forcing; masks partition answer/citation/format NLL sums and denominators.
Generation scoring preserves each output, UTF8/JSON validity, exact answer/
citation/completion matches and explicitly scoped unsupported answers on UNKNOWN
cases. Repeated/invalid/missing JSON fields fail structural scoring. Partial
coverage from finite target/case limits or STOP is explicit, never presented as
complete development accuracy.

Immutable hashes stream every tile plus Shared/Predictor, cursor, counters,
history, versions, baseline and seed before and after each evaluation/generation.
Read-only paths never call execution::step. Storage read counters may advance;
scientific continuation state may not. These checks establish immutability only,
not assistant quality, held-out independence or original-F equivalence.

generation_contracts checks deterministic ties/newlines, context/input limits,
STOP, invalid UTF8 preservation, immutable state, strict output JSON shape,
nucleus truncation and proportions, and story-opening boundary selection.

## Generic evidence-data compatibility

The historical `hexweave-docs-qa-v1`/`t1-snapshot-d0-v1` fixture remains supported
with its original provenance and `project-docs-causal-data-v1` identity. New
`hexweave-evidence-qa-v1` inputs use `evidence-causal-data-v1`, a bounded explicit
version, and truthful external_sources_fetched=true provenance. Source pins use
`hexweave-evidence-source-pins-v1`, exact candidate_files entries and a provenance
map keyed by every source ID: URL, title, publisher, UTC retrieval time, license
URL/basis and transformation description. URLs and metadata are syntax/length
checked; source authors/reviewers establish their truth and suitability.

Generic partition families are train/development/reserved_evaluation_source.
The original16-source,256-row-per-artifact, source/artifact/file/span/UTF8/render
bounds and source/fact-family disjointness remain enforced. The science author
keeps held-out sources outside the ingestion inventory. Runtime data pins are
separate from source pins, but checkpoint data identity is immutable: a different
science dataset requires a fresh run, never resume substitution. No science
corpus content or scoring rubric is compiled into the binary.

## Amortized source verification

Pattern: amortized verification, i.e. a full hash once, then a cheap metadata
guard per read with a full re-hash on any doubt. `data_guard.rs` hashes each
Dataset source in full on first use (open) and records its canonical path,
length, modified and created times. Later span reads recheck the canonical path
and metadata before and after a bounded direct read; any difference trips a full
re-hash, which fails closed on mismatch. `Dataset::open` re-hashes every verified
source once after its scan. The core path does not re-hash sources at
checkpoints (the archived DiskStore path did, through
`TrainingSource::verify_boundary`); checkpoint identity and numerics are
unaffected by the guard.

Not guaranteed: a source replaced or edited while preserving length and both
timestamps is detected only when a guard trips or at the next `Dataset::open`,
not at the moving step; opportunities completed before then may have read such
content. On Windows NTFS, LastWriteTime lags while a foreign writer holds the
file open (and for memory-mapped writes), so such a same-length edit passes the
guard until a full re-hash. Stamp carries no file identity, so a different file
swapped in with equal length/mtime/creation time also passes until then.

## Numerics profile and the tuning harness

`tuning.rs` holds the numerics profile: `Knobs::PINNED` (v3/assay numerics) and
`Knobs::core` (`§Core path`). Every build reads a thread-local; production sets
it only through `core_train::run` -> `install`, so every other path stays pinned.
`tuning_update.rs` holds `Latent`, the tuned update rules (normalized/raw
stochastic rounding, latent Adam) and the per-call draw-counter stream
(`stream`, `hash`), re-exported from `tuning` so every existing `tuning::`
call site is unchanged.
The in-memory harness `tuning_harness.rs` (ignored test `tuning_run`, driven by
`HEXWEAVE_TUNE_*` variables and `work/tuning/tune.py`) explores further knobs
under the test-only `scoped()` guard, which restores pinned knobs on drop. Pinned
defaults are IEEE-exact (divide by 1.0, multiply by 1.0, the same 0.25), and the
harness at pinned knobs reproduces production study arm 0's eight observation
files with identical SHA-256 names (`work/tuning/parity.py`). The continuous
gradient oracles (`parameter_contracts`, `numerical_tests`) also run one tuned
case (attention scale 4, mix 1, temperature 4) so tuned adjoints are checked.

Rates: `tile_rate`/`router_rate`/`embed_rate` replace the threshold-one sign
step; `update` picks how. `normalized` stochastic rounding (Gupta et al. 2015,
arXiv:1502.02551) moves each trit with probability min(rate*|g|/rms(g), 1): the
flip rate ignores loss scale, so constant rates churn at a fixed floor and only
external decay anneals them; for heavy-tailed E gradients `rate` far exceeds the
fraction actually moved. `raw` uses min(rate*|g|, 1), which self-anneals as
gradients shrink. `latent` keeps f32 shadow weights per trit, stepped by Adam
(BinaryConnect arXiv:1511.00363; Adam arXiv:1412.6980) and thresholded at ±0.5
within [-1, 1] (TWN arXiv:1605.04711); memoryless rounding lacks this search
phase (Li et al. 2017, arXiv:1706.02379). Draws come from a per-call counter
stream, so they never repeat within a run, across segments or for no-op updates.

Other knobs: `temperature` divides readout logits and caps the logit gap at
2*sqrt(d)/tau, flooring NLL at ln(1+255*exp(-2*sqrt(d)/tau)) (0.64 nats at tau 4,
2.78 at tau 8). `attention_scale` multiplies attention scores; large values make
initial attention near one-hot and starve Q/K gradients. `attention_mix`
replaces the 0.25 attention residual weight. `greedy_routes` trains on argmax
routes, but the REINFORCE router update then scores the argmax action, which is
no longer an unbiased policy gradient.

Harness limits: tuned E uses the magnitude ledger. Past the fixed 1,600-anchor
course, runs roll over: learned E and tiles (and latent state) enter a fresh
assay ledger at version 0 with fresh history, predictor and counters, carrying
the route baseline; route seeds are salted per segment, but the data order
repeats. Evaluations run at segment passes 16 (end of foundation) and 20 (after
retention stress). Rollover is not production-equivalent beyond segment 0.

Measurement: answer offsets 2, 3 and LF follow from the last byte and position
alone, so their best NLL is 0 and they test fitting without retrieval. Raw
multiple-choice argmax scores 4/16 for any prompt-blind model, so `tune.py`
also reports prior-centered choice. In this dataset the unordered pair of source
words determines the answer in every group ({soil,sand}->soil, {clay,sand}->sand,
{clay,silt}->clay, {soil,silt}->silt), so counterfactual success cannot separate
copying from pair recognition; paraphrase answers sit at untrained positions.
Results are exploratory evidence, not adopted numerics.

Probes (`tuning_sources.rs`, `tuning_measure.rs`): `Shuffled` wraps the assay
source and, per (segment salt, epoch, row), re-deals both source values and the
asked key through a pure hash, so prompt and target always agree and neither
memorization nor the pair shortcut can lower the loss; any key-blind model is
floored near 0.139 nats/byte there. The key-swap probe flips the asked key
(A<->B, C<->D); only a model that reads the key changes its answer, so swap
retrieval, not exact match on the original rows, is the retrieval evidence (a
word-order bias inside the pair scores ~12/16 on original rows). The attention
probe reports, per visit, head-major mass at the answer position on the asked
key (byte 48), both value slots and the source labels, averaged over the rows
that reached that visit (`reach`; routes can halt early). `no_halt` removes the
halt action from `Tile::route` (the REINFORCE update still scores the unmasked
distribution). With latent Adam (rate 0.003), greedy routes and shuffled data,
three seeds learned key-conditioned retrieval after ~600-1,000 passes: a later
visit attends to byte 48 (up to one-hot) after the first visit reads the slots.
