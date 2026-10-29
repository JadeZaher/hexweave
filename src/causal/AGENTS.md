# Causal-prefix-v3 numerical modules

Archived v3 owner-chain, study, comparison and diagnostic sections:
`../../archive/src/causal/AGENTS-history.md`.

## Core path: causal-latent-v1 (production training)

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
evaluation is teacher-forced held-out bits per byte over 16 fixed spans of 128
bytes in both forward modes on the same weights (`dev_*` for the run's mode,
`alternate_*` for the other), plus greedy continuations of four story openings.
`train_frozen_*` scores four fixed train spans the same frozen-weight way, run's
mode only: a generalization-gap monitor separating online adaptation during
training (prequential train loss scored on weights that just adapted to the
current story) from unrepresentative held-out spans. Both run `score_spans`,
which steps only the warm-up and scored positions of each slice; the bytes
before them are prefix context. `replay_contracts` pins its f64 result on a
fixed probe: if that moves, evaluation numerics changed, and logged numbers
from earlier commits must be re-scored before they are compared.

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
(history staleness stays bounded by age); evaluation passes zero history while
training uses the history mean, so train and held-out losses differ by more
than data. Corpus runs ignore `eval_cases`; best-checkpoint selection applies
to them as to science runs. `prepare_corpus.py` writes LF manifests and refuses
an existing corpus directory; the first TinyStories manifest (pinned by the
2026-10-28 runs) predates that and has CRLF line endings.

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
save is logged, never fatal. The newest-three rotation had pruned the 72-row
run's best state (80k anchors) once the 140k checkpoint landed.
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
the byte it just emitted) through `replay::Replay`, matching training's own
anchor schedule (two-anchor window, weights frozen; `replay_contracts`), instead
of the zero history teacher forcing used before; this costs one extra backward
(gather) pass per evaluated byte (about 3x evaluation cost). Drill evaluation
(`tuning_measure`) is deliberately unchanged, since it must stay identical to
the validated harness evaluation it is pinned against.

## Telemetry

`telemetry.rs` keeps one thread-local interval of phase wall time and realized
training routes. `core_train::record` drains it into each `log.jsonl` line under
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
path's latent update rule.

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
credit gathering. Local predicted corrections never update E or read labels.
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

Execution also allocates a seven-coordinate Mesh, one temporary <=6-neighbor
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

Generation accepts nonempty UTF8 prompts and rejects requests exceeding context.
It uses zero history, greedy legal routes and lowest-byte tie breaking, stops on
newline or byte budget, and returns raw bytes without UTF8/JSON repair. Evaluation
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
STOP, invalid UTF8 preservation, immutable state and strict output JSON shape.

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
