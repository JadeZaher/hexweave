# Causal-prefix-v3 numerical modules

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
`Knobs::core(rate, seed)`); v3, assay, text and pulse runners never install it
and stay pinned. All seven tiles and their latent state stay resident
(`resident::ResidentStore`, ~25 MB of latent f32 + Adam moments for 2.05M
coefficients); there is no per-update persistence. History corrections still
feed the forward pass; the predictor is trained and checkpointed (counter
invariants require it) but nothing reads it under anchor-only.

Checkpoints (`core_checkpoint`, payload codecs in `core_codec`):
`checkpoints/a<12-digit anchors>/` holds state, E, predictor, tiles, latents
and a SHA-256 manifest. Files are fsynced in `<name>.partial`, which is then
renamed; rename and removals retry Windows sharing errors (os 5/32, e.g. a
scanner holding a just-closed file) with backoff, and pruning to the newest
three is never fatal. Resume walks newest first: a checkpoint of another run
identity (protocol, data, seed, rate) aborts with nothing touched; one that
fails integrity is renamed `.rejected-<pid>` and the next is tried; restore
verifies every hash, the directory name, then reinstalls latents and the draw
counter, so resume is bit exact. `resume: true` with no checkpoint starts
fresh; a fresh run refuses existing checkpoints. Changing `anchors` or
`wall_seconds` extends a run. `out` must be `work/core-runs/<name>`.

Data: `science` trains on a pinned QA Dataset's train split and evaluates the
development split with `evaluation::evaluate` (teacher-forced NLL, coverage,
greedy JSON answers, exact answer/citation) plus four train-row samples.
`drill` (`core_drill`, v2) replays the assay schedule in 1,600-anchor cycles
(16 foundation passes, then 4 retention-stress passes) and re-deals values and
the asked key with `tuning_sources::redeal` under the harness's per-cycle salt,
so its stream is exactly the harness `Shuffled` stream that learned retrieval;
it is evaluated with the assay rows, key-swap and attention probes. A v1 drill
that interleaved all 32 templates per 160 anchors stayed key-blind for 96,000
anchors (0/32 swap flips; `work/core-runs/drill-acceptance`), while the three
harness seeds broke out at 28-42k, so the curriculum is part of the recipe.

`core_contracts` pins the transfer: 12 anchors of the production drill under
anchor-only match `Shuffled` under the assay ledger bit for bit (tiles, E,
predictor, latents, history, baseline, loss); the drill stream equals
`Shuffled` across cycle boundaries; resume past a corrupt newest checkpoint is
bit exact. Beyond the first cycle runs agree in distribution, not bits: the
harness rebuilds State (empty history) every 1,600 anchors and the core path
does not. Logs go to `out/log.jsonl` (lines after the last checkpoint repeat
after a resume), the summary to `out/report.json`. End a run with `out/STOP`
(final checkpoint) or the wall deadline, not by killing the trainer.

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
physical work for assay anchors. Only the assay schedule bypasses locals.
Counter validation lives in `state_counters` with the same v3 field order and
rules. Assay StateRecord schema 2 carries a required non-null descriptor; v3
schema 1 omits it and retains its former serialized bytes and numerical hash.
`checkpoint_restore` contains the existing restore transaction moved intact,
plus bounded immutable observation import. A censored observation cannot resume.

`assay_measure` polls at every teacher/greedy readout and separates content from
LF. `assay_observation` validates ordered cases and raw outputs, checks unchanged
complete numerical identity and publishes one immutable bounded artifact.
`assay_runner` checkpoints every 64 physical anchors and every observation,
through the existing native owner and three-generation retention path.
`assay_study` binds native requests to the closed conserved study declaration.

The focused new contracts execute three candidate anchors and three teacher
readouts, not the long study. The complete observation used for checkpoint
parity is deliberately fabricated metadata for the persistence fixture; its
zero losses are not model measurements. The separate partial observation is
measured and proves censoring and unchanged identity. No generated text is
needed for these software checks. Earlier closed learning experiments remain
filtered by the parent-controlled integrated sweep.

## Current bounded comparison outcome

The anchor-only/current-method fixture is complete. Across seeds 7, 17 and 29 it
executed 48 anchors and 72 locals exactly once. Anchor-only aggregate train and
held-out NLL improved in all seeds, but its seed-7 digit NLL worsened from
6.767959 to 9.988211. Current-method final held-out NLL beat anchor-only for seed
7 and lost for seeds 17/29. Retention deltas do not establish absolute retention
superiority: seed 17's lower current-method final digit loss remains relevant.
All 12 final greedy answers failed the two-byte target: eight LF-only outputs,
two strings of 14 `5` bytes plus LF and two capped 16-space strings. The 24 total
continuations include initialization. This supports no larger campaign
recommendation, adopted method change or finite useful-assistant ETA.

The 250-file executed source is `fc20f4d3...`. The initial integrated sweep passed
all four stages, including 107 Rust tests with five closed experiments filtered;
the older production sweep's failures/follow-ups remain separate history. The
3,853,050-byte logical payload bound is not measured RSS. E4 remains closed and
its STOP unchanged. No new run or private evaluation is adopted by these docs.
See [scope](../../../nonmonotonic-training-lab/work/f-anchor-comparison-drive/NODE-ZERO.md),
[interpretation](../../../nonmonotonic-training-lab/work/f-anchor-comparison-drive/DIAGNOSTIC-INTERPRETATION.md)
and [final verification](../../../nonmonotonic-training-lab/work/f-anchor-comparison-drive/FINAL-VERIFICATION.md).

The current v3 integration is complete: independent final verification
accepted its software checks, fixed-input parity and release build. Its
durable contract is
`../../docs/CAUSAL-PREFIX-V3-CONTRACT.md`. Retained v1/v2 contracts and
[original v2 validation](../../../nonmonotonic-training-lab/work/f-attention-drive/validation-receipt.json)
remain evidence for their recorded source identities. The completed frozen
joint-target fixture supports this separate development step, not a training
admission or useful-assistant estimate.
This namespace leaves text v4 unchanged; scoped lint rationale is in `LINT.md`.
Only the d16/m48/context8 numerical fixture and d128/m384/context1024 candidate
are accepted. Neither is training admission. The 81M/287M targets remain distinct.

## Current v3 joint fitting and evidence boundary

The current durable identity is `../../docs/CAUSAL-PREFIX-V3-CONTRACT.md`.
V1 and v2 documents remain byte-identical required inputs; their numerical
checkpoints cannot resume under v3. Both independent review cycles and the
software/build closure are accepted. The failed initial sweep and passing
targeted follow-ups remain distinct evidence. The executed source is the
244-file post-sweep snapshot a440b623..., not the earlier final snapshot.
E4 used this release and is now terminal after a verified cooperative resource
stop (`adaptive physical cushion reached`): 22,577 opportunities, 5,645 anchors
and 16,932 locals, ending at train-062, offset 25, epoch 0. External driver wall
was 74 min 14.528 s. Final evaluation is null (`resource_stop`), so no trained
public quality result, private eligibility or useful-model ETA follows. All
exact original processes are absent and the Job is closed; the program STOP
remains preserved. No retry or terminal-checkpoint continuation is authorized.
Installed snapshot 7bb572e9... is distinct from compiled a440b623...; this closure
changes only reporting documents. The 81M/287M targets remain unadmitted.
See [E4 terminal verification](../../../nonmonotonic-training-lab/work/f-joint-science-drive/E4-TERMINAL-VERIFICATION.md),
[E4 timing](../../../nonmonotonic-training-lab/work/f-joint-science-drive/E4-CENSORED-TIMING.md)
and [production final verification](../../../nonmonotonic-training-lab/work/f-joint-production-drive/FINAL-VERIFICATION.md).
Historical diagnostic sections below retain their pinned execution scopes.

`predictor_joint.rs` owns the production batch-1?4 arithmetic, shared by anchor,
replay and the adapted joint parity fixture. Borrow feature/correction pairs;
validate the whole batch and apply the second clip before proposing. Rank each
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
Checkpoint serialization and report code inherit the added strict Counters
fields; the existing exact identity/config equality rejects old states. Python
admission now validates every fixed configuration field/type and all three
protocol document pins; terminal reconciliation checks the new conservation
rules, version and finite metrics. Supervisor/Job/STOP/release ownership is
unchanged.

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

The original joint fitting helpers are now a thin test adapter to the production
kernel. The selected existing joint test retains the 64-update/256-pair question
and requires equality with its entire recorded final scalar checkpoint,
including packed hash/version, canonical counts and all four delivered metrics.
The sole v3 record contains 64 production joint calls / 256 pair contributions.
All 17 checkpoints and 16 blocks match the historical four-pair trajectory
except scope/protocol/call metadata. The final every-pair gate passes; its
first measured pass is update 16. This executed extraction parity does not
refresh or replace the immutable original source/observation. See the
[interpretation](../../../nonmonotonic-training-lab/work/f-joint-production-drive/DIAGNOSTIC-INTERPRETATION.md).

The old dense `Predictor::train` remains cfg(test) only, with unchanged arithmetic
for the archived commit/dense-arm questions. The old live learning-gap diagnostic
is explicitly ignored as historical v2 and ineligible under v3: it must not emit
old-labeled results using changed live anchor/local semantics, even in a default
suite. The other three unrelated numerical diagnostics are excluded from this
feature's sweep. Current execution contracts instead cover batch sizes 1?4,
repeated records, newest-four selection, pruning, target isolation and split
continuation. Persistence cases cover exact new counters/float bits/history and
v1/v2 rejection; receipt cases cover type/rule/document/counter drift. The small
zero-feature stalled call and malformed batches are boundary checks, not fitting
arms or evidence of learning. The initial sweep's two test-reference
coercions prevented Rust execution; explicit-slice fixes and the affected
format/Clippy/Rust follow-up passed, including 107 Rust tests with four
historical diagnostics filtered. Python has 29 distinct passing checks:
28 retained from the initial sweep plus the unchanged native ownership check
in a focused follow-up. Its initial memory preflight failure occurred before
child creation; the guard was not weakened. The sole release build passed,
and its plan readback has model_allocated=false, physical_admission=false
and training_admitted=false. Fixed-input parity never establishes dynamic
replay efficacy, production throughput, assistant quality or a useful ETA.

## Bounded anchor-only comparison fixture

The user adopted the fixed comparison in
`../../../nonmonotonic-training-lab/work/f-anchor-comparison-drive/NODE-ZERO.md`.
`anchor_comparison_contracts` is cfg(test) only. It mounts the existing in-memory
Store and keeps two fresh candidate states/stores per seed (7, 17, 29), released
before the next seed. The current arm calls production `execution::step` eight
times for anchors and 24 times for locals. The sibling execution child
`anchor_only_fixture` reuses forward, trace recording, pruning and anchor credit,
while omitting all locals and local replay. Production code, protocol, prior
fixtures and E4 STOP are unchanged. This is not campaign admission or checkpoint
continuation. Both independent review cycles and the sole integrated sweep are
complete: structure, format, strict Clippy and 107 Rust tests passed initially,
with five closed numerical experiments filtered. No Python ownership check,
release build, science campaign or private evaluation was rerun.

The four fixed synthetic source-copy cases have pH digits A/6, B/8, C/5, D/7
and digit-plus-LF completions. The source helper defines all exact bytes and their
length-delimited SHA256. Anchors visit A0, B0, C0, D0, A1, B1, C1, D1. Locals use
the next pending prefix, including A0 after the final anchor, without target
reads or cursor advancement. Each arm makes exactly eight target reads/advances.
All routing and coefficient eligibility remains unchanged. Anchor randomness
uses the same virtual clock 4k and seed + clock * 7919 in both arms. Anchor-only
advances that clock across omitted slots, but records zero local opportunities;
its ledger deliberately fails production schedule validation. History age and
pruning use the virtual clock at actual anchor entry, so this comparison does
not imply equality with a different contiguous-anchor schedule.

`anchor_comparison_measurement` evaluates the eight training targets and eight
unseen paraphrase-prefix targets initially, after cycle four, and after cycle
eight, ordered as four digits followed by four LFs. Every evaluation uses zero
history, the current model's greedy route and production readout/loss arithmetic.
Routes can change across arms and time. All individual f32 NLL values and f64
means are retained; digit retention is final minus cycle-four training-digit
mean. Initial and final held-out A/D generation stops at first LF or 16 bytes,
retaining raw bytes, stop reason, distinct count, longest run and exact two-byte
match. These synthetic copying conditions are not independent facts, science
benchmarks, citation checks or assistant gates.

The separate `anchor_comparison_identity` hash includes exact packed weights,
versions, history vector lengths/f32 bits, baseline bits, cursor, honest counters
and virtual clock under a test-only domain tag. It does not bypass and relabel
the production identity or generation API. Both arms' initial identities must
match. Each complete measurement preserves this identity. Current-arm boundary
validation remains mandatory; source, exposure, commit-call and version checks
apply to both arms. Per-seed reports retain all canonical counters, per-tile
update calls and versions, tied-E/predictor versions and actual pre-update anchor
losses. Learning-cycle microseconds include fixture checks and exclude evaluation;
they do not measure production throughput or equal-cost superiority.

Across all seeds there are exactly 48 anchors and 72 locals, at most 480 learning
visits/tile update calls, 120 joint predictor batches and 480 pair contributions.
The fixed upper bounds are 11,857,920 signed scalar candidates, 47,431,680
candidate-pair terms, 1,440 fit predictor forwards and 288 local correction
predictions. Candidate terms and cache constructions are analytic bounds, not
instrumented measurements. Exactly 288 teacher-forced readouts and at most 384
generated-byte readouts give at most 3,840 total tile cache constructions when
the anchor's two reverse replays and local reverse replay are included. The
largest visible prefix is 65 bytes (50-byte held-out prefix plus 15); the asserted
limit is 65. Report encoding including label/LF is capped at 64 KiB per seed and
192 KiB total. Five minutes is a cooperative abort ceiling checked between each
learning opportunity and each readout, not an asynchronous kill or prediction.

The conservative logical payload bound is 3,853,050 bytes, below 16 MiB:
1,025,920 for both packed models; 34,208 for both history payloads; 2,225,061 for
the existing 65-byte-prefix anchor-core bound; 174,645 for the joint-fit phase;
262,144 for overlapping report payloads; and 131,072 for other bounded scalar
and prefix payloads. Summing sequential scratch phases and already-counted
resident weights is deliberately conservative. Constructor scratch fits the
larger phase bounds. Container/allocator overhead and process RSS are additional
and unmeasured; this in-memory Store does not establish one-tile residency.

A software pass only validates the declared comparison. Favorable NLL, retention
or generation outcomes are never assertions and cannot select work, seeds or
extensions. No larger diagnostic or training campaign is automatically adopted.
Convergence, useful-model ETA and full-F scaling remain outside this fixture.

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
Fresh temporary Evidence is allocated per matrix during apply; no float master
or omitted nonzero optimizer state exists. A version advances once when weights
change. Full checkpoint/cursor/history/owner state belongs to the persistence
and integration lanes, not these numerical constructors.

All gradients are shape/finite-validated before tile apply mutates any matrix.
Invalid version increment fails first. A failed numerical calculation never
justifies advancing the committed checkpoint/cursor.

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
retained v2 validation linked above, scoped to that receipt's source identity.

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
| Predictor | separate new/from_packed/packed/version; features(input,output,h,q,r), predict(raw tanh), correction(unit-clipped), train(real correction) returning MSE and transition count |

Matrix/evidence helpers remain in the crate ternary module. Route actions are
legal indices mapped through sorted Mesh neighbors then halt; forced fourth
visits skip routing. Replay validates versions and retains no Forward cache.

## Execution and complete boundaries

`execution::step` takes State, a TileStore, a TrainingSource and the owner's
current STOP observation. The production TrainingSource implementation delegates
only to the pinned Dataset/Cursor API. It supplies no target to local numerical
updates; an anchor requests its target after the frozen forward trace. Cursor
advances only after complete anchor numerical/history/counter work. The owner
must poll STOP again after each returned complete opportunity and checkpoint
before its stopped exit. A mid-opportunity error sets Poisoned; that in-memory
state cannot continue or checkpoint. Recovery loads a previous complete immutable
checkpoint into a fresh owned attempt. There is no rollback claim.

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
modules; their helper files are nested by explicit path. No authored fixture has
been executed during this author pass.

## Owned integration and read-only evaluation

`causal plan` returns configuration and payload arithmetic without constructing
weights, opening data, selecting a numerical kernel or starting a process.
`causal train --owner-request <absolute path>` is the only numerical CLI entry.
It requires the Python owner's nonce/request-hash environment, immutable launch
receipt, live exact owner/child process identities and assigned Windows Job.
The child accepts one thread, candidate geometry, pinned config/data/source/
runtime/kernel and finite per-attempt quotas within cumulative owner budgets.
The launch receipt is protection against accidental unowned invocation, not a
cryptographic boundary against a hostile actor running as the same user.
Windows FFI in launch_windows only queries process creation/liveness and Job
membership; it closes only handles it opened and never terminates a process.

Config hashes use UTF8 compact JSON with lexically sorted object keys, matching
Python json.dumps(sort_keys=True,separators=(',',':')). The source manifest is
schema1 plus a map of canonical relative paths to SHA256. The owner checks the
complete source set and reviewed build receipt; the child checks every supplied
entry, required core entries, the durable protocol document and current binary.
Source/runtime pins are rechecked before a successful terminal receipt.

The worker checkpoints complete Ready state at initialization, configured
opportunity intervals, before final development evaluation and at terminal
publication. Retention preserves the latest three committed checkpoints and live
working tile references. Older checkpoint references are historical identities,
not a promise that their payloads remain retained. Failed child work never emits
a successful resumable receipt; the owner records unknown physical work and the
last complete checkpoint. Parent checkpoint/source inputs remain read-only.

Owner wall consumption includes initialization, data verification, training,
development evaluation and checkpoints. The receipt separately measures time
inside execution::step, including its real data/store I/O; initialization,
evaluation and checkpoint publication are excluded from that training subtotal.
The external owner charges launch/reap/publication overhead and guards the hard
deadline. Child STOP/deadline polling happens at complete opportunities and at
every evaluation target/generated byte, with a checkpoint reserve. A numerical
opportunity is not interrupted halfway for a cooperative STOP.

Generation accepts nonempty UTF8 prompts and rejects requests exceeding context.
It uses zero history, greedy legal routes and lowest-byte tie breaking, stops on
newline or byte budget, and returns raw bytes without UTF8/JSON repair. Evaluation
uses only the development Dataset, exact pinned prompts and answer-only teacher
forcing; masks partition answer/citation/format NLL sums and denominators.
Generation scoring preserves each output, UTF8/JSON validity, exact answer/
citation/completion matches and explicitly scoped unsupported answers on UNKNOWN
cases. Repeated/invalid/missing JSON fields fail structural scoring. Partial
coverage from finite target/case limits or STOP is explicit, never presented as
complete development accuracy. The separate owned prompt operation accepts only
externally pinned visible prompts; it never loads answer keys, rubrics or scores.

Immutable hashes stream every tile plus Shared/Predictor, cursor, counters,
history, versions, baseline and seed before and after each evaluation/generation.
Read-only paths never call execution::step. Storage read counters may advance;
scientific continuation state may not. These checks establish immutability only,
not assistant quality, held-out independence or original-F equivalence.

## Integration bounds and authored fixtures

Worker requests/source manifests are capped at64KiB; STOP and launch receipts
at2KiB/4KiB. File hashing streams8KiB. Result serialization uses a capped128KiB
writer; development generation retains at most32 cases of192 bytes per pass.
The separate prompt operation retains at most40 cases of192 bytes.
Initial and final evaluation reports coexist with continuation state, and the
result serialization buffer overlaps those reports. Two bounded Dataset ledgers,
source maps, parsed request/manifest records, checkpoint codec buffers, path and
vector headers, result reports and OS/allocator/code overhead must be included
in the owner's full physical ledger. Planned numerical payload is insufficient
as an RSS admission measurement.

generation_contracts checks deterministic ties/newlines, context/input limits,
STOP, invalid UTF8 preservation, immutable state and strict output JSON shape.
runner_contracts exercises complete checkpoint segments, strict resume, causal
exposure and cumulative budget preservation with the synthetic numerical fixture.
The real-QA worker rejects the context8 fixture. The generation/runner fixtures'
authoring-time pending status was superseded by retained v2 validation. The
later 105-test verification linked above provides separate source-bound evidence.


## Cycle-one revision: generic owned prompt evaluation

Requests explicitly select `operation:{kind:"train"}` or
`operation:{kind:"prompt_evaluation",prompts:{path,sha256}}`. The latter requires
strict resume from a clean Completed parent, unchanged checkpoint identities,
zero opportunity/checkpoint-cadence/NLL limits and a separately pinned visible
prompt file. Native process ownership, source/runtime pins and release seals are
unchanged. Owner receipts compare parent, evaluated and terminal scientific blob
identities as well as full-state hashes, and require zero training exposure/time.

Prompt JSON is at most64KiB with schema1 and 1..40 ordered `{id,prompt}` rows.
Unknown/duplicate fields, duplicate IDs, IDs over64 ASCII identifier bytes,
prompts over832 UTF8 bytes and output budgets outside1..192 are rejected.
Generation uses the same frozen decoder and reports raw bytes and explicit STOP
coverage. No keys or rubric fields exist. No quality score is produced here.
The 192-byte completion scope is short evidence-grounded educational QA, not a
broad expert assistant. Project-documentation QA is an engine fixture only.

The original train Dataset remains validated on restore; its validation scans
both train and development artifacts for disjointness. Prompt evaluation does
not numerically evaluate either Dataset, read their targets through execution,
or call step. It processes only the separately pinned visible prompts.

Prompt parsing additionally reserves one64KiB raw input, at most64KiB JSON escape
scratch, up to64KiB for a rejected overlong row, forty accepted row strings
(total<=35,840 bytes), forty row headers and forty borrowed-ID set entries. Use
256KiB plus actual container/allocator overhead as the explicit parser payload
allowance. The raw input is dropped before generation; parsed rows coexist with
up to40 generation reports (raw outputs<=7,680 bytes plus IDs/hashes/headers).
The existing128KiB capped result writer overlaps the report. Python parsing,
checkpoint codecs and process overhead still require the owned physical guard;
these bounds are not RSS measurements. Three retained generations fit only when
the configured existing disk allowance admits them; publication fails closed.

The attempt start wall is the parent's released cumulative consumption. Setup
and admission charge additional elapsed wall before child handoff, without moving
that start. Resume preserves that charge. Three retained generations preserve
evaluation N, terminal N+1 and a late-STOP terminal N+2 until owner validation.
The revision, prompt and schema fixtures' authoring-time pending status was
superseded by the retained v2 validation linked above; later fixture evidence
remains separately bound to its recorded source identity.
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

## Persistence fixture namespace

Each compiled copy of `persistence_fixture` atomically reserves a fresh run
namespace under ignored `work/causal`. Its PID/time/module-derived candidate is
accepted only by exclusive directory creation, with bounded collision retry;
PID reuse, repeated tests and parallel module copies cannot adopt prior claims.
Fixture roots share that namespace and also require exclusive creation. Synthetic
release process identities remain test data; no real process identity is implied.
Existing claims, checkpoints and failure evidence are never removed or replaced.

## Bounded learning-gap diagnostic fixture

`learning_gap_contracts` and nested `learning_gap_fixture` exist only under
cfg(test). Three sequential scenarios fix seeds 7, 17 and 29, candidate width128,
FFN384, four heads and context1024. Each starts from fresh constructors and the
36-byte ASCII prefix `Evidence: leaves are green.\nAnswer: `, with synthetic byte
targets `{`, `"`, `}`. These three bytes are a fixture, not a complete QA record.
No corpus, E1/E2/E3 weights, private prompt, file or process is opened by it.
The existing Source/Store defaults preserve the earlier `?`/`aba` context8 tests.

Each scenario executes exactly one production anchor, clones its complete
post-anchor state into an initially identical local branch, then executes three
production locals only in that branch. The total across the test is three
anchors plus nine local opportunities, at most 48 actual update-path tile visits.
There is no fifth opportunity, adaptive loop, learning-gain assertion or optimizer
replacement. Source target-read/advance counts stay at one; unchanged anchor-only
identity and shared E in the local branch are checked. State and weights of the
two post-anchor branches are compared before any local update.

The initial production anchor trace fixes the conditional measurement tile/action
route, with zero history in every comparison. Production anchor/local updates keep
their existing seeded routing and history. Initial trace loss/visit counts and
unique tile commits are checked against the actual anchor. Each local update's
tile visits are recorded by the test store; legal action indices and terminal
halt/forced-exit identity are reconstructed and checked from those visits and
production counters. These are distinct from the frozen measurement route.

Measurements before/after the anchor and after each local report signed NLL
changes on the original prefix/`{` target and advanced prefix/`"` target. Production
locals operate on the advanced cursor and never receive either target through
their update path. Worsening on the original condition is conditional retention
or interference evidence, not proof that each current local objective worsened.
The advanced-prefix NLL is also conditional on the frozen measurement route and
zero history, not an on-policy task-loss guarantee. Both labels are used only by
the immutable measurement path, whose complete state hash must not change.

Opening-minus-closing-brace logit margins use the original prefix and a second
prefix with one initial conditional greedy byte appended. That byte is frozen
across all phases, so compared inputs do not change with later predictions. This
is not a full generated continuation or reproduction of E3's output collapse.
Each scenario performs five observations of three fixed prefixes, at most 60
measurement tile forwards, plus one initial production trace and credit gather
without commits. The longest prefix is 37 bytes; each trace has at most four
visits. No readout creates additional training opportunities.

Predictor metrics hold initial anchor features and true correction targets fixed
across all five phases. Targets reproduce both production unit-clipping calls
(record construction and Predictor::train), including normal f32 rounding.
Reports accumulate f64 coordinate-mean MSE over those f32 predictions/targets,
the matched zero-predictor error, raw/clipped norms and cosine. A zero vector
produces null cosine plus an explicit reason. These frozen-probe observations
are not the run's cumulative pre-update MSE or fit on later local conditions.
There are at most four probe records per phase; finite values, coordinate bounds
and the zero-reference norm bound are asserted, not a favorable outcome.

This fixture uses the existing in-memory test store. At most two seven-tile
stores and two States coexist within a scenario; scenarios release their state
sequentially. One temporary forward cache is used per call, with bounded short
prefixes, correction records and reports. This makes no native one-resident-tile,
worker Job, RSS, disk-admission or full-training claim. The process-resolved
kernel name is reported. Output is three `LEARNING_GAP_DIAGNOSTIC` JSON lines,
each capped at32KiB (96KiB total), captured by the final test sweep. No new CLI,
controller, protocol, production constructor or runtime storage is introduced.
The completed fixture's sole integrated sweep passed 105 Rust tests, structure
and format, but failed strict Clippy on one assertion spelling. An independently
reviewed equivalent spelling then passed the affected format/Clippy-only
follow-up. Original observations and passing tests were preserved without
numerical rerun; the prior learning-gap drive's FINAL-VERIFICATION.md records
independent acceptance.

## Four-commit predictor diagnostic

`predictor_commit_contracts` and its nested `predictor_commit_oracle` are a
separate cfg(test) diagnostic. The single named test is
`fixed_predictor_commit_diagnostic`. It uses the same fresh seed7 candidate and
36-byte synthetic prefix as the earlier learning-gap fixture. Production forward
and credit gathering must reproduce the four initial visits 3,4,1,4 and actions
3,0,2,forced-exit. They gather four detached feature/correction pairs without a
tile, shared embedding or history commit. Complete initial state/store identity,
zero store updates and zero source reads/advances are checked. The original
learning-gap fixture, its measurements and its production paths are unchanged.

Freeze the pairs in that order. Store both the once-clipped correction supplied
to train and its twice-clipped target, reproducing the two production clipping
calls rather than silently applying a third clip. After gathering, all tiles,
shared state, trace and credit scratch leave scope. Only the initial predictor
and four pairs survive. Exactly four sequential production Predictor::train
calls execute, one per pair, with no replay, tile/embedding update, local,
additional seed, optimization variant or adaptive loop. This isolates predictor
commits; it does not execute a complete production anchor or task update.

Each commit observes all four fixed pairs before and after. The report includes
same-pair and all-pair coordinate-mean MSE, signed changes, projection ranges,
prediction/target norms, exact f32 tanh-saturation counts, predictor version/hash
and a bitwise pair hash. Every measurement must preserve predictor identity;
the pairs remain identical across commits. Returned production precommit MSE is
checked bitwise against the same f32 reduction. Descriptive observation MSE is
accumulated in f64 from f32 outputs; it is labeled separately.

The oracle decodes one precommit matrix to i8 and computes an independent f64
continuous extension: dot product divided by sqrt(386), tanh, coordinate-mean
squared error against the fixed twice-clipped target. Its derivative does not
call production add_outer or apply_gradient. Eight coordinates are fixed in
advance: (0,0),(1,127),(32,128),(63,255),(64,256),(95,383),(126,384),(127,385).
They span both normalized feature segments, zero initial history and spatial
coordinates. All eight are checked at each commit; none is selected from its
observed error. Central differences use epsilon1e-4 and independently recompute
the perturbed row's contribution to full mean MSE. Other rows are constant and
cancel. Tolerance is 1e-10+1e-5*abs(analytic); zero/saturated coordinates remain
visible and are not counted as evidence of a nonzero derivative.

Reference predictions must agree with production within2e-5. Every coordinate's
reference gradient is compared with a separately reconstructed production f32
rule within1e-7+1e-4*abs(reference). The latter preserves multiplication order
for exact sign/underflow accounting, not an independent derivative oracle.
Every actual trit displacement must match the declared threshold-one bounded
sign step; changed counts and version increments must match production returns.
The report separates f64-reference and reconstructed-f32 gradient dot actual
displacement, zero gradients and nonzero gradients blocked at trit endpoints.
A negative dot product is a first-order statement, not a promise that the full
unit-trit step improves loss. No assertion requires learning improvement.

Work bounds are one initial forward of four tile visits, one credit gather of
four reverse visits, four predictor commits, eight observations of four pairs,
and four derivative observations. Observations use32 predict plus32 explicit
projection calls; derivative capture adds4 predict calls and training adds4,
for72 predictor matrix forwards. There are32 fixed central differences (64
perturbed row losses), four independent full reference projections and four
complete49,408-coordinate displacement scans. Hashing reads weights but does
not run extra model forwards. No files/processes, old weights or private inputs
are opened by this test; its final stdout record is captured by validation.

Gathering uses one in-memory seven-tile Store and one State, one forward/reverse
scratch at a time, a36x128 prefix, up tofour correction vectors and the existing
tied embedding gradient. This is not native one-tile residency or physical
admission. Commit phase retains12,352 packed predictor bytes,10,272 pair f32
payload bytes, one49,408-byte i8 snapshot and1,536 bytes of derivative vectors.
Production train additionally allocates its197,632-byte gradient,98,816-byte
Evidence and bounded128-coordinate target/prediction/derivative vectors.
Reports retain four bounded commit objects, each with eight finite-difference
rows and eight pair summaries, then one encoded JSON string. Vec headers,
allocator/container overhead and JSON representations are additional; these
payload counts are not RSS measurements. There is no float master matrix or
cloned predictor arm. One `PREDICTOR_COMMIT_DIAGNOSTIC ` JSON line must fit32KiB
including label and newline. Both independent review cycles and the sole
four-stage affected sweep are complete: structure, format, strict Clippy and
one named diagnostic passed. Exactly four train calls produced one preserved
record; the earlier 105-test evidence was reused without rerunning it.
Independent interpretation and final verification retain the diagnostic's
limited scope; no optimizer change or useful-model ETA follows.

## Bounded predictor update comparison

`predictor_update_contracts` and nested `predictor_update_rules` are cfg(test)
only. The single named test is `fixed_predictor_update_comparison`. It reuses
the preceding fixture's freeze, observation and identity helpers through test
visibility changes only; the old test body and arithmetic are unchanged. No
old diagnostic is called. Gather once from fresh seed 7 using the same 36-byte
prefix, four initial pairs, both clipping stages and four-visit route. Require
the full published pair and initial predictor hashes in the test. The gathered
Store, shared state, tape and credit buffers leave scope before the update arms.
Both begin from the identical initial packed predictor and present those four
frozen pairs once, in order. The arms run sequentially.

The historical baseline is the accepted four-commit observation artifact,
SHA256 73ed7b0207d397c4ae8857ddbf25807554da3a902cf12bd10cc46369bd76704b,
executed source 9949d00d9465c83c00708af767ee949b4ff4dc804171cbb25f2b30e6a2752c00,
and log ce3ade56131b56cf388cba4bfff2cd3373507427d84ca5c471ce1cc8d6cb3946.
Its four all-pair endpoint means are embedded as explicitly historical reference
values, not new executions. Require the baseline's resolved AVX2 kernel and
exact pair/initial hashes; change no environment setting. No baseline training
sequence, corpus, old weight, private input or file is opened by the test.
No tile/embedding/local commit occurs.

The dense guard makes exactly one production Predictor::train call on a
disposable packed proposal per presentation: four calls total. It supplies the
once-clipped correction so production performs the second clip. Every proposal
row is compared with the original row on the same twice-clipped target, using
squared error in f64 converted from actual f32 predict outputs. Only strict
decreases accept; ties reject. Rejection is reported, not called learning.
No gain search, backtracking or alternative proposal occurs: 512 row decisions.

The adjacent-coordinate arm makes no production train call. For each frozen
pre-presentation row, visit columns in ascending order and directions -1 then
+1, considering every legal adjacent trit move and the original row. Reference
base projection is the left-to-right f64 dot of i8 trits and f32 features
converted to f64, multiplied by f64 reciprocal sqrt(F). Each trial uses base
plus direction*feature*scale, in that f64 order, then tanh and squared error
against the fixed f32 target converted to f64. This declares reference rounding,
not production-kernel equivalence. Nonfinite inputs/results fail the fixture.

The original row starts as best; only strict reference improvements replace it.
Thus ties prefer no move, then lowest column and -1 before +1. At most 772
signed trials occur per row and 395,264 across the arm; 512 no-move candidates
are counted separately. Chosen rows form one ternary proposal. Actual pinned
projection/tanh recomputes them, accepting only strict same-pair improvement;
otherwise retain the original row. No runner-up is tried. At most one trit
changes per row and 512 across the arm. All coefficients remain eligible;
there is no retained mask or optimizer state.

Both rules share a test-only merge selecting whole rows by trit index. F=386
has shared bytes between packed row boundaries, so byte-slice row copying would
be invalid. The merged proposal is canonical before replacing current state;
its actual predictions must equal each selected original/proposal row bitwise.
Every row must satisfy actual non-increase. Transitions count actual trit
differences, never proposed counts; version increments once iff changes exist.
A stalled presentation preserves predictor identity. Pair hashes and immutable
measurements are checked throughout. This is not a production update API.

One record contains all eight before/after comparisons: same/all-pair MSE,
signed deltas, projections, norms, versions/hashes, accepted/rejected rows,
proposed/actual moves, reference trials, stalls and aggregate worsening. Report
final-minus-initial aggregate MSE even when positive. Assertions require finite,
canonical, count and identity correctness and the defined current-pair acceptance
rule, not favorable aggregate learning. No text-learning, assistant-quality,
E3-cause or useful-model ETA claim follows from a passing fixture.

Four pure nontraining microchecks run inside the same named test: zero-feature
no-move ties, equal-improvement lowest-column ties, NaN input rejection, and a
two-row/six-column merge selecting row one while retaining row two. The merge
crosses a shared packed byte and checks every trit plus unchanged inputs. The
two tie cases evaluate eight legal signed moves total; the invalid case rejects
before candidate work. These checks are separate from eight fit presentations
and add no production train/predict call or third model arm.

Work bounds: one gather (four forward and four reverse visits, whose existing
reverse recomputes its bounded cache), eight presentations, four production
train calls, zero baseline train calls, 16 all-pair observations, and at most
395,264 fitting reference trials plus eight microcheck trials. Observations
make 64 predict and 64 explicit projection calls; eight guards make 24 predict
calls and proposal training adds four, for 156 predictor matrix forwards total.
Ranking uses 512 independent row base projections plus scalar trials, not full
matrix forwards per trial. Proposed-transition reconciliation and merging each
scan 49,408 trits once per presentation. Every loop has these fixed bounds.

Gathering uses the existing in-memory seven-tile Store and one State, not native
one-tile residency or physical admission. Afterwards keep one immutable initial
predictor and one current arm. A proposal and new merged matrix briefly coexist:
at most four 12,352-byte packed predictors/matrices. There is at most one
49,408-byte i8 assembly vector, 10,272 bytes of frozen pair payload, three
128-wide f32 prediction vectors and 128 row-acceptance booleans. Dense proposal
training has its existing 197,632-byte gradient, 98,816-byte Evidence and bounded
target/prediction/derivative scratch, released before assembly. Coordinate
ranking uses f64 scalars, no float master or persistent optimizer state. The
microchecks precede gathering and release their tiny buffers first.

Reports retain two arms of four presentation objects, 64 pair summaries and
four historical reference numbers, followed by one encoded JSON string.
Container/allocator/serialization overhead is additional, not measured RSS.
One `PREDICTOR_UPDATE_COMPARISON ` line must fit32KiB including label and newline.
No process, CLI, controller or runtime file is introduced. Both independent
review cycles and the sole affected sweep are complete: structure, format,
strict Clippy and one selected test passed with 64 filtered out. Eight fit
presentations and four pure checks produced one preserved record; independent
interpretation and final verification accepted the bounded evidence. Dense
improved direction then stalled three times; coordinate raw fit improved but
remained negatively aligned with all four targets. Earlier 105-test and
four-commit evidence was not rerun. See the
[final verification](../../../nonmonotonic-training-lab/work/f-predictor-update-drive/FINAL-VERIFICATION.md).
The separate fixed 64-presentation
[reachability scope](../../../nonmonotonic-training-lab/work/f-predictor-reachability-drive/NODE-ZERO.md)
has completed its sole affected sweep and independent final verification.
Its fixed fitting gate failed; the candidate is blocked for now. No production
rule or campaign is adopted.

## Fixed predictor reachability fixture

`predictor_reachability_contracts` is nested under the existing cfg(test)
`predictor_update_contracts` module. The single new test is
`fixed_predictor_reachability`. Nesting gives access to the unchanged adjacent
rule and existing freeze/observe/identity helpers without expanding visibility
or copying the optimizer. No old test body, helper arithmetic or production
behavior changes. The earlier tie/nonfinite/packed-row microchecks are retained
evidence and are not repeated here.

Gather once from fresh seed 7 at candidate width 128, FFN 384 and context 1024,
using the same 36-byte prefix, initial four-visit route and four frozen pairs. Require AVX2
without changing its environment. Require the complete initial predictor and
pair hashes from the preceding comparison. The initial Store, shared state,
tape and credit scratch leave scope before this fixture's update loop. Move the
single initial predictor into the loop rather than retaining a cloned arm.

Run exactly 16 complete ordered passes, each presenting all four pairs once:
64 presentations including the four needed to reconstruct the previous endpoint.
The previous artifact retained hashes, not resumable weights. After pass 1 require
predictor hash f9fd1f11a3fbad57533b99618bc05dd5c6a217a61cfb5192d1b7a0ae23b8b425
and version 4. The report binds prior observations
0c15ccc3d161926853a94f3c5742e732fcc67c31e7956bbf619c5a7fc056c43f;
the test opens no artifact or private input. There is no dense arm, old
unguarded sequence, additional seed, zero initialization, rule adjustment,
outcome-dependent extension, early learning stop or retry.

Each presentation calls only the unchanged `Rule::Adjacent` implementation.
Its deterministic f64 scalar candidate ranking and actual-kernel row acceptance
remain as documented above. All coefficients stay eligible. Existing guard
assertions check actual same-pair row non-increase, exact merged predictions,
canonical packed assembly and actual transition counts. This fixture also checks
128 accepted/rejected rows and 128 no-move candidates per presentation, one
transition per accepted row, version increment iff transitions exist, unchanged
identity on stalls, and unchanged pair identity. Stalls are counted as stalls.

Measure at initial state and after every complete pass: 17 checkpoints. Reuse the
existing immutable raw-fit observation, retaining per-pair MSE and its all-pair
mean. Separately call the actual production `Predictor::correction` for every
pair. From its returned f32 vector, accumulate squared error, target dot and
squared norms in f64 against the same frozen twice-clipped f32 target. Divide
squared error and the target's squared norm by 128 to obtain delivered MSE and
matched zero-correction MSE. The zero value is arithmetic, not a zero-initialized
model. Compute cosine from these f64 sums; serialize null if either norm is zero.
Validate finiteness, vector shape and clipped norm bounds. These are captured
delivered-vector measurements, not the preceding ideal clipped estimates.

Each checkpoint scans all 49,408 trits and records counts in [-1, 0, +1] order; their
sum must match matrix length. Measurements preserve complete predictor identity.
The report gives initial state plus 16 pass records with start/end predictor
hashes, versions, candidate/row/transition/stall counts, raw-fit summaries and
delivered metrics. Adjacent pass hashes must join; version changes must equal
the number of nonstalled presentations. Canonical counts and hashes describe
reported states, not saved packed checkpoints or resumable state.

The declared final gate is a boolean: at presentation 64, every pair has positive
delivered target dot and delivered MSE strictly below its matched zero MSE.
Also report the same boolean at each checkpoint without using it for selection.
Never assert that the gate passes or use it to control work. A passing test means
the declared invariants and bounds held, including when the fitting gate fails.
Even a passing final gate establishes only these frozen pairs' reachability;
it is not useful local-update evidence, text quality, generalization or a finite
useful-model ETA. No production optimizer, protocol or campaign is adopted.

Work bounds: one initial gather with four forward and four reverse tile visits
(the existing reverse recomputes its bounded cache); 64 adjacent presentations;
zero `Predictor::train` calls, tile/embedding commits, local opportunities or
microchecks. At most 6,324,224 legal signed scalar trials and 8,192 actual adjacent
transitions; 8,192 no-move candidates are counted separately. Ranking performs
8,192 row base projections. Each presentation's proposed-transition check and
merge each scan 49,408 trits; each checkpoint's canonical scan does likewise.
The 17 raw observations make 68 predict and 68 explicit projection calls; 68 actual
correction calls each add one predict. Existing 64 guards make 192 predict calls.
Thus there are exactly 396 predictor matrix forwards, distinct from scalar trials
and the initial tile gather. Hashing adds reads but no numerical forwards.

Gathering uses the inherited in-memory seven-tile Store and one State; it does
not claim native one-tile residency or physical admission. Afterwards retain one
current predictor and 10,272 bytes of frozen f32 pair payload. Proposal/assembly
can briefly coexist with current state: at most three 12,352-byte packed matrices
and one 49,408-byte i8 assembly vector. Guard scratch has at most three 128-wide
f32 prediction vectors and 128 row-acceptance booleans. Correction measurement
briefly has raw/returned 128-wide f32 vectors; raw observation has prediction and
projection vectors. These phases are sequential. There is no dense gradient,
Evidence, float master or persistent optimizer state in the update loop.

Retain 17 scalar measurement summaries (68 delivered-pair and 68 raw-pair MSE
summaries), 16 pass count records and bounded hashes. JSON conversion may
temporarily overlap those records and the encoded string; allocator/container
overhead is additional, not measured RSS. One `PREDICTOR_REACHABILITY ` record
must fit 32 KiB including label and newline. No files, CLI, process, controller or
runtime write is introduced. Both independent review cycles and the sole
structure/format/Clippy/new-test sweep are complete: one test passed with 65
filtered out. Exactly 64 presentations produced 4,202,152 signed trials, 8,184
transitions, eight rejected rows and no stalled presentations. The declared
final gate failed: all four target dots were positive, but pairs one and two
remained worse than their matched zero corrections. All 17 checkpoint gates
were false. Independent interpretation and final verification accepted the
record with zero material findings. Mean delivered error below mean zero does
not replace the every-pair gate. See the
[final verification](../../../nonmonotonic-training-lab/work/f-predictor-reachability-drive/FINAL-VERIFICATION.md).
The candidate is blocked for now, with no automatic extension, new seed,
initialization, production rule or campaign adopted. Earlier diagnostic
observations and passing test bodies were retained without rerun. A separate
[joint-target scope](../../../nonmonotonic-training-lab/work/f-predictor-joint-drive/NODE-ZERO.md)
completed 64 joint updates and 256 pair contributions from the original
initial state. Its software checks and fixed every-pair gate passed;
independent final verification accepted the record. This does not extend
the closed online candidate. Separate production integration is now adopted
for development, as recorded in the joint fixture's completion status below.

## Historical fixed joint-target predictor fixture (v3 adaptation above)

`predictor_joint_contracts` is a new cfg(test) child of the existing comparison
module; `predictor_joint_rules` is its private ranking/guard helper. The selected
test is `fixed_joint_predictor_fit`. Reuse the unchanged packed-row merge,
freeze/identity helpers and actual-delivered measurement arithmetic. Only the
measurement type, function and four fields gain parent-level test visibility;
all prior test bodies and production arithmetic remain unchanged. The report
adapter renames the shared measurement ordinal from `presentations` to
`joint_updates`; it changes no value or numerical operation.

The one initial read-only gather uses fresh seed 7, the original candidate
predictor, fixed 36-byte prefix, AVX2 route and four pairs already described
above. Require the full published initial predictor and pair hashes. Release
the Store, shared state, tape and credit scratch before fitting. The historical
online64 observations are identified by SHA256
8f970e14dc9590252f39101a999c6d6f2cc40b7caeede281d47f571cac104609
and explicitly marked unexecuted. No old fitting sequence or file is opened.
There is no old four-presentation endpoint constraint for this new rule.

Execute exactly 64 joint updates, grouped into 16 reporting blocks of four.
Every update uses all four frozen pairs: 256 pair contributions versus the
historical 64 online presentations. The maximum row-move budget is equal;
compute and target exposure are not equal. All coefficients remain eligible.
There is no extra arm, seed, initialization, tuning, early stop or extension.

For each frozen pre-update row, compute four f64 base dot products in ascending
column order, multiplying each by reciprocal sqrt(F). A candidate is one legal
adjacent trit movement. Its projection for pair j is base_j plus
direction*feature_j[column]*scale in that order. Rank the mean of the four
raw-tanh squared errors against the fixed twice-clipped f32 targets converted
to f64; reduce in frozen pair order. Start with no move and replace only on a
strict decrease. Thus equal minima keep no move, otherwise lowest column then
-1 before +1. Reject nonfinite or malformed input. No runner-up search occurs.

Build one canonical ternary proposal. Compute actual production predictions for
all four pairs on current/proposal state. Accept a row only if its four-pair
mean raw squared error strictly decreases under f64 accumulation. Merge whole
rows through the unchanged trit-index helper, preserving shared packed bytes.
Recompute all four predictions after assembly; each row must equal its selected
original/proposal output bitwise for every pair, with joint row non-increase.
Count actual differences, assert one transition per accepted row and adjacent
moves, and increment version once iff any transition occurs. A stalled update
preserves its complete predictor identity.

Joint row acceptance does not guarantee each pair improves. Using the actual
current/merged raw predictions already available, count per-pair MSE increases
at each update. Aggregate these raw regression counts per block and overall.
Report raw per-pair MSE deltas between block checkpoints as well. These counts
are not measurements of intermediate delivered-correction errors; actual
delivered metrics are measured only initially and after every four updates.

Reuse all 17 immutable measurements: raw per-pair/all-pair MSE, actual f32
production correction vectors summarized with f64 MSE/dot/norm/cosine, matched
zero error, canonical trit counts, versions and hashes. Check pair identity
throughout and join every block start/end hash and conditional version count.
The checkpoint adapter labels joint update numbers explicitly; pair exposure is
reported separately. No summaries are represented as resumable weights.

The final gate remains every pair's positive delivered target dot and delivered
MSE below its own zero baseline at joint update 64. All checkpoint gates are
reported, never asserted favorable or used to select/extend work. A software
pass can accompany a failed learning gate. Raw joint acceptance and the
delivered gate are distinct; neither establishes useful local learning, text
quality, equal-cost superiority, E3 causation or a viable-model ETA. No
production optimizer, protocol, campaign or private evaluation is adopted.

One pure aggregation check runs before gathering. A single zero trit sees four
identical one-coordinate features [1], with targets [-1, +1, +1, +1]. Require
the joint choice +1, despite the first target favoring -1. This adds two legal
signed candidates, eight candidate pair-error terms, four baseline pair-error
terms and four one-coordinate base projections; no predictor forward or train
call. No prior tie/nonfinite/row-assembly microcheck is repeated.

Fitting bounds: 64 updates, at most 6,324,224 legal signed candidates and
25,296,896 candidate pair-error evaluations; 8,192 no-move row candidates,
32,768 row-base projections and 32,768 baseline pair-error terms. At most 8,192
actual adjacent transitions occur. Candidate terms exclude the separately
reported pure check, actual-kernel guard reductions and measurements. Each
guard makes four current, four proposal and four merged predict calls:
768 forwards. The 17 shared checkpoints add 68 raw predict, 68 explicit project
and 68 production-correction predict calls: 204 forwards. Total is exactly
972 predictor matrix forwards and 68 actual correction calls, plus the inherited
initial gather of four tile forward/reverse visits and its cache recomputation.
Production `Predictor::train`, tile/embedding commits and local opportunities
are all zero. Proposed reconciliation/merge and canonical scans stay bounded
by 49,408 trits per scan; no model forward is hidden in identity hashing.

Gathering still uses one in-memory seven-tile Store and State, not native
one-tile residency or admission. Afterwards retain one current predictor and
10,272 bytes of frozen pair f32 payload. At most three 12,352-byte packed
matrices and one 49,408-byte i8 assembly vector coexist. The guard can retain
twelve 128-wide f32 predictions (6,144 payload bytes), 128 f64 old-row errors
(1,024 bytes) and 128 row booleans. Ranking uses at most three four-element f64
projection arrays including the by-value error argument, plus scalars, not a
float master, dense gradient, Evidence or persistent
optimizer. Measurement scratch is unchanged and sequential with updates.

The report retains 17 checkpoint JSON objects, 16 block count/delta records,
bounded identity strings and at most two typed scalar measurements during a
block transition. Conversion may overlap these with an aggregate JSON value
and the final encoded string. Container/allocator overhead is additional, not
measured RSS. One `PREDICTOR_JOINT_FIT ` line must fit 32 KiB including label/LF.
No process, CLI, controller or runtime write is added. Both independent review
cycles and the sole structure/format/Clippy/new-test sweep are complete:
one selected test passed with 66 filtered, in 4.19 seconds of harness time.
Exactly 64 joint updates produced 2,288 adjacent transitions, 25 stalls and
final version 39. The final every-pair gate passed; its first measured pass
was at update 16. Independent interpretation and final verification accepted
the record with zero material findings. Earlier diagnostic evidence remains
unchanged and was not rerun. See the
[interpretation](../../../nonmonotonic-training-lab/work/f-predictor-joint-drive/DIAGNOSTIC-INTERPRETATION.md)
and [final verification](../../../nonmonotonic-training-lab/work/f-predictor-joint-drive/FINAL-VERIFICATION.md).
The separate [production integration](../../../nonmonotonic-training-lab/work/f-joint-production-drive/NODE-ZERO.md)
has completed its own reviews, software checks, fixed-input parity and
release build, as recorded in the current v3 section and its final verification.
That closure does not convert this historical fixed-target fixture into
dynamic-learning or quality evidence. E4 subsequently ran under the unchanged
110-minute owner and 120-minute whole ceilings, ending at a resource stop
without final evaluation. It remains closed; no finite useful-model ETA or
continuation is authorized. A fresh-run proposal requires new user runtime
authorization and exact program-STOP handling. This historical technical
fixture contract remains unchanged.

## Ledger and amortized verification

Pattern: amortized verification, i.e. a running ledger with a boundary
cross-check. `DiskStore` keeps an exact `(bytes, files)` ledger of its owned
persistence directory (`storage_ledger.rs`), updated on every successful
publish and every hash-verified retention removal. Per-write budget, file-count
and peak checks use the ledger instead of rescanning the directory; IoCounters
values are unchanged because the ledger equals what a rescan reports. Any write
or removal error drops the ledger, so the next use rebuilds it from a full
scan. `reconcile` rescans at store creation, at the start of every checkpoint
publication and at retention start/end; a mismatch poisons the store and fails.

`data_guard.rs` hashes each Dataset source in full on first use (open) and
records its canonical path, length, modified and created times. Later span
reads recheck the canonical path and metadata before and after a bounded direct
read; any difference trips a full re-hash, which fails closed on mismatch.
`TrainingSource::verify_boundary` re-hashes every verified source in full at
each checkpoint publication, and `Dataset::open` does so once after its scan.
Tile/shared/predictor blob SHA256 on load/update, content-addressed publication
with the commit marker last, checkpoint identity and all numerics are unchanged.

Not guaranteed between checkpoints: a foreign file added to or removed from the
persistence directory, or a source replaced or edited while preserving length
and both timestamps, is detected only at the next boundary (or the next tripped
guard), not at the moving step. Opportunities completed before that boundary
may have read such content. Tile blob payloads remain hash-checked on every load.
Between boundaries the disk budget and FILE_COUNT_CAP are exact for owned writes
only; foreign bytes/entries can exceed them until the next reconcile poisons.
On Windows NTFS, LastWriteTime lags while a foreign writer holds the file open
(and for memory-mapped writes), so such a same-length edit passes the guard
until the boundary re-hash. Stamp carries no file identity, so a different file
swapped in with equal length/mtime/creation time also passes until then. After
a write/removal error the ledger is dropped and rebuilt from disk by the next
ledger use or reconcile without poisoning; in production every such error is
fatal anyway. A failed reconcile scan poisons like a mismatch.
`profile.rs` timers run in production builds too (two Instant reads per call,
thread-local, single-threaded; per-call cost not separately measured); only `snapshot`/`since` are test-gated and the totals are not
part of the worker result schema. `persistence_profile.rs` is the ignored
bounded harness that prints them over real steps with a real DiskStore.

## Numerics profile and the tuning harness

`tuning.rs` holds the numerics profile: `Knobs::PINNED` (v3/assay numerics) and
`Knobs::core` (`§Core path`). Every build reads a thread-local; production sets
it only through `core_train::run` -> `install`, so other runners stay pinned.
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