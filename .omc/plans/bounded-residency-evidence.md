# Bounded-residency gap and next implementation gates

Inspected 2026-11-02 while the pinned Stage 4 rate scan runs. This is an
engineering evidence note, not an implementation or a claim of large-model fit.
The owner's original variable v1 history behavior remains the current contract.

## Current barriers

1. `src/causal/geometry.rs` fixes `TILES` at seven and validates only fixture
   and candidate shapes. The 81M/287M configurations are separate planning
   inputs, not executable causal geometries.
2. `src/causal/resident.rs` owns every decoded tile. In the pinned scan binary,
   `core_batch::commit` and `core_route` access that vector directly. The
   working tree routes those accesses through `TileStore` and passed the
   integrated sweep, but still needs independent review and measurement.
3. `tuning_update.rs` retains the latent weights and both Adam moments for
   every initialized matrix in thread-local `LATENTS`. Paging only packed tile
   weights would leave optimizer memory proportional to visited parameters.
   The global draw counter, per-matrix Adam steps, tied embedding latents and
   `core_schedule::tile_steps` exploration metadata must retain their semantics.
4. `core_checkpoint::save` clones all latents, serializes complete tile and
   latent payloads to vectors, then writes them. `load` reads each whole file
   and reconstructs all tiles and latents. Thus checkpoint and restore peaks
   remain proportional to the complete state even with a paged execution path.
5. The pinned scan's `core_gradients::mean` collects one full `TileGradients` per position,
   including zeros for positions that did not visit the tile. For the current
   largest tile, 282,368 coefficients times 4 bytes times batch 32 is 34.47 MiB
   of gradient payload alone. Four position gradients plus one accumulator
   would be 5.39 MiB, excluding scratch, allocations and other state. This is
   arithmetic from shapes, not a measured peak or a complete memory ledger.
   The working tree now contains an unverified chunked reduction; the pinned
   study binary still uses the original full-batch collection.

The archived `archive/src/causal/storage.rs` is useful prior art for a
one-live-tile `TileStore` with immutable hash-addressed blobs and a poisoned
update path. Its tile blob encodes packed tile state, not the current v2
thread-local latent weights and moments. Restoring that store without making
optimizer state part of the owned tile generation would not bound training
memory or establish atomic weight/optimizer recovery.

The test-only paged prototype bounds the number of decoded tiles, but its
scratch allocations include more than one copy of a tile's optimizer payload.
The latest unrun prototype streams the optimizer values on load and publish,
retaining a packed tile buffer and decoded tile/optimizer state, and avoids
the previous full optimizer serialization in `enter`: reads now compare an
incremental hash of the same encoded values and updates do not make that copy.
Independent inspection found no definite issue in those changes; test and measure
them before claiming savings. A one-tile residency counter does
not capture these buffers, shared embedding latents, worker gradients, history,
filesystem cache or a full-state parity snapshot. Before promotion, measure
those peaks by phase in separate processes and remove or bound redundant copies;
compare bytes transferred and throughput as well as process working set and
private commit. A fixed decoded-tile count alone is not a memory-floor result.

### Version snapshot scaling investigation

`History::Record`, `execution::Trace`, `State` and checkpoint records currently
carry `[u64; TILES]` version arrays. Generalizing tile count naively makes
history and per-position trace metadata grow with tile count even if only one
weight/optimizer tile is resident. The existing history rule must still keep
the owner's original age <= 8 ticks and per-component version lag <= 1.
Within a fixed-weight batch, each `Trace` receives the same step-start version
snapshot; sharing one immutable snapshot across positions is a separate,
lower-risk way to avoid B copies, subject to lifetime, mutation and parity
checks. It does not eliminate the O(tile-count) snapshot itself.

Candidate representation for future review, not implemented: assign a strictly
increasing event epoch to each actual tile or shared-version increment. Keep
the last increment epoch per component and a global cutoff equal to the maximum
previous-last epoch encountered when any component increments. A record stores
the epoch of its version snapshot. Under monotonic versions and complete ordered
event accounting, a component advanced at least twice after the snapshot iff
the cutoff is greater than its epoch. Age pruning remains separate. This avoids
copying all versions into every history record, but still needs O(tile-count)
last-event/version metadata and does not make total RSS constant.

`work/history_epoch_probe.py` exhaustively compared this criterion for three
components (two tiles and shared E), every monotonic increment sequence through
length six, and every snapshot:1,093 histories and25,971 comparisons, zero
mismatches. This is a small mathematical check, not a proof of integration,
performance, arbitrary state corruption handling, or checkpoint compatibility.
Version decreases/future snapshots and epoch overflow must fail closed. Existing
version-array checkpoints lack the event sequence; a safe conversion or a new
identity/protocol and explicit checkpoint policy is required. No history
semantics have been changed, and the owner deferred a one-anchor window.

### Candidate batch publication and recovery design (unimplemented)

The prototype publishes each tile during `core_batch::commit` and removes its
previous blob immediately. A crash between tile updates, or before shared state
and the cursor are committed, can therefore leave a mixture of batches with no
recoverable prior tile. Its in-memory poisoned state is not crash recovery.

A future store can use immutable generations and one publication point per
complete batch. At batch start, retain the last committed manifest and stage
updated tile blobs separately; each blob owns both packed weights and that
tile's latent weights, Adam moments, versions and step metadata. Preparation,
gradient calculation and every update read use step-start tile generations,
even after earlier tile writes are staged; existing v1 history rules remain in
force. Staged outputs become visible as committed state only together.
After every tile update, the shared update, counters, history, cursor and
boundary validation succeed, write the complete state record, shared weights,
shared optimizer state and draw counter, plus predictor state when applicable,
into a new uniquely named partial generation. Its manifest records the run
identity, protocol, anchor count, hashes of all state files, and an entry for
every tile identifying its immutable blob, hash, version and optimizer steps.
Unchanged tiles may reference blobs from earlier generations. Sync all new
files before publishing the fully written manifest and rename the partial
directory to a previously unused generation name as the single commit point.
No live generation is mutated, and a failed batch must poison the process and
recover from the last published generation rather than continue in memory.
Recovery restores preparation-time mutations, including history, counters,
cursor and the global draw counter, from that same committed generation.
If rename reports an uncertain outcome, rescan and verify the destination and
its manifest before deciding whether publication succeeded; do not blindly
retry a potentially completed rename or create another commit for the batch.

On restart, ignore incomplete directories and preflight the identity of every
readable published manifest before any quarantine, fallback or other mutation,
matching the existing checkpoint behavior. Then inspect published generations
in descending sequence order and validate the complete
manifest, state and every referenced tile before installing optimizer state or
resuming. Reject malformed names, duplicate tile IDs, missing references,
path escapes, mismatched versions and counters, invalid hashes and incompatible
geometry. Preserve the existing distinction between foreign identity, integrity
failure and transient I/O: do not quarantine foreign or temporarily unreadable
data. Integrity failures may be quarantined and recovery may fall back to an
earlier verified generation; report its recovered anchor explicitly and honor
the existing owner/resume policy for any rollback. If none is valid, fail
closed instead of silently restarting training. The generation protocol must
be distinct from existing
checkpoints unless an explicit, verified conversion is provided.

Keep each blob until no retained published generation references it, including
the previous verified recovery point. Garbage collection must run only after a
new generation is published, use validated references, and tolerate interruption;
orphans and partial generations are safe to collect only when no reader or
retained generation can need them. Protect blobs referenced by quarantined or
foreign generations and active readers, or defer collection until ownership is
resolved. Unique directory names avoid relying on
Windows rename-over-existing behavior. Same-volume rename provides the intended
process-crash visibility boundary only where the target filesystem supplies the
required atomicity. File sync plus rename alone does not establish power-loss
durability: parent-directory metadata persistence must be supported and tested
on the target filesystem, or that stronger guarantee must remain unclaimed.

Implementation will need transaction hooks around `core_batch` and `TileStore`,
a generation manifest and bounded streaming reader/writer, and recovery and
garbage-collection code; the existing resident checkpoint path is not a paged
transaction. Focused tests should inject interruption before and after every
blob, state, manifest and rename boundary; verify old-or-new whole-batch state
and deterministic continuation against resident execution; exercise corrupt,
missing, foreign and transient-I/O cases; and interrupt garbage collection while
retaining an earlier recovery point. Measure actual memory and I/O separately.

## Proposed implementation sequence, subject to independent review

- Bound gradient retention with ordered batches of at most four position
  gradients and one accumulator. Reduce in original position order, use the
  full batch denominator, preserve the batch-one arithmetic path and worker
  error behavior, and verify bitwise parity for sequential/threaded batches,
  revisits, unvisited positions and checkpoint continuation. The working-tree
  implementation passed the integrated contracts. Subsequent independent
  inspection found no definite parity regression in reduction, routing or the
  resident store; changes after that sweep still require integrated validation.
  Measure throughput and peak process memory in a separate bounded comparison
  after the rate scan; the chunked path can create more short-lived threads. Do not
  infer physical savings from payload arithmetic alone.
- Separate each tile's packed weights, latent weights, moments, versions and
  step metadata into one storage-owned unit; keep shared embedding state and
  globally required metadata explicit. A fixed-capacity store must account for
  decoded tile, optimizer, temporary gradient, forward tapes, history, and any
  simultaneously live worker scratch. This must cover update, evaluation and
  float-forward modes or reject unsupported combinations explicitly.
- Preserve the fixed-weight batch transaction: route/credit preparation reads
  step-start weights; tile-major commits occur only after all positions are
  prepared. Failed or interrupted writes cannot leave a checkpoint that mixes
  old and new tile/optimizer generations. Design recovery before implementing
  eviction; test failures between every publication boundary.
- Stream checkpoint encode/hash/write and restore under a bounded buffer, with
  immutable generation references and atomic manifest publication. Validate
  corruption, cross-run identity, stale/incomplete generations and deterministic
  continuation. Account separately for the OS file cache and owned RSS/commit.
- Generalize geometry/topology only after small deterministic resident versus
  bounded-store parity and resource measurements pass. Enforce explicit
  numerical, resource, ownership and checkpoint reviews before considering any
  81M/287M execution. Those profiles remain planning-only today.

## Evidence still required for the full objective

The v1 eight-hour TinyStories runs had looping greedy completions and the
five-gram baseline scored better. The v2 rate scan has no final score yet.
Useful held-out language completions and a measured memory floor as parameter
count grows are therefore both unproven. A fixed tile cache alone does not
establish either outcome: quality, routing coverage, total training I/O,
checkpoint/restore peak and actual resident/commit memory all need bounded,
reproducible measurements.
