# Causal persistence and restart contract

Storage/checkpoint mechanics now serve causal-prefix-v3. Its identity rejects
v1/v2 numerical continuation; the unchanged causal-persistence-v1 directory marker
identifies the storage format and ownership only.
It does not admit a model, own processes or prove machine-resource fit. All
training must still pass the independent numerical/resource/owner review and the
single integrated end-of-edit sweep. Earlier fixture receipts retain their source
identities; the v3 counter/identity extensions are authored and not yet run.
Batch/pair conservation and exact float bits are checked by State restoration.
Missing added counter fields are not defaulted or migrated.

## Storage and binary encoding

`DiskStore::create(workspace, attempt, geometry, seed, disk_budget_bytes)` requires
an already-existing absolute attempt directory inside canonical workspace/work.
It rejects symlink/reparse ancestors and creates a fresh `attempt/persistence`
directory. The owner may already have request/log files outside that directory.
The persistence directory has an OWNED marker, a 4,096-entry ceiling, 128 KiB
per-file ceiling and an explicit 1 MiB..2 GiB byte budget. Those are storage
limits, not permission to allocate or run a model at the maximum.

`TileStore::read/update` holds one decoded tile within the synchronous closure.
Returned values cannot borrow its weights. Callers must not manufacture and
retain additional model copies inside a closure. `versions` is unavailable
during a live tile or after an update error. Update validates identity and exact
version change against changed packed weights. A failed update poisons this
store; recovery uses a previous complete checkpoint in a fresh attempt.

Immutable packed objects use SHA256 content names. The binary header records
magic/version, exact geometry, object kind, tile ID, weight version, canonical
evidence tag and matrix count. Each matrix carries checked rows/columns/byte
length, followed by packed trits in numerical registration order. Exact total
length, canonical trits/padding, kind, geometry, ID/version and SHA256 are checked
on decode. The threshold-one evidence rule is exactly
`threshold-one-canonical-zero-v1`; it represents complete known-zero evidence,
not discarded optimizer state. Different evidence rules fail closed.

One serialized object plus decoded packed matrices and the bounded constructor
copies fit inside the adopted conservative tile I/O reserve. Matrix payloads
are not JSON integer arrays. Shared E and predictor use the same binary format
with different object-kind tags and shapes. No parent weights or data are
implicitly imported: restore copies only a user-selected owned checkpoint's
verified immutable tile objects into the new owned attempt.

`IoCounters` are attempt-scoped, not cumulative exposure. They count direct store
blob loads, imported tile loads and immutable writes (including checkpoint
metadata writes), bytes through those operations, and one decoded tile at a time.
Checkpoint/release reinspection and filesystem metadata reads outside the store
are validation overhead and are not a complete physical disk-traffic trace.
Report that scope; do not label these counters measured device I/O. Current disk
bytes are file-length accounting; peak includes a conservative temporary plus
published-link allowance. Shared hard-link extents can be double-counted, and
filesystem metadata/OS caches are outside this bound. Owner wall time must charge
all validation, copying, checkpoint and retention work separately from training.

## Publication transaction

`checkpoint::publish(store,state,identity,owner,source)` accepts only complete
`State::Ready`. It revalidates cursor/source, schedule/exposure counters,
geometry, seed, every current tile payload, version references, history and
immutable identities before publication. It rejects InFlight and Poisoned state.

All payloads are synced immutable files. A bounded 64 KiB continuation record
pins exact cursor/masks/order/epoch, counters, seed/global opportunity, baseline,
shared/predictor versions and history ordering/dependencies. Floating state uses
IEEE bit fields, including both aggregate f64 metrics; it does not depend on
JSON float-parser rounding. The manifest is at most 16 KiB and binds every
payload length/hash plus Identity, OwnerEnvelope and a snapshot of I/O counters.
That I/O snapshot precedes its own manifest/head writes.

Files are written to uniquely named, create-new temporary files and synced.
Publication uses a same-directory hard link to a new immutable final name;
it cannot replace an existing file. Existing names are accepted only if bytes
match exactly. A filesystem without supported same-volume hard links fails
closed. This works on supported Windows filesystems without claiming that an
overwrite rename of an existing latest pointer is atomic.

The final transaction marker is `commit-{sequence:020}-{manifest_sha256}.json`,
containing the exact CheckpointRef. It is published only after the manifest and
all payloads. There is no mutable latest pointer. `latest` chooses the greatest
unambiguous committed sequence; malformed/duplicate markers fail closed. Orphan
payloads/manifests left before the marker never become resumable. Corruption of
the newest committed generation is an error, not permission to silently fall
back and lose exposure. OS/filesystem power-loss durability of directory entries
is not guaranteed by std file sync; missing/corrupt dependencies fail closed.

## Owner, release and strict resume

Exact schemas are `checkpoint_owner.rs` and `checkpoint_state.rs`, shared with
the Python owned supervisor. Identity pins protocol/config/source/runtime/data
hashes, geometry, resolved kernel, evidence and RNG policy. The source/kernel
identity is immutable; CPU implementation changes cannot silently resume old
state. OwnerEnvelope pins run/attempt/parent lineage, cumulative opportunity,
wall and attempt budgets, per-attempt disk cap, charged usage, observer sequence,
STOP category and unknown-physical-work status.

Consumed opportunities are conservative physical opportunity charges, including
known replay; they may exceed logical `global_opportunity`. They never decrease
on restart. Unknown physical work and failed outcomes are not recoverable.
Checkpointing a Ready numerical state after budget exhaustion is permitted only
with terminal BudgetStop; that checkpoint is preserved but cannot resume under
the exhausted budget. Wall time spent in cooperative cleanup remains charged.

`checkpoint::restore(workspace,parent_attempt,reference,new_attempt,identity,
request,source)` requires the parent's latest exact committed reference. It
requires hash-pinned immutable `release.json`, a matching `release-ready.json`,
absence of `release-invalid.json`, and a separate externally pinned
`release-seal.json` after the exact owner process has exited successfully. The
seal binds release hash, owner PID/creation, external finalizer PID/creation,
owner reap/exit0, source/runtime pins and whole-owner wall time no greater than
the conserved release charge. A self-published ready marker alone is insufficient.
The one-use claim lives outside the read-only parent at
`workspace/work/causal/<run_id>/claims/<releaseSHA>.json`, binding release to the
new attempt ID. The owner atomically publishes that claim; Rust reads it and
never writes or removes parent files. Seal/release/claim checks repeat before
restore returns. The Python owner also checks late revocation before admission.

Release must prove exact child reap and observer acknowledgement, known physical
work, consistent latest checkpoint, monotone usage/observer sequence and a
completed, user-stop or recognized resource-stop outcome. Failed/budget-stop
outcomes remain terminal. User STOP requires explicit renewal. Fresh owner
budgets must be identical, counters must include release-time cleanup costs,
attempt count increments once and observer sequence advances. An owner release
is evidence produced by the supervised process/observer path; these JSON
booleans alone do not independently inspect the OS or confer launch authority.

Restore verifies bit-exact state, payload hashes/shapes, all cursor/source and
counter/history invariants before returning `(DiskStore, State)`. Its newly
copied persistence directory has no committed checkpoint until the caller
publishes one. A failed restore cannot turn partial new files into a checkpoint.
Snapshots, stale lineage and external/unowned parents are not migration inputs.

## Retention and fixtures

Call `checkpoint_retention::retain(store,2)` after successful checkpoints. It
retains two to eight latest complete generations plus every current working tile
and verifies retained references before deleting anything. It removes retired
commit markers first, then only hash-verified, canonically named unreferenced
objects in this exact owned persistence directory. Unknown entries/corrupt
objects fail closed. Pending temporary names are retained for diagnosis rather
than broadly deleted. No parent directory, external snapshot, dataset or log is
a retention target. The disk cap can stop work before a checkpoint; it is not an
unbounded spill allowance or an eviction permission.

`persistence_tests.rs` and its helper cover interrupted publication preserving
the earlier commit, exact local/anchor continuation parity with nonempty history,
bit-identical floats/weights/counters/cursor, changed data/config/kernel rejection,
user STOP renewal, release revocation, InFlight/stale-version rejection,
read-only parent restore, two-generation retention, corruption and byte-budget
failure. Its in-memory comparison store is a tiny software fixture, not the
production residency implementation. Runtime fixtures stay under ignored work/.


## Historical v2 cycle-one integration correction

Resume binds attempt_start_wall_millis to the prior released cumulative wall,
while consumed.wall_millis may already include positive setup/admission elapsed
at child handoff. Removing that charge or shifting the attempt start is invalid.
The worker now retains three generations: the evaluated checkpoint and both
possible terminal publications when STOP arrives during the first publication.
All referenced payloads remain within the unchanged explicit disk budget.
Prompt-only attempts use strict Completed-parent restore, preserve every
scientific blob and charge wall/attempt/disk while advancing zero opportunities.
They require predeclared remaining wall/attempt budget and at least one unused
cumulative opportunity under the unchanged resume rules. No budget reset occurs.
These historical v2 corrections and focused fixtures have retained software
validation evidence. The later E3 [reconciled receipt](../../../nonmonotonic-training-lab/work/f-disk-reservation-drive/validation-receipt.json)
records eleven final zero-exit stages after the initial Rust failure and a
targeted fixture namespace correction. Its
[initial receipt](../../../nonmonotonic-training-lab/work/f-disk-reservation-drive/validation-initial-receipt.json)
and [targeted follow-up](../../../nonmonotonic-training-lab/work/f-disk-reservation-drive/validation-fixture-followup.json)
remain unchanged; this was not an all-green initial sweep. E3 subsequently
completed its owned pass and failed the quality gate. These historical results
do not validate the current v3 configuration, protocol-document, batch/pair
counter or production-parity changes, which still await this feature's final
integrated sweep. No current v3 runtime admission or campaign is implied.
