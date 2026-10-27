# Causal owner and external release

This is the authored `causal-prefix-v3` admission boundary on the existing
supervisor path, separate from the text v4 owner. Historical v2/E3 validation
retains its original evidence. E3's initial sweep failed the Rust stage; the
fixture namespace correction and targeted follow-up yielded eleven final
zero-exit stages in the [reconciled receipt](../../nonmonotonic-training-lab/work/f-disk-reservation-drive/validation-receipt.json).
The [initial receipt](../../nonmonotonic-training-lab/work/f-disk-reservation-drive/validation-initial-receipt.json)
and [targeted follow-up](../../nonmonotonic-training-lab/work/f-disk-reservation-drive/validation-fixture-followup.json)
remain preserved. Current v3 configuration, protocol-document, batch/pair counter
and production-parity checks await this feature's final integrated sweep;
fresh physical admission remains separately required.
It makes no useful-assistant, convergence, original-F equivalence or training-time
forecast claim. No automatic retry or calibration loop exists in these modules.

## Entry point and evidence

The supported external command is:

```powershell
python -B -m scripts.causal_launch --request <absolute-request.json> --request-sha256 <sha256>
```

Each command creates one fresh external driver and one owner attempt. A later
segment must launch a fresh driver process after the previous driver exits.
`causal_runner` is the inner entry point and requires the exact live external
driver identity environment. The numerical command is exclusively:

```text
hexweave causal train --owner-request <attempt/worker-request.json>
```

The external request is a finite JSON object with exact fields `schema:1`,
`protocol`, `binary`, `source_manifest`, `build_receipt`, `identity`, `config`,
`data_root`, `budget`, `limits`, `resume` and `operation`. File specifications contain `path`
and lowercase `sha256`. Paths resolve inside the crate workspace. Source,
runtime, protocol, configuration and train Dataset identities are supplied
explicitly; configuration hashes use compact sorted-key ASCII JSON.

The source manifest is `{schema:1,files:{relative_path:sha256}}`. The owner checks
all Rust/Python source and fixture files, Cargo metadata, `.cargo/config.toml`,
the durable causal protocol document and the two compiled tiny text inputs are
included. It verifies every listed file, bounds the manifest to 1,024 entries,
each source to 1 MiB and all source bytes to 32 MiB. `source_sha256` identifies
the exact manifest bytes. The child independently checks listed source bytes.

A separately pinned build receipt must bind `protocol`, `status:completed`,
`binary_sha256` and `source_manifest_sha256`. Its producer is the final reviewed
build orchestration. Merely hashing the binary does not establish which source
produced it. Same-user malicious forgery is outside this local workflow's trust
boundary; these are fail-closed reproducibility and ownership checks.

The owner rechecks source, runtime, request, build receipt and data manifest
after the child, before terminal success. The Rust Dataset layer verifies train
and development artifact/source pins. Generic prompt-only evaluation is an explicit owned operation; this source
contains no private keys, rubric loader or embedded held-out cases.

The external command prints the final `driver-report.json` path. That report
contains `attempt_directory`, `release_seal_sha256`, `external_owner_wall_millis`,
the exact process identities and terminal status on a clean release. The attempt
contains `report.json`, `request.json`, `worker-request.json`, `launch.json`,
`events.jsonl`, `observer.json`, `result.json`, logs and `persistence/`.

## Finite budgets and measured scope

Every request declares cumulative `maximum_opportunities`, `maximum_wall_millis`,
`maximum_attempts` and `disk_bytes_per_attempt`. No pilot values are silently
selected. Opportunities are physical completed opportunities, including any
replay; unknown physical work makes an attempt nonresumable. Resume never resets
the consumed opportunity, wall or attempt counters.

Attempt limits declare `opportunities`, `wall_millis`, `stop_reserve_millis`,
`checkpoint_every`, `evaluation_targets`, `generation_cases` and
`generation_bytes`. Limits are finite and validated; the stop reserve is at least
5,000 ms. Training permits at most32 generated cases of192 bytes and65,536
teacher-forced completion targets per evaluation pass. Prompt-only evaluation
permits at most40 generated cases of192 bytes and zero teacher-forced targets.
The effective owner
deadline is the minimum of the attempt wall and remaining cumulative wall.
Baseline/final evaluation, initialization, source validation, checkpoints and
reaping consume this owner budget. This is diagnostic-inclusive runtime.

The owned native worker has an instantaneous 512 MiB Windows Job **commit** cap,
active-process limit one, kill-on-close, below-normal priority and one numerical
thread. It is created suspended, assigned to that Job, and resumed only after a
durable launch receipt. The Rust child verifies native Job membership, its own
exact identity, the live owner identity, nonce and worker request hash. This does
not claim crash-atomic creation-to-assignment: a fatal owner crash in that short
suspended interval may leave an unresumed child, and cannot produce resume proof.

The external driver and Python owner share a sampled 256 MiB working-set guard.
Owner, driver and worker working sets are sampled separately, with an adjacent
sample sum capped at 768 MiB. These are physical RSS samples, not simultaneous
whole-tree enforcement. Terminal handle queries include historical individual
peaks; summing historical peaks would not establish a simultaneous tree peak.
The initial physical reserve includes remaining allowances plus the inherited
adaptive cushion (at least 2 GiB or 15% of physical RAM). This CPU-only path never
allocates a GPU or claims evidence about GPU reserve or placement.

Attempt disk usage is sampled with finite inventory and log bounds. The disk
allowance is not an instantaneous filesystem quota. Persistence retains three
complete checkpoints under its own bounded storage contract. Exact owner and
worker handles are used for cleanup; there is no PID/name-based broad kill.

## STOP and terminal persistence

Global `work/causal/STOP`, run/attempt `STOP`, the driver's durable stop file,
resource pressure and approaching deadlines request cooperative stopping.
`worker-stop.json` contains schema, reason and acknowledged observer sequence.
The worker completes an entered opportunity or fails it, then checkpoints a
complete Ready state; it polls between evaluation targets/generated bytes.
The grace window is bounded by both the reserve and hard owner deadline.
Unknown failures, forced termination, invalid receipts and uncertain physical
work remain terminal. Cleanup may overrun the nominal deadline to reap the exact
handle; such an attempt cannot become a successful release.

Observer events are synced before a synced sequence/hash acknowledgement. The
stream is capped at 8 MiB and 16 KiB per event. Failed publication is not retried
automatically. Observer failure, late STOP and deadline checks can revoke success.

A clean owner conservatively charges measured owner elapsed plus exactly
1,000 ms for terminal publication. That allowance is budget accounting, not a
measured training duration. Owner `report.json` separates the elapsed sample from
the charge. After provisional release/ready/report writes, the owner checks that
actual elapsed fits that charge and the hard deadline. A failed check writes
`release-invalid.json`, publishes failure and exits unsuccessfully. Invalidation
I/O can itself fail; that path still cannot obtain an external release seal.

The last receipt cannot measure its own final write. Whole-owner elapsed is
measured by the external driver from before launch through exact owner reaping.
Driver admission, sealing and its final report are separate overhead. The driver
records elapsed before its final receipt, while the pilot coordinator must
measure the complete external driver process wall for any timing forecast.
There is no recursive finalizer or claim of self-measured final write duration.

## Checkpoint release and fresh resume

The owner publishes immutable `release.json` and provisional `release-ready.json`.
These alone never authorize resume. `release.json` binds the Ready checkpoint,
cumulative consumed budgets, observer sequence, exact owner identity and exact
child reap/observer acknowledgement. `release-ready.json` binds its raw SHA.

After reaping the exact owner handle with exit code zero, the external driver
validates the release, final report, source/build/runtime pins and measured owner
wall. Only then does it create immutable `release-seal.json`:

```text
schema: 1
release_sha256, owner_identity: {pid, creation_filetime}
owner_reaped: true, owner_exit_code: 0, external_wall_millis
source_sha256, runtime_sha256
finalizer_identity: {pid, creation_filetime}
```

Missing or unpinned seal, nonzero owner exit, owner wall beyond its conservative
charge/hard deadline, uncertain exposure or any invalid marker fails closed.
A crash after provisional ready but before the owner guard cannot receive a
seal. A driver crash before sealing leaves an unusable provisional release.
The seal covers already-ended owner work; outer observer overhead is separate.

A resume request supplies `parent_directory` (the old attempt),
`release_sha256`, `release_seal_sha256`, exact `checkpoint` and
`explicit_user_renewal`. User STOP requires explicit renewal. Failed/budget-stop
parents cannot resume. The former exact owner and finalizer identities must both
be absent; ambiguous process-query errors reject admission.

The owner allocates a new attempt under the same run and creates one immutable
claim at `work/causal/<run_id>/claims/<releaseSHA>.json`:
`{schema:1,release_sha256,new_attempt_id}`. A second claim fails; an unsuccessful
attempt does not roll the claim back. Parent checkpoint, release and seal inputs
remain read-only. The child receives `parent_directory` as the parent's
`persistence` directory plus the explicit release path; it validates claim and
parent identities before and after restore. No mutable latest-checkpoint pointer
or implicit retry exists.

## Module and authored fixture inventory

`causal_launch` owns the external process and calls `causal_seal` after exact
owner reaping. `causal_runner` coordinates one attempt. `causal_prepare` allocates
fresh directories and claims. `causal_contract` validates request/source/build
pins. `causal_resources` wraps exact process identity and sampled admission.
`causal_child` owns suspended creation, Job assignment, cooperative STOP and
handle cleanup. `causal_observer` bounds and syncs evidence. `causal_checkpoint`
validates committed checkpoint blobs and parent release proofs. `causal_result`
checks child exposure, canonical state counters and evaluation evidence.
`causal_terminal` handles terminal checks and provisional release publication.

Authored Python coverage includes suspended/assigned/resumed order; exact cleanup;
resource STOP and hard deadline; late event/ready publication timing and STOP;
finite limits and external pins; missing/changed seal, source, blobs and lineage;
explicit user renewal; single-use external claims; zero-exit external owner reap
and wall charge; evaluation state/mask/byte/coverage faults; cumulative counter
resets; malformed observer JSON and protocol drift. One bounded Windows-only
nontraining native spot check creates one suspended Python child, assigns its
Job, resumes it, requests STOP, reaps its exact handle and queries terminal peak.
Mock fixtures do not establish physical ownership. Historical v2 coverage and
E3 boundary/helper checks, including the bounded native check, are recorded in
the reconciled receipt linked above, with the initial Rust failure and targeted
follow-up retained. They do not establish a pass for the pending v3 extensions.

## Native API evidence

Existing reviewed Job/working-set/resume bindings are reused from `text_resources`
and `text_win32`. New identity/liveness bindings were authored against Microsoft:

- [GetProcessTimes](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-getprocesstimes)
- [GetExitCodeProcess](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-getexitcodeprocess)
- [OpenProcess](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-openprocess)
- [GetCurrentProcess](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-getcurrentprocess)
- [SetPriorityClass](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-setpriorityclass)

The drive retains captured source evidence in its `.omc/research` directory.


## Cycle-one request and receipt revisions

`operation` is exactly `{kind:"train"}` or
`{kind:"prompt_evaluation",prompts:{path,sha256}}`. The latter reads one64KiB
pinned file under a workspace/work subdirectory, schema1 with ordered
`prompts:[{id,prompt}]`. Bounds are1..40 cases,64-byte ASCII identifier IDs,
832 UTF8 prompt bytes and1..192 generated bytes. The declared generation_cases
must equal the file count; extra fields, duplicate IDs/JSON keys, pin changes,
context overflow and silently truncated suites are rejected.

Prompt evaluation requires a clean Completed parent and identical model/config/
source/runtime/train-data identity. Its opportunities, checkpoint_every and
evaluation_targets are zero. It consumes a fresh attempt and conserved wall/disk
budget, not training exposure. Existing strict resume still requires at least
one unused cumulative opportunity; reserve that headroom, evaluation wall and an
additional attempt before training. Exhausted budgets cannot be renewed here.

Worker results now echo operation and carry
`evaluation:{kind:"train"|"prompt_evaluation",result:<variant>}`. Train carries
the existing initial/final development fields. Prompt results carry input SHA,
parent/evaluated checkpoint references, immutable before/after state hashes,
requested count, ordered ID/prompt-SHA/raw-generation cases and explicit coverage.
The owner reopens the pinned prompt file, verifies order/count/byte/UTF8/route
claims, requires zero opportunities/training time, and compares all scientific
checkpoint blobs to the parent. Keys and scores are never accepted by this path.
Short generated educational QA still needs independent external quality scoring.

The prior released wall consumption remains the new attempt start; positive
owner setup/admission time is separately conserved at worker handoff. Three
retained generations protect the evaluation reference during a late STOP. The
native launch/reap/seal sequence is unchanged. These historical v2 request and
receipt fixtures and the E3 boundary/helper checks have retained validation
evidence, including the initial E3 failure and targeted follow-up linked above.
Current v3 extensions still await their final sweep; software evidence does not
establish training quality or replace fresh physical admission.

## Historical v2 numerical identity and inherited live disk inventory

The historical v2 transition required its two durable protocol documents and
pinned factored single-query attention with its changed f32 reduction order.
It rejected v1 numerical continuation. Current v3 requires all three documents
and rejects both v1 and v2 continuation, as specified below. Byte storage markers
remain v1 because their encoding and ownership semantics did not change. The
following live inventory behavior is inherited unchanged by v3.

`causal_disk` reads each directory entry's no-follow metadata once, rejects
reparse/nonregular entries, and counts files plus directories against8192 entries.
Only the live child watcher supplies its exact Popen PID. A disappearing direct
persistence entry may trigger one rescan only when it is the matching child's
canonical `.pending-PID-u64` name, a canonical retained hashed blob or a bounded
commit marker, and the non-reparse owned parent/marker are still valid. All other
missing paths, permission errors and malformed names remain terminal failures.

Each vanished permitted entry is conservatively charged131,072 bytes, the Rust
publication-file cap. A complete second scan must have no further disappearance;
both scans share the8192-entry cap. The reported sample is the maximum of the
charged first scan and stable second scan. Present hardlinks are each charged.
Only constant counters and a bounded directory stack are retained, not a complete
file inventory. New publications after enumeration remain an inherent sampling
limit; this is not an instantaneous occupied-disk guarantee. Checkpoint payload
hash/reference validation is separate and never skips missing committed files.

The post-reap terminal inventory supplies no live-writer PID and never tolerates
disappearance. Unknown failures remain terminal with unknown physical exposure;
there is no retry, release repair or budget renewal. Deterministic filesystem
fixtures cover publication/retention churn, conservative charge, second-scan
failure, malformed ownership, permissions, reparse and aggregate entry bounds.
The historical numerical-v2/native-churn and E3 two-hour boundary fixtures have
completed validation recorded in the reconciled receipt linked above. Its
initial Rust failure and targeted fixture follow-up remain explicit; none of
these historical results substitutes for the pending v3 final sweep.

## Two-hour finite admission ceiling

Attempt `wall_millis` accepts at most7,200,000 milliseconds for either owned
operation. This is an admission ceiling, not a default run duration or a budget
renewal. Each request still supplies its smaller finite segment, cumulative
remaining wall and cooperative stop reserve; setup/evaluation/reaping remain
charged. The historical E3 coordinator declared 110 minutes training, eight
minutes optional evaluation, 118 minutes lineage and 120 minutes total external
wall; it did not resume either stopped E1/E2 experiment. E3 completed its pass
and failed the public quality gate, as retained in the
[closed outcome](../../nonmonotonic-training-lab/work/f-disk-reservation-drive/E3-OUTCOME.md).
That resource feature left numerical v2 equations and pins unchanged. Current
v3 changes the numerical identity and predictor fitting; it inherits the finite
admission ceiling, native disk sampling, STOP and exact-process ownership.
Historical boundary validation is complete with the failure/follow-up provenance
above. No current v3 campaign or executed-check pass is implied.

## V3 joint configuration and counters

The exact current rule IDs and configuration fields are checked before native
admission. All three retained protocol documents are required and pinned.
Old-protocol checkpoints cannot resume under the new identity. Worker results
reconcile batch calls separately from anchor/replay pair contributions, visits,
transitions and predictor version, with exact restored f64 metric bits. The
predictor MSE denominator is total pair contributions, not batch calls. Existing
process ownership, STOP, observer, release/seal and finite budget paths are
unchanged. No new science pilot or automatic retry is adopted.
