# Validation utilities

## Conserved retained-text study

`causal_assay_prepare` writes a pinned declaration, six requests and a plan. It
never starts a process. `causal_assay_campaign` is a separate explicit entry
point and never clears the existing program STOP. The live coordinator is the
existing external driver, included in the inherited 256 MiB driver-plus-owner
sampled allowance and 768 MiB combined sampled allowance. It does not add an
observer process. Each arm creates one fresh owner and one native Job child.

`causal_assay_study` validates exact seed/rule order, one immutable study pin,
fixed per-arm budgets and fresh starts. The launcher rejects an assay without
its live guard; the owner verifies the exact handoff marker and driver identity.
`causal_assay_guard` conserves the 120-minute whole deadline and 512 MiB inventory
across six fixed 1,100,000-ms owner ceilings. Unused arm time is untransferred.
The first non-completed arm ends the study. Requests cannot resume or retry.
Owner-free polls sample the exact live driver's RSS against the owner admission
floor; query failures terminate the study.

Admission modes (`HEXWEAVE_CAUSAL_ADMISSION`, owner decision 2026-10-26):
`strict` (default) requires the adaptive cushion (max(2 GiB, 15% of RAM)) plus
the child/owner reserve before start, and the adaptive cushion in-run. `elastic`
requires a fixed 256 MiB floor plus the 512 MiB child commit before start, and
the 256 MiB floor in-run; breach ends the run as a cooperative, checkpointed
resource_stop (graceful scaleback, resumable). The floor is sized to the host's
thrash threshold, not the workload: the measured child peak is single-digit
MiB and the 512 MiB Job commit cap is the hard protection (2026-10-26 arm 0:
7.4 MiB peak; the earlier 1 GiB floor stopped a healthy run at 748 MiB free). Both the study guard and the
per-arm owner derive their thresholds from `causal_resources.admission_floor`/
`runtime_floor`, so the gates share one floor derivation; samples stay
adjacent, not instantaneous, so this is alignment, not an absolute guarantee.
The strict-mode outer floor is deliberately stricter than the pre-2026-10-26
cushion-only poll (it adds the child/owner reserve, closing the gap that spent
the 2026-10-25 study on a guaranteed inner rejection). The child's hard
512 MiB Job commit limit and all RSS/disk caps are mode-independent. Admission
receipts record mode, floor and policy cushion; rejections name the failing
predicate with its sampled values. Before each arm, free disk must cover the
remaining 512 MiB study inventory allowance plus the separate 256 MiB cushion.
Scoring polls before and after each bounded arm summary. Samples record their
driver-only scope and elapsed time; active-arm three-process guards are unchanged.

Per-arm seals name the same still-live driver. Per-arm driver memory reports
are live samples and process-lifetime peaks; execute-call wall is an interval,
not the complete driver lifetime. Existing strict resume still requires the
old finalizer to be absent. The final complete external wall/exit check belongs
to the caller after this one study process ends. Late stops revoke success and
invalidate the affected release. Terminal publication preserves any existing
STOP or writes a terminal disposition STOP; it never archives either.
The guard tracks the last completed release separately from the mutable study
status, so stops first seen after the final arm or after report publication both
invalidate that release once. Earlier completed arms and already stopped or
failed arm dispositions are retained.

`causal_assay_result` checks physical counters, exact source cursor, bit-encoded
probability sum, immutable observation hashes, every ordered case, raw outputs,
coverage and checkpoint/result equality. `causal_assay_score` reports the fixed
copying and retention gates for every seed and each rule. These synthetic gates
do not stand in for frozen science evaluation or establish a useful-model ETA.
Checkpoint seed and rule derive the exact canonical assay config hash, matching
native restore. Terminal checking also reconciles the typed worker seed/config
before reading observations. These checks are confined to the assay protocol.

The outer study inventory samples a concurrently publishing Python owner.
`causal_disk` accepts only exact named owner-control pending files when given
that live owned handle's PID, and exact child observation/result temporary
names for the appropriate live writer. Vanished files charge 131,072 bytes and
the inherited bounded second scan must be stable. Ordinary v3 calls pass no
live-owner argument and retain their prior inventory behavior. Present files,
including hard links, are charged individually. No unrelated tree is scanned
or cleaned up.

`validate.ps1` is the integrated end-of-edit check, not a supervisor. It inherits
the caller's Rust/Python toolchain and optional Cargo cache setting, uses this
repository's `work/target`, and leaves named logs in a new validation directory.
Set `CARGO_HOME` explicitly only when intentionally reusing an existing dependency
cache; no virtual environment or model artifact is imported.

The script fails on the first failed stage. A failed sweep is not completion;
repair the combined findings, then perform and record a final validation. Builds
and subprocesses run synchronously. It does not create a persistent worker.

`check_structure.py` checks source length, parses Python, bounds Python functions
and verifies local links. Rust formatting and Clippy provide compiler-backed
checks; Rust function responsibility/length also require human/agent review.

## Bounded pulse owner

`python -B -m scripts.pulse_runner` runs only this repo's already-built
`work/target/{debug,release}/hexweave` pulse fixture. It does not build, download,
resume or retry. `pulse_io.py` owns receipt checks and persistence. See
`docs/TERNARY-PULSE.md` for bounds, STOP and local logs versus chat wake-up.
No production admission or checkpoint guarantee is implied by this fixture owner.

## Bounded text owner

`text_runner.py` coordinates one bounded run; `text_child.py` owns the exact
child handle and cleanup; `text_io.py` checks identities/receipts. The Windows
adapter is split into declarations (`text_win32.py`) and resource operations
(`text_resources.py`). Unsupported platforms fail closed, not unguarded.
The owner accepts `text-pulse-v4` receipts with the joint ternary training-pulse,
explicit route-credit, profile and diagnostic contracts. Historical v1/v2/v3 receipts are retained but
are not valid new-run receipts. Matching inspection snapshots, finite metrics,
predictor denominators, route/transition/I/O counts and exposure are checked.

The child starts suspended, gets a one-process kill-on-close job with 512 MiB
process/job committed-memory limits, then resumes. Before launching, physical
availability must exceed max(2 GiB, 15% total RAM) plus that child allowance.
RSS and remaining physical cushion are sampled every 50 ms; terminal OS peak
working set is queried even after a short run exits. Commit limits are hard;
RSS/log thresholds are detection guards. Python owner memory is outside the job.
The numerical path is single-threaded and backend thread environment variables
are one; this is not an OS CPU quota. The CPU-only path allocates no GPU memory.

Job assignment is checked before child execution, but the launch-to-assignment
window is not crash-atomic if the supervisor itself is externally killed. This
bounded fixture owner is not the production checkpoint/resume admission system.
Unknown failures and STOP are terminal; cleanup closes the job and reaps only
the owned process even if STOP writing, observations or memory queries fail.
Missing terminal memory evidence makes the run fail, not an invented RSS value.

## Explicit T2 campaign; T1 budget consumed

`python -B -m scripts.text_campaign --profile t2-credit` selects only the new
fixed six-child comparison in [T2-CREDIT-DIAGNOSTIC.md](../docs/T2-CREDIT-DIAGNOSTIC.md).
The profile is mandatory; this command cannot rerun the consumed T1 campaign.
Every child passes through
`text_runner.execute`; its absolute deadline is capped by the shared 240-second
campaign deadline. No new child starts without a full 30-second budget remaining.
Use `work/text/STOP` for the campaign STOP signal. Failure or STOP is terminal,
with no retries; partial receipts remain under `work/text/`.

The campaign records full per-run diagnostics and immutable source/binary/data
identities, alternating AB/BA/AB. Source hashes describe observed files, not proof
that a binary was built from them: retain a separate release-build receipt.
The campaign is synchronous, creates no watchdog thread and leaves child cleanup
to the existing owned Windows Job path. Cooperative cleanup can exceed the nominal
deadline to reap its child; an overrun fails. Both owner and campaign recheck STOP
and deadlines after terminal persistence, making at most one corrective report
write if success must become stopped/failed. That failure write may exceed the
budget; it cannot restore success or launch another child. A final report is
authoritative over an earlier terminal event. Campaign wall time states its
sampling boundary and excludes its own final write; record external process wall
separately. The campaign emits no time-to-quality forecast.

`text_profile.py` pins the T2 train/development identities and evaluated 72-byte
prefix; the old validation document is not read by this profile. `text_credit_io.py`
checks every raw metric, ratio, per-tile aggregate and anchor transition denominator.
Diagnostics may add overhead; they cannot change the declared 30-second child or
240-second campaign budget. The default single-run fixture stays coupled with
diagnostics off. T2 results do not automatically change defaults or launch T3.

Primary API references:
[job limits](https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-jobobject_basic_limit_information),
[extended memory limits](https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-jobobject_extended_limit_information),
[process memory information](https://learn.microsoft.com/en-us/windows/win32/api/psapi/nf-psapi-getprocessmemoryinfo).

## Performance experiments

`speed_loop.py` compares SHA256-pinned release fixtures, warmups excluded,
using medians and MAD. Only elapsed time and the descriptive execution backend
field are excluded from numerical equivalence. `examples/kernel_bench.rs`
checks full-output scalar parity before its timed shape loops. Neither helper
is a production job owner. Keep measurements in ignored `work/` and preserve
losing candidates' receipts without keeping their slower production code.


## Causal cycle-one revisions

See causal_OWNERSHIP.md for explicit train/prompt_evaluation request variants,
zero-exposure prompt receipts, conserved setup charges and three-generation
receipt retention. causal_prompts parses generic pinned visible inputs and checks
immutable output evidence; it never scores private keys or loads a rubric.
All new revision fixtures remain pending the final integrated sweep.
