# Validation utilities

## Core training owner

`python -B -m scripts.core_owner --config <core-run.json>` owns one
`hexweave train` run (see `src/causal/AGENTS.md` §core path). Before any side
effect it requires `out` = `work/core-runs/<name>` and 768 MiB available (256 MiB
floor plus the 512 MiB child commit); it then takes an exclusive
`out/OWNER.lock` (a crashed owner's lock is removed by hand after inspection).
A fresh launch refuses existing checkpoints or an already pinned binary and
copies the current release build into `out/bin/` (under a temporary name, then
an atomic replace); a resume keeps the pinned copy. A leftover `out/STOP` is
archived only if an owner wrote it (`owner:` prefix) or `--clear-stop` is
given; a previous `report.json` is archived so the end receipt can only embed
this launch's report.

The child starts suspended in its own process group, joins a one-process,
kill-on-close Job with the 512 MiB commit cap, then resumes. Each second the
owner samples the child's peak working set and host memory; ten consecutive
samples below the floor, or the wall budget plus 30 minutes, write `out/STOP`
so the trainer checkpoints and exits, and the owner kills it only after a
180-second grace. Ctrl+C in the owner does the same (STOP, grace, then kill),
and so does an owner error while supervising (for example a failed memory
query): it writes STOP if none exists and waits the grace before the kill.
Never kill the trainer directly. Start and end receipts record the config and
binary SHA-256s, git HEAD and dirty state, host memory, exit, reason, peak
working set and the report (an unreadable report is recorded as
`{"unparseable": true}`). Reasons chain with `then` so none is lost: a stop
overridden by the grace kill reads `owner_memory_stop then owner_killed_after_grace`,
and an owner error recovers this launch's owner-written STOP reason first, e.g.
`owner_memory_stop then owner_error: OSError then owner_killed_after_grace`. The end receipt is
written on every exit path the owner itself survives (not console close or
reboot, which leave `OWNER.lock` for manual removal).

### Optional recovery after a memory stop

`--resume-on-memory` enables bounded recovery for connected runs only. The
default command still launches one child. Recovery retains `OWNER.lock` across
attempts and waits, keeps the same pinned executable, and creates an immutable
`owner-attempt-<session>-<attempt>.json` config for each child. Subsequent attempts
set `resume: true`; model options and the total anchor target stay fixed. Each
attempt's elapsed wall time is subtracted from the caller's `wall_seconds`, with
the next child receiving the rounded-down remainder. Recovery waits have a
separate cumulative budget and do not replenish training time.

Only an exit-zero, exact `owner_memory_stop` receipt with a matching `stopped`
report and complete final checkpoint permits recovery. An initial explicit
resume under this mode requires the same evidence. Admission checks the pinned
binary and source-config hashes, checkpoint identity and counters, and every
payload hash. It repeats these checks after waiting, before clearing the owned
STOP. A changed, missing or corrupt checkpoint ends recovery; it is not a request
to fall back to an older checkpoint. Recovery does not migrate checkpoints or
replace a missing pinned binary.

Memory must remain at or above the existing 768 MiB start floor for 30 consecutive
one-second samples. `--max-memory-restarts` defaults to three and accepts zero
through three; `--memory-recovery-wait-seconds` defaults to 21,600 and can reduce
the six-hour cumulative wait bound. Recovery ends on these limits, an exhausted
training budget, a completed target, an error, a forced exit, or an operator
cancellation. Ctrl+C during a wait exits with status 130. Write or replace
`out/STOP` to cancel a wait; removing or changing its owned marker also cancels.
STOP requests use exclusive creation and never overwrite an existing marker.
Recovery checks the exact owned marker and its file identity when archiving it;
a raced operator replacement is preserved and cancels the attempt. A late
admission failure leaves the marker intact for another bounded wait.

Prior STOP files, reports, and per-attempt start/end receipts are retained.
`owner-recovery-<session>.json` publishes the current phase, stable-memory sample
count, wait usage, attempt/restart counts and remaining training budget. These
files are run receipts under `work/`, not model configuration fields. The helper
modules are `core_owner_recovery.py` and `core_owner_checkpoint.py`; their mocked
state-machine contracts live in `tests/test_core_owner_recovery.py`.
Connected precision defaults to `ternary` and the `causal-connected-v5`
protocol ([v5 contract](../src/causal/CONNECTED.md#label-conditioned-credit-predictor-and-gain-scaled-decay-v5)).
`connected.precision: "fp32"` selects `causal-connected-fp32-v5`.
Recovery admits the v1 through v5 generations of the configured precision (a
generation stays admitted while runs of it exist); the report, newest manifest
and older manifests must name the same protocol, so a run mixing generations is
refused. The run's pinned binary enforces that protocol when it resumes.
Recovery requires the config, report, manifest and saved common-state precision
to agree; unknown precision values and changes of precision across attempts are
rejected. Missing precision in an older ternary checkpoint retains its original
meaning. The owner does not quantize, migrate, or reinterpret its payloads.

## Corpus preparation

`prepare_corpus.py` turns a TinyStories-format file (stories separated by
`<|endoftext|>`) into `train.txt` and `heldout.txt` (every Nth story, default
20), each story ending with byte 0x03, plus a `manifest.json` recording both
files' sizes and SHA-256s and the source path, size and SHA-256. The leading
fragment before the first separator is dropped. Output goes under `work/`.

## Windows resource helpers

`owner_resources.py` samples host memory and the child's working set and builds
the one-process, kill-on-close Job with the 512 MiB process/job commit cap and
resume-from-suspended; `owner_win32.py` holds the Win32 structures and bindings.
Unsupported platforms fail closed. Both were renamed from the archived text
owner's `text_resources.py`/`text_win32.py` on 2026-10-27.

Primary API references:
[job limits](https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-jobobject_basic_limit_information),
[extended memory limits](https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-jobobject_extended_limit_information),
[process memory information](https://learn.microsoft.com/en-us/windows/win32/api/psapi/nf-psapi-getprocessmemoryinfo).

## Integrated validation

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

## Kernel microbenchmark

`examples/kernel_bench.rs` checks full-output scalar parity before its timed
shape loops. It is not a production job owner. Keep measurements in ignored
`work/`. The release-fixture comparator `speed_loop.py` is archived with the
`pulse` command it timed.

Archived owner, study, pulse and text sections:
`../archive/scripts/AGENTS-history.md`.
