# Connected inference and graceful recovery

Status: implementation, independent review, integrated validation and real
checkpoint inference complete. Recovery-enabled continuation launched and
restored-state startup verified; the training study remains active.

Owner authorization: November 4, 2026, add graceful resume, test inference, then
relaunch the stopped full study. This authorizes continuing its saved state;
the original study's fresh initialization remains recorded in its first receipt.

The first segment stopped cleanly for host memory pressure at anchor 186,164
after 44,781.094 owner seconds. Preserve its final checkpoint and receipts and
verify every payload hash before inference or further training. The original
86,400-second budget leaves 41,618 whole seconds for the continuation. Keep the
original 21,006,506 total-target cap, data, seed, numerics and pinned executable.

Implement opt-in bounded recovery in the existing owner: hold the exclusive run
lock throughout attempts and waiting, accept only clean memory-pressure stops
with a verified final checkpoint, require sustained recovered host memory, limit
retry count and total waiting, and deduct consumed attempt time. Operator STOP,
interruptions, failures and invalid checkpoints must terminate recovery. Preserve
attempt configs, stop/report archives and receipts. Keep existing Job limits.

Add connected checkpoint support to the existing completion command using the
same frozen replay policy as evaluation. Load an explicitly named verified
checkpoint into isolated scratch storage, bound generation and decoded tile
residency, and prove source/checkpoint state remains unchanged. Use the new build
for inference; training continues with its existing pinned binary.

Separate authors implement Python lifecycle and Rust inference. Independent
review covers stop ownership, time accounting, checkpoint validity, immutability
and cleanup. Apply review fixes before the integrated validation sweep. Then run
bounded real-checkpoint completions, verify source hashes remain unchanged,
launch the remaining budget under the updated owner and verify nonzero restored
state and advancing anchors. Report generation quality without conflating it
with successful inference execution.

## Executed evidence

Independent code review approved both implementations after fixes for STOP
replacement, late memory admission and checkpoint changes during recovery waits.
The integrated sweep in `work/validation/0ff3e9865b62465e9021a2957cea50c8` passed
128 Rust tests (three existing ignored), 25 Python tests, formatting, strict
Clippy, structure, all three planning profiles and smoke.

Three real checkpoint completions in `work/inference/connected-20261104` loaded
anchor 186,164, generated 128 spaces each, and preserved all 45 original training
run files byte for byte. Source-state and checkpoint fingerprints also matched;
peak decoded residency was four and all inference scratch directories were
removed. This verifies inference execution and immutability, not useful language.
Inference executable SHA-256:
`1e5f5119668ed9134ee07dc5b2c99717f466583ab20d42563e72c6f062e3ec0a`.

The hidden continuation owner is PID 114648, started at
2026-11-05T00:27:01.7880909Z (November 4, 17:27 MST). After 30 stable memory
samples it started trainer PID 110336 at 00:27:32 UTC, keeping executable
`3bc1a95146e11bedfd6dbf0be4befce2f4382f5db777c3175be1d658aadb3725`.
Recovery has at most three automatic restarts and six cumulative waiting hours;
the child receives the remaining 41,618-second training budget. Attempt and
recovery receipts are in the existing run directory. Adjacent files prefixed
`connected-full-s7-20261103-resume-20261104` record the launch, config and source
hashes. Waiting does not replenish training time, and no reboot restart is added.

The first restored log row matched every previous final mechanism counter at
anchor 186,164, with 704,273 predictor fits and 32 memory records. Initial frozen
evaluation completed 1,024 held-out targets at 4.42689 bits/byte before further
training. The startup receipt then observed anchor 186,360 with four-bundle
residency and empty error logs. This confirms restoration and advancing work;
the continuation has not finished.
