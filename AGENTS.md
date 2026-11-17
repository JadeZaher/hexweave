# Hexweave working contract

This is a new, isolated F-inspired hex-routed language-model research scaffold.
Read README.md, docs/ARCHITECTURE.md and docs/STATUS.md before changing semantics.
Do not import parent weights, data, virtual environments, secrets or Git history.

Keep source files focused (aim below 300 lines) and functions below 90 lines.
Use short one-line doc-comments; put design rationale in directory documentation.
Separate functions with a blank line. Name algorithms and link primary sources.

All runtime artifacts live under ignored work/. The default is a tiny CPU
software fixture. Owner decision 2026-10-26: an explicit in-session owner
request authorizes a bounded training study of the current small candidate,
executed through the existing mechanisms, not around them — any terminal
`work/causal/STOP` is archived with a dated receipt naming the authorization,
a fresh plan is prepared and pinned, and the in-code STOP check and plan
schema remain in force. Admission honors `HEXWEAVE_CAUSAL_ADMISSION`
(`strict` default; `elastic` starts on available memory above a fixed floor,
with graceful resource-stop scaleback). Owner decision 2026-10-27: the core
path (`hexweave train --config`, run under `scripts/core_owner.py`; see
`src/causal/AGENTS.md` §Core path) is the production training mechanism for
the current candidate, on public train/dev data only; private evaluation
material stays unread. The 2026-10-26 mechanics (`work/causal/STOP`, plan
schema, `HEXWEAVE_CAUSAL_ADMISSION`) belong to the v3 owner chain, now in
`archive/`; core runs are admitted and stopped by `scripts/core_owner.py`
(`out/STOP`). The 81M/287M profiles stay
planning-only until independent numerical, resource, ownership and checkpoint
reviews pass. Never silently download models, start servers or use cloud;
background-worker and cleanup rules are unchanged.

Keep four CPU threads, a 1 GiB device reserve and a 12 GiB owned-tree RSS ceiling
as future admission requirements, not guarantees of the current fixture.
Owner decision 2026-11-04 authorizes read-only connected checkpoint inference
and opt-in bounded recovery after clean core memory stops. Resume uses verified
saved state and the pinned executable, retains exclusive ownership through the
recovery wait, respects operator STOP and deducts consumed run time. Recovery
does not authorize checkpoint migration, reboot restart or an unbounded worker;
the lifecycle contract is in `scripts/AGENTS.md`.

Owner direction on 2026-11-04 authorizes a fresh connected FP32-forward study
and subsequent measured quantization comparisons. Preserve predictive local
credit, shared memory, curves, routing and bounded tile ownership. Implement and
independently review precision-aware numerical/checkpoint behavior before a
bounded pilot and a fresh run through the existing capped owner. Gracefully stop
the earlier study before launching the replacement; preserve its final state.
Quantized inference comparisons must leave the FP32 checkpoint unchanged and
must not claim compressed-model quality or speed without measurements.

Use focused author/reviewer tasks. Apply all review fixes before one integrated
test/lint sweep at the end. Review must be independent from authoring. Preserve
explicit distinctions between implemented behavior, planning and speculation.
No performance, convergence, ternary-kernel or F-equivalence claims without data.

## Lean tests and bounded validation

Prefer the smallest readable test change that protects a meaningful behavior or
failure mode. Extend existing fixtures and use table-driven cases for related
inputs instead of adding one file, harness or near-duplicate test per variation.
Keep failures easy to diagnose; do not collapse unrelated contracts into a giant
test or build a generic test framework for a small change. No test-count or
coverage-percentage target justifies redundant tests, trivial accessor tests,
implementation-detail assertions or re-testing library behavior.

Preserve high-value checks for numerical correctness, causal credit, versioning,
residency bounds, corruption and owned-process cleanup. Fewer tests is not a
reason to weaken those contracts, delete useful regressions or ignore failures.
New tests should identify the distinct risk they cover; reuse existing coverage
when it already proves the same behavior.

Coordinate authors so tests are not duplicated across agents. Batch all edits
and review fixes before one integrated test/lint sweep appropriate to the change.
Do not run the full suite after each fix or repeat passing stages without a
relevant change. Triage failures together, then rerun the affected checks after
the fixes; avoid tight test-fix-test loops. A focused harness check or explicitly
requested bounded performance comparison is an exception, not a new full sweep.

Documentation/comment-only changes normally need readback and relevant link or
structure checks, not new behavioral tests, a full build or training. Keep test
fixtures small, deterministic and bounded; do not launch large models, download
data or leave background workers for routine validation. Report what was checked
and any deliberately unrun checks without implying unexecuted tests passed.

