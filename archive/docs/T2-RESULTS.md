# T2 results: GO for a provisional detached T3 candidate

Completed 2026-10-24. The [predeclared development diagnostic](T2-CREDIT-DIAGNOSTIC.md)
completed its six-run allowance and independently recomputed to **GO**. Every
route-to-U-detached B arm improved development NLL; every coupled A arm worsened
it, and every A arm passed the declared frozen-credit mechanism thresholds.
This supports considering detached credit for T3. The executable text default
remains coupled. The T2 budget is consumed: no retry, extension or automatic
follow-on run is authorized. No useful-assistant training-time estimate follows.

## Paired quality and mechanism evidence

Improvement is initial minus final mean NLL, in nats/byte. B advantage is B's
improvement minus A's. Each seed's initial NLL matched exactly across arms.

| Seed | Initial NLL | A final NLL | B final NLL | A improvement | B improvement | B advantage | A differing anchors | A deficit/task descent |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | 5.519166 | 5.725301 | 5.458180 | -0.206136 | +0.060985 | +0.267121 | 46/64 | 0.358862 |
| 17 | 5.575477 | 8.450865 | 5.511137 | -2.875388 | +0.064340 | +2.939728 | 58/64 | 0.198257 |
| 29 | 5.586820 | 5.698696 | 5.526916 | -0.111876 | +0.059905 | +0.171781 | 54/64 | 0.551202 |

All seeds satisfy B improvement >= 0.01 and B advantage >= 0.01. Every A run
has at least 16 differing anchors, positive task descent and deficit/task descent
>= 0.10. Integrity checks passed. Seed 17 has a much larger coupled deterioration;
it would dominate the mean advantage, so the paired results remain explicit.
These three seeds do not establish statistical significance or convergence.

## Fixed execution and measured cost

Six distinct supervised runs used seed 7 A/B, seed 17 B/A, seed 29 A/B. Each
completed 256 opportunities at anchor interval 4: 64 actual supervised anchor
targets per run, **384 total**, without retries or resume. The new development
document was selected explicitly; the old validation document was not used.

External campaign process wall was **11.0627673 seconds**, exit zero; its internal
reported wall was 10.953 seconds. Every child and owner interval stayed below
30 seconds, and the campaign stayed within 240 seconds. Aggregate training
throughput was 38.486 supervised target bytes/s; external campaign throughput
was 34.711/s. All runs enabled diagnostics, so these include instrumentation and
do not isolate its incremental overhead or establish a speedup over T1 or another
model. Neither throughput can be extrapolated to the proposed 81M/287M models.

Peak child working set ranged from 4.340 to 4.559 MiB; this is not whole-tree RSS.
Each receipt distinguishes the largest actual-gradient payload (6,656 bytes),
three shadow bundles (19,968 bytes), and their concurrent total (26,624 bytes).
Separate evidence and excluded scratch/container/runtime costs prevent treating
these selected payload counts as complete heap or RSS.

All six children exited zero, were reaped through the owned process-handle path,
and provided terminal OS peak-memory evidence without cleanup errors. A separate
inspection at **2026-10-24T17:14:36.9789516Z** found no matching text processes;
this dated observation is distinct from receipt-reported terminal cleanup.

## Validation and artifact identity

Two independent review cycles and targeted readback of the final resource fix
preceded one integrated sweep. **All nine stages passed once:** structure,
format, strict Clippy, **62 Rust tests**, **47 Python tests**, tiny/development/local
allocation-free plans, and smoke. No passed stage was repeated. The plans remain
non-executable and unadmitted; the synthetic smoke is not assistant-quality evidence.

Independent runtime analysis reconciled all 125 reviewed source files, the
58-file campaign source fingerprint, the separate release-build binding and
before/after binary/data hashes. All six runs resolved to AVX2. Release SHA256:
`43b1cad895a4869dac6e842105afad718e39ce55ce25b1554c8463ae567c5bcc`.
The source fingerprint alone is not proof of a binary/source binding.

All 48 credit records (six totals and 42 tile rows) reconciled raw counts,
zero-denominator ratios, projection identities and per-tile sums/maxima. The
differing-result partition and selected-arm predicted/actual/applied U transitions
matched exactly. Largest decomposition residual was approximately 1.965e-9.
Final shared-file identities and all 42 selected final tile checksums matched
schema-4 inspection snapshots. Those snapshots remain non-resumable.

Within each seed, all seven initial tile files were byte-identical. **Initial
shared files were not separately persisted**, so no direct initial shared-file
comparison is possible; deterministic initialization/seed and initial evaluation
agree. This limitation is retained rather than inferred away.

## Meaning and next gate

The measurements describe a frozen-path continuous-relaxation surrogate and the
declared threshold-one ternary rule. Deficit is a task-gradient projection
difference at frozen weights, **not exact causal attribution of final loss**.
Receipts retain aggregate metrics, not every coordinate's gradient tape;
independent analysis reconciles emitted identities rather than rerunning gradients.

This remains a 19,776-coefficient, width-16 byte fixture with a fixed embedding-sum
encoder. Evaluation uses only **64 overlapping contexts, each eight input bytes**,
from the first 72 bytes of a fresh authored development document. It is not a
representative domain-assistant evaluation. Learning trajectory, exposure needed
for assistant quality, scalable state, checkpoint/resume and representative
throughput remain unmeasured.

T3's causal numerical core is authored but uncompiled in a separate proposal
namespace; execution/cursor work is underway. T4 has 60 training / 12 development
documentation-QA rows and 40 independently authored sealed cases, all unreviewed.
T5 checkpoint and owner implementation is pending. The [feature tree](FEATURE-TREE.md)
keeps these gates open, preserves the 81M/287M targets, and makes no training ETA.

## Receipts and independent verification

- [Campaign report](../work/text/t2-campaign-479b3ba8a8024087842261ce860f21eb/report.json), containing the six owner receipt paths and complete metrics.
- Seed 7: [A owner](../work/text/9f68a76bae4341da8ca6651300d4cca5/report.json), [B owner](../work/text/df1dd6569b8240b7b7e54d11a441e84c/report.json).
- Seed 17: [A owner](../work/text/89f9bfd0b32543a98e5b500a28e030a4/report.json), [B owner](../work/text/77aa66cc50a94fcfb7bf41474d447d24/report.json).
- Seed 29: [A owner](../work/text/fa083734f91945cea9fb4b7492aa7aa7/report.json), [B owner](../work/text/1cbefdd053bb4312b886ff6692a1bd12/report.json).
- Validation: [Rust](../work/validation/efe4beec193d4238b580b450b37f8836/rust-tests.log), [Python](../work/validation/efe4beec193d4238b580b450b37f8836/python-tests.log), [Clippy](../work/validation/efe4beec193d4238b580b450b37f8836/clippy.log); all nine logs share that directory.
- Separate drive evidence: [runtime verification](C:/Users/atooz/Programming/nonmonotonic-training-lab/work/f-text-drive-t2/runtime-verification.md), [machine-readable reconciliation](C:/Users/atooz/Programming/nonmonotonic-training-lab/work/f-text-drive-t2/runtime-verification.json), [validation verification](C:/Users/atooz/Programming/nonmonotonic-training-lab/work/f-text-drive-t2/validation-verification.md), [external process timing](C:/Users/atooz/Programming/nonmonotonic-training-lab/work/f-text-drive-t2/campaign-process-receipt.json), [dated process inspection](C:/Users/atooz/Programming/nonmonotonic-training-lab/work/f-text-drive-t2/post-run-processes.json).

Historical [T1 results](T1-RESULTS.md) remain a separate consumed diagnostic budget
and are not overwritten or treated as a matched timing baseline for T2.
