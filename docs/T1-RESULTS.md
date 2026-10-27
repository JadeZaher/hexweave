# T1 result: revise route-to-U credit

Observed 2026-10-24 (America/Denver). The fixed six-run comparison completed.
The predeclared decision is **revise**: every coupled run worsened held-out
next-byte NLL, while every route-to-U-detached run improved it. All three paired
differences favor detachment by more than the declared 0.01 nats/byte margin.
This closes T1 with negative evidence for this coupled update protocol. It does
not reject all shared representations or establish a useful language model.

The [experiment contract](T1-CREDIT-EXPERIMENT.md) was written before coding and
observation. Neither its thresholds nor the executable default changed after
these outcomes. A remains `coupled`; B remains `route-to-u-detached`. Both have
identical initialization, parameter count and forward geometry. B removes only
route-derived U credit, retaining task-to-U and route-to-V/R learning.

## Matched quality and cost

Each run used 256 opportunities, anchor interval 4, pulse mode, 64 actual anchor
targets, 64 held-out contexts per evaluation, width 16, seven tiles and a
256-byte vocabulary. Order was seed 7 A/B, seed 17 B/A, seed 29 A/B. All resolved
to AVX2. Initial losses match exactly within each pair; data, binary and source
identities remained stable. Routes could diverge after learning.

Improvement below means initial minus final NLL; positive is better. The
perplexity or fluency of a useful assistant was not measured.

| Seed | Initial NLL | A final NLL | B final NLL | A improvement | B improvement | A minus B improvement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | 5.509346 | 5.735745 | 5.472325 | -0.226399 | +0.037021 | -0.263420 |
| 17 | 5.581779 | 8.217102 | 5.481877 | -2.635324 | +0.099901 | -2.735225 |
| 29 | 5.606699 | 5.712605 | 5.515786 | -0.105905 | +0.090913 | -0.196818 |

| Seed | A child wall, s | B child wall, s | A training, s | B training, s | Wall A/B | Training A/B |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | 2.281 | 2.125 | 2.054966 | 2.025070 | 1.0734 | 1.0148 |
| 17 | 2.140 | 2.219 | 2.042804 | 2.113775 | 0.9644 | 0.9664 |
| 29 | 2.218 | 2.063 | 2.094597 | 1.983328 | 1.0751 | 1.0561 |

Every time ratio is within the predeclared 1.10 ceiling, but the quality criterion
fails in every seed. The revise rule also independently applies because all A
improvements are nonpositive. Three tiny seeds do not establish significance,
a general speed difference, or that B will continue improving at larger exposure.

Campaign internal wall was 13.297 seconds, sampled before terminal persistence.
External Python-process wall was **13.3849818 seconds**, including its final
publication; exit code was zero. The aggregate allowance was 240 seconds and
each supervised run had a 30-second whole-run deadline. Six runs ran once; there
were no retries and no subsequent training in this drive.

## Work and update diagnostics

These are training-path counts; evaluation does not contribute to these counts.
All rows have 64 targets and 64 predictor replay updates. MSE is the mean of
the per-update, 16-component correction MSE, including replay. It is not held-out
task loss and its denominator differs between runs.

| Seed/arm | Visits | Local updates | Returned frames | Halts / forced exits | Revisits | Route calls | Predictor MSE sum / updates | Mean MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 7 A | 647 | 480 | 167 | 87 / 169 | 55 | 478 | 22.763630 / 231 | 0.098544 |
| 7 B | 654 | 484 | 170 | 86 / 170 | 57 | 484 | 21.101101 / 234 | 0.090176 |
| 17 A | 656 | 485 | 171 | 79 / 177 | 53 | 479 | 21.804445 / 235 | 0.092785 |
| 17 B | 645 | 478 | 167 | 87 / 169 | 65 | 476 | 22.737450 / 231 | 0.098431 |
| 29 A | 635 | 484 | 151 | 95 / 161 | 63 | 474 | 17.139681 / 215 | 0.079719 |
| 29 B | 650 | 493 | 157 | 85 / 171 | 52 | 479 | 19.155956 / 221 | 0.086679 |

| Seed/arm | Transform transitions | Curve | Router R | Pulse context U | Pulse heads V | Embedding | Readout | Predictor |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 7 A | 85,169 | 10,602 | 7,279 | 107,802 | 207,959 | 5,377 | 195,661 | 112,639 |
| 7 B | 87,725 | 11,005 | 6,903 | 108,236 | 206,760 | 5,179 | 206,204 | 114,025 |
| 17 A | 88,935 | 11,823 | 7,562 | 111,670 | 213,028 | 5,426 | 193,676 | 114,603 |
| 17 B | 87,117 | 11,436 | 7,238 | 98,677 | 190,349 | 5,393 | 193,365 | 112,417 |
| 29 A | 89,066 | 11,160 | 7,092 | 108,690 | 210,171 | 4,913 | 178,215 | 102,455 |
| 29 B | 87,734 | 11,113 | 7,202 | 104,611 | 197,231 | 5,233 | 202,643 | 106,741 |

I/O counts include initialization, training and both evaluations. Initial writes
are separately identified. The decoded-tile ownership peak is one for all runs;
it is not an OS page-eviction guarantee. Peak working set below is child-only,
queried through the terminal Windows process handle; the Python owner is outside
the child Job. No GPU execution occurred.

| Seed/arm | Disk loads | Disk commits | Bytes read | Bytes written | Initial bytes written | Peak child working set, bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 7 A | 1,345 | 639 | 1,938,687 | 920,523 | 9,848 | 8,900,608 |
| 7 B | 1,350 | 640 | 1,938,961 | 918,511 | 9,848 | 4,562,944 |
| 17 A | 1,328 | 652 | 1,911,142 | 934,760 | 9,819 | 4,460,544 |
| 17 B | 1,302 | 630 | 1,881,532 | 905,700 | 9,819 | 4,575,232 |
| 29 A | 1,278 | 632 | 1,835,597 | 908,220 | 9,875 | 4,497,408 |
| 29 B | 1,306 | 643 | 1,870,767 | 918,968 | 9,875 | 4,493,312 |

The largest observed peak was 8.49 MiB. These tiny measurements cannot establish
memory or throughput for the planned 81M/287M models. All coefficients remain
trainable; the fixture does not redefine those documented targets.

## Verification, artifacts and cleanup

Two independent Dev Drive review cycles and targeted verification of their final
fixes preceded execution. One integrated validation completed **59 Rust tests,
42 Python tests, structure, formatting, strict Clippy, three allocation-free
plans and smoke**. The initial Clippy attempt could not resolve locked `memmap2`
from the default offline dependency cache, before compilation. The unfinished
stages then continued using the existing explicitly selected Cargo cache;
already-passed structure/format stages were not repeated. No dependencies or
datasets were downloaded. The release build used `--release --locked --offline`
and the owned `work/target` directory. Documentation-only completion followed;
no test rerun or binary change was required for these records.

Validation logs: `work/validation/651c7b46eff340578c8918f77cb05d3f/`.
Campaign receipt: `work/text/t1-campaign-c519b968686446e285c1a873c6068903/report.json`.
Individual owner receipts, each under `work/text/<id>/report.json`:

| Execution order | Seed/arm | Run ID |
| --- | --- | --- |
| 1 | 7 A | `b9d92e13a64a4f0db77458e1850169e0` |
| 2 | 7 B | `f4dd46ed38bb41bc8383a9937b1ce254` |
| 3 | 17 B | `7598573f182242ecac158de96f6eeb84` |
| 4 | 17 A | `57c68ce915bc407794b8e10d738f58b3` |
| 5 | 29 A | `194ac792f68a437bae05c2cc1b15f5cd` |
| 6 | 29 B | `1a091021f6b9420b93fd8fa0d74ecae5` |

SHA256 identities:

- Release binary: `21dd98b2d30248b81c8dfa088993f30620d3f63dad9c4011e2cd8a1b4566b8ab`.
- Train data: `b58add621357506a1eded9867080a76f7015789067ed089fc47818e424230f8a`.
- Validation data: `ecb0154359652e917ece411d9ad753f8900e1053446d5e13464a3073e7446367`.
- Campaign source fingerprint: `afc6bf09ac84275a52bc843b7cc308cca4e98582619c4602d6a2bbe79965f39c`.

The source fingerprint covers Rust sources, Python owner/campaign sources and
Cargo manifest/lock, not these completion documents. The separate build receipt
binds the reviewed installation/build command to its binary. Orchestration and
external timing evidence remain in
`C:/Users/atooz/Programming/nonmonotonic-training-lab/work/f-text-drive-20261024/`:
`build-and-admission.json`, `campaign-process-receipt.json`, `REVIEW-RECORD.md`
and `post-run-processes.json`.

Independent runtime verification checked all six owner/numerical/snapshot pairs,
the exact AB/BA/AB schedule, 384 total anchor targets, stable source/binary/data
identity, final shared state and all 42 final tile checksums.
Every owner receipt records exit code zero, exact-handle reaping and a successful
terminal peak-memory query. Separately, process inspection at **10:19:48 MDT
(16:19:48 UTC), 2026-10-24** found all six child PIDs absent and no matching
Hexweave/text owner process. This is a dated process observation, not a promise
about future machine state. Snapshots identify v3 but remain inspection-only;
there is no resume checkpoint.

## Next causal gate and the requested estimate

T2 should first measure how route and task corrections agree or oppose each other
in U at anchors, and how threshold-one updates convert those corrections into
trit changes. The specific hypothesis is that route-derived U updates interfere
with task learning under the current update rule. T1 supports investigating that
hypothesis; it does not identify the responsible sign, magnitude, update frequency
or causal mechanism. Seed 17 A has lower mean predictor MSE than B but much worse
held-out loss, so that single aggregate predictor metric cannot explain the result.

Predeclare the next diagnostic on a separate development split, use the recorded
detached arm as a candidate reference, and investigate one causal mechanism at a
time. Do not tune on this held-out set, silently switch defaults, combine several
optimizer changes or launch a larger run because the six children completed.

The user selected **a useful small-domain text assistant** as the eventual quality
target. Domain selection remains pending; repository documentation Q&A is the
working fallback. No assistant evaluation, representative real-data learning
trajectory, configurable full model, generation path or resumable scaling pilot
exists yet. Therefore **a strong training-time estimate is still unsupported**.
This 13.38-second diagnostic is not an estimate of time to assistant quality.
The open dependencies and completed branches are in [the feature tree](FEATURE-TREE.md).
