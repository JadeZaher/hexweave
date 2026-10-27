# Causal retained-text assay v1

This closed experimental protocol is `causal-retained-text-assay-v1`. It compares
the existing dense tied-embedding commit with one magnitude-dependent commit.
It is not production `causal-prefix-v3` and cannot consume its checkpoints.
No useful-assistant capability, convergence or training-time estimate follows
from software checks or from passing these synthetic diagnostic gates.

## Fixed method

Use the candidate geometry width 128, FFN width 384, four heads, seven streamed tiles,
four maximum visits, context 1024 and one numerical thread. All parameter groups
remain trainable. Both arms execute exactly 1600 real anchors and zero locals.
Persist physical opportunity count n; derive the route/history clock as 4*n.
Route initialization, frozen gradient gather, unique tile commits, detached-U
route credit, baseline update and guarded v3 joint predictor fitting are reused.
There is no local replay. Shared input and output embedding E stays tied.

`retained-dense` invokes the existing threshold-one fresh-evidence E commit.
`retained-magnitude` first validates the whole f32 gradient and shape. For g,
convert exactly to f64 and set p=min(1,abs(g)); g=0 never attempts a transition.
Direction is -sign(g). No float master, momentum, residual or persistent
per-coordinate evidence is introduced; tile and predictor rules are unchanged.

For zero-based anchor k, hash ASCII
`hexweave/retained-text/embedding-adjacent-v1`, one NUL, seed u64 little endian,
then k u64 little endian using SHA256. Interpret its first eight bytes as a
little-endian u64 base. For logical row-major E coordinate i, set
x=base+(i+1)*0x9e3779b97f4a7c15. Mix x=(x^(x>>30))*0xbf58476d1ce4e5b9,
then x=(x^(x>>27))*0x94d049bb133111eb, then x=x^(x>>31). All u64 arithmetic wraps.
Set u=(x>>11)*2^-53 in f64; attempt iff u<p. Clamp the adjacent movement to
[-1,1]. E version increments once iff any trit actually changes. Sampling
consumes no route RNG. Seed, anchor and coordinate fully determine the draw.

Report attempted, boundary-clamped and actual transitions; attempts equal
clamped plus actual. Sum the probabilities in row-major f64 order and persist
the sum's exact u64 bits. Dense metrics use deterministic p=1 for nonzero g.
With a uniformly distributed underlying 64-bit word, finite-grid selection
probability differs from ideal continuous selection by at most 2^-53. Fixed
seeds give deterministic pseudorandom decisions, not evidence of uniformity or
unbiased trajectories. Clipping and ternary boundaries preclude an unrestricted
unbiased SGD claim. No descent or convergence guarantee is made. The relevant
primary reference is Li et al., [Training Quantized Nets: A Deeper Understanding](https://proceedings.neurips.cc/paper/2017/file/1c303b0eed3133200cf715285011b4e4-Paper.pdf).

## Fixed source and exposure

The distinct synthetic source pins manifest
`0a77b8fe038c8139314e7331417ced5aa853805ec26562680c3d770a01ade6db`, rows
`0ad6de82f45fb0cff217ad34ac2ebe60eaff2ef1ce32399ef892b36ecbfc40b9`, and schedule
`7ceefd7f72531f01c3336af6779d89e4e1ac5cfd649ba2b5f94f62004501ad3c`.
They contain 16 each foundation, retention-stress, paraphrase and counterfactual
rows. Completions are soil, sand, clay or silt followed by LF: four content bytes
and one format byte. No citations, scientific claims or private examples occur.
Science Dataset parsing stays unchanged and cannot accept these plain answers.

Train 16 foundation passes then four retention-stress passes. For zero-based
global pass index j, rotate indices 0..15 left by j mod 16, then reverse on odd j.
Finish all five completion bytes before advancing a case. Probes never train.
Cursor epoch means the completed pass count, row_index means selected case
index and offset means the next answer byte. At n=1600, epoch 20, row 0, offset 0,
example_id `complete` is terminal. The full cursor is derived from n; content
count is 4*floor(n/5)+min(n mod5,4), LF count floor(n/5), citation count zero.

## Immutable observations and persistence

Observe all 48 foundation/paraphrase/counterfactual rows, in that group and
ascending index order, after passes 0,1,2,4,8,16,17,20. Each case gets five
teacher readouts, separately reported four content losses and LF loss, plus
first-byte rank with lower byte index breaking equal-logit ties. Generate after
passes 0,4,8,16,17,20 using greedy lowest-index ties, zero history, first LF stop
and an eight-byte ceiling. Preserve raw bytes; never repair output. Prefix caps
are teacher 89, greedy 92 and final generated buffer 93 bytes.

Every readout polls owned control. Before/after hashes cover the complete
numerical state, rule, physical clock, update metrics and prior observation
ledger. Observation performs no updates. One bounded immutable artifact per
checkpoint is SHA-referenced in the ledger. Each artifact is at most 65,536 bytes;
the existing terminal result cap remains 131,072. Partial observations are marked
censored, persisted, and cannot resume; they never count as completed coverage.
Only complete prior observation references may be imported by strict restore.
Missing earlier observations, changed source/rule/clock, wrong state schema,
null descriptors, unknown fields or malformed ledgers reject.

Assay state uses schema 2 and a required assay descriptor. V3 state remains
schema 1 with no descriptor; even an explicit null is rejected. Packed numerical
blob codecs are reused unchanged under the distinct protocol/config identity.
Checkpoints occur at least every 64 completed physical anchors, at every
observation boundary and terminal state; retain the existing three generations.
Historical observation hashes do not promise retained historical weights.

## Owned study and interpretation

The prescribed six fresh arms are seeds 7, 17 and 29, each dense then magnitude, with
identical initialization within each seed. Total ceiling is 9,600 anchors,
11,520 teacher readouts and 13,824 generated-byte readouts. No current/local arm,
seed selection, adaptive extension, retry or resumed campaign is permitted.
The six fixed owner wall ceilings are 1,100,000 ms each, totaling 110 minutes.
One conserved study deadline is 120 minutes including setup and finalization.
Unused allowance is not recycled. Maximum opportunity budget 1601 provides one
unexecutable headroom slot; the fixed source ends at 1600, so genuine deadline,
resource or user stops remain terminal rather than being relabeled Completed.

Each numerical arm passes through the existing external driver, Python owner,
Windows Job, physical admission, durable observer, strict release and exact
handle cleanup. The study coordinator is the external driver itself, not a
fourth process. Existing 256 MiB driver-plus-owner sampled RSS and 512 MiB child
Job commit limits remain; sampled combined RSS is 768 MiB. Observation has no
separate process. Each attempt has 64 MiB disk allowance; the study inventory
ceiling is 512 MiB with a separate 256 MiB free-disk cushion. Disk bounds are
conservative allocation/accounting limits, not measured residency claims.

At passes 8 and 16 each seed must reach 16/16 foundation exact, at least 14/16
each probe class and at least 3/4 per answer word in each probe class. Preserve
these thresholds at passes 17 and 20. No empty, capped or repeated-word generation
is allowed across the 48 cases. Report all seeds and all observations; the
every-seed conjunction is prospective diagnostic evidence only. Frozen science
public/private gates remain distinct and private inputs remain unread.

Preparation and software validation authorize no numerical campaign. Existing
program STOP is honored byte-for-byte. Fresh user runtime authorization and an
exact reviewed STOP archival decision are required before a study launch.

Each request also pins the immutable six-arm study declaration and its ordinal.
The external launcher rejects an assay request without the live conserved study
guard; the owner checks its exact driver handoff. All six arms are fresh starts.
The shared study driver is still alive at each per-arm release seal, so existing
strict-resume finalizer-absence requirements remain unsatisfied until the whole
driver exits. No per-arm report claims that driver has exited. The fixed owner
ceilings are not renewed or transferred between arms. The software checkpoint
restore API supports complete prior observations; this study does not admit a
resume request. Terminal study publication preserves an existing STOP or writes
its own terminal disposition STOP without automatically clearing either.

The study's outer inventory adds narrowly identified live Python-owner control
publication files to the existing native-writer disappearance accounting. Each
accepted disappearance charges a full bounded publication allowance, and a
second unstable scan fails closed. The study driver samples and records its own
control inventory, all six exact attempt directories and all six driver-log
directories. A successful final study receipt still requires an external check
that the one exact study process exited and a measurement of its complete wall
time including final persistence.
