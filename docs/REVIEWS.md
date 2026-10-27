# Independent review record

## SIMD and disk-local text extension, 2026-10-24 UTC

Two author lanes implemented SIMD/fixture optimizations and the text/store/owner
path; a third, read-only reviewer checked them independently. Root integrated
CLI, measurement harness and documentation, also reviewed separately. Findings
fixed before the integrated sweep included removing a local activation tape,
work-directory junction containment, cleanup despite observer/STOP/query errors,
and preserving truthful memory/reap evidence. The slower packed-byte updater
was rejected rather than shipped; original update semantics remain.

Independent verification read all validation logs (58 Rust / 38 Python tests,
zero failures), the six final numerical-parity comparisons, both native text
receipts/events and the final empty Hexweave process inventory. See
[performance loops](PERFORMANCE-LOOPS.md) and [text results](TEXT-RESULTS.md).
Approval covers bounded fixtures only, not full-size admission or production
checkpoint/resume. Both text arms worsened held-out loss; no efficient-learning
or time-to-quality claim was approved.

## Historical scaffold review

Date: 2026-10-23. Static review and author fix passes precede the final integrated
checks. See [VALIDATION.md](VALIDATION.md) for executable evidence.

## Independent lanes

The foundation author reviewed numerical code; the numerical author reviewed
foundation code and the separately authored benchmark. The benchmark author
reviewed integration documentation and scripts supportively. Root integrated the
findings. Authors did not independently approve their own implementation.

The supportive lens gives the strongest defensible interpretation of the idea,
not a flattering endorsement. It confirmed hard learned routing, continuous
curves, spatial tiles and trained recurrent pulse have real entry points, while
requiring the local-residency and concurrent-wave goals to remain visible gates.

## Findings and fixes

| Review finding | Resolution |
| --- | --- |
| CLI allowed 32 updates, numerical API allowed 16 | Both use 1 through 16 |
| Evaluation example influenced initial optimizer baseline | Training baseline made independent of evaluation |
| Xorshift had an absorbing-zero seed and a rounded endpoint risk | Seed and draw-range handling hardened with regressions |
| Invalid route probabilities could silently take a fallback | Nonfinite/invalid policy values fail closed |
| Weighted policy term could be mistaken for exact expected-loss gradient | Coefficient one and fixed-incoming-history scope now explicit |
| Fixture token count could resemble durable experiment exposure | Output distinguishes fixture evaluations from production exposure |
| Diagram reversed pulse and routing order | Mandatory center, pulse-before-tile, route-after-tile now shown |
| Byte-token wording overstated synthetic IDs | Documentation now names the integer-ID fixture |
| Pulse history description omitted prior-example state | One detached terminal pulse is documented; no correction FIFO claimed |
| Metadata fanout differed from prior Hex experiment | FIFO budget bias and conserved branch mass explicitly distinguished |
| Python JSON accepted UTF-16/32 despite UTF-8 contract | Strict UTF-8 decode and regression cases |
| Deep nesting / huge numbers escaped clean validation | Bounded-input errors normalized to validation failures |
| Benchmark prose named thread/KV settings absent from schema | Explicit bounded thread and KV-dtype fields, labeled unverified |
| Granite example mislabeled verified instruct candidate as base | Template kind corrected to instruct |
| Inherited Cargo target could write outside owned work | Validation and editor tasks override target directory |
| Candle barrier pool had a separate unbounded default | CANDLE_NUM_THREADS pinned separately from Rayon/BLAS |
| Validation might read a stale native exit code | Clear exit evidence before invocation; reject missing/nonzero exit |
| Several Python installations made command resolution return an array | Use the first application in normal PATH precedence |
| Function-local exit variable shadowed the automatic native result | Reset and read the explicit global exit code; checked 0 and 7 |
| Strict Clippy flagged manual divisibility syntax | Equivalent is_multiple_of call, after the nonzero-head guard |
| Concurrent waves were an aspiration without an acceptance test | Roadmap requires derivative reference, actual overlap and transfer metrics |

## Positive evidence and remaining boundaries

Review found checked sizing arithmetic, bounded config ingestion, legal axial
neighbors and capped planner work. First-position-only route conditioning avoids
sequence-wide lookahead; causal attention and the differentiated realized path
are explicitly tested. Benchmark requests use direct loopback sockets, no
redirects/proxies, explicit deadlines, capped responses and exclusive outputs.

No review establishes full F equivalence, independent node training, reduced
large-model residency, GPU performance, convergence, usable pretrained quality or
Granite superiority. The generic endpoint cannot prove which artifact it loaded;
benchmark output marks that binding unverified and remains leaderboard-ineligible.

Independent reinspection closed the static findings before the final successful
integrated sweep: 20 Rust and 12 Python tests passed. The supportive integration
review and adversarial numerical/foundation/benchmark reviews approve this
bounded scaffold, not the future full training protocol.

## Ternary pulse implementation pass: 2026-10-23

Separate author lanes implemented packed ternary state and the node-local pulse
protocol. A different reviewer inspected the numerical protocol; the root
independently inspected the packed-state primitive. The root authored the bounded
run owner, reviewed independently by both the primitive author and a separate
lower-cost GPT-6 Luna agent. The lower-cost agent successfully read files and
sent findings back to this active conversation. No closed-session wake-up was
demonstrated.

The strongest supportive case is concrete: direct ternary storage avoids hidden
floating-point masters, selected tiles receive local updates, and task-derived
corrections can supervise a shared predictor. The adversarial boundary is equally
important: all packed weights are still host-resident, local predictions can be
wrong, and a small working set does not establish lower total training cost.

Findings applied before the integrated check:

- Tightened receipts to reject duplicate keys, bool/integer identity confusion,
  incorrect run settings, missing metrics and inconsistent evidence bounds.
- Changed per-child timeout into a shared campaign deadline; documented the
  polled log threshold and lack of a persistent chat notification integration.
- Identified anchor/data-schedule aliasing, excessive evidence hysteresis under
  eviction, unreachable router rows and incomplete correction dependency stamps.
  The pulse contract records the resulting protocol choices.
- Removed wasted non-anchor execution from the anchor-only ablation. It is a
  control for added local updates, not an ordinary pretraining performance baseline.

Static review validates the scope and software design, not convergence, exact
discrete gradients, full F equivalence, out-of-core residency or efficiency.
Integrated verification and bounded observations are recorded separately.

## Joint tile pulse: two-cycle Dev Drive, 2026-10-24

Node zero was the staged ternary joint-pulse implementation. PowerShell's RNG
selected four distinct review types without replacement, with independent
reviewer tasks and randomized stances:

| Cycle | Branch | Stance | Synthesis returned to node zero |
| --- | --- | --- | --- |
| 1 | Architecture | Enthusiastic | Linked the pulse guide; extended the existing coupling fixture for task-to-U credit without direct route-head credit |
| 1 | Correctness | Discriminating | No blocking finding; checked derivatives, frozen commits, storage schema, accounting and receipt protocol |
| 2 | Documentation/contracts | Enthusiastic | Corrected exact forward, credit-join, projection and commit function names; independently verified wording |
| 2 | Numerical/learning validity | Enthusiastic | Clarified bounded forward modulation is not a small learning step; deferred benefit claims pending a matched coupling ablation |

Enthusiastic reviews sought useful improvements under the same evidence standard
as discriminating reviews. No review type was repeated and no recursive review
tree was spawned. The first architecture read overlapped final authoring; its
final findings incorporated the completed directory contract. Cycle two reviewed
the revised, frozen code. Static review approved bounded integrated verification,
not convergence or performance. Installer checks independently confirmed target
containment, reparse-point refusal, hash-guarded copies and no deletion.

The reusable `dev-drive` skill is one local Codex skill file, with two cycles as
its default contract and one integrated validation sweep after changes. Actual
validation/run evidence is recorded in [STATUS.md](STATUS.md).
