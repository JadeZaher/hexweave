# T1: route credit into the shared pulse context

Contract declared before implementation and observation, 2026-10-24
(America/Denver). This is a tiny diagnostic of one credit mechanism, not a
viable-language-model pilot or authorization to scale.

## Fixed intervention

Arm A is `--route-credit coupled`, the inherited computation. Arm B is
`--route-credit route-to-u-detached`. Both use identical ternary initialization,
parameter count, forward U/V matrices, task-to-U credit, route-to-R credit and
route-to-V credit. Only B suppresses the routing contribution in the U adjoint:
`dU = ((V_task^T dhead_task) * (1 - latent^2) / sqrt(W)) outer features / sqrt(2W)`.
A additionally includes `V_route^T dhead_route` in that latent correction.
Task and route contributions to V are unchanged. This is not wholly uncoupled
routing: task updates to U can still alter subsequent routing.

The implementation records the option in CLI, owner events/report, numerical
receipt and inspection snapshot. Protocol `text-pulse-v3` distinguishes required
credit identity from v2 receipts; tile schema and initialization remain unchanged.
Snapshots remain explicitly non-resumable. The coupled arm retains the previous
arithmetic order; no alternate initialization or geometry is introduced.

## Matched schedule and budgets

Six sequential supervised runs, no retries: seed 7 A/B, seed 17 B/A, seed 29 A/B.
Each has 256 step opportunities, anchor interval 4, pulse mode, backend auto,
and 64 actual anchor targets. The existing authored train/validation byte files,
binary SHA256 and resolved kernel must match throughout the campaign. Initial
held-out NLL must agree exactly within each seed; a mismatch stops immediately
when that pair's second arm completes, before another child can start. Routes and visit counts may
diverge after updates; equal exposure is not equal compute.

Each child has a 30-second whole-run deadline. The aggregate operational budget
is 240 seconds, including six run deadlines and owner/cleanup overhead. A shared
monotonic campaign deadline constrains every run; no child starts with less than
30 seconds remaining. Deadline/STOP is checked before and after every run and
inside the existing owner, including after receipt validation and final identity
hashing, and after terminal event/report persistence. STOP is also rechecked
after those phases. Remaining campaign time is rechecked immediately at child handoff.
Cleanup can exceed the nominal budget when required
to reap an owned child; any overrun is failure, never successful completion.
Failure, missing evidence, user STOP or any identity mismatch terminates the
campaign and preserves partial receipts. A terminal persistence overrun converts
success to failure and makes one corrective durable report write; that failure
write may itself exceed the budget and is never timed in a publication loop.
The final report overrides an earlier completion event. Campaign `wall_seconds`
is sampled before terminal persistence, or after its first publication when
correcting a late failure; it excludes its own final write. Record external
process wall time separately. No external data, scaling or retries.

## Measurements and decision declared in advance

For each seed, retain both full owner receipts and compare initial/final NLL,
NLL improvement (`initial - final`), child wall/training seconds, actual targets,
local updates, visits/halts/forced exits, U/V/router/other trit transitions,
predictor squared-error sum and update denominator, disk loads/commits/bytes,
peak working set, resolved kernel, source/input identities and terminal cleanup.
Predictor error is the mean over predictor updates, including replay, of the
per-update mean squared correction error; it is not evaluation loss.

Practical paired quality margin: 0.01 nats/byte greater improvement for A.
Practical paired cost margin: A/B at most 1.10 for both child wall and training
time. These are diagnostic thresholds, not statistical significance tests.

- **Retain provisionally:** all three pairs show A's quality advantage at least
  0.01, A itself improves held-out NLL in every pair, and both time ratios are
  at most 1.10 in every pair.
- **Revise:** all three pairs favor B by at least 0.01; or A fails to improve
  held-out NLL in every seed; or A exceeds either time ratio of 1.10 in every
  pair while providing no positive quality advantage in any pair.
- **Inconclusive:** every other complete, valid outcome. Mixed results remain
  inconclusive; a failed/incomplete campaign has no scientific decision.

No parameter or threshold changes follow held-out observations. Further tuning
requires a separate development split and a new declared contract. Three seeds,
tiny authored data and a byte-sum context encoder cannot establish convergence,
useful language quality, 81M/287M throughput, or time to a viable F text model.

## Verification gate

Extend the existing coupling fixture to preserve task-to-U and route-to-V/R
while removing only route-to-U. Extend existing receipt/owner fixtures for arm
identity, accounting, aggregate deadline and terminal failure behavior. Two
independent static review cycles precede one final integrated software sweep.
Only after that gate may the fixed campaign use `scripts.text_campaign`; it
delegates every child to `scripts.text_runner` and the existing admitted owner.
