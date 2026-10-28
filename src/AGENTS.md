# Source map and implementation rules

`main.rs` is the executable entry point. `config.rs` validates bounded profile
metadata and computes an allocation-free main-matrix estimate. `topology.rs`
assigns deterministic axial coordinates. `planner.rs` is a metadata-only branch
planner. The tensor implementation lives in `model/`, with its own directory
notes; `smoke.rs` runs the bounded numerical fixture.

`ternary.rs` supplies packed two-bit matrix values and signed i16 update
evidence. `model/` remains the FP32 reference. `ternary/` contains checked
scalar/AVX2 kernels and evidence contracts. `causal/` is the ternary byte-level
candidate and its core training path (`hexweave train`, `causal/AGENTS.md`);
`causal_cli.rs` parses `causal plan`. The earlier `pulse` and `text` protocols
are archived under `../archive/src/` and no longer build.

## Ternary state

Values are -1, 0 or +1 from initialization. Scalar products decode trits into
add/subtract/skip operations, with wider activations/accumulation and no floating
weight masters. Evidence adds the negative gradient sign; opposing signs cancel.
At threshold it takes one adjacent step and resets, including saturated endpoints.
Zero gradients do not accumulate. This is a research rule, not an exact
discrete-gradient theorem. Full validation precedes update mutation. Payload
byte counters exclude containers, allocator overhead and transient computation;
they are not measured memory. Runtime AVX2 preserves scalar accumulation order;
small shapes use scalar. See `docs/PERFORMANCE-LOOPS.md` for measured scope.

Keep source files below 300 lines and functions below 90 lines where feasible.
Prefer descriptive names, blank lines between functions, and terse one-line
doc-comments. Put rationale and limitations in directory documents. Never hide
unsupported behavior behind a successful result. Add fixes together before the
integrated test sweep and require a reviewer distinct from the author.

## Profile accounting

The planned matrix count is `vocab * hidden + tiles * (4 * hidden^2 +
3 * hidden * ffn)`. It assumes tied input/output embeddings, four square
attention projections and three gated-FFN matrices per tile. Norms, activation
curves, controller and pulse are additional parameters, reported as unknown
overhead rather than zero. The estimate is independent of route occupancy:
inactive weights still exist. The 16-byte FP32 Adam state floor is only weights,
gradients and two moments for the counted matrices. It excludes intermediates,
attention workspaces, runtime, allocations and checkpoint copies. It is neither
a physical-memory admission test nor a training-time forecast.

Radius is capped before topology allocation. Other dimensions are bounded and
arithmetic is checked. Configuration input is limited to 64 KiB and unknown
fields are rejected. The planner never constructs tensors. The CLI cannot load
these larger profiles into the numerical fixture or train them; `train` runs only
the fixed causal candidate.

## Topology

The axial disk contains `1 + 3 * radius * (radius + 1)` tiles. IDs use increasing
`q`, then increasing `r`; the center therefore need not be ID zero. Neighbors
have axial distance one, stable ascending IDs, no duplicates and no self edge.
Hex coordinates constrain route legality, not physical tensor placement.

## Planner

The planner evaluates an explicitly supplied policy callback and supports Stop,
Single and a distinct-neighbor two-way Fork. It is not a learned policy or a
tensor fanout executor. Repeated visits are legal and appear in branch history.
Root depth is zero; `max_depth` counts visited tiles along a branch, including
the root. `max_visits` includes the root and all reserved child visits.

A FIFO pending queue has a hard `max_frontier` cap. A requested expansion that
cannot fit the remaining visit/frontier budget becomes an exit, with an explicit
reason; it is never partially accepted or silently changed to another route.
Depth/no-neighbor exits bypass the policy. Legal action validation still applies
before visit/frontier enforcement. This simple queue has an intentional ordering
bias under tight global budgets; it is not a symmetric parallel-wave scheduler.

Forks split parent mass equally and each terminal branch retains its mass, so
exit weights sum to one, including forced exits. This is a conserved branch-mass
merge, not an equal average over leaves of different depths. Numerical fanout
must later define branch-local activations/pulse snapshots, correction merging,
causal context and routing credit before using this metadata plan for training.
