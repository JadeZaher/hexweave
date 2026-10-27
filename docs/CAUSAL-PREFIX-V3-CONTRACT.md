# Causal-prefix-v3 joint predictor contract

This document defines `causal-prefix-v3`. Independent reviews and integrated
validation are pending at authoring; no science campaign is admitted. It inherits
the v2 attention contract, SHA256
`b6d725b398867675701b9f9bf23484e1f2e72547f3029086d38561859c10fab3`,
and retained v1 contract, SHA256
`762b6ae6abacae6e047ed5fc2db50cb95471475878ee2a20b8f8f2629fbcd478`,
except for predictor fitting, batch selection, counters and predictor scratch
specified here. Both earlier documents remain unchanged required source inputs.

## Numerical rule

The admitted predictor rule is `joint-adjacent-raw-row-mean-v1`; anchor selection
is `current-anchor-forward-v1` and replay selection is `newest4-compatible-v1`.
These strings are exact configuration fields bound by config/source/runtime and
checkpoint identity. All previous numerical protocols are rejected. Packed
weight and ownership formats remain unchanged; no checkpoint migration exists.

A fit consumes 1â€“4 borrowed feature/correction pairs in canonical input order.
Features and targets are finite and have the admitted geometry. Stored real
corrections are once unit-clipped; fitting applies the existing f32 unit clip
once more, exactly as v2 did. Initialization, all eligible coefficients, raw
f32 projection/tanh and delivered correction clipping remain unchanged.

Freeze the pre-update predictor. For each row, compute each pair's scalar f64
dot in ascending column order and multiply by reciprocal sqrt(feature width).
Rank legal adjacent trit changes by equal-weight mean raw-tanh squared target
error, reducing in pair order. Candidate projection is base plus
`direction * feature[column] * scale` in that order. Start with no move and
replace only on strict improvement; ties retain no move, otherwise the lowest
column and then -1 before +1. There is one proposal per row and no runner-up,
inner epoch, gain search, float master or persistent optimizer state.

Evaluate current and proposed matrices with actual f32 predictor calls on every
pair. Accept a row only when its f64-reduced mean raw squared error strictly
decreases. Merge by trit index, preserving packed bytes shared by row boundaries.
Recompute every merged prediction, require bitwise agreement with the selected
row and nonincreasing joint row error, and commit only after all checks succeed.
Count one adjacent transition per accepted row. Increment predictor version
once iff any trit changes; a stall preserves version and packed identity.
Individual pair error may increase despite joint row descent. There is no
claim about intermediate delivered error or downstream task improvement.

## Anchor, replay and persistence

After frozen credit gathering, tile commits and tied-E commit, construct the
anchor's 1â€“4 records in forward visit order. Repeated tile visits remain separate
records. Fit this batch once, then push the same records into history in order.
No earlier history record enters the anchor batch. Features still use the
original frozen trace/history; the tied-E gradient is dropped before fitting.

Local execution retains the existing entry prune, then fits the newest at most
four retained records, oldest-to-newest within that suffix, before any local tile
update. Empty history makes no fit. Capacity eight, FIFO eviction, age <=8,
nonfuture shared/all-tile version lag <=1, later per-visit pruning and history
mean are unchanged. Replay borrows stored features and never reads the pending
target. Causal cursor, routes, target-free local updates and tile/E fitting are
unchanged. `threshold-one-canonical-zero-v1` still describes tile/E updates and
packed evidence encoding, not the new predictor fitting rule.

`predictor_anchor_updates` and `predictor_replay_updates` count batch calls,
including stalls. `predictor_anchor_pairs` and `predictor_replay_pairs` count
pair contributions. Each pair's precommit MSE retains f32 subtraction, square,
coordinate reduction and division by width; promote it to f64 and add in pair
order to `predictor_mse_sum`. Its denominator is total pair contributions.
Exact checkpoint float-bit storage remains mandatory.

At Ready: anchor batches equal anchors; anchor pairs lie in [anchors,4*anchors];
replay pairs lie in [replay batches,4*replay batches]; replay batches cannot
exceed local opportunities. Anchor pairs plus local tile updates equal visits.
Anchor tile commits are between anchors and anchor pairs. Predictor version
cannot exceed total batches and transitions cannot exceed width*total batches.
Rust restoration and Python terminal reconciliation enforce these invariants.
All old identities and missing new counters fail closed.

## Work, memory and evidence limits

For width d and F=3d+2, a batch proposes at most 2*d*F signed candidates and d
adjacent transitions; each candidate uses at most four pair-error terms.
At d128/F386 this is 98,816 candidates and 395,264 pair terms per batch.
An anchor and its three locals use at most four batches, not four inner epochs.
Actual guard calls are three times batch size; clipping/proposal/merge passes
are separately bounded linear work. No claim of equal cost to the old rule.

The predictor phase ledger reserves current shared packed weights, two extra
packed predictor matrices, one d*F i8 row assembly, capacity-eight history,
up to four new anchor feature/correction records, four second-clipped targets,
borrowed pair/input descriptors, twelve d-wide f32 prediction buffers, d f64
row errors, d selection flags and bounded projection/metadata scratch. Existing
prefix, trace and retained gathered corrections overlap this phase. Replay
borrows its records. Container headroom is explicit; allocator overhead, native
resident memory and file cache require physical admission, not ledger inference.
The embedding gradient is already dropped, while unchanged anchor/tile phases
retain their existing conservative bounds. No decoded tile is held by fitting.

The fitting fixture's exact four-pair endpoint must survive extraction into the
shared production kernel. Batches 1â€“3, selection, target isolation, counters,
continuation and old-identity rejection require current-protocol contracts.
Historical dense diagnostics remain test-only; the old live learning-gap test
is explicitly ineligible under v3 and reproducible only from its frozen source.
A passing fixed-target gate is neither assistant quality nor a useful-model ETA.
A new owned pilot requires separate review, physical admission, configured
reserve/cap fit and a finite declaration within the user's two-hour whole-pass
ceiling. This contract does not mandate a separate empirical maximum-opportunity
or STOP experiment. New throughput and latency remain unmeasured until an
explicitly admitted run supplies them.
