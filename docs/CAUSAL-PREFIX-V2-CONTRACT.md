# Causal-prefix-v2 factored attention contract

2026-10-24. This successor changes the floating-point evaluation of single-query
attention and its full conditional reverse. At authoring, independent review,
integrated checks and fresh training admission are pending. This document is
the `causal-prefix-v2` protocol identity. It incorporates the unchanged v1
contract in `CAUSAL-PREFIX-CONTRACT.md`, SHA256
`762b6ae6abacae6e047ed5fc2db50cb95471475878ee2a20b8f8f2629fbcd478`,
except for the attention evaluation and scratch terms replaced below.

The byte codec, fixed profiles, immutable position-plus-byte prefix memory,
ternary parameters/evidence, normalization, FFN, pulse, detached route policy,
predictor, target isolation, cursor schedule, two correction phases and generation
rules are unchanged. The memory is not a contextual representation of the source.
Source-supplied short science QA is the accepted working capability; no quality,
convergence, original-F equivalence or 81M/287M admission follows from this change.

## Version boundary

Every owner request, worker request, configuration, checkpoint identity, result,
release and build receipt uses `causal-prefix-v2`. Both durable protocol files
are mandatory source inputs; their exact hashes are checked. The v2 document hash
is `Identity.protocol_sha256`. Existing strict identity equality also binds
configuration, source, binary, resolved kernel and data. A v1 checkpoint cannot
resume under v2. The binary/evidence storage format and its owned-directory
marker remain `causal-persistence-v1`; that marker identifies storage ownership,
not numerical compatibility. A new numerical experiment needs a fresh declared
budget and attempt lineage. Failed v1 attempts remain terminal historical facts.

## Factored single-query forward

Matrices have output rows and input columns. Set `s=1/sqrt(d)`, `k=d/H` and
`t=1/sqrt(k)`. `K_h,V_h` are the contiguous k-row blocks for head h. Prefix
`M` has L rows and d columns and remains immutable within an opportunity.

```text
n = rms(x)
q = s Q n
for h in ascending head order:
  b_h = (s*t) K_h^T q_h                  // d values
  p_h = softmax(M b_h)                   // L values
  r_h = M^T p_h                          // d values
  z_h = s V_h r_h                        // k values
z = concat(z_h)
o = s O z
u = x + 0.25 o
```

This is equivalent over exact real arithmetic to projecting every prefix row
through K and V, then taking the original head dot products and weighted sums.
The leading prefix work changes from O(L*d*d) to O(d*d + H*L*d). This is an
operation-count result, not a measured end-to-end speedup.

The implementation forms `score_scale = s / sqrt(k)` in f32. Row-block kernels
read packed trits directly with the existing add/subtract/skip arithmetic. They
use scalar ascending column order for forward, ascending row order for transpose,
regardless of the resolved full-matrix kernel. Q/O and other full projections
retain existing kernel dispatch. Dot products reduce ascending columns. Prefix
weighted sums accumulate ascending rows. Heads run ascending. Softmax retains
maximum subtraction. All stored intermediates are f32 and finite-checked.

Moving projections across sums changes f32 rounding. Trit-gradient signs near
zero, route ties and all subsequent trajectories can change. Neither bitwise
v1 parity nor matching trained weights is required or claimed. The new numerical
identity is mandatory even though the exact-real equations agree.

## Full conditional reverse and sinks

Given `g = dLoss/du`, all equations use frozen parameters and the realized route.

```text
go = 0.25 g
grad_O += s outer(go,z)
v = s O^T go
for h in ascending head order:
  c_h = s V_h^T v_h
  w_i = dot(M_i,c_h)
  delta_i = p_hi * (w_i - sum_j(p_hj*w_j))
  a_h = M^T delta
  gq_h = (s*t) K_h a_h
  grad_K_h += (s*t) outer(q_h,a_h)
  grad_V_h += s outer(v_h,r_h)
  gM_i += delta_i*b_h + p_hi*c_h
grad_Q += s outer(gq,n)
gx = g + rms_back(n,inv_x,s Q^T gq)
```

The two contributions to gM are added separately, key then value, for each head
and prefix row. Matrix-gradient outer products use the existing scaled-left
then multiply-right accumulation. Input adjoints are always returned. Prefix
and parameter sinks are independently optional and additive; omitting a sink
cannot affect the other values. Parameter-free phase one gathers gM once.
Phase two aggregates repeated visits into one tile gradient with no prefix sink.
No route RNG is replayed. No sink choice changes target visibility or updates E.

The caller adds gx0 into the last prefix row, reverses each prefix RMS and sums
all byte occurrences together with tied readout E credit before its single
commit. Local updates never request a shared-E update or read a target. Route
R/P credit stays detached and returns no upstream derivative.

## Bounded allocation replacement

The old cache of projected K/V rows, `2Ld` floats, is removed. The forward cache
contains normalized input and its inverse, q, b/r arrays of Hd each, HL
probabilities, z and o. It lives only for the current tile calculation/replay;
Frames retain only input and metadata. Only one decoded tile and one tile
gradient may exist at a time, under the unchanged store closure ownership.

```text
B_arena_v2 = 4*(2Hd + 2HL + 9m + 64d + 8r + 4)
```

The 2Hd term is the retained score vectors and summaries. The 2HL term bounds
probabilities plus the one-head L reverse vector. Forward temporaries are one
d score vector and one k value vector. Reverse temporaries are output/attended/
query adjoints, one d summary adjoint, one L score adjoint, one d score-vector
adjoint and one k query adjoint, followed by Q transpose/RMS/residual vectors.
These d/k vectors, the retained normalized input/q/z/o and unchanged target/
pulse vectors remain within the existing 64d reservation. FFN 9m, route 8r and
scalar reservations remain unchanged. No Ld K/V adjoint or decoded float weight
array is allocated. Parameters and optional prefix adjoints remain outside arena.

At candidate L1024 this is 83,696 bytes, replacing 1,128,176 bytes. Cache K/V
payload is replaced by 4,096 bytes of b/r instead of 1,048,576 bytes. Immutable
prefix plus prefix adjoint still occupies 1,053,696 bytes. Substitute the new
arena into the unchanged anchor/predictor ledger formulas; do not subtract other
state, history, trace, metadata, allocator, data, codec or I/O reservations.
This is a capacity ledger, not a measured heap/RSS bound or physical admission.

## Required independent evidence

An independent test-only f64 legacy forward projects each row through K/V.
Finite differences check every Q/K/V/O coefficient and every x/M coordinate
on a deterministic fixture. Optional prefix/parameter sinks are compared in
all four combinations, including additive reuse and a zero-memory case.
Both profiles exercise all heads and the context cap. Existing full-tile
parameter, tied-E, causality, repeated-visit, strict resume and one-tile store
fixtures remain applicable. No production float master weights are introduced.
Review and the final integrated sweep must precede any separately admitted run.
