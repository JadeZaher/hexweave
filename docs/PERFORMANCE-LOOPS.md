# Three bounded performance loops

## Intention and acceptance rule

The owner's request was to reduce local compute while retaining ternary-first
weights, learned routing and activation, non-monotonic hex visits, and localized
updates. The greedy rule is: propose from a concrete hot operation, measure the
same fixture, retain a gain only when numerical behavior remains unchanged.
Structural inspection and kernel timing guide these loops; this is not a sampled
whole-program CPU profile or a time-to-language-quality experiment.

## Fixed comparison

Release builds use the same Cargo lockfile and compiler settings. The original
scalar executable is frozen and SHA256-pinned before any candidate replaces its
build output. Each case uses 256 step opportunities, resident evidence limit one,
anchor interval four, seeds 7/8/9 and both pulse/anchor-only modes. One warmup is
excluded; nine repeated samples per case yield median and median absolute
deviation. Internal numerical time and whole-process wall time are separate.

Every loop's six receipts matched the original in **every non-timing field**.
Those historical receipts still contain the old scalar-only execution description.
The final reporter corrects that description; future comparisons exclude exactly
`seconds` and `arithmetic.execution`, never losses, routes, counts or state bounds.
The harness is the archived [speed_loop.py](../archive/scripts/speed_loop.py),
which timed the archived `pulse` command. Runs were sequential,
not randomized/interleaved A/B; thermal/background variation remains possible.

| Version | Pulse median across three case medians | Anchor-only median | Decision |
| --- | ---: | ---: | --- |
| Original scalar release | 3.074 ms | 1.011 ms | Baseline |
| Loop 1: AVX2 | 2.587 ms | 0.875 ms | Retain |
| Loop 2: sparse input / fewer allocations | 2.467 ms | 0.770 ms | Retain |
| Loop 3: packed update candidate | 2.972 ms | 0.915 ms | Reject; regression repeated |
| Final retained code, scalar small-shape fallback | 2.492 ms | 0.791 ms | Retain prior gains; no additional end-to-end gain established |
| Final integrated binary, corrected backend metadata | 2.589 ms | 0.767 ms | Six numerical receipts match; timing variation remains |

The final integrated binary is about **1.19x pulse / 1.32x anchor-only** faster
than baseline, or roughly 16% / 24% less internal time. The preceding retained
build measured 1.23x / 1.28x; neither establishes a precise universal speedup.
Millisecond fixtures have
limited timing resolution and exclude process startup; these are not large-model
forecasts, energy measurements or a comparison with conventional dense training.

## What changed

1. **Packed matvec:** AVX2 works across eight independent output lanes, retaining
   scalar summation order within every lane. No FMA, reassociated reduction,
   float master weights or decoded full-model cache was introduced. Unsupported
   CPUs fall back to scalar; `HEXWEAVE_KERNEL=scalar|auto|avx2` controls dispatch.
2. **Avoidable work:** embedding input reads one packed column instead of building
   one-hot input and scanning the matrix. Anchor updates allocate gradients only
   for visited tiles, preserving tile-ID commit order and repeated-visit sums.
   Anchor-only runs no longer build unused predictor-history records.
3. **Measured rejection:** packing four evidence transitions into one byte loop
   regressed the fixture and was removed. Its adversarial evidence tests remain.
   A forward matrix with fewer than eight rows cannot use a full SIMD lane group;
   the 7x16 kernel was slower, so such shapes now dispatch directly to scalar
   (likewise transpose with fewer than eight columns).

The [kernel microbenchmark](../examples/kernel_bench.rs) measures fixed shapes,
validation and allocation together, with seven timed samples after warmup. Its
current version checks every output bit against scalar before timing. Earlier
raw microbench receipts checked only a first-output checksum; the broader SIMD
parity tests supply correctness coverage, not that checksum alone.

## Evidence locations and next bottleneck

Copies of the receipts are in this checkout's `work/performance-simd/`, including
[baseline](../work/performance-simd/baseline.json) and
[final integrated comparison](../work/performance-simd/final-integrated.json).
Source staging and frozen binaries remain in the parent's ignored workspace:
`nonmonotonic-training-lab/work/hexweave-simd/work/bench/`. Receipts include
`baseline.json`, `loop1-scalar.json`, `loop1-avx2.json`, `loop2-avx2.json`,
`loop2-recheck.json`, `loop3-avx2.json`, `loop3-recheck.json`, and
`loop3-retained.json`. Every receipt identifies its executable SHA256.
No Git history, model weights or environment was migrated between projects.

The archived [disk-local text path](../archive/docs/TEXT-PULSE.md) is a separate protocol, not relabeled
evidence for this speedup. It must count reloads, serialized copies, synchronous
file commits, predictor work, anchors and evaluation. Safe next candidates are
compact tile records and a bounded resident cache; either must preserve version
semantics and measure actual RSS and I/O, not merely arithmetic time.
