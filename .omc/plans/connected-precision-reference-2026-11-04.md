# Connected precision reference: authorized implementation

Owner direction: "lets start with the highest precision we can and quantize down".
Implement FP32 forward weights and activations with the existing higher-precision
optimizer intermediates, then assess lower-precision views separately. This is
the practical full-precision CPU reference, not an FP64-everywhere implementation.
The old study remains on its original binary until graceful replacement. No new
run is claimed until independent review, validation and launch evidence exist.

## Evidence and competing explanations

The corrected decoder faithfully emits checkpoint-186164 score maxima. Twelve
additional true-corpus prefix probes still emit spaces after supplied correct
spaces and letters. Two active-route FFN projections have at most three row
directions, with nearly all shadow weights far outside fixed +/-0.5 thresholds.
See docs/STATUS.md and work/inference/decoder-audit-20261104 for independently
verified artifacts. This supports investigating the interaction between local
credit, optimization and ternary projection. It does not establish which caused
the redundancy, nor inevitability for ternary models generally.

Current connected arithmetic already retains FP32 masters, Adam moments,
activations and predictor parameters. Continuous forward weights are the
missing control. Reading this badly drifted checkpoint's shadows as float would
change its scale and is not equivalent to training an FP32 reference.

## Implementation boundary

- Add an explicit connected precision identity; reject mixed checkpoint modes.
- Supply continuous weights directly from resident leases and shared optimizer
  state to every forward, input-adjoint and readout operation. Never silently
  fall back to trits when continuous weights are missing.
- Preserve the four-bundle bound, exclusive leases, checkpoint hash checking and
  explicit ownership. The old TLS float hook is insufficient for this store.
- Initialize all continuous weights before their first read. Advance numerical
  revision/identity on continuous updates even when no ternary threshold is
  crossed; memory freshness and fingerprints must track the actual computation.
- Preserve the predictor, shared-memory policy, activation curves, local lanes,
  routing policy, gains, gradient/update bounds and corrected generation policy.
  Do not restore an anchor-only learning path to obtain float support.

## Comparison and launch sequence

The earlier ternary study already started fresh at seed 7. Start the FP32 study
fresh with that same seed and initial coefficients (initial trits represented
exactly as FP32), public corpus order and optimizer hyperparameters. Use the
earlier study as a historical comparator at matched saved checkpoints/observed
byte counts where available; this sequence does not launch another ternary arm.
Record wall time separately. A new paired/repeated control may later be needed
for stronger attribution. The recipe is not claimed optimal for either mode.

Assess fixed-context next-byte predictions, greedy free generation, sensitivity
to changed prefixes, route usage, recent predictor error versus zero/cosine,
per-matrix row redundancy, shadow magnitude and ternary transition rates. Use
quality and representation evidence together; loss improvement alone does not
establish useful language generation. Repeat before making broad conclusions.

FP32 is the initial CPU numerical reference. BF16/FP8 or gradual compression are
subsequent efficiency experiments after continuous learning is demonstrated;
FP8 introduces scaling, rounding and format choices of its own. Higher precision
does not remove the need to audit unstable or uninformative local updates.

Before any run, implement and independently review numerical, checkpoint,
resource and ownership behavior; then perform one integrated validation sweep
and use the existing capped owner with a separate fresh output directory.

The explicit weight view adds at most one FP32 coefficient buffer per resident
tile and one for shared E. It does not create an all-tile resident weight map;
the added four-tile array payload is about 4.5 MiB for this candidate. Canonical
masters and Adam state remain checkpointed, fingerprinted and independently
owned. The extra resident bytes must be included in measured pilot usage.

Prepared configs:
- work/core-runs/connected-fp32-pilot-s7-20261104.json: fresh 640 anchors or 900
  seconds, four workers, eight lanes, 512-byte window; it crosses a full context
  window and saves intermediate/final checkpoints.
- work/core-runs/connected-fp32-full-s7-20261104.json: separate fresh weights,
  seed 7, rate 0.003, 24 hours or one 21,006,506-byte corpus pass. Same core
  numerical options as the earlier connected study except forward precision.

After independent source review, request the old run's graceful STOP, verify
final checkpoint/receipt/owner cleanup, and preserve its state. Run the integrated
sweep and release build once fixes are complete. Execute the bounded pilot under
the existing 512 MiB Job cap, verify fresh counters, precision identity, forward
progress, saved-state integrity and resource usage. Exercise read-only FP32,
scaled-int8 and scaled-ternary completion views on the pilot checkpoint. These
round tile/embedding weights while leaving the predictor and arithmetic FP32.
They do not provide native FP8 kernels or compressed checkpoint exports. A pilot
result does not establish language quality.

Launch the full fresh config only after the pilot and independent verification
pass. Use bounded memory recovery, retain source/config/binary receipts, confirm
the initial state and advancing counters, and report the live owner/trainer.

## Execution evidence

Independent source review passed after complete-master validation and bounded
view-refresh fixes. The old ternary owner stopped cleanly at 236,228 anchors;
all final payload hashes and process/lock cleanup were independently verified.
Final state and receipts are preserved under
work/core-runs/connected-full-s7-20261103-preserved/a000000236228.

Initial sweep befea10123434aadade18aab67e0641b passed structure/format and found
one unused Rust import in Clippy. After that sole fix, affected and remaining
checks in 31c0aacf40284a84bc3ed8b35ea22da5 passed: strict Clippy, 133 Rust tests
(three existing ignored), 26 Python tests, three planning profiles and smoke.
Release build and actual pilot/quantization checks are the remaining pre-launch
requirements; their outcomes must be recorded before the full fresh launch.
