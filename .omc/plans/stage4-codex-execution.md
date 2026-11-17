# Stage 4 execution detail — prepared 2026-11-02

This implements core-next-stages.md after dc785bf and the owner's explicit
decision to preserve original v1 variable history and defer one-anchor history.
No run in this document is complete merely because its config is prepared.

## Completed software gate

Independent source and post-commit review approved dc785bf. All nonignored
library/integration contracts and Python checks passed across the integrated
sweep and affected reruns. Saved public v1 scores reproduced exactly.

## Immediate profiles

Sequential B=1,8,32; 3008 anchors, rate .003, seed7, ternary, history on,
explore0, constant decay. Existing profile config shape (16 spans, eval and
checkpoint interval1000), fresh profile-v2-bB-20261102 outputs. Each goes
through scripts.core_owner, 900s cap. Compare training time separately from
end-to-end wall because interval crossings/final evaluation differ by B.

## Rate scan (first arm complete; second arm active)

Two sequential B32 arms, rates .003 and .017. Same seed7/corpus/window512,
history on, explore0, constant decay; 64 held-out spans. Ninety-minute loop
deadline per arm (finalization adds elapsed time), initial and final evaluation, checkpoints approximately every
15 minutes of training (anchor interval set from measured B32 throughput).
The scan compares final frozen held-out scores at matched wall budgets,
reporting actual bytes/optimizer steps too. Small differences remain provisional;
if within .02 bits/byte, re-score both final checkpoints on128 spans before
selection, keeping the .02 tie preference for the lower rate.

If that tie-break is needed, use the existing owner-controlled frozen-rescore
path sequentially for each arm: copy its preserved final checkpoint and the
exact scan-pinned executable into a fresh `work/core-runs/<rescore>` output;
copy its config with that output, `resume=true`, `anchors` equal to the saved
anchor count, and `eval_spans=128`. The scan-era source at `dc785bf` checks
the anchor limit before another batch and excludes these fields from identity.
Verify copied hashes against the scan receipts before launch, and afterward
require terminal `completed`, unchanged anchors and optimizer steps, and 16,384
held-out targets. The owner can mutate or quarantine a bad copy in the fresh
output, so retain both preserved originals separately. Prepopulate the pinned
binary: allowing the owner to copy the current build could change numerics.

Continue the selected arm under its pinned binary for an EIGHT-HOUR TOTAL
budget including its scan, not eight additional hours. The other scan arm
remains a bounded ninety-minute comparison. No cloud, downloads or larger
models. Choose recurring eval_every >=20*E*R, rounded upward to a multiple32,
where E is measured64-span evaluation seconds and R measured training anchors/s.
Owner wall limits, Job cap, STOP and receipts remain the admission mechanisms.

The .003 arm completed and its final checkpoint is preserved; .017 is active
under owner PID79720 (trainer72380, session46168). See the cutover handoff.
Require its current owner end receipt, zero exit, expected terminal, finite
final metrics, matching pinned hashes, checkpoint/report anchor agreement,
and released lock before launching the next arm. Resource stops and owner
errors halt the sequence for inspection. Compare final frozen own-mode scores.
Before resuming a selected arm, preserve both scan-final checkpoints and
their manifests/reports/receipts outside ordinary newest-three rotation.
The eight-hour total uses actual owner elapsed time: subtract the selected
scan receipt's wall_seconds and allow for continuation finalization explicitly.

The preregistered B1 history-on/off re-ablation (100k anchors,128 spans at80k
and100k,same seed,+.02 mean difference rule) remains outstanding. With the
owner's history decision it does not remove the variable train/eval window
asymmetry. Record that limitation explicitly rather than claiming it does.

## Required report

Best/final frozen held-out and alternate mode, per-hour and per-byte v1 and
n-gram comparisons, train-ahead pairs, per-tile routes/optimizer steps,
throughput, owned-process memory, and four continuations. No learning or
performance claims from software tests or the short profiling runs.

Comparison caveat found during the scan: historical v1 curves and
`ngram-curve.json` use 16 held-out spans; the rate scan uses 64. Do not present
their differences as matched-sample effects. Re-score selected saved v1 states
and n-gram baselines on matching spans before a strong cross-protocol claim,
or report the historical curves explicitly as contextual, unmatched estimates.
Avoid CPU-heavy re-scoring concurrently with timed training arms.
