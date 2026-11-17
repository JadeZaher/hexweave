---
type: plan
date: 2026-11-06
status: authorized
---

# Calibrated connected predictor (protocol v2) and removal of the gate option

Owner direction (2026-11-06): ablation first, then make the predictor learnable;
launch fresh from-scratch runs; remove the previous study's artifacts and code
rather than keeping options that confuse context; keep the code clean.

## Diagnosis this targets

`connected_predictor.rs` is one `tanh(W f / sqrt(F))` layer with `unit_clip`,
trained by Adam at a fixed rate 0.003 directly on raw real corrections. Real
corrections shrank to norm ~0.01 (component ~1e-3). Adam's step is scale-free
(`m/sqrt(v)`), so each fit moves the output by roughly the same amount whatever
the target size; the predictor's output jitter therefore exceeded the targets
(fit error 12–189 times the zero-prediction error across both FP32 studies,
cosine 0.002–0.04). Predicted records were also ~100 times larger than observed
records in shared memory. Without the target byte the predictable part of a
correction is only its conditional expectation, so low single-sample cosine is
expected even for a good predictor; magnitude is what must be calibrated.

## Design (no option; one behavior)

Per tile k, keep running statistics over observed fits (running mean for the
first 1,000 observations, then EMA with weight 0.001):
- `target_energy[k]` = mean |t|^2; normalization scale `nu_k = sqrt(target_energy[k])`
  (use the current |t| on the first observation).
- `alignment[k]` = mean p_hat . t_hat; `prediction_energy[k]` = mean p_hat . p_hat,
  where `t_hat = t / nu_k` and `p_hat = unit_clip(tanh(...))` (the network output).
- `observations[k]`.

The network trains on `t_hat` (unit-scale targets), not raw `t`.
Trust: `trust_k = clamp(alignment[k] / max(prediction_energy[k], 1e-12), 0, 1)`;
zero before the first observation. This is the least-squares shrinkage slope.
Calibrated prediction for tile k: `nu_k * trust_k * p_hat`. It is what lanes use
for local credit, what predicted records publish, and what FitMetrics compare
with the real target (so the Lambert gain sees calibrated evidence).
No-evidence fits (|t|^2/d <= 1e-12, the gain's existing threshold) still count
in `fits` (accounting invariant) but change neither weights nor statistics.

Local update step: `Lambert gain (value_at) * trust_k`. Zero skips the job through
the existing skip path (`connected_lane_job.rs` `train && job.gain > 0.0`, lanes
commit only when updated). Route updates from actual anchor credit keep the plain
Lambert gain. State validation: `local_updates <= predicted_records` always.
Check `connected_update` gain validation accepts `0 < gain <= gain_max`.

Protocol bump: `causal-connected-v2` and `causal-connected-fp32-v2` (Rust and
`scripts/core_owner_checkpoint.py`). v1 checkpoints are rejected by the new
binary; earlier runs stay usable through their own pinned binaries.

Remove entirely: `Gate` enum and `connected.gate` option, `Gain::local`, gate
validation, log/report `gate` fields, gate tests, CONNECTED.md fail-closed
section, src/causal/AGENTS.md gate pointer. Log/report add per-tile `trust`.

## Tests (lean, flow-shaped)

- Predictor learning flow, table-driven over a fixed feature stream: consistent
  target `0.01*u` -> trust >= 0.5, cosine(pred, u) >= 0.9, |pred| within 2x of
  0.01; sign-alternating `+/-0.01*u` -> trust <= 0.1, |pred| <= 0.002.
- Keep threaded/sequential/checkpoint parity for ternary and FP32.
- Keep a test proving predicted credit changes real local weights (pre-fit
  consistent observations so trust > 0 if needed).
- Update existing tests for the new `predict` signature and protocol strings.

## Sequence

Ablation (gate "off", 8 h) launches first from the pre-change binary. Then
implement, independent review, one sweep, release build, 640-anchor pilot,
fresh 24-hour FP32 run, seed 7, identical data/optimizer settings.
