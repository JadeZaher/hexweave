# Connected protocol v3: scale control, saturating curve, unigram bias (plan)

Status: proposed 2026-11-07, not implemented. Supersedes nothing; v2 stays the live protocol until
v3 passes review, sweep and a bounded pilot. Evidence: scratch probes `collapse_probe.py` /
`collapse_probe2.py` (session scratchpad) over checkpoints `a000000086016` (v2) and
`a000000585728` (ablation); summarized in `docs/STATUS.md` §"Calibrated predictor v2 FP32 study".

## Diagnosis (measured, not inferred)

1. Residual-stream explosion. Center tile FP32 masters at 86k anchors: `A` RMS 32.9 (40x the
   ternary init 0.82), `CURVE` 30.4 (37x), `DOWN` 6.5, `V` 2.6, `O` 1.7. One center visit maps a
   prefix state of RMS 1 to RMS 1.1e5; 100% of FFN preactivations sit beyond the outer knots, where
   `ffn.rs::interpolation` extrapolates linearly, so FFN output scales like |A|·|CURVE|·|DOWN|.
2. Hidden from the loss, fatal to the signal. `shared.rs::readout` RMS-normalizes the output, so
   held-out is unaffected (v2 is ahead of the frozen ablation at equal anchors), but
   `shared.rs::task` returns `inverse * (...)` with `inverse = 1/RMS(output)`: the real correction is
   1.4e-5 (probe) vs 0.42 in the ablation; `zero_mse` 1.6e-12 (probe) vs ~1e-12 logged at 90k.
3. Later visits are bypassed: after visit one, `out ≈ x` (cosine 1.0); the route is one tile + E.
4. Driver: all 8 lanes start from the same prefix state (`connected_lanes.rs::drive`), collide on the
   center tile and run sequentially on near-identical inputs; ~10 local updates per anchor hit tile 3
   with the same predicted direction, and scale-free Adam (eps 1e-8) turns tiny consistent gradients
   into full `rate`-sized steps. Cumulative applied norm reached 3.0e3 by anchor 1,007 and 5.7e3 by
   2,003 while `zero_mse` fell 1e-4 → 1.2e-6. The pilot (640 anchors) ended before this showed.
5. E rank collapse onto the unigram: top singular direction correlates |ρ|=0.96 with log byte
   frequency in both runs; v2 σ1/σ2 = 9.7 (ablation 2.7), effective rank 2.3 (4.2), E RMS 3.0 (1.3),
   cos(h, top direction) 0.49 (0.31). The tied readout stores the unigram prior in E's rank-1 part.
6. Amplifiers: calibration `target_energy` lags the live target by ~1000x (EMA 0.001); the 1e-12
   no-evidence threshold is now being crossed, which silently freezes predictor fits and calibration.
7. Memory: 30/32 records are predicted with ~zero corrections (trust 0.01–0.05), so retrieved
   history is ~0 and curve conditioning is inert. Consequence, not cause.

## Changes (one behavior each; no option toggles)

A. Master-weight scale control (`connected_update.rs`, `connected_adam.rs`): decoupled weight decay
   `w *= 1 - decay*rate` on every FP32 master touched by an update (tiles and E), plus a hard
   per-matrix RMS cap (project the matrix back onto RMS ≤ cap after commit). Config:
   `connected.weight_decay` (default 0.01, range (0, 1]) and `connected.master_rms_cap`
   (default 1.5, range [1, 8]); both enter run identity. Log per-tile max matrix RMS.
B. Saturating activation curve (`ffn.rs::interpolation`): clamp `z` to [-2, 2] before segment
   lookup; slope is zero outside the knots. Table test over z ∈ {-5, -2, -1, 0, 1, 2, 5}.
C. Readout bias (`shared.rs`): learned per-byte logit bias `b[256]`, gradient `delta`, own Adam slot,
   decayed like E, persisted in `connected.json` common state, included in state hashes and
   completion/eval views. Finite-difference check against the readout loss.
D. One local Adam step per tile per anchor (`connected_lanes.rs`): accumulate lane gradients per
   tile during the local phase and apply the mean once after all lanes finish (gain = Lambert ×
   trust as now). Lanes still forward, route and publish per visit. This removes the 8–10x coherent
   repetition on the center tile without changing lane count or memory semantics. Counter
   `local_updates` then counts tiles updated per anchor; adjust `connected_state_validate.rs`.
E. Calibration cliff (as implemented): the predictor skips only effectively-zero targets
   (`zero_mse <= 1e-30`; the Lambert gain keeps its 1e-12 evidence guard), clips the normalized
   target `t_hat` to norm 4 against calibration lag, and the calibration EMA is 0.01 after warmup.
F. Observability (`connected_log.rs`, `connected_train.rs`): per log row, mean readout `inverse`
   (1/RMS of the final route output), fraction of FFN preactivations beyond the knots on anchor
   routes, per-tile max master RMS, E RMS and σ1/σ2 (power iteration, cheap). State validation
   fails the step if any master RMS exceeds the cap or output RMS exceeds 1e3: fail loudly.
G. Protocol strings `causal-connected-v3` / `causal-connected-fp32-v3`; loader rejects v2;
   `scripts/core_owner_checkpoint.py` admits v3 (keep v1/v2 admission while those runs exist).

## Pilot and acceptance

30-minute FP32 pilot, seed 7, same config as the v2 full run plus A–G. Pass criteria at ≥2,000
anchors (the point where v2 had already collapsed): max master RMS < 1.5·init, readout output RMS
in [0.5, 5], `zero_mse` ≥ 1e-5, trust not monotonically falling, fits never skipped, E σ1/σ2 < 4.
Then a 24 h run through `scripts/core_owner.py`. Compare against the ablation's windowed held-out
means; no claims beyond measured bits/byte.

## Pilot verdict (2026-11-07, `connected-fp32-v3-pilot-s7-20261107`, 8,580 anchors)

Passed: master RMS ≤ 1.02 (cap never bound), fits never skipped, held-out 4.185 (ahead of v2 and
the ablation near 8k). Failed: readout output RMS 1.45 → 12.7 (probe 18.7 after four visits);
E σ1/σ2 1.5 → 9.7; trust 0.40 → 0.03 and `zero_mse` 2.7e-3 → 1.5e-5 still falling. Probe of
`a000000008580`: masters at init scale but the center tile's attention output has RMS 25 on an
RMS-1 input; V/G/DOWN at 3.0–3.8× the random spectral norm; cos(h, E top direction) 0.69.
Mechanism: alignment, not scale — the RMS cap bounds the Frobenius norm, not the spectral norm,
and decay is isotropic, so neither touches a rank ratio. Decay 0.01 also shrank untouched
outer-tile masters to 0.8× init in 8.6k anchors (e^-6 over a 24 h run). 24 h launch held.

## v3.1 candidates (not implemented)

1. Per-visit residual normalization: scale each tile's output to RMS 1 under the connected
   knob (backward through `norm::backward`), so no visit is bypassed by magnitude and the
   readout correction no longer scales as 1/RMS. Removes the amplification channel.
2. Readout bias initialized to the public training corpus' byte log-frequencies at `State::new`
   (data-derived prior; the corpus manifest already sits in the run identity), so E never needs
   to encode the unigram and the mean readout gradient is near zero from anchor 0. Removes the
   rank-1 drive on E instead of waiting for Adam to learn the bias.
3. `weight_decay` 1e-3 in the configs (e^-0.6 over 200k anchors; the cap stays the hard bound).
4. If σ1/σ2 still climbs in the pilot: Adam epsilon as a scale floor for E (steps proportional
   to the gradient for small consistent coordinates), the deferred item below.

## v4 pilot verdict (2026-11-07, `connected-fp32-v4-pilot2-s7-20261107`, 7,049 anchors)

Passed: E effective rank 66.7/128 (σ1/σ2 3.15), zero_mse flat ~1.4e-3, trust rising to 0.071,
masters ≤ 0.95, held-out 4.206. Raw per-visit RMS mean peaked 2.11 at 4k then fell to 1.29; run
max 3.21 (7% over the 3.0 guess, trend reversed) — accepted. Probe: center attention RMS 2.7
(v3: 25). 24 h run launched. Next review point: the 24 h run's held-out against the frozen-tile
baseline (ablation 400–600k mean 4.009, best 3.940) at equal anchors, and whether trust keeps
rising; if visit_rms_max climbs past 4 or E σ1/σ2 past 5, revisit the deferred items.

## Deferred

- Adam epsilon as a scale floor (1e-8 → 1e-4): revisit if applied/update stays ≥ 0.1 while
  `zero_mse` < 1e-6 in the pilot.
- Per-visit RMS normalization of the residual stream: revisit if the pilot's output RMS drifts
  above 5 despite A+B.
- Unit-normalized correction targets (direction-only credit): revisit only if A–D leave the
  correction scale unstable.
- Memory crowding by zero predicted records: revisit once trust is informative.
