# Connected v3 implementation seams (binding for all lanes)

Companion to `connected-v3-scale-control-2026-11-07.md` (the diagnosis and change list A–G).
Three lanes edit one working tree concurrently. Each owns its files exclusively; cross-lane needs
go through the interfaces below, exactly as written. If a lane must deviate, it reports the
deviation instead of editing another lane's file.

## Ownership

Lane 1 (Rust: scale, curve, bias, protocol, persistence):
`connected_config.rs`, `connected_update.rs`, `connected_adam.rs`, `ffn.rs`, `tile.rs`,
`shared.rs`, `weights.rs`, `connected_state.rs`, `connected_precision.rs`,
`connected_checkpoint.rs`, `connected_eval.rs`, `connected_completion*.rs`,
`connected_numeric_contracts.rs`, `connected_storage_contracts.rs`, `connected_curve_contracts.rs`,
`connected_precision_contracts.rs`. New module (if needed): `connected_update_scale.rs`, registered
by inserting `mod connected_update_scale;` directly after `mod connected_update;` in `mod.rs`.

Lane 2 (Rust: lanes, step, predictor, log, validation):
`connected_lanes.rs`, `connected_lane_job.rs`, `connected_step.rs`, `connected_anchor.rs`,
`connected_predictor.rs`, `connected_calibration.rs`, `connected_gain.rs`, `connected_log.rs`,
`connected_train.rs`, `connected_state_validate.rs`, `connected_contracts.rs`,
`connected_predictor_contracts.rs`. New module (if needed): `connected_lane_apply.rs`, registered
by inserting `mod connected_lane_apply;` directly after `mod connected_lanes;` in `mod.rs`.

Lane 3 (Python, docs, configs): `scripts/core_owner_checkpoint.py`, `scripts/AGENTS.md`,
`tests/test_core_owner_recovery.py`, `src/causal/CONNECTED.md`, `src/causal/AGENTS.md`,
`README.md`, `docs/ARCHITECTURE.md`, `work/core-runs/connected-fp32-v3-*.json`.
`docs/STATUS.md` is reserved for the orchestrator.

Everything else is read-only for every lane. Nobody runs the full sweep, `cargo test` across the
crate, or commits. Allowed: `cargo check` at the end (report only errors in owned files; the other
Rust lane's half-finished code may fail), one run of the lane's own new tests
(`cargo test <module_name>`), and `python scripts/check_structure.py`.

## Interfaces (Lane 1 provides, Lane 2 consumes)

```rust
// connected_config.rs — both fields serde-defaulted, validated, part of run identity.
pub struct Options { /* existing */ pub weight_decay: f64 /* default 0.01, finite, (0, 1] */,
                     pub master_rms_cap: f64 /* default 1.5, finite, [1, 8] */ }

// connected_update.rs — `max_norm: f64` parameters are replaced by `options: &Options`.
pub(super) fn apply(tile: &mut Tile, latents: &mut Latents, gradients: &TileGradients,
                    rate: f64, gain: f32, options: &Options, mode: Mode) -> Result<UpdateMetrics>;
pub(super) fn apply_embedding(shared: &mut Shared, optimizer: &mut Option<Latent>,
                    gradient: &[f32], rate: f64, options: &Options) -> Result<UpdateMetrics>;
pub(super) fn apply_bias(shared: &mut Shared, optimizer: &mut Option<Latent>,
                    gradient: &[f32], rate: f64, options: &Options) -> Result<UpdateMetrics>;
pub(super) struct UpdateMetrics { /* existing */ pub max_rms: f64 /* max master RMS over the
                    matrices this call touched, measured after projection */ }
```
Semantics: before adding Adam's delta, every coefficient in each updated region is multiplied by
`1 - options.weight_decay * rate` (decoupled decay; `Mode::RouteOnly` decays only route regions).
After commit, any touched matrix whose master RMS exceeds `options.master_rms_cap` is scaled as a
whole to exactly the cap (ternary trits re-derived from the scaled masters; version advances as
today). `apply_bias` is Adam plus decay on the 256-entry readout bias with no RMS cap; `max_rms`
reports the bias RMS. The uniform `max_update_norm` cap and the gain stay as they are.

```rust
// shared.rs
pub(super) struct Shared { /* existing */ pub bias: Option<Vec<f32>> /* VOCAB, None outside the
                    connected protocol */ }
pub(super) struct Task { /* existing */ pub bias_gradient: Option<Vec<f32>> /* = softmax - onehot
                    when bias is Some */, pub output_rms: f32 /* RMS of the routed output before
                    normalization, i.e. 1/inverse */ }
// Readout::logits include the bias when Some. `Shared::new` leaves bias None; connected
// `State::new` (connected_state.rs) sets Some(vec![0.0; VOCAB]).

// tile.rs
pub(super) struct Forward { /* existing */ pub knot_overflow: f32 /* fraction of FFN preactivations
                    with |z| > 2 on this forward */ }

// connected_state.rs
pub(super) struct State { /* existing */ pub bias_latent: Option<Latent> /* Adam for the bias,
                    persisted like shared_latent */ }
pub(super) struct Counters { /* existing */ pub readout_rms_sum: f64, pub knot_overflow_sum: f64,
                    pub master_rms_max: [f32; TILES] /* running max per tile */ }
```
`ffn::interpolation` clamps `z` to `[-2, 2]` whenever `tuning::current().connected` is set; the
value and slope returned by `curve` are both computed from the clamped `z` (slope 0 outside).
Non-connected numerics stay bit-identical. Protocol strings become `causal-connected-v3` and
`causal-connected-fp32-v3`; `Precision::from_protocol` accepts only those.

## Lane 2 obligations

- `connected_anchor::Frame` gains `knot_overflow: f32` from `Forward`; `Credit` gains
  `bias_gradient: Option<Vec<f32>>` and `output_rms: f32` from `Task`.
- `connected_step.rs`: after `apply_embedding`, call `apply_bias` when `credit.bias_gradient` is
  Some; `ensure!(credit.output_rms <= 1e3, "connected residual stream exceeded its bound")`;
  accumulate `readout_rms_sum`, `knot_overflow_sum` (mean over the anchor's frames) and
  `master_rms_max` (from every `UpdateMetrics.max_rms`).
- `connected_lanes.rs` / `connected_lane_job.rs`: jobs never apply updates. A job with
  `train && gain > 0` returns its local `TileGradients` in `Outcome.gradient`; `drive` sums
  gradients per tile with a count, releases every lease read-only, and after all lanes finish
  applies the mean gradient once per tile (`gain = Lambert value_at × trust` evaluated then,
  `Mode::Local`), updating `local_updates` (now ≤ TILES per anchor), transitions, opposed momentum,
  applied/base norm sums, boosted/damped and `master_rms_max`. Record publication order and
  `local_visits` are unchanged.
- `connected_predictor.rs` / `connected_calibration.rs`: the skip threshold in `fit` becomes
  `zero_mse <= 1e-30` (exact zeros only; the gain keeps its 1e-12 guard); the normalized target
  `t_hat` is clipped to norm ≤ 4 before alignment and Adam; after the 1,000-observation warmup the
  calibration EMA weight is 0.01.
- `connected_log.rs` / `connected_train.rs`: rows and report add `readout_rms` and
  `knot_overflow` (interval means), `master_rms` (per tile), `embedding_rms` and
  `embedding_spectral_ratio` (σ1/σ2 of the E master by power iteration with one deflation, 32
  iterations, computed at log time from `state.shared_latent`).
- `connected_state_validate.rs`: keep `local_updates <= predicted_records`; add
  `master_rms_max[t] <= options.master_rms_cap * (1 + 1e-5)` for every tile.

## Lane 3 obligations

- `scripts/core_owner_checkpoint.py`: `PROTOCOLS` adds the v3 strings after v2 (keep v1/v2 while
  those runs exist); `tests/test_core_owner_recovery.py` covers v3 admitted and mixed generations
  refused, extending the existing table rather than adding a harness.
- `src/causal/CONNECTED.md`: header says v3; new section "Scale control, saturating curve, readout
  bias and per-anchor local steps (v3)" stating A–F as implemented behavior with the measured v2
  failure as motivation, one paragraph each, design statements only (no evidence claims).
  `src/causal/AGENTS.md`, `README.md`, `scripts/AGENTS.md`, `docs/ARCHITECTURE.md`: replace v2
  protocol mentions with v3 where they describe the current binary; do not rewrite history.
- Configs: `work/core-runs/connected-fp32-v3-pilot-s7-20261107.json` (wall 1800 s) and
  `connected-fp32-v3-full-s7-20261107.json` (wall 86400 s) copied from
  `connected-fp32-v2-full-s7-20261107.json` with `out` renamed and `connected` gaining
  `"weight_decay": 0.01, "master_rms_cap": 1.5` explicitly.

## v4 addendum (2026-11-07, after the v3 pilot; see the plan's "Pilot verdict")

Two lanes. Lane R (Rust): `tile.rs`, `connected_state.rs`, `connected_state_validate.rs`,
`connected_train.rs`, the corpus source file that can count bytes (`core_corpus.rs` or the
`TrainingSource` impl it uses), `connected_precision.rs`, `connected_log.rs`, and the test files
`connected_curve_contracts.rs`, `connected_precision_contracts.rs`, `connected_storage_contracts.rs`,
`connected_contracts.rs`, `connected_numeric_contracts.rs`, `connected_completion_contracts.rs`,
`connected_precision_contracts.rs`. Lane P (Python/docs/configs): `scripts/core_owner_checkpoint.py`,
`tests/test_core_owner_recovery.py`, `src/causal/CONNECTED.md`, `src/causal/AGENTS.md`, `README.md`,
`scripts/AGENTS.md`, `work/core-runs/connected-fp32-v4-*.json`. `docs/STATUS.md` stays with the
orchestrator.

Interfaces and semantics:
- `tile.rs`: under `tuning::current().connected`, the tile output is RMS-normalized
  (`norm::rms(raw).values`, RMS 1); `Cache` keeps that `Rms`; `backward` first maps `upstream`
  through `norm::backward(normalized_output, inverse, upstream)` and then runs the existing adjoint
  chain on the raw output. `Forward.output` is the normalized output. Non-connected numerics stay
  bit-identical. Name it: post-visit RMS normalization of the residual stream.
- `connected_state.rs`: new `State::install_readout_prior(&mut self, prior: &[f32]) -> Result<()>`
  (VOCAB entries, finite), allowed only while `anchors == 0` and `bias_latent` is None; it sets
  `shared.bias`. `validate_bias`'s no-latent branch requires only `anchors == 0` and a finite bias
  (the all-zero requirement goes away). The first `apply_bias` creates the master from the current
  bias as today.
- `connected_train.rs`: on a fresh start only (not resume), compute
  `prior[b] = ln((count_b + 0.5) / (total + 128))` over the configured training corpus bytes (the
  same train file the cursor reads; one bounded pass, no new config) and install it before the
  first step; the report and the first log row carry `"readout_prior": "corpus-log-unigram"`.
  Resume leaves the restored bias alone.
- Protocol strings `causal-connected-v4` / `causal-connected-fp32-v4`; `from_protocol` accepts only
  those; Python `PROTOCOLS` appends v4 after v3 (v1–v3 stay admitted while their runs exist).
- Configs: `connected-fp32-v4-pilot-s7-20261107.json` (1800 s) and
  `connected-fp32-v4-full-s7-20261107.json` (86400 s), copied from the v3 configs with `out`
  renamed and `weight_decay` 0.001 (decay 0.01 shrank untouched masters 20% in 8.6k anchors).

Pilot acceptance (v4, at ≥ 8,000 anchors), restated after the v4 review: `readout_rms` ≈ 1 is only
the check that normalization is on (it is ≈1.000005 by construction); the alignment signal is the
raw per-visit RMS before normalization, logged as `visit_rms` (interval mean) and `visit_rms_max`
(run maximum) — require `visit_rms_max` ≤ 3 (the v3 probe's amplified center visit would read
about 6.3); E σ1/σ2 < 4; mean trust ≥ 0.05; interval `zero_mse` ≥ 1e-4; masters ≤ cap; held-out
≤ 4.3 at 8,192. Review note: post-visit normalization damps residual-path credit through a visit by
that visit's raw 1/RMS (Post-LN property), so a rising `visit_rms` is also a credit-damping signal.

## Style (all lanes)

Files under 300 lines, functions under 90, one-line doc comments, rationale in CONNECTED.md or the
directory AGENTS.md, blank line between functions, name the algorithm where one is used. Tests:
smallest change that protects a real flow; extend existing fixtures; table-driven for related
inputs; assert observable outcomes (RMS bound held, fits not skipped, one update per tile per
anchor), never counters alone when a behavior is visible.
