//! Numerics profile: pinned threshold-one (v3/assay) or the core latent path; see AGENTS.md §core.

use serde::Serialize;
use std::cell::Cell;

/// How a tuned rate moves trits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[cfg_attr(not(test), allow(dead_code))]
pub(super) enum Update {
    /// Stochastic rounding, probability min(rate*|g|/rms(g), 1); flip rate is loss-independent.
    Normalized,
    /// Stochastic rounding, probability min(rate*|g|, 1).
    Raw,
    /// Persistent latent weights stepped by Adam; trits threshold latents at ±0.5.
    Latent,
}

/// Numerics profile; `PINNED` reproduces threshold-one numerics bit for bit. Serialized into
/// the core run identity, so changing a field or `Knobs::core` makes old checkpoints unresumable.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub(super) struct Knobs {
    /// Tile-matrix rate; None keeps the threshold-one sign step.
    pub tile_rate: Option<f64>,
    /// Router-matrix override of `tile_rate`.
    pub router_rate: Option<f64>,
    /// Tied-E rate; None keeps the pinned rule.
    pub embed_rate: Option<f64>,
    pub update: Update,
    /// Readout logit divisor; exactly 1 keeps pinned numerics.
    pub temperature: f32,
    /// Attention score multiplier; exactly 1 keeps pinned numerics.
    pub attention_scale: f32,
    /// Residual weight of the attention output; pinned 0.25.
    pub attention_mix: f32,
    /// Argmax instead of sampled routes during training.
    pub greedy_routes: bool,
    /// Remove the halt action so every route uses all visits.
    pub no_halt: bool,
    /// Added to the training route seed; pinned 0.
    pub route_salt: u64,
    pub seed: u64,
    /// Compute with each matrix's f32 latent weights instead of its trits (`weights.rs`).
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub float_forward: bool,
    /// Zero history input; skips records, the history push and predictor training; see AGENTS.md §core.
    #[serde(skip_serializing_if = "is_true")]
    pub history: bool,
    pub batch: u32,
    pub decay: super::core_schedule::Decay,
    pub explore: f64,
    pub history_ticks: u64,
    pub fit_predictor: bool,
    /// Enable the connected protocol's memory-conditioned curve.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub connected: bool,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub connected_fp32: bool,
    #[serde(skip_serializing_if = "super::connected_config::SliceMode::is_default")]
    pub slice_mode: super::connected_config::SliceMode,
    /// Post-visit RMS normalization when `connected`; false is `connected.visit_norm: false`.
    #[serde(skip_serializing_if = "is_true")]
    pub visit_norm: bool,
}

impl Knobs {
    pub const PINNED: Self = Self {
        tile_rate: None,
        router_rate: None,
        embed_rate: None,
        update: Update::Normalized,
        temperature: 1.0,
        attention_scale: 1.0,
        attention_mix: 0.25,
        greedy_routes: false,
        no_halt: false,
        route_salt: 0,
        seed: 0,
        float_forward: false,
        history: true,
        batch: 1,
        decay: super::core_schedule::Decay::Constant,
        explore: 0.0,
        history_ticks: 8,
        fit_predictor: true,
        connected: false,
        connected_fp32: false,
        slice_mode: super::connected_config::SliceMode::Full,
        visit_norm: true,
    };

    /// Core path (causal-latent-v1): latent Adam on tiles and E, greedy routes, pinned forward.
    pub fn core(rate: f64, seed: u64) -> Self {
        Self {
            tile_rate: Some(rate),
            embed_rate: Some(rate),
            update: Update::Latent,
            greedy_routes: true,
            seed,
            ..Self::PINNED
        }
    }

    /// Non-default numerics the gradient oracles also check (temperature, scale, mix).
    #[cfg(test)]
    pub const ORACLE: Self = Self {
        temperature: 4.0,
        attention_scale: 4.0,
        attention_mix: 1.0,
        ..Self::PINNED
    };

    /// Tuned rate for one tile matrix, or None for the pinned sign step.
    pub fn rate(&self, matrix: usize) -> Option<f64> {
        if matrix == super::tile_gradients::ROUTER {
            self.router_rate.or(self.tile_rate)
        } else {
            self.tile_rate
        }
    }
}

/// Serde helper: omit a `bool` field from the identity payload when it holds its default `true`.
fn is_true(value: &bool) -> bool {
    *value
}

thread_local! {
    static KNOBS: Cell<Knobs> = const { Cell::new(Knobs::PINNED) };
}

/// The profile for this thread; pinned unless `install` was called.
pub(super) fn current() -> Knobs {
    KNOBS.with(Cell::get)
}

/// Install a profile for this thread; restart draw streams and discard latent state.
pub(super) fn install(knobs: Knobs) {
    KNOBS.with(|cell| cell.set(knobs));
    super::tuning_update::reset();
}

/// Switch float-forward mode only, keeping latents and draws (for the alternate-mode evaluation).
pub(super) fn set_float_forward(on: bool) {
    KNOBS.with(|cell| {
        cell.set(Knobs {
            float_forward: on,
            ..cell.get()
        })
    });
}

/// Restores pinned knobs when dropped, so tuned settings never outlive their test.
#[cfg(test)]
pub(super) struct Scoped;

#[cfg(test)]
impl Drop for Scoped {
    fn drop(&mut self) {
        install(Knobs::PINNED);
    }
}

/// Install knobs until the returned guard drops.
#[cfg(test)]
pub(super) fn scoped(knobs: Knobs) -> Scoped {
    install(knobs);
    Scoped
}

/// Switch connected slice mode and visit normalization to `options`, keeping latents and draws.
pub(super) fn set_connected(options: &super::connected_config::Options) {
    KNOBS.with(|cell| {
        cell.set(Knobs {
            slice_mode: options.slice_mode,
            visit_norm: options.visit_norm,
            ..cell.get()
        })
    });
}

/// Replace knobs mid-run, continuing draw streams and latent state.
pub(super) fn anneal(knobs: Knobs) {
    KNOBS.with(|cell| cell.set(knobs));
}

// `Latent`, the tuned update rules and the per-call draw-counter stream live in
// `tuning_update.rs` (see AGENTS.md §Numerics profile); re-exported so existing `tuning::`
// call sites are unchanged.
#[cfg(test)]
pub(super) use super::tuning_update::{export, ripple_latents};
pub(super) use super::tuning_update::{
    hash, import, steps, update, with_float, with_latents, Latent, Latents,
};
pub(super) use super::tuning_update::{put_tile, take_tile};
