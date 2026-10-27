//! Test-only optimizer knobs; `cfg(not(test))` builds always run `Knobs::PINNED`. See AGENTS.md §tuning.

use crate::ternary::TernaryMatrix;
use anyhow::{ensure, Result};

/// How a tuned rate moves trits; only test builds construct non-default variants.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(not(test), allow(dead_code))]
pub(super) enum Update {
    /// Stochastic rounding, probability min(rate*|g|/rms(g), 1); flip rate is loss-independent.
    Normalized,
    /// Stochastic rounding, probability min(rate*|g|, 1).
    Raw,
    /// Persistent latent weights stepped by Adam; trits threshold latents at ±0.5.
    Latent,
}

/// Exploration overrides; `PINNED` reproduces threshold-one numerics bit for bit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Knobs {
    /// Tile-matrix rate; None keeps the threshold-one sign step.
    pub tile_rate: Option<f64>,
    /// Router-matrix override of `tile_rate`.
    pub router_rate: Option<f64>,
    /// Tied-E rate; None keeps the assay rule.
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
    };

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

/// Latent weights and interleaved Adam moments for one ternary matrix.
pub(super) struct Latent {
    pub weights: Vec<f32>,
    pub moments: Vec<f32>,
    pub step: u64,
}

#[cfg(test)]
thread_local! {
    static KNOBS: std::cell::Cell<Knobs> = const { std::cell::Cell::new(Knobs::PINNED) };
    static CALLS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    static LATENTS: std::cell::RefCell<std::collections::HashMap<(usize, usize), Latent>> =
        std::cell::RefCell::new(std::collections::HashMap::new());
}

#[cfg(not(test))]
pub(super) fn current() -> Knobs {
    Knobs::PINNED
}

#[cfg(test)]
pub(super) fn current() -> Knobs {
    KNOBS.with(std::cell::Cell::get)
}

/// Install knobs for this thread; restart draw streams and discard latent state.
#[cfg(test)]
pub(super) fn set(knobs: Knobs) {
    KNOBS.with(|cell| cell.set(knobs));
    CALLS.with(|cell| cell.set(0));
    LATENTS.with(|cell| cell.borrow_mut().clear());
}

/// Restores pinned knobs when dropped, so tuned settings never outlive their test.
#[cfg(test)]
pub(super) struct Scoped;

#[cfg(test)]
impl Drop for Scoped {
    fn drop(&mut self) {
        set(Knobs::PINNED);
    }
}

/// Install knobs until the returned guard drops.
#[cfg(test)]
pub(super) fn scoped(knobs: Knobs) -> Scoped {
    set(knobs);
    Scoped
}

/// Replace knobs mid-run, continuing draw streams and latent state.
#[cfg(test)]
pub(super) fn anneal(knobs: Knobs) {
    KNOBS.with(|cell| cell.set(knobs));
}

#[cfg(test)]
fn next_call() -> u64 {
    CALLS.with(|cell| {
        let call = cell.get();
        cell.set(call + 1);
        call
    })
}

#[cfg(not(test))]
fn next_call() -> u64 {
    0
}

#[cfg(test)]
fn take_latent(key: (usize, usize)) -> Option<Latent> {
    LATENTS.with(|cell| cell.borrow_mut().remove(&key))
}

#[cfg(not(test))]
fn take_latent(_key: (usize, usize)) -> Option<Latent> {
    None
}

#[cfg(test)]
fn put_latent(key: (usize, usize), latent: Latent) {
    LATENTS.with(|cell| cell.borrow_mut().insert(key, latent));
}

#[cfg(not(test))]
fn put_latent(_key: (usize, usize), _latent: Latent) {}

/// One tuned update of `matrix` keyed by (owner, matrix index); returns trit transitions.
pub(super) fn update(
    knobs: &Knobs,
    rate: f64,
    key: (usize, usize),
    matrix: &mut TernaryMatrix,
    gradient: &[f32],
) -> Result<usize> {
    ensure!(rate.is_finite() && rate >= 0.0, "invalid tuned rate");
    let stream = || stream(knobs.seed, key.0, key.1);
    match knobs.update {
        Update::Normalized => {
            matrix.apply_stochastic(gradient, effective(rate, gradient), stream())
        }
        Update::Raw => matrix.apply_stochastic(gradient, rate, stream()),
        Update::Latent => {
            let mut latent = match take_latent(key) {
                Some(latent) => latent,
                None => Latent {
                    weights: (0..matrix.len())
                        .map(|index| matrix.value(index).map(f32::from))
                        .collect::<Result<_>>()?,
                    moments: vec![0.0; 2 * matrix.len()],
                    step: 0,
                },
            };
            latent.step += 1;
            let changed = matrix.apply_latent(
                &mut latent.weights,
                &mut latent.moments,
                gradient,
                rate,
                latent.step,
            );
            put_latent(key, latent);
            changed
        }
    }
}

/// Fresh counter-based draw stream per stochastic update call.
pub(super) fn stream(seed: u64, owner: usize, matrix: usize) -> u64 {
    hash(seed, &[owner as u64, matrix as u64, next_call()])
}

/// Pure SplitMix64 fold of a seed and coordinates; equal inputs always give equal output.
pub(super) fn hash(seed: u64, values: &[u64]) -> u64 {
    values
        .iter()
        .fold(mix(seed), |state, &value| mix(state ^ value))
}

/// Rate divided by this update's gradient RMS.
pub(super) fn effective(rate: f64, gradient: &[f32]) -> f64 {
    let square = gradient.iter().map(|g| f64::from(*g).powi(2)).sum::<f64>();
    let rms = (square / gradient.len().max(1) as f64).sqrt();
    if rms > 0.0 {
        rate / rms
    } else {
        0.0
    }
}

// SplitMix64 finalizer: https://prng.di.unimi.it/splitmix64.c
fn mix(value: u64) -> u64 {
    let mut x = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}
