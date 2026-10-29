//! Numerics profile: pinned threshold-one (v3/assay) or the core latent path; see AGENTS.md §core.

use crate::ternary::TernaryMatrix;
use anyhow::{ensure, Result};
use serde::Serialize;
use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;

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

/// Latent weights and interleaved Adam moments for one ternary matrix.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct Latent {
    pub weights: Vec<f32>,
    pub moments: Vec<f32>,
    pub step: u64,
}

thread_local! {
    static KNOBS: Cell<Knobs> = const { Cell::new(Knobs::PINNED) };
    static CALLS: Cell<u64> = const { Cell::new(0) };
    static LATENTS: RefCell<BTreeMap<(usize, usize), Latent>> = const { RefCell::new(BTreeMap::new()) };
}

/// The profile for this thread; pinned unless `install` was called.
pub(super) fn current() -> Knobs {
    KNOBS.with(Cell::get)
}

/// Install a profile for this thread; restart draw streams and discard latent state.
pub(super) fn install(knobs: Knobs) {
    KNOBS.with(|cell| cell.set(knobs));
    CALLS.with(|cell| cell.set(0));
    LATENTS.with(|cell| cell.borrow_mut().clear());
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

/// Run `f` on `key`'s latent weights when float-forward mode is on and they exist.
pub(super) fn with_float<R>(key: (usize, usize), f: impl FnOnce(&[f32]) -> R) -> Option<R> {
    if !current().float_forward {
        return None;
    }
    LATENTS.with(|cell| cell.borrow().get(&key).map(|latent| f(&latent.weights)))
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

/// Replace knobs mid-run, continuing draw streams and latent state.
#[cfg(test)]
pub(super) fn anneal(knobs: Knobs) {
    KNOBS.with(|cell| cell.set(knobs));
}

fn next_call() -> u64 {
    CALLS.with(|cell| {
        let call = cell.get();
        cell.set(call + 1);
        call
    })
}

/// Latent states keyed (owner, matrix); owner `TILES` is the tied E.
pub(super) type Latents = Vec<((usize, usize), Latent)>;

/// Test helper: off-lattice latents (half the trit plus a sine ripple) for the given matrices.
#[cfg(test)]
pub(super) fn ripple_latents(matrices: &[((usize, usize), &TernaryMatrix)]) -> Result<()> {
    let mut latents = Vec::with_capacity(matrices.len());
    for (key, matrix) in matrices {
        let weights = (0..matrix.len())
            .map(|i| {
                Ok(
                    (0.5 * f32::from(matrix.value(i)?) + 0.3 * (i as f32 * 0.7).sin())
                        .clamp(-1.0, 1.0),
                )
            })
            .collect::<Result<Vec<f32>>>()?;
        let moments = vec![0.0; 2 * weights.len()];
        latents.push((
            *key,
            Latent {
                weights,
                moments,
                step: 0,
            },
        ));
    }
    import(latents, 0);
    Ok(())
}

/// Each tracked latent's Adam step count, keyed like `LATENTS`; no weight or moment data copied.
pub(super) fn steps() -> Vec<((usize, usize), u64)> {
    LATENTS.with(|cell| {
        cell.borrow()
            .iter()
            .map(|(&key, latent)| (key, latent.step))
            .collect()
    })
}

/// Copy of the latent state and draw counter, for checkpoints.
pub(super) fn export() -> (Latents, u64) {
    let latents = LATENTS.with(|cell| {
        cell.borrow()
            .iter()
            .map(|(key, latent)| (*key, latent.clone()))
            .collect()
    });
    (latents, CALLS.with(Cell::get))
}

/// Replace the latent state and draw counter from a checkpoint.
pub(super) fn import(latents: Latents, calls: u64) {
    LATENTS.with(|cell| *cell.borrow_mut() = latents.into_iter().collect());
    CALLS.with(|cell| cell.set(calls));
}

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
            let existing = LATENTS.with(|cell| cell.borrow_mut().remove(&key));
            let mut latent = match existing {
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
            LATENTS.with(|cell| cell.borrow_mut().insert(key, latent));
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
