//! Decoupled weight decay, master RMS cap and readout-bias steps; see CONNECTED.md §Scale control.

use super::connected_adam as adam;
use super::connected_config::Options;
use super::connected_signal::length;
use super::connected_update::{bound_scale, UpdateMetrics};
use super::geometry::VOCAB;
use super::shared::Shared;
use super::tuning::Latent;
use super::{connected_simd, norm};
use crate::ternary::TernaryMatrix;
use anyhow::{ensure, Context, Result};

/// One committed master region: trit transitions, continuous change and final RMS.
pub(super) struct Committed {
    pub transitions: usize,
    pub changed: bool,
    pub rms: f64,
}

/// Fraction of each touched master kept by decoupled weight decay (AdamW; Loshchilov & Hutter
/// 2019, https://arxiv.org/abs/1711.05101): `1 - weight_decay * rate` at the effective rate.
pub(super) fn retention(options: &Options, rate: f64) -> Result<f64> {
    let keep = 1.0 - options.weight_decay * rate;
    ensure!(
        options.weight_decay > 0.0 && rate > 0.0 && keep.is_finite() && (0.0..=1.0).contains(&keep),
        "invalid connected weight decay or rate"
    );
    Ok(keep)
}

/// Decay the region, add Adam's delta, cap the region's RMS, then re-derive its trits.
pub(super) fn commit(
    matrix: &mut TernaryMatrix,
    latent: &mut Latent,
    start: usize,
    delta: &[f32],
    keep: f64,
    cap: f64,
) -> Result<Committed> {
    ensure!(
        delta.len() <= latent.weights.len() && start + delta.len() <= matrix.len(),
        "connected commit shape mismatch"
    );
    let region = &mut latent.weights[..delta.len()];
    let moved = decay_add(region, delta, keep)?;
    let (rms, projected) = project(region, cap)?;
    let mut values = (0..matrix.len())
        .map(|i| matrix.value(i))
        .collect::<Result<Vec<_>>>()?;
    let mut transitions = 0;
    for (i, weight) in region.iter().enumerate() {
        let trit = adam::trit(*weight);
        transitions += usize::from(values[start + i] != trit);
        values[start + i] = trit;
    }
    *matrix = TernaryMatrix::from_values(matrix.rows(), matrix.cols(), &values)?;
    Ok(Committed {
        transitions,
        changed: moved || projected,
        rms,
    })
}

/// `w = keep * w + delta` with one f64 rounding per coefficient; reports whether any moved.
fn decay_add(weights: &mut [f32], delta: &[f32], keep: f64) -> Result<bool> {
    ensure!(
        weights.len() == delta.len(),
        "connected decay shape mismatch"
    );
    let mut moved = false;
    for (weight, update) in weights.iter_mut().zip(delta) {
        let next = (keep * f64::from(*weight) + f64::from(*update)) as f32;
        ensure!(next.is_finite(), "nonfinite connected latent");
        moved |= next != *weight;
        *weight = next;
    }
    Ok(moved)
}

/// Euclidean projection onto the RMS ball: uniform rescale to the cap when above it.
fn project(values: &mut [f32], cap: f64) -> Result<(f64, bool)> {
    ensure!(
        cap.is_finite() && cap > 0.0,
        "invalid connected master RMS cap"
    );
    let before = rms(values);
    if before <= cap {
        return Ok((before, false));
    }
    // The same small floating-point margin as the uniform update-norm bound.
    let factor = cap / before * (1.0 - 2.0 * f64::from(f32::EPSILON));
    for value in values.iter_mut() {
        *value = (f64::from(*value) * factor) as f32;
    }
    Ok((rms(values), true))
}

/// Root-mean-square of a master region, accumulated in f64.
pub(super) fn rms(values: &[f32]) -> f64 {
    let square: f64 = values.iter().map(|v| f64::from(*v).powi(2)).sum();
    (square / values.len().max(1) as f64).sqrt()
}

/// `apply_bias` for the one-weight context gate master `g = latent.weights[0]`.
pub(super) fn apply_gate(
    latent: &mut Latent,
    gradient: f32,
    rate: f64,
    options: &Options,
) -> Result<UpdateMetrics> {
    let keep = retention(options, rate)?;
    let bound = options.max_update_norm;
    ensure!(
        bound.is_finite() && bound > 0.0,
        "invalid context gate update bound"
    );
    adam::validate(latent, 1)?;
    let (mut delta, opposed_momentum) = adam::propose(latent, &[gradient], rate)?;
    let base_norm = length(&delta);
    connected_simd::scale(&mut delta, bound_scale(1.0, base_norm, bound))?;
    let applied_norm = length(&delta);
    decay_add(&mut latent.weights, &delta, keep)?;
    Ok(UpdateMetrics {
        transitions: 0,
        base_norm,
        applied_norm,
        gain: 1.0,
        opposed_momentum,
        max_rms: rms(&latent.weights),
        weight_norm: length(&latent.weights),
    })
}

/// Adam plus decoupled decay on the FP32 readout bias; uniform norm bound, no RMS cap.
pub(super) fn apply_bias(
    shared: &mut Shared,
    optimizer: &mut Option<Latent>,
    gradient: &[f32],
    rate: f64,
    options: &Options,
) -> Result<UpdateMetrics> {
    let keep = retention(options, rate)?;
    let bound = options.max_update_norm;
    ensure!(
        bound.is_finite() && bound > 0.0,
        "invalid readout bias update bound"
    );
    let bias = shared
        .bias
        .as_mut()
        .context("missing connected readout bias")?;
    norm::vector(gradient, VOCAB)?;
    norm::vector(bias, VOCAB)?;
    let latent = optimizer.get_or_insert_with(|| Latent {
        weights: bias.clone(),
        moments: vec![0.0; 2 * VOCAB],
        step: 0,
    });
    ensure!(
        latent.weights == *bias,
        "stale connected readout bias master"
    );
    let (mut delta, opposed_momentum) = adam::propose(latent, gradient, rate)?;
    let base_norm = length(&delta);
    connected_simd::scale(&mut delta, bound_scale(1.0, base_norm, bound))?;
    let applied_norm = length(&delta);
    decay_add(&mut latent.weights, &delta, keep)?;
    bias.copy_from_slice(&latent.weights);
    Ok(UpdateMetrics {
        transitions: 0,
        base_norm,
        applied_norm,
        gain: 1.0,
        opposed_momentum,
        max_rms: rms(&latent.weights),
        weight_norm: length(&latent.weights),
    })
}
