//! Gated FFN with three learned curve knots per channel; see STUDY.md.

use super::connected_config::SliceMode;
use super::norm;
use super::tile::Tile;
use super::tile_gradients::{self as grad, TileGradients, A, CURVE, DOWN, G};
use anyhow::Result;

pub(super) struct Cache {
    pub gate: Vec<f32>,
    pub preactivation: Vec<f32>,
    pub curved: Vec<f32>,
    pub product: Vec<f32>,
    pub output: Vec<f32>,
    pub controls: [f32; 3],
    pub mask: Vec<f32>,
    pub down_scale: f32,
}

impl Cache {
    /// Fraction of active FFN channels whose preactivation exceeds the outer knots (|z| > 2).
    pub fn knot_overflow(&self) -> f32 {
        let mut active = 0usize;
        let mut beyond = 0usize;
        for (z, &w) in self.preactivation.iter().zip(&self.mask) {
            if w > 0.0 {
                active += 1;
                beyond += usize::from(z.abs() > 2.0);
            }
        }
        beyond as f32 / active.max(1) as f32
    }
}

/// Compute DOWN(sigmoid(Gx) * curve(Ax)) at one routed state.
#[cfg_attr(not(test), allow(dead_code))]
pub(super) fn forward(tile: &Tile, normalized: &[f32], controls: [f32; 3]) -> Result<Cache> {
    forward_visit(
        tile, normalized, controls, normalized, normalized, &[0.0; 16],
    )
}

/// Compute the FFN forward pass with optional `(input, tile_id)` active-window slicing.
pub(super) fn forward_visit(
    tile: &Tile,
    normalized: &[f32],
    controls: [f32; 3],
    input: &[f32],
    memory: &[f32],
    history: &[f32],
) -> Result<Cache> {
    let scale = (tile.geometry.width as f32).sqrt().recip();
    let mut gate = tile.project(G, normalized, scale)?;
    for value in &mut gate {
        *value = if *value >= 0.0 {
            1.0 / (1.0 + (-*value).exp())
        } else {
            let e = value.exp();
            e / (1.0 + e)
        };
    }
    let preactivation = tile.project(A, normalized, scale)?;
    let mut curved = Vec::with_capacity(tile.geometry.ffn);
    for (j, &z) in preactivation.iter().enumerate() {
        curved.push(curve(tile, j, z, &controls)?.0);
    }
    let visit_seed = seed(tile.id, tile.geometry.width, input, memory, history);
    let (mask, down_scale) = slice_mask(tile.geometry.ffn, tile.id, visit_seed);
    let product: Vec<_> = gate
        .iter()
        .zip(&curved)
        .zip(&mask)
        .map(|((g, c), w)| g * c * w)
        .collect();
    let output = tile.project(DOWN, &product, down_scale)?;
    Ok(Cache {
        gate,
        preactivation,
        curved,
        product,
        output,
        controls,
        mask,
        down_scale,
    })
}

/// Deterministic context/state signature: last 4 prefix rows, routed state signs, history and tile ID.
fn seed(tile_id: usize, width: usize, input: &[f32], memory: &[f32], history: &[f32]) -> u64 {
    let bucket = |slice: &[f32]| -> u64 {
        slice
            .iter()
            .take(32)
            .enumerate()
            .fold(0u64, |acc, (i, &v)| {
                let code = if v > 0.05 {
                    2u64
                } else if v < -0.05 {
                    1u64
                } else {
                    0u64
                };
                acc ^ (code << (2 * i))
            })
    };
    let rows = (memory.len() / width.max(1)).max(1);
    let tail_start = rows.saturating_sub(4) * width.min(memory.len());
    let prefix_sig = bucket(&memory[tail_start..]) ^ ((rows as u64) << 48);
    let input_sig = bucket(input);
    let history_sig = bucket(history);
    super::tuning::hash(tile_id as u64, &[prefix_sig, input_sig, history_sig])
}

/// Active channel weights and `DOWN` scale for the configured `SliceMode`.
fn slice_mask(m: usize, tile_id: usize, visit_seed: u64) -> (Vec<f32>, f32) {
    let mode = super::tuning::current().slice_mode;
    if mode == SliceMode::Full || m == 0 {
        return (vec![1.0; m], (m.max(1) as f32).sqrt().recip());
    }
    let offset = (visit_seed as usize) % m;
    let active = match ((visit_seed >> 32) as usize) % 3 {
        0 => m / 2,
        1 => (3 * m) / 4,
        _ => (7 * m) / 8,
    }
    .max(1);
    let mut mask = if mode == SliceMode::Hybrid {
        let shift = tile_id + ((visit_seed >> 48) as usize);
        (0..m)
            .map(|j| {
                if (j + shift).is_multiple_of(2) {
                    0.5
                } else {
                    0.25
                }
            })
            .collect()
    } else {
        vec![0.0; m]
    };
    for k in 0..active {
        mask[(offset + k) % m] = 1.0;
    }
    let energy: f32 = mask.iter().map(|w| w * w).sum();
    (mask, energy.max(1.0).sqrt().recip())
}

pub(super) fn backward(
    tile: &Tile,
    normalized: &[f32],
    cache: &Cache,
    upstream: &[f32],
    mut gradients: Option<&mut TileGradients>,
) -> Result<(Vec<f32>, [f32; 3])> {
    let (d, m) = (tile.geometry.width, tile.geometry.ffn);
    let scale = (d as f32).sqrt().recip();
    let down_scale = cache.down_scale;
    grad::outer(
        gradients.as_deref_mut(),
        DOWN,
        upstream,
        &cache.product,
        down_scale,
    )?;
    let product_adjoint = tile.transpose(DOWN, upstream, down_scale)?;
    let mut gate_adjoint = vec![0.0; m];
    let mut preactivation_adjoint = vec![0.0; m];
    let mut control_adjoint = [0.0; 3];
    for j in 0..m {
        let masked_adjoint = product_adjoint[j] * cache.mask[j];
        gate_adjoint[j] = masked_adjoint * cache.curved[j] * cache.gate[j] * (1.0 - cache.gate[j]);
        let curve_adjoint = masked_adjoint * cache.gate[j];
        preactivation_adjoint[j] =
            curve_adjoint * curve(tile, j, cache.preactivation[j], &cache.controls)?.1;
        let (segment, fraction, _) = interpolation(cache.preactivation[j]);
        control_adjoint[segment] += curve_adjoint * (1.0 - fraction);
        control_adjoint[segment + 1] += curve_adjoint * fraction;
        if let Some(gradient) = gradients.as_deref_mut() {
            // Each value depends on two adjacent knot offsets, both scaled by 0.25.
            gradient.values[CURVE][segment * m + j] += curve_adjoint * (1.0 - fraction) * 0.25;
            gradient.values[CURVE][(segment + 1) * m + j] += curve_adjoint * fraction * 0.25;
        }
    }
    grad::outer(
        gradients.as_deref_mut(),
        G,
        &gate_adjoint,
        normalized,
        scale,
    )?;
    grad::outer(gradients, A, &preactivation_adjoint, normalized, scale)?;
    let mut result = tile.transpose(G, &gate_adjoint, scale)?;
    let extra = tile.transpose(A, &preactivation_adjoint, scale)?;
    norm::add(&mut result, &extra)?;
    Ok((result, control_adjoint))
}

/// Return the interpolated height and input slope for one channel.
pub(super) fn curve(
    tile: &Tile,
    channel: usize,
    z: f32,
    controls: &[f32; 3],
) -> Result<(f32, f32)> {
    let (segment, fraction, slope) = interpolation(z);
    let m = tile.geometry.ffn;
    let left = segment as f32 - 1.0
        + tile.weight(CURVE, segment * m + channel)? * 0.25
        + controls[segment];
    let right = segment as f32
        + tile.weight(CURVE, (segment + 1) * m + channel)? * 0.25
        + controls[segment + 1];
    Ok((
        left * (1.0 - fraction) + right * fraction,
        (right - left) * 0.5 * slope,
    ))
}

/// Segment, fraction and slope factor for knots at -2, 0 and 2. Connected curves saturate
/// outside the knots (clamped point, slope 0); other paths extrapolate linearly (factor 1).
fn interpolation(value: f32) -> (usize, f32, f32) {
    let (value, slope) = if super::tuning::current().connected && value.abs() > 2.0 {
        (value.clamp(-2.0, 2.0), 0.0)
    } else {
        (value, 1.0)
    };
    if value < 0.0 {
        (0, (value + 2.0) * 0.5, slope)
    } else {
        (1, value * 0.5, slope)
    }
}
