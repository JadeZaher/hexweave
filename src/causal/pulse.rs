//! Learned history-conditioned gain, shift and route heads; see STUDY.md.

use super::tile::Tile;
use super::tile_gradients::{self as grad, TileGradients, P, U};
use anyhow::Result;

pub(super) struct Cache {
    pub features: Vec<f32>,
    pub latent: Vec<f32>,
    pub heads: Vec<f32>,
}

impl Cache {
    pub fn gain(&self, index: usize) -> f32 {
        1.0 + 0.125 * self.heads[index].tanh()
    }

    pub fn shift(&self, index: usize, width: usize) -> f32 {
        0.125 * self.heads[width + index].tanh()
    }

    /// Three learned knot controls share the pulse projection in the connected protocol.
    pub fn curve_controls(&self) -> [f32; 3] {
        if super::tuning::current().connected {
            std::array::from_fn(|k| 0.125 * self.heads[k].tanh())
        } else {
            [0.0; 3]
        }
    }
}

/// Project current state plus detached correction history into modulation heads.
pub(super) fn forward(tile: &Tile, normalized: &[f32], history: &[f32]) -> Result<Cache> {
    let d = tile.geometry.width;
    let features: Vec<_> = normalized.iter().chain(history).copied().collect();
    let mut latent = tile.project(U, &features, (2.0 * d as f32).sqrt().recip())?;
    for value in &mut latent {
        *value = value.tanh();
    }
    // P rows are [gain: d, shift: d, route bias: r]; routing consumes the final r.
    let heads = tile.project(P, &latent, (d as f32).sqrt().recip())?;
    Ok(Cache {
        features,
        latent,
        heads,
    })
}

pub(super) fn backward(
    tile: &Tile,
    cache: &Cache,
    ffn: &[f32],
    upstream: &[f32],
    curve_adjoint: &[f32; 3],
    mut gradients: Option<&mut TileGradients>,
) -> Result<Vec<f32>> {
    let d = tile.geometry.width;
    let mut heads_adjoint = vec![0.0; cache.heads.len()];
    for j in 0..d {
        heads_adjoint[j] =
            0.25 * ffn[j] * upstream[j] * 0.125 * (1.0 - cache.heads[j].tanh().powi(2));
        heads_adjoint[d + j] =
            0.25 * upstream[j] * 0.125 * (1.0 - cache.heads[d + j].tanh().powi(2));
    }
    if super::tuning::current().connected {
        for k in 0..3 {
            heads_adjoint[k] += curve_adjoint[k] * 0.125 * (1.0 - cache.heads[k].tanh().powi(2));
        }
    }
    let scale = (d as f32).sqrt().recip();
    grad::outer(
        gradients.as_deref_mut(),
        P,
        &heads_adjoint,
        &cache.latent,
        scale,
    )?;
    let mut latent_adjoint = tile.transpose(P, &heads_adjoint, scale)?;
    for (value, &latent) in latent_adjoint.iter_mut().zip(&cache.latent) {
        *value *= 1.0 - latent * latent;
    }
    let context_scale = (2.0 * d as f32).sqrt().recip();
    grad::outer(
        gradients,
        U,
        &latent_adjoint,
        &cache.features,
        context_scale,
    )?;
    let mut input = tile.transpose(U, &latent_adjoint, context_scale)?;
    // Historical corrections are observations: no gradient flows into earlier anchors.
    input.truncate(d);
    Ok(input)
}
