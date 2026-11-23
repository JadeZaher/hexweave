//! Masked explicit Adam and positive whole-tile update scaling; see AGENTS.md.

use super::connected_adam as adam;
use super::connected_config::Options;
use super::connected_signal::length;
use super::connected_update_scale as scale;
use super::geometry::MATRIX_COUNT;
use super::shared::Shared;
use super::tile::Tile;
use super::tile_gradients::{TileGradients, P, ROUTER};
use super::tuning::{Latent, Latents};
use super::{connected_simd, norm};
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};

pub(super) use super::connected_update_scale::{apply_bias, apply_gate};

pub(super) const ROUTE_P_SLOT: usize = MATRIX_COUNT;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Mode {
    Local,
    RouteOnly,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub(super) struct UpdateMetrics {
    pub transitions: usize,
    pub base_norm: f64,
    pub applied_norm: f64,
    pub gain: f32,
    pub opposed_momentum: usize,
    /// Largest master RMS over the regions this call updated, after the RMS-cap projection.
    pub max_rms: f64,
    /// L2 norm of the updated master regions after commit, from their committed RMS.
    pub weight_norm: f64,
}

struct Proposal {
    matrix: usize,
    slot: usize,
    start: usize,
    delta: Vec<f32>,
}

/// Decoupled decay at the realized step multiplier and the RMS cap act per updated region: P's
/// local and route rows separately.
#[allow(clippy::too_many_arguments)]
pub(super) fn apply(
    tile: &mut Tile,
    latents: &mut Latents,
    gradients: &TileGradients,
    rate: f64,
    gain: f32,
    options: &Options,
    mode: Mode,
) -> Result<UpdateMetrics> {
    gradients.validate(tile.geometry, tile.id)?;
    ensure!(
        tile.weights
            .as_ref()
            .is_none_or(|w| w.mode == super::connected_precision::View::Fp32),
        "cannot update a quantized diagnostic view"
    );
    let max_norm = options.max_update_norm;
    ensure!(
        gain.is_finite() && gain > 0.0 && max_norm.is_finite() && max_norm > 0.0,
        "invalid connected update bound"
    );
    let mut metrics = UpdateMetrics {
        gain,
        ..UpdateMetrics::default()
    };
    let proposals = proposals(tile, latents, gradients, rate, mode, &mut metrics)?;
    metrics.base_norm = metrics.base_norm.sqrt();
    let factor = bound_scale(gain, metrics.base_norm, max_norm);
    // AdamW decays at the realized multiplier, the gain after the norm cap (CONNECTED.md §v5).
    let keep = scale::retention(options, rate * f64::from(factor))?;
    let mut changed = false;
    for mut proposal in proposals {
        connected_simd::scale(&mut proposal.delta, factor)?;
        metrics.applied_norm += length(&proposal.delta).powi(2);
        let committed = scale::commit(
            &mut tile.matrices[proposal.matrix],
            &mut latents[proposal.slot].1,
            proposal.start,
            &proposal.delta,
            keep,
            options.master_rms_cap,
        )?;
        metrics.transitions += committed.transitions;
        metrics.max_rms = metrics.max_rms.max(committed.rms);
        metrics.weight_norm += committed.rms.powi(2) * proposal.delta.len() as f64;
        changed |= committed.changed;
    }
    sync_p(latents, tile);
    latents.sort_by_key(|(key, _)| *key);
    metrics.applied_norm = metrics.applied_norm.sqrt();
    metrics.weight_norm = metrics.weight_norm.sqrt();
    if metrics.transitions > 0 || (tile.weights.is_some() && changed) {
        tile.version = tile
            .version
            .checked_add(1)
            .context("tile version overflow")?;
    }
    if tile.weights.is_some() {
        super::connected_precision::attach_tile(
            tile,
            latents,
            super::connected_precision::View::Fp32,
        )?;
    }
    Ok(metrics)
}

fn proposals(
    tile: &Tile,
    latents: &mut Latents,
    gradients: &TileGradients,
    rate: f64,
    mode: Mode,
    metrics: &mut UpdateMetrics,
) -> Result<Vec<Proposal>> {
    let mut proposals = Vec::new();
    for matrix in 0..MATRIX_COUNT {
        let Some((slot, start, count)) = region(tile, matrix, mode) else {
            continue;
        };
        let index = match latents.iter().position(|(key, _)| *key == (tile.id, slot)) {
            Some(index) => index,
            None => {
                let size = if slot == ROUTE_P_SLOT {
                    count
                } else {
                    tile.matrices[matrix].len()
                };
                let offset = if slot == ROUTE_P_SLOT { start } else { 0 };
                latents.push((
                    (tile.id, slot),
                    adam::fresh(&tile.matrices[matrix], offset, size)?,
                ));
                latents.len() - 1
            }
        };
        let (delta, opposed) = adam::propose(
            &mut latents[index].1,
            &gradients.values[matrix][start..start + count],
            rate,
        )?;
        metrics.opposed_momentum += opposed;
        metrics.base_norm += length(&delta).powi(2);
        proposals.push(Proposal {
            matrix,
            slot: index,
            start,
            delta,
        });
    }
    Ok(proposals)
}

fn region(tile: &Tile, matrix: usize, mode: Mode) -> Option<(usize, usize, usize)> {
    let length = tile.matrices[matrix].len();
    let route_start = 2 * tile.geometry.width * tile.geometry.width;
    match (mode, matrix) {
        (Mode::RouteOnly, ROUTER) => Some((matrix, 0, length)),
        (Mode::RouteOnly, P) => Some((ROUTE_P_SLOT, route_start, length - route_start)),
        (Mode::RouteOnly, _) | (Mode::Local, ROUTER) => None,
        (Mode::Local, P) => Some((matrix, 0, route_start)),
        (Mode::Local, _) => Some((matrix, 0, length)),
    }
}

fn sync_p(latents: &mut Latents, tile: &Tile) {
    let Some(route) = latents
        .iter()
        .position(|(key, _)| *key == (tile.id, ROUTE_P_SLOT))
    else {
        return;
    };
    let Some(local) = latents.iter().position(|(key, _)| *key == (tile.id, P)) else {
        return;
    };
    let start = 2 * tile.geometry.width * tile.geometry.width;
    let route_values = latents[route].1.weights.clone();
    latents[local].1.weights[start..].copy_from_slice(&route_values);
}

/// Tied E takes the tile masters' decay at gain 1 and their RMS cap.
pub(super) fn apply_embedding(
    shared: &mut Shared,
    optimizer: &mut Option<Latent>,
    gradient: &[f32],
    rate: f64,
    options: &Options,
) -> Result<UpdateMetrics> {
    norm::vector(gradient, shared.embedding.len())?;
    ensure!(
        shared
            .weights
            .as_ref()
            .is_none_or(|w| w.mode == super::connected_precision::View::Fp32),
        "cannot update a quantized diagnostic view"
    );
    let max_norm = options.max_update_norm;
    ensure!(
        max_norm.is_finite() && max_norm > 0.0,
        "invalid embedding update bound"
    );
    let keep = scale::retention(options, rate)?;
    if optimizer.is_none() {
        *optimizer = Some(adam::fresh(&shared.embedding, 0, shared.embedding.len())?);
    }
    let latent = optimizer.as_mut().context("missing embedding optimizer")?;
    let (mut delta, opposed_momentum) = adam::propose(latent, gradient, rate)?;
    let base_norm = length(&delta);
    connected_simd::scale(&mut delta, bound_scale(1.0, base_norm, max_norm))?;
    let applied_norm = length(&delta);
    let committed = scale::commit(
        &mut shared.embedding,
        latent,
        0,
        &delta,
        keep,
        options.master_rms_cap,
    )?;
    if committed.transitions > 0 || (shared.weights.is_some() && committed.changed) {
        shared.version = shared
            .version
            .checked_add(1)
            .context("embedding version overflow")?;
    }
    if shared.weights.is_some() {
        super::connected_precision::attach_shared(
            shared,
            latent,
            super::connected_precision::View::Fp32,
        )?;
    }
    Ok(UpdateMetrics {
        transitions: committed.transitions,
        base_norm,
        applied_norm,
        gain: 1.0,
        opposed_momentum,
        max_rms: committed.rms,
        weight_norm: committed.rms * (delta.len() as f64).sqrt(),
    })
}

pub(super) fn bound_scale(gain: f32, norm: f64, bound: f64) -> f32 {
    let factor = f64::from(gain).min(bound / norm.max(f64::MIN_POSITIVE));
    // Leave a small floating-point margin around the uniform norm bound.
    (factor * (1.0 - 2.0 * f64::from(f32::EPSILON))) as f32
}
