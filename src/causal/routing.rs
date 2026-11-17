//! Learned neighbor-or-halt scores and detached policy credit; see STUDY.md.

use super::norm;
use super::tile::Tile;
use super::tile_gradients::{self as grad, TileGradients, P, ROUTER};
use anyhow::{ensure, Result};

/// Score normalized output plus correction history, with a learned pulse bias.
pub(super) fn probabilities(
    tile: &Tile,
    output: &[f32],
    heads: &[f32],
    history: &[f32],
) -> Result<Vec<f32>> {
    let (d, r) = (tile.geometry.width, tile.geometry.routes(tile.id)?);
    norm::vector(output, d)?;
    norm::vector(history, d)?;
    norm::vector(heads, 2 * d + r)?;
    let normalized = norm::rms(output)?;
    let features: Vec<_> = normalized.values.iter().chain(history).copied().collect();
    let mut values = tile.project(ROUTER, &features, (2.0 * d as f32).sqrt().recip())?;
    for (i, value) in values.iter_mut().enumerate() {
        *value += 0.25 * heads[2 * d + i];
    }
    norm::softmax(&mut values)?;
    Ok(values)
}

pub(super) fn choose(probabilities: &[f32], random: &mut Option<u64>) -> Result<usize> {
    norm::finite(probabilities)?;
    ensure!(
        !probabilities.is_empty() && probabilities.iter().all(|p| *p >= 0.0),
        "invalid route probabilities"
    );
    if let Some(state) = random {
        // SplitMix64: https://prng.di.unimi.it/splitmix64.c
        *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = *state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^= value >> 31;
        let draw = ((value >> 41) as f32 + 0.5) / 8_388_608.0;
        let mut sum = 0.0;
        for (index, &probability) in probabilities.iter().enumerate() {
            sum += probability;
            if draw < sum {
                return Ok(index);
            }
        }
        Ok(probabilities.len() - 1)
    } else {
        // Core routing is greedy; equal scores select the lowest legal action index.
        Ok(probabilities
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(&a.0)))
            .unwrap()
            .0)
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn policy(
    tile: &Tile,
    output: &[f32],
    heads: &[f32],
    latent: &[f32],
    history: &[f32],
    selected: usize,
    advantage: f32,
    gradient: &mut TileGradients,
) -> Result<()> {
    let (d, r) = (tile.geometry.width, tile.geometry.routes(tile.id)?);
    ensure!(
        selected < r && advantage.is_finite(),
        "invalid detached policy observation"
    );
    // Detached score-function credit: (loss - baseline) * grad(log p[selected]).
    let mut score = probabilities(tile, output, heads, history)?;
    for (index, value) in score.iter_mut().enumerate() {
        *value = advantage
            * (if index == selected {
                1.0 - *value
            } else {
                -*value
            });
    }
    let normalized = norm::rms(output)?;
    let features: Vec<_> = normalized.values.iter().chain(history).copied().collect();
    grad::outer(
        Some(&mut *gradient),
        ROUTER,
        &score,
        &features,
        (2.0 * d as f32).sqrt().recip(),
    )?;
    // Policy credit updates ROUTER and P's route rows, not U or upstream state.
    grad::add_outer(
        &mut gradient.values[P][2 * d * d..],
        &score,
        latent,
        0.25 / (d as f32).sqrt(),
    )
}
