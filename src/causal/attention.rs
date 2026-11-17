//! Single-query attention over immutable prefix rows; see STUDY.md.

use super::norm::{self, Rms};
use super::tile::Tile;
use super::tile_gradients::{TileGradients, K, O, Q, V};
use anyhow::Result;

/// Replay scratch: per-head vectors and probabilities, without per-token K/V projections.
pub(super) struct Cache {
    pub normalized: Rms,
    pub query: Vec<f32>,
    pub score_vectors: Vec<f32>,
    pub summaries: Vec<f32>,
    pub probabilities: Vec<f32>,
    pub attended: Vec<f32>,
    pub output: Vec<f32>,
}

/// Attend from one evolving state to the shared byte-plus-position prefix.
pub(super) fn forward(tile: &Tile, input: &[f32], memory: &[f32]) -> Result<Cache> {
    let (d, h) = (tile.geometry.width, tile.geometry.heads);
    let (length, k, scale) = (memory.len() / d, d / h, (d as f32).sqrt().recip());
    let score_scale = scale / (k as f32).sqrt() * super::tuning::current().attention_scale;
    let normalized = norm::rms(input)?;
    let query = tile.project(Q, &normalized.values, scale)?;
    let mut score_vectors = Vec::with_capacity(h * d);
    let mut summaries = vec![0.0; h * d];
    let mut probabilities = vec![0.0; h * length];
    let mut attended = vec![0.0; d];
    for head in 0..h {
        let start = head * k;
        // Factor q_h dot (K_h row) into row dot (K_h^T q_h), then scan the prefix.
        let mut score_vector = tile.transpose_rows(K, start, k, &query[start..start + k])?;
        norm::scale_in_place(&mut score_vector, score_scale)?;
        let scores = &mut probabilities[head * length..(head + 1) * length];
        for (score, row) in scores.iter_mut().zip(memory.chunks_exact(d)) {
            *score = dot(row, &score_vector);
        }
        norm::softmax(scores)?;
        let summary = &mut summaries[head * d..(head + 1) * d];
        // Linearity lets V_h project one weighted summary instead of every prefix row.
        weighted_sum(summary, memory, scores);
        norm::finite(summary)?;
        let mut value = tile.project_rows(V, start, k, summary)?;
        norm::scale_in_place(&mut value, scale)?;
        attended[start..start + k].copy_from_slice(&value);
        score_vectors.extend(score_vector);
    }
    let output = tile.project(O, &attended, scale)?;
    Ok(Cache {
        normalized,
        query,
        score_vectors,
        summaries,
        probabilities,
        attended,
        output,
    })
}

/// Propagate conditional adjoints, optionally accumulating prefix or parameter credit.
pub(super) fn backward(
    tile: &Tile,
    memory: &[f32],
    cache: &Cache,
    upstream: &[f32],
    prefix: Option<&mut [f32]>,
    gradients: Option<&mut TileGradients>,
) -> Result<Vec<f32>> {
    super::attention_reverse::backward(tile, memory, cache, upstream, prefix, gradients)
}

pub(super) fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

pub(super) fn weighted_sum(output: &mut [f32], memory: &[f32], weights: &[f32]) {
    for (row, &weight) in memory.chunks_exact(output.len()).zip(weights) {
        for (sum, &value) in output.iter_mut().zip(row) {
            *sum += weight * value;
        }
    }
}
