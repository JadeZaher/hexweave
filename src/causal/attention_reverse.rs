use super::attention::{dot, weighted_sum, Cache};
use super::norm;
use super::tile::Tile;
use super::tile_gradients::{self as grad, TileGradients, K, O, Q, V};
use anyhow::Result;

pub(super) fn backward(
    tile: &Tile,
    memory: &[f32],
    cache: &Cache,
    upstream: &[f32],
    mut prefix: Option<&mut [f32]>,
    mut gradients: Option<&mut TileGradients>,
) -> Result<Vec<f32>> {
    let (d, h) = (tile.geometry.width, tile.geometry.heads);
    let (k, scale) = (d / h, (d as f32).sqrt().recip());
    let mix = super::tuning::current().attention_mix;
    let output_adjoint: Vec<_> = upstream.iter().map(|b| mix * b).collect();
    grad::outer(
        gradients.as_deref_mut(),
        O,
        &output_adjoint,
        &cache.attended,
        scale,
    )?;
    let attended_adjoint = tile.transpose(O, &output_adjoint, scale)?;
    let mut query_adjoint = vec![0.0; d];
    for head in 0..h {
        let query = head_backward(
            tile,
            memory,
            cache,
            &attended_adjoint,
            head,
            prefix.as_deref_mut(),
            gradients.as_deref_mut(),
        )?;
        query_adjoint[head * k..(head + 1) * k].copy_from_slice(&query);
    }
    if let Some(prefix) = prefix {
        norm::finite(prefix)?;
    }
    grad::outer(
        gradients,
        Q,
        &query_adjoint,
        &cache.normalized.values,
        scale,
    )?;
    let normalized_adjoint = tile.transpose(Q, &query_adjoint, scale)?;
    let mut result = norm::backward(
        &cache.normalized.values,
        cache.normalized.inverse,
        &normalized_adjoint,
    )?;
    norm::add(&mut result, upstream)?;
    Ok(result)
}

fn head_backward(
    tile: &Tile,
    memory: &[f32],
    cache: &Cache,
    attended_adjoint: &[f32],
    head: usize,
    prefix: Option<&mut [f32]>,
    gradients: Option<&mut TileGradients>,
) -> Result<Vec<f32>> {
    let d = tile.geometry.width;
    let (length, k, scale) = (
        memory.len() / d,
        d / tile.geometry.heads,
        (d as f32).sqrt().recip(),
    );
    let score_scale = scale / (k as f32).sqrt() * super::tuning::current().attention_scale;
    let start = head * k;
    let value_adjoint = &attended_adjoint[start..start + k];
    let mut summary_adjoint = tile.transpose_rows(V, start, k, value_adjoint)?;
    norm::scale_in_place(&mut summary_adjoint, scale)?;
    let probabilities = &cache.probabilities[head * length..(head + 1) * length];
    let scores = score_adjoint(memory, &summary_adjoint, probabilities)?;
    let mut score_vector_adjoint = vec![0.0; d];
    weighted_sum(&mut score_vector_adjoint, memory, &scores);
    norm::finite(&score_vector_adjoint)?;
    let mut query = tile.project_rows(K, start, k, &score_vector_adjoint)?;
    norm::scale_in_place(&mut query, score_scale)?;
    if let Some(gradient) = gradients {
        grad::add_outer(
            &mut gradient.values[K][start * d..(start + k) * d],
            &cache.query[start..start + k],
            &score_vector_adjoint,
            score_scale,
        )?;
        grad::add_outer(
            &mut gradient.values[V][start * d..(start + k) * d],
            value_adjoint,
            &cache.summaries[head * d..(head + 1) * d],
            scale,
        )?;
    }
    if let Some(prefix) = prefix {
        add_prefix(
            prefix,
            &scores,
            probabilities,
            &cache.score_vectors[head * d..(head + 1) * d],
            &summary_adjoint,
        );
    }
    Ok(query)
}

fn score_adjoint(memory: &[f32], summary: &[f32], probabilities: &[f32]) -> Result<Vec<f32>> {
    let mut scores: Vec<_> = memory
        .chunks_exact(summary.len())
        .map(|row| dot(row, summary))
        .collect();
    let mean = dot(probabilities, &scores);
    for (score, &probability) in scores.iter_mut().zip(probabilities) {
        *score = probability * (*score - mean);
    }
    norm::finite(&scores)?;
    Ok(scores)
}

fn add_prefix(
    prefix: &mut [f32],
    scores: &[f32],
    probabilities: &[f32],
    key: &[f32],
    value: &[f32],
) {
    for ((row, &score), &probability) in prefix
        .chunks_exact_mut(key.len())
        .zip(scores)
        .zip(probabilities)
    {
        for ((out, &k), &v) in row.iter_mut().zip(key).zip(value) {
            *out += score * k;
            *out += probability * v;
        }
    }
}
