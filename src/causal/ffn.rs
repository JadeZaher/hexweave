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
}

pub(super) fn forward(tile: &Tile, normalized: &[f32]) -> Result<Cache> {
    let scale = (tile.geometry.width as f32).sqrt().recip();
    let mut gate = norm::project(&tile.matrices[G], normalized, scale)?;
    for value in &mut gate {
        *value = if *value >= 0.0 {
            1.0 / (1.0 + (-*value).exp())
        } else {
            let e = value.exp();
            e / (1.0 + e)
        };
    }
    let preactivation = norm::project(&tile.matrices[A], normalized, scale)?;
    let mut curved = Vec::with_capacity(tile.geometry.ffn);
    for (j, &z) in preactivation.iter().enumerate() {
        curved.push(curve(tile, j, z)?.0);
    }
    let product: Vec<_> = gate.iter().zip(&curved).map(|(g, c)| g * c).collect();
    let output = norm::project(
        &tile.matrices[DOWN],
        &product,
        (tile.geometry.ffn as f32).sqrt().recip(),
    )?;
    Ok(Cache {
        gate,
        preactivation,
        curved,
        product,
        output,
    })
}

pub(super) fn backward(
    tile: &Tile,
    normalized: &[f32],
    cache: &Cache,
    upstream: &[f32],
    mut gradients: Option<&mut TileGradients>,
) -> Result<Vec<f32>> {
    let (d, m) = (tile.geometry.width, tile.geometry.ffn);
    let scale = (d as f32).sqrt().recip();
    let down_scale = (m as f32).sqrt().recip();
    grad::outer(
        gradients.as_deref_mut(),
        DOWN,
        upstream,
        &cache.product,
        down_scale,
    )?;
    let product_adjoint = norm::transpose(&tile.matrices[DOWN], upstream, down_scale)?;
    let mut gate_adjoint = vec![0.0; m];
    let mut preactivation_adjoint = vec![0.0; m];
    for j in 0..m {
        gate_adjoint[j] =
            product_adjoint[j] * cache.curved[j] * cache.gate[j] * (1.0 - cache.gate[j]);
        let curve_adjoint = product_adjoint[j] * cache.gate[j];
        preactivation_adjoint[j] = curve_adjoint * curve(tile, j, cache.preactivation[j])?.1;
        if let Some(gradient) = gradients.as_deref_mut() {
            let (segment, fraction) = interpolation(cache.preactivation[j]);
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
    let mut result = norm::transpose(&tile.matrices[G], &gate_adjoint, scale)?;
    let extra = norm::transpose(&tile.matrices[A], &preactivation_adjoint, scale)?;
    norm::add(&mut result, &extra)?;
    Ok(result)
}

fn curve(tile: &Tile, channel: usize, z: f32) -> Result<(f32, f32)> {
    let (segment, fraction) = interpolation(z);
    let m = tile.geometry.ffn;
    let left =
        segment as f32 - 1.0 + f32::from(tile.matrices[CURVE].value(segment * m + channel)?) * 0.25;
    let right =
        segment as f32 + f32::from(tile.matrices[CURVE].value((segment + 1) * m + channel)?) * 0.25;
    Ok((
        left * (1.0 - fraction) + right * fraction,
        (right - left) * 0.5,
    ))
}

fn interpolation(value: f32) -> (usize, f32) {
    if value < 0.0 {
        (0, (value + 2.0) * 0.5)
    } else {
        (1, value * 0.5)
    }
}
