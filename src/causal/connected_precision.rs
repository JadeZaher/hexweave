//! Explicit connected weight precision and bounded frozen views; see CONNECTED.md.

use super::geometry::MATRIX_COUNT;
use super::shared::Shared;
use super::tile::Tile;
use super::tuning::{Latent, Latents};
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum Precision {
    #[default]
    Ternary,
    Fp32,
}

impl Precision {
    pub fn is_ternary(&self) -> bool {
        *self == Self::Ternary
    }

    pub fn protocol(self) -> &'static str {
        match self {
            Self::Ternary => "causal-connected-v5",
            Self::Fp32 => "causal-connected-fp32-v5",
        }
    }

    /// Only current protocols parse; v1-v4 checkpoints need their pinned binaries.
    pub fn from_protocol(value: &str) -> Result<Self> {
        match value {
            "causal-connected-v5" => Ok(Self::Ternary),
            "causal-connected-fp32-v5" => Ok(Self::Fp32),
            _ => anyhow::bail!("unknown connected precision protocol"),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum View {
    #[default]
    Fp32,
    Int8,
    Ternary,
}

#[derive(Clone)]
pub(super) struct Weights {
    pub version: u64,
    pub mode: View,
    pub matrices: Vec<Vec<f32>>,
}

impl Weights {
    pub fn matrix(&self, index: usize) -> Result<&[f32]> {
        self.matrices
            .get(index)
            .map(Vec::as_slice)
            .context("missing explicit weight matrix")
    }

    pub fn bytes(&self) -> usize {
        self.matrices.iter().map(|matrix| matrix.len() * 4).sum()
    }
}

/// Symmetric per-matrix nearest-level quantization; views retain FP32 arithmetic.
pub(super) fn rounded(values: &[f32], mode: View) -> Result<Vec<f32>> {
    super::norm::finite(values)?;
    ensure!(!values.is_empty(), "empty continuous matrix");
    if mode == View::Fp32 {
        return Ok(values.to_vec());
    }
    let scale = scale(values, mode);
    if scale == 0.0 {
        return Ok(vec![0.0; values.len()]);
    }
    let bound = if mode == View::Int8 { 127.0 } else { 1.0 };
    let result: Vec<_> = values
        .iter()
        .map(|v| (v / scale).round().clamp(-bound, bound) * scale)
        .collect();
    super::norm::finite(&result)?;
    Ok(result)
}

pub(super) fn scale(values: &[f32], mode: View) -> f32 {
    match mode {
        View::Int8 => values.iter().map(|v| v.abs()).fold(0.0f32, f32::max) / 127.0,
        View::Ternary => {
            (values.iter().map(|v| f64::from(v.abs())).sum::<f64>() / values.len() as f64) as f32
        }
        View::Fp32 => 1.0,
    }
}

pub(super) fn fresh(tile: &Tile) -> Result<Latents> {
    let mut values = Vec::new();
    for (matrix, weights) in tile.matrices.iter().enumerate() {
        values.push((
            (tile.id, matrix),
            super::connected_adam::fresh(weights, 0, weights.len())?,
        ));
    }
    let p = &tile.matrices[super::tile_gradients::P];
    let start = 2 * tile.geometry.width * tile.geometry.width;
    values.push((
        (tile.id, MATRIX_COUNT),
        super::connected_adam::fresh(p, start, p.len() - start)?,
    ));
    Ok(values)
}

pub(super) fn validate_masters(tile: &Tile, values: &Latents) -> Result<()> {
    ensure!(
        values.len() == MATRIX_COUNT + 1,
        "FP32 tile requires all master matrices before forward"
    );
    super::connected_store_codec::validate(tile, values)?;
    for (matrix, ((owner, index), _)) in values.iter().enumerate() {
        ensure!(
            *owner == tile.id && *index == matrix,
            "missing FP32 master matrix"
        );
    }
    let start = 2 * tile.geometry.width * tile.geometry.width;
    ensure!(
        values[super::tile_gradients::P].1.weights[start..] == values[MATRIX_COUNT].1.weights,
        "inconsistent FP32 P master rows"
    );
    Ok(())
}

pub(super) fn attach_tile(tile: &mut Tile, values: &Latents, mode: View) -> Result<()> {
    validate_masters(tile, values)?;
    tile.weights = None;
    let matrices = values[..MATRIX_COUNT]
        .iter()
        .map(|(_, latent)| rounded(&latent.weights, mode))
        .collect::<Result<_>>()?;
    tile.weights = Some(Weights {
        version: tile.version,
        mode,
        matrices,
    });
    Ok(())
}

pub(super) fn attach_shared(shared: &mut Shared, latent: &Latent, mode: View) -> Result<()> {
    super::connected_adam::validate(latent, shared.embedding.len())?;
    shared.weights = None;
    shared.weights = Some(Weights {
        version: shared.version,
        mode,
        matrices: vec![rounded(&latent.weights, mode)?],
    });
    Ok(())
}

pub(super) fn matrix(
    weights: &Option<Weights>,
    index: usize,
    version: u64,
) -> Result<Option<&[f32]>> {
    ensure!(
        !super::tuning::current().connected_fp32 || weights.is_some(),
        "missing explicit connected FP32 weight view"
    );
    ensure!(
        weights
            .as_ref()
            .is_none_or(|weights| weights.version == version),
        "stale explicit weight view"
    );
    weights
        .as_ref()
        .map(|weights| weights.matrix(index))
        .transpose()
}
