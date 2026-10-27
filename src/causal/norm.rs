use crate::ternary::TernaryMatrix;
use anyhow::{ensure, Result};

#[derive(Debug)]
pub(super) struct Rms {
    pub values: Vec<f32>,
    pub inverse: f32,
}

pub(super) fn finite(values: &[f32]) -> Result<()> {
    ensure!(
        values.iter().all(|v| v.is_finite()),
        "nonfinite causal arithmetic"
    );
    Ok(())
}

pub(super) fn vector(values: &[f32], length: usize) -> Result<()> {
    ensure!(values.len() == length, "causal vector shape mismatch");
    finite(values)
}

pub(super) fn rms(values: &[f32]) -> Result<Rms> {
    ensure!(!values.is_empty(), "empty RMS input");
    finite(values)?;
    let square: f32 = values.iter().map(|v| v * v).sum();
    ensure!(square.is_finite(), "nonfinite RMS square sum");
    let inverse = (square / values.len() as f32 + 1e-5).sqrt().recip();
    let normalized = values.iter().map(|v| v * inverse).collect();
    Ok(Rms {
        values: normalized,
        inverse,
    })
}

pub(super) fn backward(normalized: &[f32], inverse: f32, upstream: &[f32]) -> Result<Vec<f32>> {
    vector(upstream, normalized.len())?;
    finite(normalized)?;
    ensure!(
        !normalized.is_empty() && inverse.is_finite() && inverse > 0.0,
        "invalid RMS cache"
    );
    let mean = upstream
        .iter()
        .zip(normalized)
        .map(|(b, z)| b * z)
        .sum::<f32>()
        / normalized.len() as f32;
    let result: Vec<_> = upstream
        .iter()
        .zip(normalized)
        .map(|(b, z)| inverse * (b - z * mean))
        .collect();
    finite(&result)?;
    Ok(result)
}

pub(super) fn project(matrix: &TernaryMatrix, input: &[f32], scale: f32) -> Result<Vec<f32>> {
    let mut output = matrix.forward(input)?;
    scale_in_place(&mut output, scale)?;
    Ok(output)
}

pub(super) fn transpose(matrix: &TernaryMatrix, input: &[f32], scale: f32) -> Result<Vec<f32>> {
    let mut output = matrix.transpose(input)?;
    scale_in_place(&mut output, scale)?;
    Ok(output)
}

pub(super) fn scale_in_place(values: &mut [f32], scale: f32) -> Result<()> {
    ensure!(scale.is_finite(), "nonfinite causal scale");
    for value in values.iter_mut() {
        *value *= scale;
    }
    finite(values)
}

pub(super) fn add(destination: &mut [f32], source: &[f32]) -> Result<()> {
    vector(source, destination.len())?;
    for (out, value) in destination.iter_mut().zip(source) {
        *out += value;
    }
    finite(destination)
}

pub(super) fn softmax(values: &mut [f32]) -> Result<()> {
    ensure!(!values.is_empty(), "empty causal softmax");
    finite(values)?;
    let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    for value in values.iter_mut() {
        *value = (*value - maximum).exp();
    }
    let sum: f32 = values.iter().sum();
    ensure!(sum.is_finite() && sum > 0.0, "invalid causal softmax mass");
    for value in values {
        *value /= sum;
    }
    Ok(())
}

pub(super) fn unit_clip(values: &[f32]) -> Result<Vec<f32>> {
    finite(values)?;
    let norm = values.iter().map(|v| v * v).sum::<f32>().sqrt();
    ensure!(norm.is_finite(), "nonfinite correction norm");
    Ok(values.iter().map(|v| v / norm.max(1.0)).collect())
}
