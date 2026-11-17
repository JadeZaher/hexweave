//! Explicit Adam proposals, before whole-update gain and norm limiting; see AGENTS.md.

use super::norm;
use super::tuning::Latent;
use crate::ternary::TernaryMatrix;
use anyhow::{ensure, Context, Result};

pub(super) fn fresh(matrix: &TernaryMatrix, start: usize, length: usize) -> Result<Latent> {
    ensure!(
        start
            .checked_add(length)
            .is_some_and(|end| end <= matrix.len()),
        "invalid latent region"
    );
    Ok(Latent {
        weights: (start..start + length)
            .map(|i| matrix.value(i).map(f32::from))
            .collect::<Result<_>>()?,
        moments: vec![0.0; 2 * length],
        step: 0,
    })
}

pub(super) fn validate(latent: &Latent, length: usize) -> Result<()> {
    norm::vector(&latent.weights, length)?;
    norm::vector(&latent.moments, 2 * length)?;
    ensure!(
        latent.moments.iter().skip(1).step_by(2).all(|v| *v >= 0.0),
        "negative Adam second moment"
    );
    Ok(())
}

/// Adam uses separate clocks for local P rows and policy P rows; https://arxiv.org/abs/1412.6980.
pub(super) fn propose(
    latent: &mut Latent,
    gradient: &[f32],
    rate: f64,
) -> Result<(Vec<f32>, usize)> {
    ensure!(
        rate.is_finite() && rate > 0.0 && rate <= 1.0,
        "invalid connected update rate"
    );
    ensure!(
        gradient.len() <= latent.weights.len(),
        "gradient exceeds latent region"
    );
    validate(latent, latent.weights.len())?;
    norm::finite(gradient)?;
    latent.step = latent.step.checked_add(1).context("Adam step overflow")?;
    let exponent = latent.step.min(1_048_576) as i32;
    let (b1, b2) = (1.0 - 0.9f64.powi(exponent), 1.0 - 0.999f64.powi(exponent));
    let mut delta = Vec::with_capacity(gradient.len());
    let mut opposed = 0;
    for (index, gradient) in gradient.iter().enumerate() {
        let g = f64::from(*gradient);
        let m = 0.9 * f64::from(latent.moments[2 * index]) + 0.1 * g;
        let v = 0.999 * f64::from(latent.moments[2 * index + 1]) + 0.001 * g * g;
        latent.moments[2 * index] = m as f32;
        latent.moments[2 * index + 1] = v as f32;
        let update = (-rate * (m / b1) / ((v / b2).sqrt() + 1e-8)) as f32;
        opposed += usize::from(f64::from(update) * g > 0.0);
        delta.push(update);
    }
    norm::finite(&delta)?;
    validate(latent, latent.weights.len())?;
    Ok((delta, opposed))
}

pub(super) fn trit(value: f32) -> i8 {
    if value > 0.5 {
        1
    } else if value < -0.5 {
        -1
    } else {
        0
    }
}
