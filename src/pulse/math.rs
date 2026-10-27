use anyhow::{ensure, Result};

pub(super) struct Random(u64);

impl Random {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn unit(&mut self) -> f32 {
        // SplitMix64 with midpoint conversion strictly inside (0, 1).
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^= value >> 31;
        ((value >> 41) as f32 + 0.5) / 8_388_608.0
    }
}

pub(super) fn softmax(values: &[f32]) -> Result<Vec<f32>> {
    ensure!(!values.is_empty(), "softmax cannot be empty");
    finite(values)?;
    let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probabilities: Vec<_> = values.iter().map(|v| (v - maximum).exp()).collect();
    let total: f32 = probabilities.iter().sum();
    ensure!(
        total.is_finite() && total > 0.0,
        "invalid softmax normalizer"
    );
    for probability in &mut probabilities {
        *probability /= total;
    }
    Ok(probabilities)
}

pub(super) fn finite(values: &[f32]) -> Result<()> {
    ensure!(
        values.iter().all(|v| v.is_finite()),
        "nonfinite pulse arithmetic"
    );
    Ok(())
}

pub(super) fn add_outer(destination: &mut [f32], left: &[f32], right: &[f32]) {
    assert_eq!(destination.len(), left.len() * right.len());
    for (row, &value) in destination.chunks_exact_mut(right.len()).zip(left) {
        for (entry, &input) in row.iter_mut().zip(right) {
            *entry += value * input;
        }
    }
}

pub(super) fn clipped(values: &[f32]) -> Result<Vec<f32>> {
    finite(values)?;
    let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
    ensure!(norm.is_finite(), "nonfinite correction norm");
    let scale = 1.0 / norm.max(1.0);
    Ok(values.iter().map(|value| value * scale).collect())
}

pub(super) fn loss_and_gradient(logits: &[f32], target: usize) -> Result<(f32, Vec<f32>)> {
    ensure!(target < logits.len(), "target outside vocabulary");
    let mut derivative = softmax(logits)?;
    let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = logits.iter().map(|value| (value - maximum).exp()).sum();
    let loss = maximum + sum.ln() - logits[target];
    derivative[target] -= 1.0;
    ensure!(loss.is_finite(), "nonfinite task loss");
    Ok((loss, derivative))
}
