use anyhow::{ensure, Result};

pub(super) fn finite(values: &[f32]) -> Result<()> {
    ensure!(
        values.iter().all(|v| v.is_finite()),
        "nonfinite text arithmetic"
    );
    Ok(())
}

pub(super) fn softmax(values: &[f32]) -> Result<Vec<f32>> {
    finite(values)?;
    ensure!(!values.is_empty(), "empty logits");
    let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut result: Vec<_> = values.iter().map(|v| (v - maximum).exp()).collect();
    let total: f32 = result.iter().sum();
    ensure!(total.is_finite() && total > 0.0, "invalid probability mass");
    for value in &mut result {
        *value /= total;
    }
    Ok(result)
}

pub(super) fn loss(logits: &[f32], target: usize) -> Result<(f32, Vec<f32>)> {
    ensure!(target < logits.len(), "invalid byte target");
    let mut gradient = softmax(logits)?;
    let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let normalizer: f32 = logits.iter().map(|v| (v - maximum).exp()).sum();
    let result = maximum + normalizer.ln() - logits[target];
    gradient[target] -= 1.0;
    ensure!(result.is_finite(), "nonfinite loss");
    Ok((result, gradient))
}

pub(super) fn outer(destination: &mut [f32], left: &[f32], right: &[f32]) {
    assert_eq!(destination.len(), left.len() * right.len());
    for (row, &a) in destination.chunks_exact_mut(right.len()).zip(left) {
        for (value, &b) in row.iter_mut().zip(right) {
            *value += a * b;
        }
    }
}

pub(super) fn clipped(values: &[f32]) -> Result<Vec<f32>> {
    finite(values)?;
    let norm = values.iter().map(|v| v * v).sum::<f32>().sqrt();
    ensure!(norm.is_finite(), "nonfinite norm");
    Ok(values.iter().map(|v| v / norm.max(1.0)).collect())
}

pub(super) fn choose(probabilities: &[f32], random: &mut Option<u64>) -> usize {
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
                return index;
            }
        }
        probabilities.len() - 1
    } else {
        probabilities
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(&a.0)))
            .map(|(index, _)| index)
            .unwrap_or(0)
    }
}

/// FNV-1a is a deterministic identity check, not a cryptographic integrity proof.
pub(super) fn fingerprint(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100_0000_01b3)
    })
}
