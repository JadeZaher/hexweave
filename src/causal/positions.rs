use anyhow::{ensure, Result};

pub(super) fn add(position: usize, embedding: &mut [f32]) -> Result<()> {
    ensure!(
        matches!(embedding.len(), 16 | 128) && position < 1024,
        "invalid sinusoidal position shape"
    );
    let width = embedding.len();
    for j in 0..width / 2 {
        let phase = position as f64 * 10000_f64.powf(-2.0 * j as f64 / width as f64);
        embedding[2 * j] += phase.sin() as f32;
        embedding[2 * j + 1] += phase.cos() as f32;
    }
    super::norm::finite(embedding)
}
