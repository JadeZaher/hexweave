use super::math::{clipped, fingerprint, loss, outer};
use super::node::Frame;
use super::{CONTEXT, FEATURES, VOCAB, WIDTH};
use crate::ternary::{Evidence, TernaryMatrix};
use crate::topology::Mesh;
use anyhow::{ensure, Result};
use serde_json::{json, Value};

pub(super) struct Shared {
    pub embedding: TernaryMatrix,
    pub readout: TernaryMatrix,
    pub predictor: TernaryMatrix,
    pub version: u64,
    pub transitions: [usize; 3],
    embedding_evidence: Evidence,
    readout_evidence: Evidence,
    predictor_evidence: Evidence,
}

impl Shared {
    pub fn new(seed: u64) -> Result<Self> {
        Ok(Self {
            embedding: TernaryMatrix::new(WIDTH, VOCAB, seed)?,
            readout: TernaryMatrix::new(VOCAB, WIDTH, seed.wrapping_add(1))?,
            predictor: TernaryMatrix::new(WIDTH, FEATURES, seed.wrapping_add(2))?,
            version: 0,
            transitions: [0; 3],
            embedding_evidence: Evidence::new(WIDTH * VOCAB)?,
            readout_evidence: Evidence::new(WIDTH * VOCAB)?,
            predictor_evidence: Evidence::new(WIDTH * FEATURES)?,
        })
    }

    pub fn input(&self, context: &[u8]) -> Result<Vec<f32>> {
        ensure!(context.len() == CONTEXT, "wrong context length");
        let mut hidden = vec![0.0; WIDTH];
        for (position, &token) in context.iter().enumerate() {
            let column = self.embedding.column(token as usize)?;
            for (value, embedding) in hidden.iter_mut().zip(column) {
                *value += embedding * position_scale(position);
            }
        }
        Ok(hidden)
    }

    pub fn task(&self, hidden: &[f32], target: usize) -> Result<(f32, Vec<f32>)> {
        let logits: Vec<_> = self
            .readout
            .forward(hidden)?
            .iter()
            .map(|v| v / 4.0)
            .collect();
        loss(&logits, target)
    }

    pub fn features(&self, frame: &Frame, history: &[f32], mesh: &Mesh) -> Vec<f32> {
        let coordinate = mesh.coordinates()[frame.id];
        frame
            .input
            .iter()
            .chain(&frame.output)
            .chain(history)
            .copied()
            .chain([coordinate.q as f32, coordinate.r as f32])
            .collect()
    }

    pub fn predict(&self, features: &[f32]) -> Result<Vec<f32>> {
        Ok(self
            .predictor
            .forward(features)?
            .iter()
            .map(|v| 0.5 * (v / (FEATURES as f32).sqrt()).tanh())
            .collect())
    }

    pub fn train_predictor(&mut self, features: &[f32], correction: &[f32]) -> Result<f32> {
        let target = clipped(correction)?;
        let predicted = self.predict(features)?;
        let error = predicted
            .iter()
            .zip(&target)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / WIDTH as f32;
        let derivative: Vec<_> = predicted
            .iter()
            .zip(target)
            .map(|(p, t)| {
                (p - t) / WIDTH as f32 * (1.0 - (2.0 * p).powi(2)) / (FEATURES as f32).sqrt()
            })
            .collect();
        let mut gradient = vec![0.0; WIDTH * FEATURES];
        outer(&mut gradient, &derivative, features);
        self.transitions[2] +=
            self.predictor
                .apply_gradient(&mut self.predictor_evidence, &gradient, 1)?;
        Ok(error)
    }

    pub fn update(
        &mut self,
        context: &[u8],
        input_gradient: &[f32],
        output: &[f32],
        logits_gradient: &[f32],
    ) -> Result<()> {
        let mut embedding_gradient = vec![0.0; WIDTH * VOCAB];
        for (position, &token) in context.iter().enumerate() {
            for (row, value) in input_gradient.iter().enumerate() {
                embedding_gradient[row * VOCAB + token as usize] +=
                    value * position_scale(position);
            }
        }
        let mut readout_gradient = vec![0.0; VOCAB * WIDTH];
        outer(&mut readout_gradient, logits_gradient, output);
        let input =
            self.embedding
                .apply_gradient(&mut self.embedding_evidence, &embedding_gradient, 1)?;
        let output =
            self.readout
                .apply_gradient(&mut self.readout_evidence, &readout_gradient, 1)?;
        self.transitions[0] += input;
        self.transitions[1] += output;
        if input + output > 0 {
            self.version += 1;
        }
        Ok(())
    }

    pub fn snapshot(&self) -> Value {
        json!({
            "schema": 2, "width": WIDTH, "vocab": VOCAB, "features": FEATURES,
            "shared_version": self.version,
            "embedding": self.embedding.packed_bytes(),
            "readout": self.readout.packed_bytes(),
            "predictor": self.predictor.packed_bytes(),
            "resumable": false
        })
    }

    pub fn identity(&self) -> Result<u64> {
        Ok(fingerprint(&serde_json::to_vec(&self.snapshot())?))
    }

    pub fn packed_bytes(&self) -> usize {
        self.embedding.storage_bytes()
            + self.readout.storage_bytes()
            + self.predictor.storage_bytes()
    }

    pub fn evidence_bytes(&self) -> usize {
        self.embedding_evidence.bytes()
            + self.readout_evidence.bytes()
            + self.predictor_evidence.bytes()
    }
}

fn position_scale(position: usize) -> f32 {
    0.5 * (position + 1) as f32 / ((CONTEXT * (CONTEXT + 1) / 2) as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn input_is_order_sensitive_and_only_reads_the_context() -> Result<()> {
        let shared = Shared::new(7)?;
        let prefix = *b"bees fly";
        let mut reversed = prefix;
        reversed.reverse();
        assert_ne!(shared.input(&prefix)?, shared.input(&reversed)?);
        let first = b"bees fly east";
        let second = b"bees fly west";
        assert_eq!(
            shared.input(&first[..CONTEXT])?,
            shared.input(&second[..CONTEXT])?
        );
        Ok(())
    }
}
