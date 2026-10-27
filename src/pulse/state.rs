use super::math::{add_outer, clipped, loss_and_gradient};
use super::node::{Frame, Node};
use super::{FEATURES, THRESHOLD, TILES, VOCAB, WIDTH};
use crate::ternary::{Evidence, TernaryMatrix};
use crate::topology::Mesh;
use anyhow::{ensure, Result};

pub(super) struct Model {
    pub nodes: Vec<Node>,
    pub embedding: TernaryMatrix,
    pub readout: TernaryMatrix,
    pub predictor: TernaryMatrix,
    pub embedding_evidence: Evidence,
    pub readout_evidence: Evidence,
    pub predictor_evidence: Evidence,
    pub mesh: Mesh,
    pub shared_version: u64,
    pub embedding_transitions: usize,
    pub readout_transitions: usize,
    pub predictor_transitions: usize,
}

impl Model {
    pub fn new(seed: u64) -> Result<Self> {
        let mesh = Mesh::new(1)?;
        let nodes = (0..TILES)
            .map(|id| {
                Node::new(
                    seed.wrapping_add(100 + id as u64 * 3),
                    mesh.neighbors(id)?.len() + 1,
                )
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            nodes,
            embedding: TernaryMatrix::new(WIDTH, VOCAB, seed)?,
            readout: TernaryMatrix::new(VOCAB, WIDTH, seed.wrapping_add(1))?,
            predictor: TernaryMatrix::new(WIDTH, FEATURES, seed.wrapping_add(2))?,
            embedding_evidence: Evidence::new(WIDTH * VOCAB)?,
            readout_evidence: Evidence::new(VOCAB * WIDTH)?,
            predictor_evidence: Evidence::new(WIDTH * FEATURES)?,
            mesh,
            shared_version: 0,
            embedding_transitions: 0,
            readout_transitions: 0,
            predictor_transitions: 0,
        })
    }

    pub fn input(&self, token: usize) -> Result<Vec<f32>> {
        ensure!(token < VOCAB, "input token outside vocabulary");
        let mut input = self.embedding.column(token)?;
        for value in &mut input {
            *value *= 0.5;
        }
        Ok(input)
    }

    pub fn task(&self, hidden: &[f32], target: usize) -> Result<(f32, Vec<f32>)> {
        let logits: Vec<_> = self
            .readout
            .forward(hidden)?
            .iter()
            .map(|v| v / (WIDTH as f32).sqrt())
            .collect();
        loss_and_gradient(&logits, target)
    }

    pub fn center(&self) -> usize {
        self.mesh
            .coordinates()
            .iter()
            .position(|point| point.q == 0 && point.r == 0)
            .unwrap()
    }

    pub fn features(&self, frame: &Frame, history: &[f32]) -> Vec<f32> {
        let coordinate = self.mesh.coordinates()[frame.node];
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
            .map(|value| 0.5 * (value / (FEATURES as f32).sqrt()).tanh())
            .collect())
    }

    pub fn train_predictor(&mut self, features: &[f32], target: &[f32]) -> Result<f32> {
        ensure!(target.len() == WIDTH, "wrong correction width");
        let target = clipped(target)?;
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
            .map(|(prediction, target)| {
                (2.0 / WIDTH as f32)
                    * (prediction - target)
                    * 0.5
                    * (1.0 - (2.0 * prediction).powi(2))
                    / (FEATURES as f32).sqrt()
            })
            .collect();
        let mut gradients = vec![0.0; WIDTH * FEATURES];
        add_outer(&mut gradients, &derivative, features);
        self.predictor_transitions += self.predictor.apply_gradient(
            &mut self.predictor_evidence,
            &clipped(&gradients)?,
            THRESHOLD,
        )?;
        Ok(error)
    }

    pub fn all_matrices(&self) -> Vec<&TernaryMatrix> {
        let mut matrices = vec![&self.embedding, &self.readout, &self.predictor];
        for node in &self.nodes {
            matrices.extend(node.matrices());
        }
        matrices
    }

    pub fn validate_weights(&self) -> Result<()> {
        for matrix in self.all_matrices() {
            for index in 0..matrix.len() {
                ensure!(
                    (-1..=1).contains(&matrix.value(index)?),
                    "nonternary model weight"
                );
            }
        }
        Ok(())
    }

    pub fn shared_evidence_bytes(&self) -> usize {
        self.embedding_evidence.bytes()
            + self.readout_evidence.bytes()
            + self.predictor_evidence.bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_embedding_lookup_matches_one_hot_bits() -> Result<()> {
        for seed in [0, 7, u64::MAX, 1 << 32] {
            let model = Model::new(seed)?;
            for token in 0..VOCAB {
                let mut one_hot = vec![0.0; VOCAB];
                one_hot[token] = 0.5;
                let expected = model.embedding.forward_scalar(&one_hot)?;
                let actual = model.input(token)?;
                assert_eq!(
                    actual
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    expected
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>()
                );
            }
            assert!(model.input(VOCAB).is_err());
        }
        Ok(())
    }
}
