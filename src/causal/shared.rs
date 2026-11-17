//! Tied byte embeddings, immutable prefix rows and next-byte readout; see STUDY.md.

use super::geometry::{Geometry, EVIDENCE_RULE, VOCAB};
use super::tile_gradients::add_outer;
use super::{norm, positions};
use crate::ternary::{Evidence, TernaryMatrix};
use anyhow::{ensure, Context, Result};

pub(super) struct Shared {
    pub geometry: Geometry,
    pub embedding: TernaryMatrix,
    pub version: u64,
    pub weights: Option<super::connected_precision::Weights>,
    /// Learned FP32 per-byte logit bias (VOCAB entries); None outside the connected protocol.
    pub bias: Option<Vec<f32>>,
}

pub(super) struct Prefix {
    pub bytes: Vec<u8>,
    /// Row-major byte-plus-position features; tiles read these without updating them.
    pub memory: Vec<f32>,
    pub inverse: Vec<f32>,
    pub shared_version: u64,
}

pub(super) struct Readout {
    pub normalized: norm::Rms,
    pub logits: Vec<f32>,
}

pub(super) struct Task {
    pub loss: f32,
    pub input_correction: Vec<f32>,
    pub embedding_gradient: Vec<f32>,
    /// Softmax minus one-hot (the logit gradient) when a readout bias is present.
    pub bias_gradient: Option<Vec<f32>>,
    /// RMS of the routed output before readout normalization (1 / inverse).
    pub output_rms: f32,
}

impl Readout {
    pub fn loss(&self, target: u8) -> Result<f32> {
        norm::vector(&self.logits, VOCAB)?;
        let maximum = self
            .logits
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = self.logits.iter().map(|v| (v - maximum).exp()).sum();
        let loss = maximum + sum.ln() - self.logits[usize::from(target)];
        ensure!(loss.is_finite(), "nonfinite causal loss");
        Ok(loss)
    }
}

impl Prefix {
    /// The current routed state starts at the last visible byte's embedding row.
    pub fn initial(&self, geometry: Geometry) -> &[f32] {
        &self.memory[self.memory.len() - geometry.width..]
    }
}

impl Shared {
    pub fn new(geometry: Geometry, seed: u64) -> Result<Self> {
        geometry.validate()?;
        Ok(Self {
            geometry,
            weights: None,
            embedding: TernaryMatrix::new(geometry.width, VOCAB, seed)?,
            version: 0,
            bias: None,
        })
    }

    pub fn from_packed(
        geometry: Geometry,
        version: u64,
        packed: &[u8],
        rule: &str,
    ) -> Result<Self> {
        geometry.validate()?;
        ensure!(rule == EVIDENCE_RULE, "shared evidence encoding mismatch");
        Ok(Self {
            geometry,
            weights: None,
            version,
            embedding: TernaryMatrix::from_packed(geometry.width, VOCAB, packed)?,
            bias: None,
        })
    }

    pub fn packed(&self) -> &[u8] {
        self.embedding.packed_bytes()
    }

    /// Rebuild all visible embedding rows for this anchor, including overlapping positions.
    pub fn prefix(&self, bytes: &[u8]) -> Result<Prefix> {
        ensure!(
            (1..=self.geometry.context).contains(&bytes.len()),
            "invalid visible prefix length"
        );
        let mut memory = Vec::with_capacity(bytes.len() * self.geometry.width);
        let mut inverse = Vec::with_capacity(bytes.len());
        for (position, &byte) in bytes.iter().enumerate() {
            let mut column = self.embedding_column(usize::from(byte))?;
            positions::add(position, &mut column)?;
            let row = norm::rms(&column)?;
            memory.extend(row.values);
            inverse.push(row.inverse);
        }
        Ok(Prefix {
            bytes: bytes.to_vec(),
            memory,
            inverse,
            shared_version: self.version,
        })
    }

    /// RMS-normalized column `E[:, byte]` with no position: the connected predictor's label feature.
    pub fn target_feature(&self, byte: u8) -> Result<Vec<f32>> {
        Ok(norm::rms(&self.embedding_column(usize::from(byte))?)?.values)
    }

    /// Every readout path (training, replay, decoding, quantized views) adds the FP32 bias here.
    pub fn readout(&self, output: &[f32]) -> Result<Readout> {
        norm::vector(output, self.geometry.width)?;
        let normalized = norm::rms(output)?;
        let mut logits = self.embedding_transpose(&normalized.values, self.logit_scale())?;
        if let Some(bias) = &self.bias {
            norm::add(&mut logits, bias)?;
        }
        Ok(Readout { normalized, logits })
    }

    /// Tied readout scale 1/(sqrt(d)*temperature); dividing by the pinned 1.0 is exact.
    fn logit_scale(&self) -> f32 {
        (self.geometry.width as f32).sqrt().recip() / super::tuning::current().temperature
    }

    /// Differentiate next-byte loss into the routed state and tied readout matrix E.
    pub fn task(&self, output: &[f32], target: u8) -> Result<Task> {
        let readout = self.readout(output)?;
        let loss = readout.loss(target)?;
        let mut delta = readout.logits;
        norm::softmax(&mut delta)?;
        delta[usize::from(target)] -= 1.0;
        let scale = self.logit_scale();
        let input = self.embedding_project(&delta, scale)?;
        let input_correction = norm::backward(
            &readout.normalized.values,
            readout.normalized.inverse,
            &input,
        )?;
        let mut embedding_gradient = vec![0.0; self.embedding.len()];
        add_outer(
            &mut embedding_gradient,
            &readout.normalized.values,
            &delta,
            scale,
        )?;
        Ok(Task {
            loss,
            input_correction,
            embedding_gradient,
            bias_gradient: self.bias.is_some().then_some(delta),
            output_rms: readout.normalized.inverse.recip(),
        })
    }

    /// Sum normalized prefix occurrences into the existing tied-readout gradient.
    pub fn add_prefix_gradient(
        &self,
        prefix: &Prefix,
        prefix_adjoint: &[f32],
        gradient: &mut [f32],
    ) -> Result<()> {
        let d = self.geometry.width;
        ensure!(
            prefix.shared_version == self.version,
            "frozen prefix E version mismatch"
        );
        ensure!(
            (1..=self.geometry.context).contains(&prefix.bytes.len()),
            "invalid prefix state length"
        );
        norm::vector(&prefix.memory, prefix.bytes.len() * d)?;
        norm::vector(&prefix.inverse, prefix.bytes.len())?;
        norm::vector(prefix_adjoint, prefix.memory.len())?;
        norm::vector(gradient, self.embedding.len())?;
        for (i, &byte) in prefix.bytes.iter().enumerate() {
            let row = norm::backward(
                &prefix.memory[i * d..(i + 1) * d],
                prefix.inverse[i],
                &prefix_adjoint[i * d..(i + 1) * d],
            )?;
            for (j, &value) in row.iter().enumerate() {
                gradient[j * VOCAB + usize::from(byte)] += value;
            }
        }
        norm::finite(gradient)
    }

    pub fn apply(&mut self, gradient: &[f32]) -> Result<usize> {
        norm::vector(gradient, self.embedding.len())?;
        let next = self
            .version
            .checked_add(1)
            .context("shared version overflow")?;
        let transitions = self.embedding.apply_gradient(
            &mut Evidence::new(self.embedding.len())?,
            gradient,
            1,
        )?;
        if transitions > 0 {
            self.version = next;
        }
        Ok(transitions)
    }
}
