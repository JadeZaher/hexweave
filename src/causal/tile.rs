use super::geometry::{Geometry, EVIDENCE_RULE, MATRIX_COUNT};
use super::tile_gradients::TileGradients;
use super::{attention, ffn, norm, pulse, routing};
use crate::ternary::{Evidence, TernaryMatrix};
use anyhow::{ensure, Context, Result};

pub(super) struct Tile {
    pub geometry: Geometry,
    pub id: usize,
    pub version: u64,
    pub matrices: [TernaryMatrix; MATRIX_COUNT],
}

pub(super) struct Forward {
    pub output: Vec<f32>,
    #[allow(dead_code)]
    pub latent: Vec<f32>,
    pub heads: Vec<f32>,
}

struct Cache {
    attention: attention::Cache,
    normalized: norm::Rms,
    ffn: ffn::Cache,
    pulse: pulse::Cache,
    output: Vec<f32>,
}

impl Tile {
    pub fn new(geometry: Geometry, id: usize, seed: u64) -> Result<Self> {
        geometry.validate()?;
        let shapes = geometry.shapes(id)?;
        let base = seed
            .wrapping_add(16)
            .wrapping_add((id as u64).wrapping_mul(16));
        let mut matrices = Vec::with_capacity(MATRIX_COUNT);
        for (offset, (rows, cols)) in shapes.into_iter().enumerate() {
            matrices.push(TernaryMatrix::new(
                rows,
                cols,
                base.wrapping_add(offset as u64),
            )?);
        }
        Ok(Self {
            geometry,
            id,
            version: 0,
            matrices: matrices
                .try_into()
                .map_err(|_| anyhow::anyhow!("tile matrix count"))?,
        })
    }

    pub fn from_packed(
        geometry: Geometry,
        id: usize,
        version: u64,
        packed: &[Vec<u8>],
        evidence_rule: &str,
    ) -> Result<Self> {
        geometry.validate()?;
        ensure!(
            packed.len() == MATRIX_COUNT && evidence_rule == EVIDENCE_RULE,
            "tile state schema mismatch"
        );
        let mut matrices = Vec::with_capacity(MATRIX_COUNT);
        for ((rows, cols), bytes) in geometry.shapes(id)?.into_iter().zip(packed) {
            matrices.push(TernaryMatrix::from_packed(rows, cols, bytes)?);
        }
        Ok(Self {
            geometry,
            id,
            version,
            matrices: matrices
                .try_into()
                .map_err(|_| anyhow::anyhow!("tile matrix count"))?,
        })
    }

    pub fn packed(&self) -> [&[u8]; MATRIX_COUNT] {
        std::array::from_fn(|index| self.matrices[index].packed_bytes())
    }

    pub fn forward(&self, input: &[f32], memory: &[f32], history: &[f32]) -> Result<Forward> {
        let cache = self.cache(input, memory, history)?;
        Ok(Forward {
            output: cache.output,
            latent: cache.pulse.latent,
            heads: cache.pulse.heads,
        })
    }

    pub fn route(
        &self,
        forward: &Forward,
        history: &[f32],
        random: &mut Option<u64>,
    ) -> Result<usize> {
        let mut probabilities =
            routing::probabilities(self, &forward.output, &forward.heads, history)?;
        if super::tuning::current().no_halt {
            let halt = probabilities.len() - 1;
            probabilities[halt] = 0.0;
            let total: f32 = probabilities.iter().sum();
            probabilities.iter_mut().for_each(|p| *p /= total);
        }
        routing::choose(&probabilities, random)
    }

    /// Replay one frozen frame; prefix and parameter sinks select the anchor phase.
    #[allow(clippy::too_many_arguments)]
    pub fn backward(
        &self,
        input: &[f32],
        memory: &[f32],
        history: &[f32],
        upstream: &[f32],
        prefix_adjoint: Option<&mut [f32]>,
        mut gradients: Option<&mut TileGradients>,
        policy: Option<(usize, f32)>,
    ) -> Result<Vec<f32>> {
        norm::vector(upstream, self.geometry.width)?;
        if let Some(prefix) = prefix_adjoint.as_deref() {
            norm::vector(prefix, memory.len())?;
        }
        if let Some(gradient) = gradients.as_deref() {
            gradient.validate(self.geometry, self.id)?;
        }
        ensure!(
            policy.is_none() || gradients.is_some(),
            "policy needs a parameter sink"
        );
        let cache = self.cache(input, memory, history)?;
        let ffn_adjoint: Vec<_> = upstream
            .iter()
            .enumerate()
            .map(|(j, b)| 0.25 * cache.pulse.gain(j) * b)
            .collect();
        let mut normalized_adjoint = ffn::backward(
            self,
            &cache.normalized.values,
            &cache.ffn,
            &ffn_adjoint,
            gradients.as_deref_mut(),
        )?;
        let pulse_adjoint = pulse::backward(
            self,
            &cache.pulse,
            &cache.ffn.output,
            upstream,
            gradients.as_deref_mut(),
        )?;
        norm::add(&mut normalized_adjoint, &pulse_adjoint)?;
        let mut attention_adjoint = norm::backward(
            &cache.normalized.values,
            cache.normalized.inverse,
            &normalized_adjoint,
        )?;
        norm::add(&mut attention_adjoint, upstream)?;
        let result = attention::backward(
            self,
            memory,
            &cache.attention,
            &attention_adjoint,
            prefix_adjoint,
            gradients.as_deref_mut(),
        )?;
        if let (Some((selected, advantage)), Some(gradient)) = (policy, gradients.as_deref_mut()) {
            routing::policy(
                self,
                &cache.output,
                &cache.pulse.heads,
                &cache.pulse.latent,
                history,
                selected,
                advantage,
                gradient,
            )?;
        }
        if let Some(gradient) = gradients {
            gradient.validate(self.geometry, self.id)?;
        }
        Ok(result)
    }

    pub fn apply(&mut self, gradients: &TileGradients) -> Result<[usize; MATRIX_COUNT]> {
        gradients.validate(self.geometry, self.id)?;
        let next = self
            .version
            .checked_add(1)
            .context("tile version overflow")?;
        let knobs = super::tuning::current();
        let mut transitions = [0; MATRIX_COUNT];
        for (index, (matrix, gradient)) in
            self.matrices.iter_mut().zip(&gradients.values).enumerate()
        {
            transitions[index] = match knobs.rate(index) {
                Some(rate) => {
                    super::tuning::update(&knobs, rate, (self.id, index), matrix, gradient)?
                }
                None => matrix.apply_gradient(&mut Evidence::new(matrix.len())?, gradient, 1)?,
            };
        }
        if transitions.iter().any(|&count| count > 0) {
            self.version = next;
        }
        Ok(transitions)
    }

    fn cache(&self, input: &[f32], memory: &[f32], history: &[f32]) -> Result<Cache> {
        let d = self.geometry.width;
        norm::vector(input, d)?;
        norm::vector(history, d)?;
        ensure!(
            !memory.is_empty()
                && memory.len().is_multiple_of(d)
                && memory.len() / d <= self.geometry.context,
            "invalid immutable prefix shape"
        );
        norm::finite(memory)?;
        ensure!(
            history.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>() <= 1.000002,
            "history exceeds unit correction bound"
        );
        let attention = attention::forward(self, input, memory)?;
        let mix = super::tuning::current().attention_mix;
        let u: Vec<_> = input
            .iter()
            .zip(&attention.output)
            .map(|(x, o)| x + mix * o)
            .collect();
        let normalized = norm::rms(&u)?;
        let ffn = ffn::forward(self, &normalized.values)?;
        let pulse = pulse::forward(self, &normalized.values, history)?;
        let output: Vec<_> = u
            .iter()
            .enumerate()
            .map(|(j, u)| u + 0.25 * (pulse.gain(j) * ffn.output[j] + pulse.shift(j, d)))
            .collect();
        norm::finite(&output)?;
        Ok(Cache {
            attention,
            normalized,
            ffn,
            pulse,
            output,
        })
    }
}
