//! One routed attention/FFN/modulation block and its two-phase adjoint; see STUDY.md.

use super::geometry::{Geometry, EVIDENCE_RULE, MATRIX_COUNT};
use super::tile_gradients::TileGradients;
use super::{attention, ffn, norm, pulse, routing};
use crate::ternary::{Evidence, TernaryMatrix};
use anyhow::{ensure, Context, Result};
use cache::Cache;

#[path = "tile_cache.rs"]
mod cache;

pub(super) struct Tile {
    pub geometry: Geometry,
    pub id: usize,
    pub version: u64,
    pub weights: Option<super::connected_precision::Weights>,
    pub matrices: [TernaryMatrix; MATRIX_COUNT],
}

pub(super) struct Forward {
    /// Next state; RMS 1 under `connected` (post-visit normalization), raw otherwise.
    pub output: Vec<f32>,
    /// RMS of the raw output before post-visit normalization (`1/inverse` under `connected`).
    pub raw_rms: f32,
    #[allow(dead_code)]
    pub latent: Vec<f32>,
    pub heads: Vec<f32>,
    /// Fraction of FFN preactivations beyond the outer knots (|z| > 2) on this forward.
    pub knot_overflow: f32,
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
            weights: None,
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
            weights: None,
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

    /// Return the next state and routing features, discarding the internal activation cache.
    pub fn forward(&self, input: &[f32], memory: &[f32], history: &[f32]) -> Result<Forward> {
        let cache = self.cache(input, memory, history)?;
        Ok(Forward {
            knot_overflow: cache.ffn.knot_overflow(),
            raw_rms: cache.raw_rms(),
            output: cache.output_norm.map_or(cache.output, |rms| rms.values),
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
        // Both correction gathering and parameter fitting rebuild this full forward cache.
        let cache = self.cache(input, memory, history)?;
        // Below, `upstream` is the raw-output adjoint: post-visit normalization comes first.
        let raw = cache.raw_adjoint(upstream)?;
        let upstream: &[f32] = &raw;
        let ffn_adjoint: Vec<_> = upstream
            .iter()
            .enumerate()
            .map(|(j, b)| 0.25 * cache.pulse.gain(j) * b)
            .collect();
        let (mut normalized_adjoint, curve_adjoint) = ffn::backward(
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
            &curve_adjoint,
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
        // Route credit is detached: it changes parameters, not the returned input correction.
        if let (Some((selected, advantage)), Some(gradient)) = (policy, gradients.as_deref_mut()) {
            routing::policy(
                self,
                cache.visible(),
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
        // Versions track packed-trit changes; latent Adam can advance without a version bump.
        if transitions.iter().any(|&count| count > 0) {
            self.version = next;
        }
        Ok(transitions)
    }

    /// Attention, gated FFN, gain/shift, then post-visit RMS normalization (connected only).
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
        let knobs = super::tuning::current();
        let u: Vec<_> = input
            .iter()
            .zip(&attention.output)
            .map(|(x, o)| x + knobs.attention_mix * o)
            .collect();
        let normalized = norm::rms(&u)?;
        let pulse = pulse::forward(self, &normalized.values, history)?;
        let ffn = ffn::forward_visit(
            self,
            &normalized.values,
            pulse.curve_controls(),
            input,
            memory,
            history,
        )?;
        let output: Vec<_> = u
            .iter()
            .enumerate()
            .map(|(j, u)| u + 0.25 * (pulse.gain(j) * ffn.output[j] + pulse.shift(j, d)))
            .collect();
        norm::finite(&output)?;
        let output_norm = if knobs.connected {
            Some(norm::rms(&output)?)
        } else {
            None
        };
        Ok(Cache {
            attention,
            normalized,
            ffn,
            pulse,
            output,
            output_norm,
        })
    }
}
