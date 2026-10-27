//! Joint tile conditioning and credit projection; see AGENTS.md, Joint training pulse.

use super::math::{finite, outer};
use super::{RouteCredit, WIDTH};
use crate::ternary::TernaryMatrix;
use anyhow::{ensure, Result};

pub(super) struct TrainingPulse {
    pub context: TernaryMatrix,
    pub heads: TernaryMatrix,
}

pub(super) struct PulseFrame {
    pub features: Vec<f32>,
    pub latent: Vec<f32>,
    pub heads: Vec<f32>,
}

impl TrainingPulse {
    pub fn new(seed: u64, routes: usize) -> Result<Self> {
        Ok(Self {
            context: TernaryMatrix::new(WIDTH, 2 * WIDTH, seed)?,
            heads: TernaryMatrix::new(2 * WIDTH + routes, WIDTH, seed.wrapping_add(1))?,
        })
    }

    /// Produce one shared latent and its stacked gain, curve and route heads.
    pub fn forward(&self, input: &[f32], history: &[f32]) -> Result<PulseFrame> {
        ensure!(
            input.len() == WIDTH && history.len() == WIDTH,
            "invalid pulse context"
        );
        let features: Vec<_> = input.iter().chain(history).copied().collect();
        let latent: Vec<_> = self
            .context
            .forward(&features)?
            .iter()
            .map(|value| (value / context_scale()).tanh())
            .collect();
        let heads: Vec<_> = self
            .heads
            .forward(&latent)?
            .iter()
            .map(|value| value / hidden_scale())
            .collect();
        finite(&heads)?;
        Ok(PulseFrame {
            features,
            latent,
            heads,
        })
    }

    /// Return the task-only pulse adjoint to the current tile input.
    pub fn input_correction(&self, frame: &PulseFrame, heads: &[f32]) -> Result<Vec<f32>> {
        let latent = self.context_derivative(frame, heads)?;
        let features = self.context.transpose(&latent)?;
        Ok(features[..WIDTH]
            .iter()
            .map(|value| value / context_scale())
            .collect())
    }

    /// Accumulate V credit and the selected route-to-U projection.
    pub fn add_gradients(
        &self,
        frame: &PulseFrame,
        head_correction: &[f32],
        context: &mut [f32],
        heads: &mut [f32],
        route_credit: RouteCredit,
    ) -> Result<()> {
        let head_derivative: Vec<_> = head_correction
            .iter()
            .map(|value| value / hidden_scale())
            .collect();
        outer(heads, &head_derivative, &frame.latent);
        let detached;
        let context_credit = match route_credit {
            RouteCredit::Coupled => head_correction,
            RouteCredit::RouteToUDetached => {
                detached = head_correction[..2 * WIDTH]
                    .iter()
                    .copied()
                    .chain(std::iter::repeat_n(0.0, head_correction.len() - 2 * WIDTH))
                    .collect::<Vec<_>>();
                &detached
            }
        };
        let latent: Vec<_> = self
            .context_derivative(frame, context_credit)?
            .iter()
            .map(|value| value / context_scale())
            .collect();
        outer(context, &latent, &frame.features);
        Ok(())
    }

    fn context_derivative(&self, frame: &PulseFrame, head_correction: &[f32]) -> Result<Vec<f32>> {
        let hidden = self.heads.transpose(head_correction)?;
        let derivative: Vec<_> = hidden
            .iter()
            .zip(&frame.latent)
            .map(|(value, latent)| value / hidden_scale() * (1.0 - latent * latent))
            .collect();
        finite(&derivative)?;
        Ok(derivative)
    }
}

impl PulseFrame {
    pub fn gain(&self, index: usize) -> f32 {
        1.0 + 0.125 * self.heads[index].tanh()
    }

    pub fn curve_shift(&self, index: usize) -> f32 {
        0.125 * self.heads[WIDTH + index].tanh()
    }

    pub fn route_bias(&self, index: usize) -> f32 {
        0.25 * self.heads[2 * WIDTH + index]
    }
}

pub(super) fn context_scale() -> f32 {
    (2.0 * WIDTH as f32).sqrt()
}

pub(super) fn hidden_scale() -> f32 {
    (WIDTH as f32).sqrt()
}
