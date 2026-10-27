use super::gradients::{task_heads, transform_derivative, Gradients};
use super::math::{choose, finite, softmax};
use super::training_pulse::{context_scale, hidden_scale, PulseFrame, TrainingPulse};
use super::{TILES, WIDTH};
use crate::ternary::{Evidence, TernaryMatrix};
use anyhow::{ensure, Result};

pub(super) struct Node {
    pub transform: TernaryMatrix,
    pub curve: TernaryMatrix,
    pub router: TernaryMatrix,
    pub pulse: TrainingPulse,
    pub version: u64,
}

pub(super) struct Frame {
    pub id: usize,
    pub version: u64,
    pub input: Vec<f32>,
    pub output: Vec<f32>,
    pub transformed: Vec<f32>,
    pub preactivation: Vec<f32>,
    pub slopes: Vec<f32>,
    pub pulse: PulseFrame,
    pub route: Option<Route>,
}

pub(super) struct Route {
    pub destinations: Vec<usize>,
    pub features: Vec<f32>,
    pub probabilities: Vec<f32>,
    pub selected: usize,
}

impl Node {
    pub fn new(seed: u64, routes: usize) -> Result<Self> {
        Ok(Self {
            transform: TernaryMatrix::new(WIDTH, WIDTH, seed)?,
            curve: TernaryMatrix::new(3, WIDTH, seed.wrapping_add(1))?,
            router: TernaryMatrix::new(routes, WIDTH * 2, seed.wrapping_add(2))?,
            pulse: TrainingPulse::new(seed.wrapping_add(3), routes)?,
            version: 0,
        })
    }

    /// Apply the tile-wide pulse to transformation and activation; routing reuses it.
    pub fn forward(&self, id: usize, input: &[f32], history: &[f32]) -> Result<Frame> {
        let pulse = self.pulse.forward(input, history)?;
        let transformed: Vec<_> = self
            .transform
            .forward(input)?
            .iter()
            .map(|value| value / hidden_scale())
            .collect();
        let mut preactivation = Vec::with_capacity(WIDTH);
        let mut output = input.to_vec();
        let mut slopes = Vec::with_capacity(WIDTH);
        for index in 0..WIDTH {
            let value = transformed[index] * pulse.gain(index) + history[index] * 0.125;
            preactivation.push(value);
            let (segment, fraction) = interpolation(value);
            let left =
                segment as f32 - 1.0 + self.curve.value(segment * WIDTH + index)? as f32 * 0.25;
            let right =
                segment as f32 + self.curve.value((segment + 1) * WIDTH + index)? as f32 * 0.25;
            output[index] +=
                0.25 * (left * (1.0 - fraction) + right * fraction + pulse.curve_shift(index));
            slopes.push((right - left) / 2.0);
        }
        finite(&output)?;
        Ok(Frame {
            id,
            version: self.version,
            input: input.to_vec(),
            output,
            transformed,
            preactivation,
            slopes,
            pulse,
            route: None,
        })
    }

    pub fn route(
        &self,
        frame: &Frame,
        history: &[f32],
        neighbors: &[usize],
        random: &mut Option<u64>,
    ) -> Result<Route> {
        let features: Vec<_> = frame.output.iter().chain(history).copied().collect();
        let logits: Vec<_> = self
            .router
            .forward(&features)?
            .iter()
            .enumerate()
            .map(|(index, value)| value / context_scale() + frame.pulse.route_bias(index))
            .collect();
        let probabilities = softmax(&logits)?;
        let selected = choose(&probabilities, random);
        Ok(Route {
            destinations: neighbors.iter().copied().chain([TILES]).collect(),
            features,
            probabilities,
            selected,
        })
    }

    pub fn returned(&self, frame: &Frame, correction: &[f32]) -> Result<Vec<f32>> {
        ensure!(
            frame.version == self.version,
            "frozen anchor version mismatch"
        );
        let transformed = self
            .transform
            .transpose(&transform_derivative(frame, correction))?;
        let pulse = self
            .pulse
            .input_correction(&frame.pulse, &task_heads(frame, correction))?;
        let result: Vec<_> = correction
            .iter()
            .zip(transformed)
            .zip(pulse)
            .map(|((direct, transform), context)| direct + transform + context)
            .collect();
        finite(&result)?;
        Ok(result)
    }

    pub fn apply(&mut self, gradients: &Gradients) -> Result<[usize; 5]> {
        let mut transitions = [0; 5];
        for (index, (matrix, gradient)) in [
            (&mut self.transform, &gradients.transform),
            (&mut self.curve, &gradients.curve),
            (&mut self.router, &gradients.router),
            (&mut self.pulse.context, &gradients.pulse_context),
            (&mut self.pulse.heads, &gradients.pulse_heads),
        ]
        .into_iter()
        .enumerate()
        {
            transitions[index] =
                matrix.apply_gradient(&mut Evidence::new(matrix.len())?, gradient, 1)?;
        }
        if transitions.iter().sum::<usize>() > 0 {
            self.version += 1;
        }
        Ok(transitions)
    }
}

pub(super) fn interpolation(value: f32) -> (usize, f32) {
    if value < 0.0 {
        (0, (value + 2.0) / 2.0)
    } else {
        (1, value / 2.0)
    }
}

pub(super) fn parameter_count(routes: usize) -> usize {
    WIDTH * WIDTH
        + 3 * WIDTH
        + routes * 2 * WIDTH
        + WIDTH * 2 * WIDTH
        + (2 * WIDTH + routes) * WIDTH
}
