use super::math::outer;
use super::node::{interpolation, Frame, Node};
use super::training_pulse::{context_scale, hidden_scale};
use super::{RouteCredit, WIDTH};
use anyhow::{ensure, Result};

pub(super) struct Gradients {
    pub transform: Vec<f32>,
    pub curve: Vec<f32>,
    pub router: Vec<f32>,
    pub pulse_context: Vec<f32>,
    pub pulse_heads: Vec<f32>,
}

impl Gradients {
    pub fn zeros(node: &Node) -> Self {
        Self {
            transform: vec![0.0; node.transform.len()],
            curve: vec![0.0; node.curve.len()],
            router: vec![0.0; node.router.len()],
            pulse_context: vec![0.0; node.pulse.context.len()],
            pulse_heads: vec![0.0; node.pulse.heads.len()],
        }
    }

    /// Accumulate task and policy credit with the selected route-to-U ablation.
    pub fn add(
        &mut self,
        node: &Node,
        frame: &Frame,
        correction: &[f32],
        advantage: Option<f32>,
        route_credit: RouteCredit,
    ) -> Result<()> {
        ensure!(
            frame.version == node.version,
            "joint-pulse gradient version mismatch"
        );
        outer(
            &mut self.transform,
            &transform_derivative(frame, correction),
            &frame.input,
        );
        for (index, value) in correction.iter().enumerate() {
            let (segment, fraction) = interpolation(frame.preactivation[index]);
            self.curve[segment * WIDTH + index] += value * 0.0625 * (1.0 - fraction);
            self.curve[(segment + 1) * WIDTH + index] += value * 0.0625 * fraction;
        }
        let mut head_correction = task_heads(frame, correction);
        if let (Some(route), Some(advantage)) = (&frame.route, advantage) {
            let route_gradient: Vec<_> = route
                .probabilities
                .iter()
                .enumerate()
                .map(|(index, probability)| {
                    let indicator = if index == route.selected { 1.0 } else { 0.0 };
                    let score = advantage * (indicator - probability);
                    head_correction[2 * WIDTH + index] += 0.25 * score;
                    score / context_scale()
                })
                .collect();
            outer(&mut self.router, &route_gradient, &route.features);
        }
        node.pulse.add_gradients(
            &frame.pulse,
            &head_correction,
            &mut self.pulse_context,
            &mut self.pulse_heads,
            route_credit,
        )
    }
}

pub(super) fn transform_derivative(frame: &Frame, correction: &[f32]) -> Vec<f32> {
    correction
        .iter()
        .zip(&frame.slopes)
        .enumerate()
        .map(|(index, (value, slope))| {
            value * 0.25 * slope * frame.pulse.gain(index) / hidden_scale()
        })
        .collect()
}

pub(super) fn task_heads(frame: &Frame, correction: &[f32]) -> Vec<f32> {
    let mut heads = vec![0.0; frame.pulse.heads.len()];
    for (index, value) in correction.iter().enumerate() {
        heads[index] = value
            * 0.25
            * frame.slopes[index]
            * frame.transformed[index]
            * 0.125
            * (1.0 - frame.pulse.heads[index].tanh().powi(2));
        heads[WIDTH + index] =
            value * 0.25 * 0.125 * (1.0 - frame.pulse.heads[WIDTH + index].tanh().powi(2));
    }
    heads
}
