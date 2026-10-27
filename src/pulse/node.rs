use super::math::{add_outer, finite, softmax, Random};
use super::{TILES, WIDTH};
use crate::ternary::TernaryMatrix;
use anyhow::Result;

pub(super) const ROUTE_FEATURES: usize = 2 * WIDTH;
const TRANSFORM_SCALE: f32 = 0.353_553_38;
const RESIDUAL_SCALE: f32 = 0.25;

pub(super) struct Node {
    pub transform: TernaryMatrix,
    pub activation: TernaryMatrix,
    pub router: TernaryMatrix,
    pub version: u64,
}

#[derive(Clone)]
pub(super) struct Route {
    pub rows: Vec<usize>,
    pub probabilities: Vec<f32>,
    pub selected: usize,
    pub features: Vec<f32>,
}

#[derive(Clone)]
pub(super) struct Frame {
    pub node: usize,
    pub version: u64,
    pub input: Vec<f32>,
    pub output: Vec<f32>,
    pub preactivation: Vec<f32>,
    pub coefficients: Vec<f32>,
    pub route: Option<Route>,
}

pub(super) struct Gradients {
    pub transform: Vec<f32>,
    pub activation: Vec<f32>,
    pub router: Vec<f32>,
}

impl Gradients {
    pub fn zeros(router_len: usize) -> Self {
        Self {
            transform: vec![0.0; WIDTH * WIDTH],
            activation: vec![0.0; WIDTH],
            router: vec![0.0; router_len],
        }
    }

    pub fn add_transform(&mut self, frame: &Frame, correction: &[f32]) {
        let local = local_derivative(frame, correction);
        add_outer(&mut self.transform, &local, &frame.input);
        for (index, value) in self.activation.iter_mut().enumerate() {
            *value += correction[index] * RESIDUAL_SCALE * 0.25 * frame.preactivation[index].tanh();
        }
    }

    pub fn add_route(&mut self, route: &Route, advantage: f32) {
        for position in 0..route.rows.len() {
            let indicator = if position == route.selected { 1.0 } else { 0.0 };
            let derivative = advantage * (indicator - route.probabilities[position]) / 4.0;
            for (column, &feature) in route.features.iter().enumerate() {
                self.router[position * ROUTE_FEATURES + column] += derivative * feature;
            }
        }
    }
}

impl Node {
    pub fn new(seed: u64, route_count: usize) -> Result<Self> {
        Ok(Self {
            transform: TernaryMatrix::new(WIDTH, WIDTH, seed)?,
            activation: TernaryMatrix::new(1, WIDTH, seed.wrapping_add(1))?,
            router: TernaryMatrix::new(route_count, ROUTE_FEATURES, seed.wrapping_add(2))?,
            version: 0,
        })
    }

    pub fn forward(&self, id: usize, input: &[f32]) -> Result<Frame> {
        let preactivation: Vec<_> = self
            .transform
            .forward(input)?
            .iter()
            .map(|value| value * TRANSFORM_SCALE)
            .collect();
        let coefficients: Vec<_> = (0..WIDTH)
            .map(|index| {
                self.activation
                    .value(index)
                    .map(|value| 1.0 + 0.25 * value as f32)
            })
            .collect::<Result<_>>()?;
        let output: Vec<_> = (0..WIDTH)
            .map(|index| {
                input[index] + RESIDUAL_SCALE * coefficients[index] * preactivation[index].tanh()
            })
            .collect();
        finite(&output)?;
        Ok(Frame {
            node: id,
            version: self.version,
            input: input.to_vec(),
            output,
            preactivation,
            coefficients,
            route: None,
        })
    }

    pub fn route(
        &self,
        output: &[f32],
        history: &[f32],
        neighbors: &[usize],
        random: Option<&mut Random>,
    ) -> Result<Route> {
        let features: Vec<_> = output.iter().chain(history).copied().collect();
        let all_logits = self.router.forward(&features)?;
        let rows: Vec<_> = neighbors.iter().copied().chain([TILES]).collect();
        let logits: Vec<_> = all_logits.iter().map(|value| value / 4.0).collect();
        let probabilities = softmax(&logits)?;
        let selected = choose(&probabilities, random);
        Ok(Route {
            rows,
            probabilities,
            selected,
            features,
        })
    }

    pub fn returned_correction(&self, frame: &Frame, correction: &[f32]) -> Result<Vec<f32>> {
        let local = local_derivative(frame, correction);
        let transformed = self.transform.transpose(&local)?;
        let result: Vec<_> = correction
            .iter()
            .zip(transformed)
            .map(|(a, b)| a + b)
            .collect();
        finite(&result)?;
        Ok(result)
    }

    pub fn matrices(&self) -> [&TernaryMatrix; 3] {
        [&self.transform, &self.activation, &self.router]
    }
}

fn local_derivative(frame: &Frame, correction: &[f32]) -> Vec<f32> {
    (0..WIDTH)
        .map(|index| {
            correction[index]
                * RESIDUAL_SCALE
                * frame.coefficients[index]
                * (1.0 - frame.preactivation[index].tanh().powi(2))
                * TRANSFORM_SCALE
        })
        .collect()
}

fn choose(probabilities: &[f32], random: Option<&mut Random>) -> usize {
    if let Some(random) = random {
        let draw = random.unit();
        let mut total = 0.0;
        for (index, &probability) in probabilities.iter().enumerate() {
            total += probability;
            if draw < total {
                return index;
            }
        }
        probabilities.len() - 1
    } else {
        probabilities
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(right.1).then_with(|| right.0.cmp(&left.0)))
            .map(|(index, _)| index)
            .unwrap_or(0)
    }
}
