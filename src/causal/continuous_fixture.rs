use super::super::tile::Tile;
use super::super::tile_gradients::{A, CURVE, DOWN, G, K, O, P, Q, ROUTER, U, V};
use anyhow::Result;

pub(super) struct Reference {
    pub weights: Vec<Vec<f64>>,
    pub width: usize,
    pub ffn: usize,
    pub heads: usize,
}

pub(super) struct Output {
    pub values: Vec<f64>,
    pub latent: Vec<f64>,
    pub route_features: Vec<f64>,
}

pub(super) fn rms(values: &[f64]) -> Vec<f64> {
    let inverse = (values.iter().map(|v| v * v).sum::<f64>() / values.len() as f64 + 1e-5)
        .sqrt()
        .recip();
    values.iter().map(|value| value * inverse).collect()
}

fn probabilities(logits: &[f64]) -> Vec<f64> {
    let maximum = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut values: Vec<_> = logits.iter().map(|value| (value - maximum).exp()).collect();
    let total: f64 = values.iter().sum();
    for value in &mut values {
        *value /= total;
    }
    values
}

impl Reference {
    pub fn new(tile: &Tile) -> Result<Self> {
        let weights = tile
            .matrices
            .iter()
            .enumerate()
            .map(|(matrix_index, matrix)| {
                (0..matrix.len())
                    .map(|index| Ok(f64::from(tile.weight(matrix_index, index)?)))
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            weights,
            width: tile.geometry.width,
            ffn: tile.geometry.ffn,
            heads: tile.geometry.heads,
        })
    }

    fn project(&self, matrix: usize, input: &[f64]) -> Vec<f64> {
        self.weights[matrix]
            .chunks_exact(input.len())
            .map(|row| {
                row.iter()
                    .zip(input)
                    .map(|(weight, value)| weight * value)
                    .sum::<f64>()
                    / (input.len() as f64).sqrt()
            })
            .collect()
    }

    fn attention(&self, input: &[f64], memory: &[f64]) -> Vec<f64> {
        let knobs = super::super::tuning::current();
        let query = self.project(Q, &rms(input));
        let keys: Vec<_> = memory
            .chunks_exact(self.width)
            .map(|row| self.project(K, row))
            .collect();
        let values: Vec<_> = memory
            .chunks_exact(self.width)
            .map(|row| self.project(V, row))
            .collect();
        let head_width = self.width / self.heads;
        let mut attended = vec![0.0; self.width];
        for head in 0..self.heads {
            let start = head * head_width;
            let scores: Vec<_> = keys
                .iter()
                .map(|key| {
                    (start..start + head_width)
                        .map(|j| query[j] * key[j])
                        .sum::<f64>()
                        / (head_width as f64).sqrt()
                        * f64::from(knobs.attention_scale)
                })
                .collect();
            for (probability, value) in probabilities(&scores).iter().zip(&values) {
                for j in start..start + head_width {
                    attended[j] += probability * value[j];
                }
            }
        }
        let projected = self.project(O, &attended);
        input
            .iter()
            .zip(projected)
            .map(|(x, output)| x + f64::from(knobs.attention_mix) * output)
            .collect()
    }

    fn ffn(&self, normalized: &[f64]) -> Vec<f64> {
        let gates = self.project(G, normalized);
        let activation = self.project(A, normalized);
        let product: Vec<_> = activation
            .iter()
            .enumerate()
            .map(|(j, &z)| {
                let (segment, fraction) = if z < 0.0 {
                    (0, (z + 2.0) / 2.0)
                } else {
                    (1, z / 2.0)
                };
                let left = segment as f64 - 1.0 + self.weights[CURVE][segment * self.ffn + j] / 4.0;
                let right =
                    segment as f64 + self.weights[CURVE][(segment + 1) * self.ffn + j] / 4.0;
                (left * (1.0 - fraction) + right * fraction) / (1.0 + (-gates[j]).exp())
            })
            .collect();
        self.project(DOWN, &product)
    }

    pub fn forward(&self, input: &[f64], memory: &[f64], history: &[f64]) -> Output {
        let attention = self.attention(input, memory);
        let normalized = rms(&attention);
        let ffn = self.ffn(&normalized);
        let features: Vec<_> = normalized.iter().chain(history).copied().collect();
        let latent: Vec<_> = self
            .project(U, &features)
            .iter()
            .map(|value| value.tanh())
            .collect();
        let heads = self.project(P, &latent);
        let values: Vec<_> = attention
            .iter()
            .enumerate()
            .map(|(j, value)| {
                value
                    + 0.25
                        * ((1.0 + 0.125 * heads[j].tanh()) * ffn[j]
                            + 0.125 * heads[self.width + j].tanh())
            })
            .collect();
        let route_features = rms(&values)
            .into_iter()
            .chain(history.iter().copied())
            .collect();
        Output {
            values,
            latent,
            route_features,
        }
    }

    pub fn detached_policy(&self, frozen: &Output, selected: usize, advantage: f64) -> f64 {
        let mut logits = self.project(ROUTER, &frozen.route_features);
        let heads = self.project(P, &frozen.latent);
        for (index, value) in logits.iter_mut().enumerate() {
            *value += 0.25 * heads[2 * self.width + index];
        }
        advantage * probabilities(&logits)[selected].ln()
    }
}
