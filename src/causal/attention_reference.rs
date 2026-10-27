use super::super::tile::Tile;
use anyhow::Result;

pub(super) struct Reference {
    pub weights: [Vec<f64>; 4],
    pub width: usize,
    pub heads: usize,
}

impl Reference {
    pub fn new(tile: &Tile) -> Result<Self> {
        let mut weights: [Vec<f64>; 4] = std::array::from_fn(|_| Vec::new());
        for (destination, matrix) in weights.iter_mut().zip(&tile.matrices) {
            *destination = (0..matrix.len())
                .map(|i| Ok(f64::from(matrix.value(i)?)))
                .collect::<Result<_>>()?;
        }
        Ok(Self {
            weights,
            width: tile.geometry.width,
            heads: tile.geometry.heads,
        })
    }

    fn project(&self, index: usize, values: &[f64]) -> Vec<f64> {
        self.weights[index]
            .chunks_exact(self.width)
            .map(|row| {
                row.iter().zip(values).map(|(w, x)| w * x).sum::<f64>() / (self.width as f64).sqrt()
            })
            .collect()
    }

    /// Independent legacy row-projected f64 attention, including residual.
    pub fn forward(&self, input: &[f64], memory: &[f64]) -> Vec<f64> {
        let inverse = (input.iter().map(|x| x * x).sum::<f64>() / self.width as f64 + 1e-5)
            .sqrt()
            .recip();
        let normalized: Vec<_> = input.iter().map(|x| x * inverse).collect();
        let query = self.project(0, &normalized);
        let keys: Vec<_> = memory
            .chunks_exact(self.width)
            .map(|row| self.project(1, row))
            .collect();
        let values: Vec<_> = memory
            .chunks_exact(self.width)
            .map(|row| self.project(2, row))
            .collect();
        let k = self.width / self.heads;
        let knobs = super::super::tuning::current();
        let mut attended = vec![0.0; self.width];
        for head in 0..self.heads {
            let start = head * k;
            let mut scores: Vec<_> = keys
                .iter()
                .map(|key| {
                    (start..start + k).map(|j| query[j] * key[j]).sum::<f64>() / (k as f64).sqrt()
                        * f64::from(knobs.attention_scale)
                })
                .collect();
            let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            scores.iter_mut().for_each(|x| *x = (*x - maximum).exp());
            let total: f64 = scores.iter().sum();
            for (score, value) in scores.iter().zip(&values) {
                for j in start..start + k {
                    attended[j] += score / total * value[j];
                }
            }
        }
        input
            .iter()
            .zip(self.project(3, &attended))
            .map(|(x, o)| x + f64::from(knobs.attention_mix) * o)
            .collect()
    }

    pub fn objective(&self, input: &[f64], memory: &[f64], upstream: &[f32]) -> f64 {
        self.forward(input, memory)
            .iter()
            .zip(upstream)
            .map(|(y, g)| y * f64::from(*g))
            .sum()
    }
}
