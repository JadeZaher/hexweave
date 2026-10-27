use candle_core::{Device, Tensor, Var};

use super::ops::{log_softmax, Random};

pub(super) const PULSE_WIDTH: usize = 8;

pub(super) struct Controller {
    pulse_input: Var,
    pulse_recurrent: Var,
    route: Var,
    spatial: Var,
}

impl Controller {
    pub fn new(width: usize, tiles: usize, random: &mut Random) -> candle_core::Result<Self> {
        Ok(Self {
            pulse_input: random.matrix(width, PULSE_WIDTH)?,
            pulse_recurrent: random.matrix(PULSE_WIDTH, PULSE_WIDTH)?,
            route: random.matrix(width + PULSE_WIDTH, tiles + 1)?,
            spatial: random.matrix(2, 1)?,
        })
    }

    pub fn pulse(&self, first: &Tensor, previous: &Tensor) -> candle_core::Result<Tensor> {
        first
            .matmul(&self.pulse_input)?
            .add(&previous.matmul(&self.pulse_recurrent)?)?
            .tanh()
    }

    pub fn choose(
        &self,
        first: &Tensor,
        pulse: &Tensor,
        candidates: &[usize],
        displacement: &[f32],
        random: Option<&mut Random>,
    ) -> candle_core::Result<(usize, Tensor)> {
        let features = Tensor::cat(&[first, pulse], 1)?;
        let logits = features.matmul(&self.route)?.squeeze(0)?;
        let indices: Vec<_> = candidates.iter().map(|&id| id as u32).collect();
        let indices = Tensor::from_vec(indices, candidates.len(), &Device::Cpu)?;
        let offsets = Tensor::from_slice(displacement, (candidates.len(), 2), &Device::Cpu)?;
        let legal = logits
            .index_select(&indices, 0)?
            .add(&offsets.matmul(&self.spatial)?.squeeze(1)?)?;
        if !legal
            .to_vec1::<f32>()?
            .iter()
            .all(|value| value.is_finite())
        {
            candle_core::bail!("nonfinite legal route logits");
        }
        let log_probabilities = log_softmax(&legal, 0)?;
        let probabilities = log_probabilities.exp()?.to_vec1::<f32>()?;
        let total: f64 = probabilities.iter().map(|&value| f64::from(value)).sum();
        if probabilities.is_empty()
            || !total.is_finite()
            || total <= 0.0
            || probabilities
                .iter()
                .any(|&value| !value.is_finite() || value < 0.0)
        {
            candle_core::bail!("invalid route probability distribution");
        }
        let selected = if let Some(random) = random {
            let threshold = f64::from(random.unit()) * total;
            let mut cumulative = 0_f64;
            probabilities
                .iter()
                .position(|probability| {
                    cumulative += f64::from(*probability);
                    cumulative >= threshold
                })
                .unwrap_or_else(|| {
                    probabilities
                        .iter()
                        .rposition(|&value| value > 0.0)
                        .expect("positive probability mass")
                })
        } else {
            probabilities
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(index, _)| index)
                .expect("nonempty probability vector")
        };
        Ok((
            candidates[selected],
            log_probabilities.narrow(0, selected, 1)?.reshape(())?,
        ))
    }

    pub fn parameters(&self) -> Vec<(&'static str, Var)> {
        vec![
            ("pulse_input", self.pulse_input.clone()),
            ("pulse_recurrent", self.pulse_recurrent.clone()),
            ("router", self.route.clone()),
            ("router_spatial", self.spatial.clone()),
        ]
    }
}
