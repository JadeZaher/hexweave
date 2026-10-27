use candle_core::{Device, Tensor, Var};

pub(super) struct Random(u64);

impl Random {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn unit(&mut self) -> f32 {
        // SplitMix64; a 23-bit midpoint conversion stays strictly inside (0, 1).
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^= value >> 31;
        ((value >> 41) as f32 + 0.5) / 8_388_608.0
    }

    pub fn matrix(&mut self, rows: usize, cols: usize) -> candle_core::Result<Var> {
        let scale = (3.0 / rows as f32).sqrt();
        let values: Vec<_> = (0..rows * cols)
            .map(|_| (self.unit() * 2.0 - 1.0) * scale)
            .collect();
        Var::from_vec(values, (rows, cols), &Device::Cpu)
    }
}

pub(super) fn log_softmax(values: &Tensor, dim: usize) -> candle_core::Result<Tensor> {
    let normalizer = values.log_sum_exp(dim)?.unsqueeze(dim)?;
    values.broadcast_sub(&normalizer)
}

pub(super) fn rms(values: &Tensor) -> candle_core::Result<Tensor> {
    let denominator = values.sqr()?.mean_keepdim(1)?.affine(1.0, 1e-5)?.sqrt()?;
    values.broadcast_div(&denominator)
}

/// Three-knot linear spline with linear endpoint extrapolation.
pub(super) fn curve(values: &Tensor, knots: &Tensor) -> candle_core::Result<Tensor> {
    let low = knots.narrow(1, 0, 1)?;
    let middle = knots.narrow(1, 1, 1)?;
    let high = knots.narrow(1, 2, 1)?;
    let left_slope = middle.sub(&low)?.affine(0.5, 0.0)?;
    let right_slope = high.sub(&middle)?.affine(0.5, 0.0)?;
    let linear = values.affine(1.0, 2.0)?.broadcast_mul(&left_slope)?;
    let hinge = values
        .relu()?
        .broadcast_mul(&right_slope.sub(&left_slope)?)?;
    linear.broadcast_add(&low)?.add(&hinge)
}

#[cfg(test)]
mod tests {
    use super::Random;

    #[test]
    fn adversarial_seeds_are_repeatable_nonconstant_and_inside_unit_interval() {
        for seed in [0, 0x9e37_79b9_7f4a_7c15, u64::MAX] {
            let mut first = Random::new(seed);
            let mut second = Random::new(seed);
            let samples: Vec<_> = (0..256).map(|_| first.unit()).collect();
            assert!(samples.iter().all(|&value| value > 0.0 && value < 1.0));
            assert!(samples.windows(2).any(|pair| pair[0] != pair[1]));
            for value in samples {
                assert_eq!(value, second.unit());
            }
        }
    }
}
