//! Packed ternary weights and bounded sign evidence; see src/AGENTS.md.

use anyhow::{ensure, Result};

mod head_rows;
mod kernels;

#[cfg(test)]
mod update_contracts;

pub use kernels::{avx2_available, kernel_name};

const MAX_VALUES: usize = 1_048_576;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TernaryMatrix {
    rows: usize,
    cols: usize,
    packed: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Evidence {
    values: Vec<i16>,
}

impl Evidence {
    pub fn new(len: usize) -> Result<Self> {
        ensure!((1..=MAX_VALUES).contains(&len), "invalid evidence length");
        Ok(Self {
            values: vec![0; len],
        })
    }

    /// Count the evidence payload, excluding allocation metadata.
    pub fn bytes(&self) -> usize {
        self.values.len() * std::mem::size_of::<i16>()
    }
}

impl TernaryMatrix {
    pub fn new(rows: usize, cols: usize, seed: u64) -> Result<Self> {
        let len = checked_length(rows, cols)?;
        let mut matrix = Self {
            rows,
            cols,
            packed: vec![0; len.div_ceil(4)],
        };
        let mut state = seed;
        for index in 0..len {
            matrix.write(index, (splitmix64(&mut state) % 3) as i8 - 1);
        }
        Ok(matrix)
    }

    pub fn from_values(rows: usize, cols: usize, values: &[i8]) -> Result<Self> {
        let len = checked_length(rows, cols)?;
        ensure!(values.len() == len, "ternary matrix length mismatch");
        ensure!(
            values.iter().all(|value| (-1..=1).contains(value)),
            "invalid trit"
        );
        let mut matrix = Self {
            rows,
            cols,
            packed: vec![0; len.div_ceil(4)],
        };
        for (index, &value) in values.iter().enumerate() {
            matrix.write(index, value);
        }
        Ok(matrix)
    }

    /// Decode canonical packed storage; see ternary/AGENTS.md.
    pub fn from_packed(rows: usize, cols: usize, bytes: &[u8]) -> Result<Self> {
        let len = checked_length(rows, cols)?;
        ensure!(bytes.len() == len.div_ceil(4), "packed length mismatch");
        for index in 0..len {
            ensure!(
                (bytes[index / 4] >> ((index % 4) * 2)) & 3 != 3,
                "invalid packed trit"
            );
        }
        if !len.is_multiple_of(4) {
            let padding_shift = (len % 4) * 2;
            ensure!(
                bytes[bytes.len() - 1] >> padding_shift == 0,
                "nonzero padding"
            );
        }
        Ok(Self {
            rows,
            cols,
            packed: bytes.to_vec(),
        })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn len(&self) -> usize {
        self.rows * self.cols
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Count packed weight bytes, excluding allocation metadata.
    pub fn storage_bytes(&self) -> usize {
        self.packed.len()
    }

    /// Borrow the canonical packed payload without decoding the matrix.
    pub fn packed_bytes(&self) -> &[u8] {
        &self.packed
    }

    pub fn value(&self, index: usize) -> Result<i8> {
        ensure!(index < self.len(), "ternary index out of bounds");
        Ok(self.read(index))
    }

    /// Read an unscaled column without a one-hot vector or full matrix decode.
    pub fn column(&self, index: usize) -> Result<Vec<f32>> {
        ensure!(index < self.cols, "column index out of bounds");
        Ok((0..self.rows)
            .map(|row| f32::from(self.read(row * self.cols + index)))
            .collect())
    }

    /// Accumulate negative-gradient signs; threshold crossings reset evidence.
    pub fn apply_gradient(
        &mut self,
        evidence: &mut Evidence,
        gradient: &[f32],
        threshold: i16,
    ) -> Result<usize> {
        ensure!(threshold > 0, "evidence threshold must be positive");
        ensure!(
            evidence.values.len() == self.len(),
            "evidence length mismatch"
        );
        validate_vector(gradient, self.len())?;
        let mut changed = 0;
        for (index, (&gradient, accumulated)) in
            gradient.iter().zip(&mut evidence.values).enumerate()
        {
            let direction = if gradient > 0.0 {
                -1
            } else if gradient < 0.0 {
                1
            } else {
                0
            };
            *accumulated = accumulated.saturating_add(direction);
            let movement = if *accumulated >= threshold {
                1
            } else if *accumulated <= -threshold {
                -1
            } else {
                continue;
            };
            let before = self.read(index);
            let after = (before + movement).clamp(-1, 1);
            self.write(index, after);
            changed += usize::from(after != before);
            *accumulated = 0;
        }
        Ok(changed)
    }

    /// Stochastic rounding (Gupta et al. 2015, arXiv:1502.02551): step against each gradient with probability min(rate*|g|, 1).
    /// Memoryless; see Li et al. 2017 (arXiv:1706.02379) for why it stalls without latent weights.
    pub fn apply_stochastic(&mut self, gradient: &[f32], rate: f64, stream: u64) -> Result<usize> {
        ensure!(rate.is_finite() && rate >= 0.0, "invalid stochastic rate");
        validate_vector(gradient, self.len())?;
        let mut changed = 0;
        for (index, &value) in gradient.iter().enumerate() {
            if value == 0.0 || uniform(stream, index) >= (rate * f64::from(value).abs()).min(1.0) {
                continue;
            }
            let before = self.read(index);
            let after = (before + if value > 0.0 { -1 } else { 1 }).clamp(-1, 1);
            self.write(index, after);
            changed += usize::from(after != before);
        }
        Ok(changed)
    }

    /// Latent-weight Adam step (BinaryConnect arXiv:1511.00363; Adam arXiv:1412.6980);
    /// trits threshold latents clipped to [-1, 1] at ±0.5 (TWN, arXiv:1605.04711).
    pub fn apply_latent(
        &mut self,
        latent: &mut [f32],
        moments: &mut [f32],
        gradient: &[f32],
        rate: f64,
        step: u64,
    ) -> Result<usize> {
        ensure!(
            rate.is_finite() && rate >= 0.0 && step > 0,
            "invalid latent step"
        );
        ensure!(
            latent.len() == self.len() && moments.len() == 2 * self.len(),
            "latent state length mismatch"
        );
        validate_vector(gradient, self.len())?;
        let exponent = i32::try_from(step.min(1 << 20))?;
        let (first, second) = (1.0 - 0.9f64.powi(exponent), 1.0 - 0.999f64.powi(exponent));
        let mut changed = 0;
        for (index, (&value, weight)) in gradient.iter().zip(latent.iter_mut()).enumerate() {
            let g = f64::from(value);
            let m = 0.9 * f64::from(moments[2 * index]) + 0.1 * g;
            let v = 0.999 * f64::from(moments[2 * index + 1]) + 0.001 * g * g;
            moments[2 * index] = m as f32;
            moments[2 * index + 1] = v as f32;
            let step = rate * (m / first) / ((v / second).sqrt() + 1e-8);
            let next = (f64::from(*weight) - step).clamp(-1.0, 1.0);
            *weight = next as f32;
            let after = if next >= 0.5 {
                1
            } else if next <= -0.5 {
                -1
            } else {
                0
            };
            if after != self.read(index) {
                self.write(index, after);
                changed += 1;
            }
        }
        Ok(changed)
    }

    fn read(&self, index: usize) -> i8 {
        ((self.packed[index / 4] >> ((index % 4) * 2)) & 3) as i8 - 1
    }

    fn write(&mut self, index: usize, value: i8) {
        let shift = (index % 4) * 2;
        let byte = &mut self.packed[index / 4];
        *byte = (*byte & !(3 << shift)) | (((value + 1) as u8) << shift);
    }
}

fn checked_length(rows: usize, cols: usize) -> Result<usize> {
    let len = rows.checked_mul(cols).unwrap_or(0);
    ensure!(
        (1..=MAX_VALUES).contains(&len),
        "invalid ternary matrix shape"
    );
    Ok(len)
}

fn validate_vector(values: &[f32], expected: usize) -> Result<()> {
    ensure!(values.len() == expected, "vector length mismatch");
    ensure!(
        values.iter().all(|value| value.is_finite()),
        "nonfinite vector"
    );
    Ok(())
}

fn accumulate(sum: &mut f32, trit: i8, value: f32) {
    match trit {
        -1 => *sum -= value,
        1 => *sum += value,
        _ => {}
    }
}

/// Counter-based uniform draw in [0, 1) for one coordinate of a stream.
fn uniform(stream: u64, index: usize) -> f64 {
    let mut state = stream.wrapping_add((index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15));
    (splitmix64(&mut state) >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
}

// SplitMix64: https://prng.di.unimi.it/splitmix64.c
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut value = *state;
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}
