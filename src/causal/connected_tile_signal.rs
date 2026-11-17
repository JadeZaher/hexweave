//! A tile's own training signal: input EMA, reference, distance, gradient and loss EMAs.
//! See CONNECTED.md §Per-tile training state.

use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};

/// EMA horizon in observations; the first `HORIZON` use the running mean (bias-free start).
pub(super) const HORIZON: f64 = 256.0;

/// Visits before the first reference: one full horizon, so it is a many-sample mean.
pub(super) const WARMUP: u64 = HORIZON as u64;

/// Denominator floor of the relative distance, as in `connected_restart::relative`.
fn tiny() -> f64 {
    f64::MIN_POSITIVE.sqrt()
}

/// Per-tile observations; empty vectors and zero counts until the first visit.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub(super) struct TileSignal {
    /// EMA of the visit input concatenated with its history input (width 2 x d).
    pub input_ema: Vec<f64>,
    /// `input_ema` snapshot at the last restart, or at warm-up; empty before warm-up.
    pub reference: Vec<f64>,
    pub visits: u64,
    /// EMA of the pre-rate L2 norm of the tile's mean local gradient per step.
    pub gradient_norm_ema: f64,
    pub steps: u64,
    /// EMA of the anchor loss over anchors that visited the tile.
    pub loss_ema: f64,
    pub anchors: u64,
}

/// Bias-free EMA weight: running mean for the first `HORIZON` samples, then 1 / HORIZON.
fn weight(count: u64) -> f64 {
    (1.0 / count as f64).max(1.0 / HORIZON)
}

impl TileSignal {
    /// Whether nothing has been observed; such a signal is omitted from checkpoints.
    pub fn is_unused(&self) -> bool {
        *self == Self::default()
    }

    /// Whether the warm-up has passed and a reference exists.
    pub fn is_warm(&self) -> bool {
        !self.reference.is_empty()
    }

    /// Folds one visit's input and history into the EMA; the `WARMUP`-th sets the reference.
    pub fn observe_input(&mut self, input: &[f32], history: &[f32]) -> Result<()> {
        let width = input.len() + history.len();
        if self.visits == 0 {
            self.input_ema = vec![0.0; width];
        }
        ensure!(
            self.input_ema.len() == width,
            "connected tile signal width changed"
        );
        self.visits += 1;
        let alpha = weight(self.visits);
        let sample = input.iter().chain(history).map(|&v| f64::from(v));
        for (mean, value) in self.input_ema.iter_mut().zip(sample) {
            *mean += alpha * (value - *mean);
        }
        if !self.is_warm() {
            self.snapshot();
        }
        Ok(())
    }

    pub fn observe_gradient(&mut self, norm: f64) {
        self.steps += 1;
        self.gradient_norm_ema += weight(self.steps) * (norm - self.gradient_norm_ema);
    }

    pub fn observe_loss(&mut self, loss: f64) {
        self.anchors += 1;
        self.loss_ema += weight(self.anchors) * (loss - self.loss_ema);
    }

    /// ||input_ema - reference|| / max(||reference||, tiny); 0 before warm-up.
    pub fn distance(&self) -> f64 {
        let gap = self
            .input_ema
            .iter()
            .zip(&self.reference)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        let scale = self.reference.iter().map(|v| v * v).sum::<f64>().sqrt();
        gap / scale.max(tiny())
    }

    /// Restart: the reference becomes the current EMA (distance exactly 0); no-op before warm-up.
    pub fn snapshot(&mut self) {
        if self.visits >= WARMUP {
            self.reference.clone_from(&self.input_ema);
        }
    }

    /// Moves the reference `fraction` of the way to the EMA (Lambert law's chase).
    pub fn chase(&mut self, fraction: f64) {
        for (reference, mean) in self.reference.iter_mut().zip(&self.input_ema) {
            *reference += fraction * (mean - *reference);
        }
    }

    /// Shapes, counts and finiteness; `width` is the concatenated input width.
    pub fn validate(&self, width: usize) -> Result<()> {
        let used = self.visits > 0;
        ensure!(
            self.input_ema.len() == if used { width } else { 0 }
                && (!self.is_warm() || (self.reference.len() == width && self.visits >= WARMUP)),
            "connected tile signal shape"
        );
        ensure!(
            self.input_ema
                .iter()
                .chain(&self.reference)
                .chain([&self.gradient_norm_ema, &self.loss_ema])
                .all(|v| v.is_finite())
                && self.gradient_norm_ema >= 0.0
                && self.loss_ema >= 0.0
                && (self.steps > 0 || self.gradient_norm_ema == 0.0)
                && (self.anchors > 0 || self.loss_ema == 0.0),
            "nonfinite connected tile signal"
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{TileSignal, WARMUP};

    #[test]
    fn reference_waits_for_warmup_then_distance_is_relative_to_it() -> anyhow::Result<()> {
        let mut signal = TileSignal::default();
        for _ in 1..WARMUP {
            signal.observe_input(&[1.0], &[0.0])?;
            signal.snapshot();
            assert!(!signal.is_warm() && signal.distance() == 0.0);
        }
        // (input, history, expected EMA, distance from the warm-up reference [1, 0])
        for (input, history, ema, distance) in [
            ([1.0f32], [0.0f32], [1.0, 0.0], 0.0),
            ([257.0], [0.0], [2.0, 0.0], 1.0),
            ([2.0], [256.0], [2.0, 1.0], 2f64.sqrt()),
        ] {
            signal.observe_input(&input, &history)?;
            assert_eq!(
                (signal.input_ema.as_slice(), signal.distance()),
                (&ema[..], distance)
            );
        }
        signal.validate(2)?;
        signal.chase(0.5);
        assert_eq!(signal.reference, [1.5, 0.5]);
        signal.snapshot();
        assert_eq!(signal.distance(), 0.0);
        Ok(())
    }
}
