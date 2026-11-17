//! Per-tile predictor target scale and least-squares trust; see CONNECTED.md.

use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};

const WARMUP: u64 = 1_000;
const EMA: f64 = 0.01;

/// Running means over evidence-bearing observed fits of one tile.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Calibration {
    pub target_energy: f64,
    pub alignment: f64,
    pub prediction_energy: f64,
    pub observations: u64,
}

impl Calibration {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            [self.target_energy, self.alignment, self.prediction_energy]
                .iter()
                .all(|value| value.is_finite())
                && self.prediction_energy >= 0.0
                && ((self.observations > 0 && self.target_energy > 0.0)
                    || (self.observations == 0 && *self == Self::default())),
            "invalid predictor calibration"
        );
        Ok(())
    }

    /// Target normalization `nu`; the first observation normalizes by its own norm.
    pub fn scale(&self, energy: f64) -> f64 {
        if self.observations == 0 {
            energy.sqrt()
        } else {
            self.target_energy.sqrt()
        }
    }

    /// Least-squares shrinkage slope of normalized targets on network outputs, clamped to [0, 1].
    pub fn trust(&self) -> f64 {
        if self.observations == 0 {
            return 0.0;
        }
        (self.alignment / self.prediction_energy.max(1e-12)).clamp(0.0, 1.0)
    }

    /// Output multiplier `nu * trust` mapping a unit-scale network output to correction scale.
    pub fn gain(&self) -> f64 {
        self.target_energy.sqrt() * self.trust()
    }

    /// Running mean for the first 1,000 observations, then an EMA with weight 0.01.
    pub fn observe(&mut self, energy: f64, alignment: f64, prediction_energy: f64) -> Result<()> {
        self.observations = self
            .observations
            .checked_add(1)
            .context("predictor calibration count overflow")?;
        let weight = if self.observations <= WARMUP {
            (self.observations as f64).recip()
        } else {
            EMA
        };
        self.target_energy += weight * (energy - self.target_energy);
        self.alignment += weight * (alignment - self.alignment);
        self.prediction_energy += weight * (prediction_energy - self.prediction_energy);
        self.validate()
    }
}
