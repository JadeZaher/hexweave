//! Positive Lambert W gains from observed correction quality; see AGENTS.md.

use super::connected_config::Options;
use super::connected_predictor::FitMetrics;
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Gain {
    pub multiplier: f32,
    pub quality_ema: f64,
    pub observations: u64,
    pub epoch: u64,
    pub revision: u64,
}

impl Gain {
    pub fn new() -> Self {
        Self {
            multiplier: 1.0,
            quality_ema: 0.0,
            observations: 0,
            epoch: 0,
            revision: 0,
        }
    }

    pub fn validate(&self, options: &Options) -> Result<()> {
        ensure!(
            self.multiplier.is_finite()
                && self.multiplier >= options.gain_min
                && self.multiplier <= options.gain_max
                && self.quality_ema.is_finite()
                && (-1.0..=1.0).contains(&self.quality_ema),
            "invalid connected gain"
        );
        Ok(())
    }

    pub fn observe(
        &mut self,
        metrics: &FitMetrics,
        epoch: u64,
        revision: u64,
        options: &Options,
    ) -> Result<()> {
        ensure!(
            epoch >= self.epoch && revision >= self.revision,
            "gain evidence moved backwards"
        );
        ensure!(
            [
                metrics.mse,
                metrics.zero_mse,
                metrics.cosine,
                metrics.target_norm,
                metrics.prediction_norm
            ]
            .iter()
            .all(|x| x.is_finite())
                && metrics.mse >= 0.0
                && metrics.zero_mse >= 0.0
                && (-1.000001..=1.000001).contains(&metrics.cosine),
            "invalid gain evidence"
        );
        // No reference direction means no new gain evidence; the old observation still ages.
        if metrics.zero_mse <= 1e-12 {
            return self.validate(options);
        }
        let improvement = (1.0 - metrics.mse / (metrics.zero_mse + 1e-12)).clamp(-1.0, 1.0);
        let quality = (0.5 * metrics.cosine + 0.5 * improvement).clamp(-1.0, 1.0);
        self.quality_ema = 0.75 * self.quality_ema + 0.25 * quality;
        let control = 6.0 * (0.75 * self.quality_ema + 0.25 * quality);
        self.multiplier = (control.signum() * lambert_w(control.abs())?)
            .exp()
            .clamp(f64::from(options.gain_min), f64::from(options.gain_max))
            as f32;
        self.observations = self
            .observations
            .checked_add(1)
            .context("gain count overflow")?;
        self.epoch = epoch;
        self.revision = revision;
        self.validate(options)
    }

    pub fn value(&self, epoch: u64, options: &Options) -> f32 {
        let Some(age) = epoch.checked_sub(self.epoch) else {
            return 1.0;
        };
        if self.observations == 0 || age > 64 {
            return 1.0;
        }
        (f64::from(self.multiplier).ln() / (1.0 + age as f64 / 16.0))
            .exp()
            .clamp(f64::from(options.gain_min), f64::from(options.gain_max)) as f32
    }

    pub fn value_at(&self, epoch: u64, revision: u64, options: &Options) -> f32 {
        if revision
            .checked_sub(self.revision)
            .is_none_or(|lag| lag > 8)
        {
            return 1.0;
        }
        self.value(epoch, options)
    }
}

/// Principal real Lambert W, safeguarded Newton iteration; https://dlmf.nist.gov/4.13.
pub(super) fn lambert_w(value: f64) -> Result<f64> {
    ensure!(
        value.is_finite() && (0.0..=1e6).contains(&value),
        "Lambert W input outside bound"
    );
    if value == 0.0 {
        return Ok(0.0);
    }
    let (mut low, mut high) = (0.0, value.ln_1p());
    let mut w = 0.5 * high;
    for _ in 0..48 {
        let exponential = w.exp();
        let residual = w * exponential - value;
        if residual.abs() <= 4.0 * f64::EPSILON * value.max(1.0) {
            return Ok(w);
        }
        if residual < 0.0 {
            low = w;
        } else {
            high = w;
        }
        let candidate = w - residual / (exponential * (w + 1.0));
        w = if candidate > low && candidate < high {
            candidate
        } else {
            0.5 * (low + high)
        };
    }
    ensure!(
        (w * w.exp() - value).abs() <= 1e-10 * value.max(1.0),
        "Lambert W failed to converge"
    );
    Ok(w)
}
