//! Calibrated observed-credit FP32 predictor with explicit Adam state; see CONNECTED.md.

use super::connected_calibration::Calibration;
use super::connected_signal::{cosine, length, Kind, Record};
use super::geometry::{Geometry, TILES};
use super::{norm, tuning};
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};

/// Largest normalized-target norm seen by alignment and Adam.
const TARGET_CLIP: f64 = 4.0;

/// Connected feature count `4d + 2`; `Geometry::features` keeps sizing the core predictor.
fn feature_count(geometry: Geometry) -> usize {
    4 * geometry.width + 2
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct FitMetrics {
    pub mse: f64,
    pub zero_mse: f64,
    pub cosine: f64,
    pub prediction_norm: f64,
    pub target_norm: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Predictor {
    pub geometry: Geometry,
    pub weights: Vec<f32>,
    pub moments: Vec<f32>,
    pub fits: u64,
    pub calibration: [Calibration; TILES],
}

impl Predictor {
    pub fn new(geometry: Geometry, seed: u64) -> Result<Self> {
        geometry.validate()?;
        let count = geometry.width * feature_count(geometry);
        let weights = (0..count)
            .map(|i| {
                let bits = tuning::hash(seed, &[0x50524544, i as u64]) >> 40;
                ((bits as f32 / 16_777_215.0) * 2.0 - 1.0) * 0.01
            })
            .collect();
        Ok(Self {
            geometry,
            weights,
            moments: vec![0.0; 2 * count],
            fits: 0,
            calibration: [Calibration::default(); TILES],
        })
    }

    pub fn validate(&self, geometry: Geometry) -> Result<()> {
        geometry.validate()?;
        ensure!(self.geometry == geometry, "predictor geometry mismatch");
        norm::vector(&self.weights, geometry.width * feature_count(geometry))?;
        norm::vector(&self.moments, 2 * self.weights.len())?;
        ensure!(
            self.moments.iter().skip(1).step_by(2).all(|v| *v >= 0.0),
            "negative predictor second moment"
        );
        for calibration in &self.calibration {
            calibration.validate()?;
        }
        ensure!(
            self.steps() <= self.fits,
            "predictor calibration exceeds fits"
        );
        Ok(())
    }

    /// Adam steps taken: one per evidence-bearing fit, summed over tiles.
    fn steps(&self) -> u64 {
        self.calibration
            .iter()
            .fold(0, |sum, tile| sum.saturating_add(tile.observations))
    }

    /// Least-squares trust in `tile`'s predictions; zero before its first observation.
    pub fn trust(&self, tile: usize) -> f64 {
        self.calibration.get(tile).map_or(0.0, Calibration::trust)
    }

    /// `[rms(input); rms(output); history; target; q; r]`, `target` the observed byte's label
    /// feature (conditional DNI, https://proceedings.mlr.press/v70/jaderberg17a.html).
    pub fn features(
        &self,
        input: &[f32],
        output: &[f32],
        history: &[f32],
        target: &[f32],
        tile_id: usize,
    ) -> Result<Vec<f32>> {
        let d = self.geometry.width;
        for values in [input, output, history, target] {
            norm::vector(values, d)?;
        }
        ensure!(tile_id < TILES, "invalid predictor tile");
        let mesh = super::geometry::mesh();
        let point = mesh.coordinates()[tile_id];
        let x = norm::rms(input)?;
        let y = norm::rms(output)?;
        Ok(x.values
            .iter()
            .chain(&y.values)
            .chain(history)
            .chain(target)
            .copied()
            .chain([point.q as f32, point.r as f32])
            .collect())
    }

    fn raw(&self, features: &[f32]) -> Result<Vec<f32>> {
        let f = feature_count(self.geometry);
        norm::vector(features, f)?;
        let scale = (f as f64).sqrt().recip();
        let values: Vec<f32> = self
            .weights
            .chunks_exact(f)
            .map(|row| {
                (row.iter()
                    .zip(features)
                    .map(|(w, x)| f64::from(*w) * f64::from(*x))
                    .sum::<f64>()
                    * scale)
                    .tanh() as f32
            })
            .collect();
        norm::vector(&values, self.geometry.width)?;
        Ok(values)
    }

    /// Calibrated correction `nu * trust * p_hat` for `tile`, where `p_hat` is the network output.
    pub fn predict(&self, features: &[f32], tile: usize) -> Result<Vec<f32>> {
        ensure!(tile < TILES, "invalid predictor tile");
        self.calibrated(&norm::unit_clip(&self.raw(features)?)?, tile)
    }

    fn calibrated(&self, output: &[f32], tile: usize) -> Result<Vec<f32>> {
        let gain = self.calibration[tile].gain();
        let scaled: Vec<f32> = output
            .iter()
            .map(|p| (f64::from(*p) * gain) as f32)
            .collect();
        norm::unit_clip(&scaled)
    }

    /// Scores the calibrated prediction, then fits the network to the clipped normalized target;
    /// `target` is the label feature of the byte whose readout produced `record.correction`.
    pub fn fit(&mut self, record: &Record, target: &[f32], rate: f64) -> Result<FitMetrics> {
        ensure!(
            record.kind == Kind::Observed,
            "predictor targets require observed task credit"
        );
        record.validate(self.geometry)?;
        ensure!(
            rate.is_finite() && rate > 0.0 && rate <= 0.1,
            "invalid predictor rate"
        );
        let features = self.features(
            &record.input,
            &record.output,
            &record.history,
            target,
            record.tile_id,
        )?;
        let raw = self.raw(&features)?;
        let output = norm::unit_clip(&raw)?;
        let tile = record.tile_id;
        let metrics = metrics(&self.calibrated(&output, tile)?, &record.correction);
        self.fits = self
            .fits
            .checked_add(1)
            .context("predictor step overflow")?;
        // Only an exactly zero target is skipped: counted, but weights and statistics stay unchanged.
        if metrics.zero_mse <= 1e-30 {
            return Ok(metrics);
        }
        let energy = length(&record.correction).powi(2);
        let t_hat = normalized(&record.correction, self.calibration[tile].scale(energy));
        let alignment = output
            .iter()
            .zip(&t_hat)
            .map(|(p, t)| f64::from(*p) * t)
            .sum::<f64>();
        self.calibration[tile].observe(energy, alignment, length(&output).powi(2))?;
        self.apply(&features, &upstream(&raw, &output, &t_hat), rate)?;
        Ok(metrics)
    }

    fn apply(&mut self, features: &[f32], upstream: &[f64], rate: f64) -> Result<()> {
        let f = feature_count(self.geometry);
        let scale = (f as f64).sqrt().recip();
        let exponent = self.steps().min(1_048_576) as i32;
        let b1 = 1.0 - 0.9f64.powi(exponent);
        let b2 = 1.0 - 0.999f64.powi(exponent);
        for (index, weight) in self.weights.iter_mut().enumerate() {
            let g = upstream[index / f] * f64::from(features[index % f]) * scale;
            let m = 0.9 * f64::from(self.moments[2 * index]) + 0.1 * g;
            let v = 0.999 * f64::from(self.moments[2 * index + 1]) + 0.001 * g * g;
            self.moments[2 * index] = m as f32;
            self.moments[2 * index + 1] = v as f32;
            *weight = (f64::from(*weight) - rate * (m / b1) / ((v / b2).sqrt() + 1e-8)) as f32;
        }
        self.validate(self.geometry)
    }
}

/// Unit-scale target `t / nu`, clipped to norm 4: target clipping against calibration lag.
fn normalized(correction: &[f32], scale: f64) -> Vec<f64> {
    let target: Vec<f64> = correction.iter().map(|t| f64::from(*t) / scale).collect();
    let norm = target.iter().map(|t| t * t).sum::<f64>().sqrt();
    let factor = if norm > TARGET_CLIP {
        TARGET_CLIP / norm
    } else {
        1.0
    };
    target.into_iter().map(|t| t * factor).collect()
}

fn metrics(prediction: &[f32], target: &[f32]) -> FitMetrics {
    let d = prediction.len() as f64;
    FitMetrics {
        mse: prediction
            .iter()
            .zip(target)
            .map(|(p, t)| (f64::from(*p) - f64::from(*t)).powi(2))
            .sum::<f64>()
            / d,
        zero_mse: length(target).powi(2) / d,
        cosine: cosine(prediction, target),
        prediction_norm: length(prediction),
        target_norm: length(target),
    }
}

/// Mean-square-error adjoint through `unit_clip(tanh(z))` for a unit-scale target.
fn upstream(raw: &[f32], output: &[f32], target: &[f64]) -> Vec<f64> {
    let d = output.len() as f64;
    let residual: Vec<f64> = output
        .iter()
        .zip(target)
        .map(|(p, t)| f64::from(*p) - t)
        .collect();
    let raw_norm = length(raw);
    let projection = residual
        .iter()
        .zip(output)
        .map(|(r, p)| r * f64::from(*p))
        .sum::<f64>();
    residual
        .iter()
        .zip(output)
        .zip(raw)
        .map(|((r, p), z)| {
            let clipped = if raw_norm > 1.0 {
                (r - f64::from(*p) * projection) / raw_norm
            } else {
                *r
            };
            (2.0 / d) * clipped * (1.0 - f64::from(*z).powi(2))
        })
        .collect()
}
