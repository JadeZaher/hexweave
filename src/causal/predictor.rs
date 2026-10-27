use super::geometry::{Geometry, EVIDENCE_RULE};
use super::norm;
#[cfg(test)]
use super::tile_gradients::add_outer;
#[cfg(test)]
use crate::ternary::Evidence;
use crate::ternary::TernaryMatrix;
#[cfg(test)]
use anyhow::Context;
use anyhow::{ensure, Result};

pub(super) struct Predictor {
    pub geometry: Geometry,
    pub matrix: TernaryMatrix,
    pub version: u64,
}

#[cfg(test)]
pub(super) struct PredictorUpdate {
    pub mean_squared_error: f32,
    pub transitions: usize,
}

impl Predictor {
    pub fn new(geometry: Geometry, seed: u64) -> Result<Self> {
        geometry.validate()?;
        Ok(Self {
            geometry,
            version: 0,
            matrix: TernaryMatrix::new(geometry.width, geometry.features(), seed.wrapping_add(1))?,
        })
    }

    pub fn from_packed(
        geometry: Geometry,
        version: u64,
        packed: &[u8],
        rule: &str,
    ) -> Result<Self> {
        geometry.validate()?;
        ensure!(
            rule == EVIDENCE_RULE,
            "predictor evidence encoding mismatch"
        );
        Ok(Self {
            geometry,
            version,
            matrix: TernaryMatrix::from_packed(geometry.width, geometry.features(), packed)?,
        })
    }

    pub fn packed(&self) -> &[u8] {
        self.matrix.packed_bytes()
    }

    pub fn features(
        &self,
        input: &[f32],
        output: &[f32],
        history: &[f32],
        q: i32,
        r: i32,
    ) -> Result<Vec<f32>> {
        let d = self.geometry.width;
        norm::vector(input, d)?;
        norm::vector(output, d)?;
        norm::vector(history, d)?;
        ensure!(
            (-1..=1).contains(&q) && (-1..=1).contains(&r) && (-1..=1).contains(&(q + r)),
            "invalid predictor coordinate"
        );
        let x = norm::rms(input)?;
        let y = norm::rms(output)?;
        Ok(x.values
            .iter()
            .chain(&y.values)
            .chain(history)
            .copied()
            .chain([q as f32, r as f32])
            .collect())
    }

    pub fn predict(&self, features: &[f32]) -> Result<Vec<f32>> {
        norm::vector(features, self.geometry.features())?;
        let mut output = norm::project(
            &self.matrix,
            features,
            (self.geometry.features() as f32).sqrt().recip(),
        )?;
        for value in &mut output {
            *value = value.tanh();
        }
        Ok(output)
    }

    pub fn correction(&self, features: &[f32]) -> Result<Vec<f32>> {
        norm::unit_clip(&self.predict(features)?)
    }

    pub fn train_joint(
        &mut self,
        pairs: &[(&[f32], &[f32])],
    ) -> Result<super::predictor_joint::Update> {
        super::predictor_joint::apply(self, pairs)
    }

    /// Historical v2 dense update, retained only for archived diagnostic questions.
    #[cfg(test)]
    pub fn train(&mut self, features: &[f32], real_correction: &[f32]) -> Result<PredictorUpdate> {
        norm::vector(real_correction, self.geometry.width)?;
        let next = self
            .version
            .checked_add(1)
            .context("predictor version overflow")?;
        let target = norm::unit_clip(real_correction)?;
        let prediction = self.predict(features)?;
        let error = prediction
            .iter()
            .zip(&target)
            .map(|(p, t)| (p - t).powi(2))
            .sum::<f32>()
            / self.geometry.width as f32;
        ensure!(error.is_finite(), "nonfinite predictor error");
        let derivative: Vec<_> = prediction
            .iter()
            .zip(&target)
            .map(|(p, t)| 2.0 / self.geometry.width as f32 * (p - t) * (1.0 - p * p))
            .collect();
        let mut gradient = vec![0.0; self.matrix.len()];
        add_outer(
            &mut gradient,
            &derivative,
            features,
            (self.geometry.features() as f32).sqrt().recip(),
        )?;
        let transitions =
            self.matrix
                .apply_gradient(&mut Evidence::new(self.matrix.len())?, &gradient, 1)?;
        if transitions > 0 {
            self.version = next;
        }
        Ok(PredictorUpdate {
            mean_squared_error: error,
            transitions,
        })
    }
}
