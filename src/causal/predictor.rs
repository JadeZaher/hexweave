use super::geometry::{Geometry, EVIDENCE_RULE, RADIUS};
use super::norm;
use crate::ternary::TernaryMatrix;
use anyhow::{ensure, Result};

pub(super) struct Predictor {
    pub geometry: Geometry,
    pub matrix: TernaryMatrix,
    pub version: u64,
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
            [q, r, q + r]
                .iter()
                .all(|value| value.unsigned_abs() as usize <= RADIUS),
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

    #[cfg(test)]
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

    #[cfg(test)]
    pub fn correction(&self, features: &[f32]) -> Result<Vec<f32>> {
        norm::unit_clip(&self.predict(features)?)
    }

    #[cfg(test)]
    pub fn train_joint(
        &mut self,
        pairs: &[(&[f32], &[f32])],
    ) -> Result<super::predictor_joint::Update> {
        super::predictor_joint::apply(self, pairs)
    }
}
