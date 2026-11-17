//! A bounded, detached correction memory, not token activations or a KV cache; see STUDY.md.

use super::geometry::{Geometry, TILES};
use super::norm;
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};

/// Capacity counts route-frame records, not anchors or input bytes.
pub(super) const HISTORY_CAP: usize = 8;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Record {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub features: Vec<f32>,
    pub correction: Vec<f32>,
    pub tile_versions: [u64; TILES],
    pub shared_version: u64,
    pub opportunity: u64,
    pub tile_id: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct History {
    pub records: Vec<Record>,
}

impl Record {
    pub fn validate(&self, geometry: Geometry) -> Result<()> {
        ensure!(self.tile_id < TILES, "invalid history tile");
        if super::tuning::current().fit_predictor {
            norm::vector(&self.features, geometry.features())?;
        } else {
            ensure!(
                self.features.is_empty(),
                "v2 history contains predictor features"
            );
        }
        norm::vector(&self.correction, geometry.width)?;
        let squared: f64 = self.correction.iter().map(|v| f64::from(*v).powi(2)).sum();
        ensure!(squared <= 1.000002, "history correction exceeds unit bound");
        Ok(())
    }

    /// Keep age-bounded records only while every tile and E are at most one version ahead.
    pub fn compatible(&self, opportunity: u64, versions: &[u64; TILES], shared: u64) -> bool {
        fn near(old: u64, current: u64) -> bool {
            current.checked_sub(old).is_some_and(|lag| lag <= 1)
        }
        opportunity
            .checked_sub(self.opportunity)
            .is_some_and(|age| age <= super::tuning::current().history_ticks)
            && near(self.shared_version, shared)
            && self
                .tile_versions
                .iter()
                .zip(versions)
                .all(|(&old, &current)| near(old, current))
    }
}

impl History {
    #[cfg(test)]
    pub fn replay(&self) -> &[Record] {
        &self.records[self.records.len().saturating_sub(4)..]
    }

    pub fn validate(
        &self,
        geometry: Geometry,
        opportunity: u64,
        versions: &[u64; TILES],
        shared: u64,
    ) -> Result<()> {
        ensure!(
            self.records.len() <= HISTORY_CAP,
            "history exceeds capacity"
        );
        let mut previous = None;
        for record in &self.records {
            record.validate(geometry)?;
            ensure!(
                record.opportunity < opportunity,
                "history record is not a completed anchor"
            );
            ensure!(
                record.opportunity % 4 == 0,
                "history record from a local opportunity"
            );
            ensure!(
                previous.is_none_or(|old| old <= record.opportunity),
                "history order mismatch"
            );
            ensure!(
                record.shared_version <= shared
                    && record
                        .tile_versions
                        .iter()
                        .zip(versions)
                        .all(|(old, current)| old <= current),
                "history version is in the future"
            );
            previous = Some(record.opportunity);
        }
        Ok(())
    }

    pub fn prune(&mut self, opportunity: u64, versions: &[u64; TILES], shared: u64) -> usize {
        let before = self.records.len();
        self.records
            .retain(|record| record.compatible(opportunity, versions, shared));
        before - self.records.len()
    }

    /// Average route-frame corrections; anchors with more retained visits contribute more terms.
    pub fn mean(&self, geometry: Geometry) -> Result<Vec<f32>> {
        let mut mean = vec![0.0; geometry.width];
        if self.records.is_empty() {
            return Ok(mean);
        }
        for record in &self.records {
            record.validate(geometry)?;
            for (value, correction) in mean.iter_mut().zip(&record.correction) {
                *value += correction / self.records.len() as f32;
            }
        }
        norm::finite(&mean)?;
        Ok(mean)
    }

    pub fn push(&mut self, record: Record, geometry: Geometry) -> Result<()> {
        record.validate(geometry)?;
        ensure!(
            self.records.len() <= HISTORY_CAP,
            "history exceeds capacity"
        );
        if self.records.len() == HISTORY_CAP {
            self.records.remove(0);
        }
        if self.records.capacity() < HISTORY_CAP {
            self.records.reserve_exact(HISTORY_CAP - self.records.len());
        }
        self.records.push(record);
        Ok(())
    }
}
