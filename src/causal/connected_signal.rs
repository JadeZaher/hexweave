//! Structured cross-lane evidence and contextual detached retrieval; see AGENTS.md.

use super::geometry::{Geometry, TILES};
use super::norm;
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};

#[path = "connected_retrieval.rs"]
mod retrieval;
#[cfg(test)]
pub(super) use retrieval::eligible;

pub(super) const MEMORY_AGE: u64 = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(super) enum Kind {
    Observed,
    Predicted,
    SelfGenerated,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Record {
    pub input: Vec<f32>,
    pub output: Vec<f32>,
    pub history: Vec<f32>,
    pub correction: Vec<f32>,
    pub tile_id: usize,
    pub revision: u64,
    pub shared_revision: u64,
    pub epoch: u64,
    pub kind: Kind,
    pub lane: Option<usize>,
}

impl Record {
    pub fn validate(&self, geometry: Geometry) -> Result<()> {
        ensure!(self.tile_id < TILES, "invalid connected evidence tile");
        ensure!(
            self.lane.is_none_or(|lane| lane < 8),
            "invalid memory source lane"
        );
        for vector in [&self.input, &self.output, &self.history, &self.correction] {
            norm::vector(vector, geometry.width)?;
        }
        ensure!(
            length(&self.correction) <= 1.000001,
            "unbounded connected correction"
        );
        ensure!(
            length(&self.history) <= 1.000001,
            "unbounded connected history"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Memory {
    pub geometry: Geometry,
    pub capacity: usize,
    pub records: Vec<Record>,
    pub publications: u64,
    pub observed: u64,
    pub predicted: u64,
    pub self_generated: u64,
    /// Full-memory eviction takes the busiest tile's oldest record (CONNECTED.md §Visit balance).
    #[serde(default, skip_serializing_if = "super::connected_balance::is_false")]
    pub balanced: bool,
    /// Largest shared-revision lag a read admits (CONNECTED.md §Memory reach); omitted at 1.
    #[serde(
        default = "super::connected_reach::unit_lag",
        skip_serializing_if = "super::connected_reach::is_unit_lag"
    )]
    pub lag: u64,
}

impl Memory {
    pub fn new(geometry: Geometry, capacity: usize) -> Result<Self> {
        geometry.validate()?;
        ensure!(
            (8..=64).contains(&capacity),
            "invalid connected memory capacity"
        );
        Ok(Self {
            geometry,
            capacity,
            records: Vec::new(),
            publications: 0,
            observed: 0,
            predicted: 0,
            self_generated: 0,
            balanced: false,
            lag: 1,
        })
    }

    pub fn validate(&self) -> Result<()> {
        self.geometry.validate()?;
        ensure!(
            (8..=64).contains(&self.capacity)
                && self.records.len() <= self.capacity
                && super::connected_reach::LAGS.contains(&self.lag),
            "invalid connected memory shape"
        );
        let total = self
            .observed
            .checked_add(self.predicted)
            .and_then(|n| n.checked_add(self.self_generated))
            .context("memory count overflow")?;
        ensure!(
            total == self.publications && self.publications >= self.records.len() as u64,
            "invalid memory publication counters"
        );
        for record in &self.records {
            record.validate(self.geometry)?;
        }
        ensure!(
            self.records.windows(2).all(|p| p[0].epoch <= p[1].epoch),
            "memory publication time reversed"
        );
        Ok(())
    }

    pub fn publish(&mut self, record: Record) -> Result<()> {
        record.validate(self.geometry)?;
        ensure!(
            self.records.last().is_none_or(|r| r.epoch <= record.epoch),
            "memory publication time reversed"
        );
        self.publications = self
            .publications
            .checked_add(1)
            .context("memory count overflow")?;
        let counter = match record.kind {
            Kind::Observed => &mut self.observed,
            Kind::Predicted => &mut self.predicted,
            Kind::SelfGenerated => &mut self.self_generated,
        };
        *counter = counter.checked_add(1).context("memory count overflow")?;
        self.records
            .retain(|r| record.epoch.saturating_sub(r.epoch) <= MEMORY_AGE);
        if self.records.len() == self.capacity {
            let oldest = super::connected_balance::evict(&self.records, self.balanced);
            self.records.remove(oldest);
        }
        self.records.push(record);
        Ok(())
    }

    /// Drops every retained record `tile` published; publication counters keep their history.
    pub fn forget(&mut self, tile: usize) -> usize {
        let before = self.records.len();
        self.records.retain(|record| record.tile_id != tile);
        before - self.records.len()
    }
}

pub(super) fn length(values: &[f32]) -> f64 {
    values
        .iter()
        .map(|v| f64::from(*v).powi(2))
        .sum::<f64>()
        .sqrt()
}

pub(super) fn cosine(left: &[f32], right: &[f32]) -> f64 {
    let denominator = length(left) * length(right);
    if denominator <= 1e-30 {
        return 0.0;
    }
    (left
        .iter()
        .zip(right)
        .map(|(a, b)| f64::from(*a) * f64::from(*b))
        .sum::<f64>()
        / denominator)
        .clamp(-1.0, 1.0)
}
