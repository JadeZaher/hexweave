//! Structured cross-lane evidence and contextual detached retrieval; see AGENTS.md.

use super::geometry::{Geometry, TILES};
use super::norm;
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};

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

pub(super) struct Retrieval {
    pub values: Vec<f32>,
    pub records: usize,
    pub cross_lane_records: usize,
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
        })
    }

    pub fn validate(&self) -> Result<()> {
        self.geometry.validate()?;
        ensure!(
            (8..=64).contains(&self.capacity) && self.records.len() <= self.capacity,
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
            self.records.remove(0);
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

    pub fn query(
        &self,
        input: &[f32],
        tile_id: usize,
        epoch: u64,
        revisions: &[u64; TILES],
        shared_revision: u64,
    ) -> Result<Vec<f32>> {
        Ok(self
            .query_report(input, tile_id, epoch, revisions, shared_revision, None)?
            .values)
    }

    /// `query` restricted to records published by anchors strictly before `published_before`.
    pub fn query_before(
        &self,
        input: &[f32],
        tile_id: usize,
        epoch: u64,
        revisions: &[u64; TILES],
        published_before: u64,
    ) -> Result<Vec<f32>> {
        let cutoff = Some(published_before);
        Ok(self
            .retrieve(
                input,
                tile_id,
                epoch,
                revisions,
                published_before,
                None,
                cutoff,
            )?
            .values)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn query_report(
        &self,
        input: &[f32],
        tile_id: usize,
        epoch: u64,
        revisions: &[u64; TILES],
        shared_revision: u64,
        lane: Option<usize>,
    ) -> Result<Retrieval> {
        self.retrieve(
            input,
            tile_id,
            epoch,
            revisions,
            shared_revision,
            lane,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn retrieve(
        &self,
        input: &[f32],
        tile_id: usize,
        epoch: u64,
        revisions: &[u64; TILES],
        shared_revision: u64,
        lane: Option<usize>,
        published_before: Option<u64>,
    ) -> Result<Retrieval> {
        norm::vector(input, self.geometry.width)?;
        ensure!(tile_id < TILES, "invalid memory reader tile");
        let mut result = vec![0.0f64; self.geometry.width];
        let mut mass = 0.0;
        let (mut records, mut cross_lane_records) = (0, 0);
        for record in &self.records {
            let Some(age) = epoch.checked_sub(record.epoch) else {
                continue;
            };
            if !eligible(record, epoch, revisions, shared_revision)
                || published_before.is_some_and(|cutoff| record.shared_revision >= cutoff)
            {
                continue;
            }
            let similarity =
                0.75 * cosine(input, &record.input) + 0.25 * cosine(input, &record.output);
            let provenance = match record.kind {
                Kind::Observed => 1.0,
                Kind::Predicted => 0.25,
                Kind::SelfGenerated => 0.125,
            };
            let locality = if tile_id == record.tile_id { 1.0 } else { 0.75 };
            let weight = (3.0 * similarity).exp() * provenance * locality / (1.0 + age as f64);
            for (sum, value) in result.iter_mut().zip(&record.correction) {
                *sum += weight * f64::from(*value);
            }
            mass += weight;
            records += 1;
            if lane.is_some()
                && record.lane.is_some()
                && lane != record.lane
                && length(&record.correction) > 0.0
            {
                cross_lane_records += 1;
            }
        }
        let output: Vec<f32> = result
            .into_iter()
            .map(|v| (v / mass.max(1e-30)) as f32)
            .collect();
        Ok(Retrieval {
            values: norm::unit_clip(&output)?,
            records,
            cross_lane_records,
        })
    }
}

pub(super) fn eligible(record: &Record, epoch: u64, revisions: &[u64; TILES], shared: u64) -> bool {
    epoch
        .checked_sub(record.epoch)
        .is_some_and(|age| age <= MEMORY_AGE)
        && revisions[record.tile_id]
            .checked_sub(record.revision)
            .is_some_and(|lag| lag <= 8)
        && shared
            .checked_sub(record.shared_revision)
            .is_some_and(|lag| lag <= 1)
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
