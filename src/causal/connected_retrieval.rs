//! Eligibility-filtered, similarity-weighted shared-memory reads; see CONNECTED.md §Shared signal.

use super::super::geometry::TILES;
use super::super::norm;
use super::{cosine, length, Kind, Memory, Record, MEMORY_AGE};
use anyhow::{ensure, Result};

/// One read: the retrieved correction mean, the records it averaged and their shared-revision lags.
pub(in super::super) struct Retrieval {
    pub values: Vec<f32>,
    pub records: usize,
    pub cross_lane_records: usize,
    /// Sum and maximum of `shared - record.shared_revision` over the averaged records.
    pub lag_sum: u64,
    pub lag_max: u64,
}

impl Memory {
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
        let (mut records, mut cross_lane_records, mut lag_sum, mut lag_max) = (0, 0, 0, 0);
        for record in &self.records {
            let Some(age) = epoch.checked_sub(record.epoch) else {
                continue;
            };
            if !eligible(record, epoch, revisions, shared_revision, self.lag)
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
            // `eligible` has checked the subtraction.
            let lag = shared_revision - record.shared_revision;
            lag_sum += lag;
            lag_max = lag_max.max(lag);
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
            lag_sum,
            lag_max,
        })
    }
}

/// Age at most 64 epochs, source-tile revision lag at most 8, shared-revision lag at most `lag`.
pub(in super::super) fn eligible(
    record: &Record,
    epoch: u64,
    revisions: &[u64; TILES],
    shared: u64,
    lag: u64,
) -> bool {
    epoch
        .checked_sub(record.epoch)
        .is_some_and(|age| age <= MEMORY_AGE)
        && revisions[record.tile_id]
            .checked_sub(record.revision)
            .is_some_and(|tile_lag| tile_lag <= 8)
        && shared
            .checked_sub(record.shared_revision)
            .is_some_and(|shared_lag| shared_lag <= lag)
}
