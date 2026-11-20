//! Shared-memory reach: `memory_lag` admission, live-read accounting and the history input; see
//! CONNECTED.md §Memory reach and visit normalization switches.

use super::connected_config::Options;
use super::connected_state::State;
use super::geometry::TILES;
use anyhow::{ensure, Result};
use serde_json::{json, Value};
use std::ops::RangeInclusive;

/// Admitted `connected.memory_lag` values.
pub(super) const LAGS: RangeInclusive<u64> = 1..=64;

/// Serde default for a memory record without `lag`: the one-anchor limit of every earlier run.
pub(super) fn unit_lag() -> u64 {
    1
}

/// Serde skip: the default lag stays out of options identity and the memory record.
pub(super) fn is_unit_lag(lag: &u64) -> bool {
    *lag == 1
}

/// Admission of `connected.memory_lag`.
pub(super) fn validate(options: &Options) -> Result<()> {
    ensure!(
        LAGS.contains(&options.memory_lag),
        "connected memory_lag must be in 1..=64"
    );
    Ok(())
}

/// Session-only tally of live training-anchor memory reads; never checkpointed.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Reach {
    reads: u64,
    records: u64,
    lag_sum: u64,
    lag_max: u64,
}

impl Reach {
    /// Reads, mean records per read and mean shared-revision lag since `previous`; session max.
    pub fn report(&self, previous: &Self, memory_lag: u64) -> Value {
        let reads = self.reads - previous.reads;
        let records = self.records - previous.records;
        let lags = (self.lag_sum - previous.lag_sum) as f64;
        json!({"memory_lag": memory_lag, "reads": reads,
            "records_per_read": (reads > 0).then(|| records as f64 / reads as f64),
            "mean_lag": (records > 0).then(|| lags / records as f64),
            "session_lag_max": self.lag_max})
    }

    /// Largest shared-revision lag any live read has averaged this session.
    #[cfg(test)]
    pub fn lag_max(&self) -> u64 {
        self.lag_max
    }
}

/// A live training anchor's read at `tile` under the anchor's own shared revision, tallied.
pub(super) fn anchor_read(
    state: &mut State,
    input: &[f32],
    tile: usize,
    revisions: &[u64; TILES],
    options: &Options,
) -> Result<Vec<f32>> {
    state.counters.memory_reads += 1;
    let retrieval =
        state
            .memory
            .query_report(input, tile, state.epoch, revisions, state.anchors, None)?;
    let reach = &mut state.reach;
    reach.reads += 1;
    reach.records += retrieval.records as u64;
    reach.lag_sum += retrieval.lag_sum;
    reach.lag_max = reach.lag_max.max(retrieval.lag_max);
    Ok(options.history_input(retrieval.values))
}

impl Options {
    /// The tile history input: retrieved memory, or zeros of its width under the ablation.
    pub fn history_input(&self, retrieved: Vec<f32>) -> Vec<f32> {
        if self.history {
            retrieved
        } else {
            vec![0.0; retrieved.len()]
        }
    }
}
