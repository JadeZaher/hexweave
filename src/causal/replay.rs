//! Rebuild local correction history under frozen weights; see STUDY.md for evaluation costs.

use super::credit;
use super::execution::Trace;
use super::history::History;
use super::shared::Prefix;
use super::state::State;
use super::store_api::TileStore;
use super::tuning;
use anyhow::Result;

/// Training's clock advances four ticks per anchor; see AGENTS.md §core for the prune rule.
const TICKS_PER_ANCHOR: u64 = 4;
/// Warm-up positions scored before a probe span, and generation's own prompt warm-up length; see AGENTS.md §core.
pub(super) const SPAN_WARMUP: usize = 2;

#[derive(Default)]
pub(super) struct Replay {
    history: History,
    clock: u64,
}

impl Replay {
    /// Return zero when history is disabled; otherwise prune and average retained corrections.
    pub fn input(&mut self, state: &State) -> Result<Vec<f32>> {
        if !tuning::current().history {
            return Ok(vec![0.0; state.geometry.width]);
        }
        self.history
            .prune(self.clock, &state.tile_versions, state.shared.version);
        self.history.mean(state.geometry)
    }

    /// Observe a known or emitted target without updating weights; see STUDY.md for the distinction.
    pub fn observe<S: TileStore>(
        &mut self,
        state: &State,
        store: &mut S,
        prefix: &Prefix,
        trace: &Trace,
        target: u8,
    ) -> Result<f32> {
        if !tuning::current().history {
            self.clock += TICKS_PER_ANCHOR;
            return state.shared.readout(&trace.output)?.loss(target);
        }
        // History-on inference still runs correction backward; it also computes unused E credit.
        let credit = credit::gather(state, store, prefix, trace, target)?;
        let records = credit::records(
            state,
            trace,
            &credit.corrections,
            prefix.shared_version,
            self.clock,
        )?;
        for record in records {
            self.history.push(record, state.geometry)?;
        }
        self.clock += TICKS_PER_ANCHOR;
        Ok(credit.loss)
    }
}
