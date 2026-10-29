//! History replay for evaluation and generation: rebuilds training's `history` input from each
//! position's actual next byte with frozen weights, on training's own schedule; see AGENTS.md §core.

use super::credit;
use super::execution::Trace;
use super::history::History;
use super::shared::Prefix;
use super::state::State;
use super::store_api::TileStore;
use anyhow::Result;

/// Training's clock advances four ticks per anchor; records live for two anchors.
const TICKS_PER_ANCHOR: u64 = 4;

#[derive(Default)]
pub(super) struct Replay {
    history: History,
    clock: u64,
}

impl Replay {
    /// The `history` input for the next forward pass: training's prune, then mean.
    pub fn input(&mut self, state: &State) -> Result<Vec<f32>> {
        self.history
            .prune(self.clock, &state.tile_versions, state.shared.version);
        self.history.mean(state.geometry)
    }

    /// Record this position's corrections toward `target`, as training does after each anchor,
    /// without touching any weight.
    pub fn observe<S: TileStore>(
        &mut self,
        state: &State,
        store: &mut S,
        prefix: &Prefix,
        trace: &Trace,
        target: u8,
    ) -> Result<f32> {
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
