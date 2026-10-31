//! History replay for evaluation and generation: rebuilds training's `history` input from each position's actual next byte with frozen weights; see AGENTS.md §core.

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
    /// The `history` input for the next forward pass: zero when the knob is off, else training's
    /// prune then mean.
    pub fn input(&mut self, state: &State) -> Result<Vec<f32>> {
        if !tuning::current().history {
            return Ok(vec![0.0; state.geometry.width]);
        }
        self.history
            .prune(self.clock, &state.tile_versions, state.shared.version);
        self.history.mean(state.geometry)
    }

    /// Record this position's corrections toward `target`, as training does after each anchor,
    /// without touching any weight. When the knob is off this skips `credit::gather`'s full
    /// backward pass entirely and returns the same loss straight from the readout (matching
    /// `Shared::task`, which `gather` would otherwise compute identically).
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
