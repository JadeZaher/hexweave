//! Replay on restart: re-run a restarted tile's recent positions, stepping only that tile.
//! See CONNECTED.md §Restart options (replay).

use super::connected_lane_apply::{record, update_tile, Accumulator};
use super::connected_replay::replay_salted;
use super::connected_state::State;
use super::connected_store::Store;
use super::execution::TrainingSource;
use anyhow::{ensure, Context, Result};

/// Route-sampling salt for restart replays ("RSTR"), distinct from prioritized replay's.
pub(super) const RESTART_SALT: u64 = 0x5253_5452;

/// Re-runs `tile`'s recent positions oldest first, each followed by one step of that tile only.
pub(super) fn run<S: TrainingSource>(
    state: &mut State,
    store: &mut Store,
    source: &S,
    tile: usize,
) -> Result<()> {
    let tiles = state.restarts.as_ref().context("missing restart state")?;
    let positions: Vec<u64> = tiles[tile].recent.iter().copied().collect();
    for position in positions {
        // `replay_one` reads memory only below `position` (`Memory::query_before`).
        let (loss, frames) = replay_salted(state, store, source, position, RESTART_SALT)?;
        let before = state.counters.local_updates;
        let mut gradients = Accumulator::default();
        for (visited, gradient) in frames {
            if visited == tile {
                gradients.add_exact(tile, gradient)?;
            }
        }
        let stepped = match gradients.into_mean(tile) {
            Some(mean) => {
                let options = state.options;
                // The fresh clock (origin = anchors) gives the full post-restart rate.
                let rate = options.local_rate
                    * super::connected_balance::rate_factor(&options, state, tile)?;
                let update = update_tile(store, tile, &mean, rate, 1.0, &options)?;
                record(&mut state.counters, tile, &update);
                true
            }
            None => false,
        };
        ensure!(
            state.counters.local_updates - before == u64::from(stepped),
            "connected restart replay step accounting"
        );
        let entry = &mut state.restarts.as_mut().context("missing restart state")?[tile];
        entry.replays += 1;
        entry.replay_steps += u64::from(stepped);
        entry.replay_loss_sum += f64::from(loss);
    }
    store.quiescent()
}
