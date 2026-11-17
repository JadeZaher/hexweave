//! Bounded prioritized replay of past anchor-exact positions; see CONNECTED.md §Prioritized replay.

use super::connected_anchor;
use super::connected_lane_apply::Accumulator;
use super::connected_state::State;
use super::connected_store::Store;
use super::execution::TrainingSource;
use super::tile_gradients::TileGradients;
use super::tuning;
use anyhow::{ensure, Result};
use serde_json::{json, Value};
use std::cmp::Ordering;

/// Route-sampling salt for replayed anchors ("REPL").
const REPLAY_SALT: u64 = 0x5245_504c;

/// One past training anchor: its cursor position and most recent loss as priority.
struct Entry {
    position: u64,
    priority: f32,
}

/// Session-only replay buffer and counters; resume starts it empty.
#[derive(Default)]
pub(super) struct Buffer {
    entries: Vec<Entry>,
    anchors: u64,
    visits: u64,
    loss_sum: f64,
    /// Times a tile restart emptied the buffer.
    clears: u64,
}

/// Replay order: highest priority first, then the oldest position.
fn replay_order(a: &Entry, b: &Entry) -> Ordering {
    b.priority
        .total_cmp(&a.priority)
        .then(a.position.cmp(&b.position))
}

/// Forgetting order: lowest priority first, then the oldest position.
fn forget_order(a: &Entry, b: &Entry) -> Ordering {
    a.priority
        .total_cmp(&b.priority)
        .then(a.position.cmp(&b.position))
}

impl Buffer {
    /// Session counters and current size for logs and the report.
    pub fn report(&self) -> Value {
        json!({"buffer_size": self.entries.len(), "replay_anchors": self.anchors,
            "replay_visits": self.visits, "replay_loss_sum": self.loss_sum, "buffer_clears": self.clears,
            "replay_loss_mean": (self.anchors > 0).then(|| self.loss_sum / self.anchors as f64)})
    }

    /// Inserts the live anchor, then forgets the least important entry beyond `capacity`.
    pub fn admit(&mut self, position: u64, loss: f32, capacity: usize) {
        self.entries.push(Entry {
            position,
            priority: loss,
        });
        if self.entries.len() > capacity {
            let lowest = (0..self.entries.len())
                .min_by(|&a, &b| forget_order(&self.entries[a], &self.entries[b]))
                .unwrap_or(0);
            self.entries.remove(lowest);
        }
    }

    /// Empties the buffer after a tile restart; session counters keep their totals.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.clears += 1;
    }

    /// Indices of the `count` entries to replay, in replay order.
    fn ranked(&self, count: usize) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..self.entries.len()).collect();
        indices.sort_by(|&a, &b| replay_order(&self.entries[a], &self.entries[b]));
        indices.truncate(count);
        indices
    }
}

/// Re-runs the top entries under the live anchor's weights; only tile gradients leave here.
pub(super) fn run<S: TrainingSource>(
    state: &mut State,
    store: &mut Store,
    source: &S,
    gradients: &mut Accumulator,
) -> Result<()> {
    let Some(replay) = state.options.replay else {
        return Ok(());
    };
    for index in state.replay.ranked(replay.per_anchor) {
        let position = state.replay.entries[index].position;
        let (loss, frames) = replay_one(state, store, source, position)?;
        let visits = frames.len() as u64;
        for (tile, gradient) in frames {
            gradients.add_exact(tile, gradient)?;
        }
        let buffer = &mut state.replay;
        buffer.entries[index].priority = loss;
        buffer.anchors += 1;
        buffer.visits += visits;
        buffer.loss_sum += f64::from(loss);
    }
    Ok(())
}

/// One replayed anchor's loss and per-frame tile gradients; memory is cut off at `position`.
pub(super) fn replay_one<S: TrainingSource>(
    state: &State,
    store: &mut Store,
    source: &S,
    position: u64,
) -> Result<(f32, Vec<(usize, TileGradients)>)> {
    replay_salted(state, store, source, position, REPLAY_SALT)
}

/// `replay_one` with its own route-sampling `salt`.
pub(super) fn replay_salted<S: TrainingSource>(
    state: &State,
    store: &mut Store,
    source: &S,
    position: u64,
    salt: u64,
) -> Result<(f32, Vec<(usize, TileGradients)>)> {
    ensure!(
        position < state.anchors,
        "connected replay of an untrained anchor"
    );
    let revisions = std::array::from_fn(|id| store.slots[id].meta.revision);
    let cursor = source.replay_cursor(position)?;
    let prefix = state.shared.prefix(&source.pending_prefix(&cursor)?)?;
    let random = tuning::hash(state.seed, &[state.anchors, position, salt]);
    // Records from anchor `position` on carry its own or later labels (CONNECTED.md §Replay).
    let trace = connected_anchor::forward(
        store,
        state.geometry,
        &prefix,
        state.options.visits,
        Some(random),
        |tile, input| {
            let retrieved =
                state
                    .memory
                    .query_before(input, tile, state.epoch, &revisions, position)?;
            Ok(state.options.history_input(retrieved))
        },
    )?;
    let target = source.anchor_target(&cursor)?;
    let credit = connected_anchor::gather(&state.shared, store, &prefix, &trace, target.byte)?;
    let tiles = trace.frames.iter().map(|frame| frame.tile);
    Ok((credit.loss, tiles.zip(credit.gradients).collect()))
}

#[cfg(test)]
mod tests {
    use super::Buffer;

    #[test]
    fn buffer_replays_highest_first_and_forgets_lowest_oldest() {
        // (admitted (position, loss), capacity, replayed count, expected positions in replay order)
        for (admitted, capacity, count, expected) in [
            // Highest priority first; equal priorities replay the oldest position first.
            (
                vec![(0, 1.0), (1, 3.0), (2, 2.0), (3, 3.0)],
                4,
                4,
                vec![1, 3, 2, 0],
            ),
            (
                vec![(0, 1.0), (1, 3.0), (2, 2.0), (3, 3.0)],
                4,
                2,
                vec![1, 3],
            ),
            // Over capacity the lowest priority is forgotten; on a tie, the oldest.
            (vec![(0, 1.0), (1, 1.0), (2, 2.0)], 2, 2, vec![2, 1]),
            // A just-admitted entry below every kept priority is itself forgotten.
            (vec![(0, 5.0), (1, 4.0), (2, 0.5)], 2, 2, vec![0, 1]),
        ] {
            let mut buffer = Buffer::default();
            for (position, loss) in admitted {
                buffer.admit(position, loss, capacity);
            }
            let order: Vec<u64> = buffer
                .ranked(count)
                .into_iter()
                .map(|index| buffer.entries[index].position)
                .collect();
            assert_eq!(order, expected);
        }
    }
}
