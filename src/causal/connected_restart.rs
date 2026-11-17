//! Per-tile local-rate floor and warm restarts; see CONNECTED.md §Warm restarts.

use super::connected_config::{Options, Restart};
use super::connected_restart_config::{Law, Signal};
use super::connected_restart_due as due;
use super::connected_state::State;
use super::connected_store::Store;
use super::connected_tile_signal::TileSignal;
use super::connected_update::UpdateMetrics;
use super::execution::TrainingSource;
use super::geometry::TILES;
use super::tile_gradients::TileGradients;
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

#[path = "connected_restart_check.rs"]
mod check;
pub(super) use check::{replay_steps, report, validate};

/// One tile's training-side state (CONNECTED.md §Per-tile training state); never in a tile bundle.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct TileRestart {
    /// Anchor index the tile's local-rate schedule restarted at; 0 before any restart.
    pub origin: u64,
    /// Accumulated shared change, or the tile's own input distance under `signal: tile`.
    pub drift: f64,
    /// Current restart threshold on `drift`; 0 when restarts are off (floor only).
    pub bar: f64,
    /// Restarts this tile has taken.
    pub restarts: u64,
    /// Shared-memory records dropped by this tile's restarts.
    pub forgotten_records: u64,
    /// Summed anchor loss and visited-anchor count of the cycle in progress.
    pub cycle_loss: f64,
    pub cycle_anchors: u64,
    /// Mean anchor loss of the last finished cycle; None before the first restart.
    pub previous_mean: Option<f64>,
    /// Multiplier of the tile's local-rate factor, in [0, 4].
    #[serde(default = "one", skip_serializing_if = "is_one")]
    pub plasticity: f64,
    /// Sum of plasticity over visited anchors under the Lambert law.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub plasticity_sum: f64,
    #[serde(default, skip_serializing_if = "TileSignal::is_unused")]
    pub signal: TileSignal,
    /// Restart-replayed positions, the steps they took and their summed loss.
    #[serde(default, skip_serializing_if = "is_zero_count")]
    pub replays: u64,
    #[serde(default, skip_serializing_if = "is_zero_count")]
    pub replay_steps: u64,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub replay_loss_sum: f64,
    /// Anchor since which a due tile has waited under `stagger`; None when not waiting.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub due_since: Option<u64>,
    /// Session-only: the most recent visited anchor positions, oldest first.
    #[serde(skip)]
    pub recent: VecDeque<u64>,
    /// Session-only: the tile took a local step during the anchor in flight.
    #[serde(skip)]
    pub stepped: bool,
}

fn one() -> f64 {
    1.0
}

fn is_one(value: &f64) -> bool {
    *value == 1.0
}

fn is_zero(value: &f64) -> bool {
    *value == 0.0
}

fn is_zero_count(value: &u64) -> bool {
    *value == 0
}

impl TileRestart {
    /// A tile that has never restarted, with restart threshold `bar` and dial `plasticity`.
    pub fn new(bar: f64, plasticity: f64) -> Self {
        Self {
            origin: 0,
            drift: 0.0,
            bar,
            restarts: 0,
            forgotten_records: 0,
            cycle_loss: 0.0,
            cycle_anchors: 0,
            previous_mean: None,
            plasticity,
            plasticity_sum: 0.0,
            signal: TileSignal::default(),
            replays: 0,
            replay_steps: 0,
            replay_loss_sum: 0.0,
            due_since: None,
            recent: VecDeque::new(),
            stepped: false,
        }
    }
}

/// Per-anchor snapshot taken before the anchor: cumulative change and per-tile visits.
pub(super) struct Mark {
    change: f64,
    visits: [u64; TILES],
}

impl Mark {
    pub fn of(state: &State) -> Self {
        let counters = &state.counters;
        Self {
            change: counters.change_sum,
            visits: std::array::from_fn(|tile| {
                counters.local_visits[tile] + counters.anchor_visits[tile]
            }),
        }
    }
}

/// Fresh per-tile state when `options.restart` is configured.
pub(super) fn fresh(options: &Options) -> Option<Vec<TileRestart>> {
    let restart = options.restart?;
    let bar = restart.threshold.unwrap_or(0.0);
    let dial = restart.plasticity.unwrap_or([1.0; TILES]);
    Some(dial.iter().map(|&p| TileRestart::new(bar, p)).collect())
}

/// `tile`'s local-rate factor at `anchor`: the plain schedule, or its per-tile law.
pub(super) fn factor(
    options: &Options,
    restarts: Option<&[TileRestart]>,
    anchor: u64,
    tile: usize,
) -> Result<f64> {
    match (options.restart, restarts) {
        (Some(restart), Some(tiles)) => {
            let entry = &tiles[tile];
            Ok(match restart.law {
                Law::Restart => {
                    let since = anchor.saturating_sub(entry.origin);
                    options.local_decay.factor(since).max(restart.floor) * entry.plasticity
                }
                Law::Lambert => entry.plasticity.max(restart.floor),
            })
        }
        (None, None) => Ok(options.local_decay.factor(anchor)),
        _ => bail!("connected restart state/options mismatch"),
    }
}

/// Applied update norm relative to the committed master norm, its denominator bounded below.
pub(super) fn relative(update: &UpdateMetrics) -> f64 {
    update.applied_norm / update.weight_norm.max(f64::MIN_POSITIVE.sqrt())
}

/// Marks `tile` as locally stepped during the anchor in flight.
pub(super) fn mark_stepped(state: &mut State, tile: usize) {
    if let Some(tiles) = &mut state.restarts {
        tiles[tile].stepped = true;
    }
}

/// Folds one live anchor visit into `tile`'s signal and its recent-position ring.
pub(super) fn observe_visit(
    state: &mut State,
    tile: usize,
    input: &[f32],
    history: &[f32],
) -> Result<()> {
    let (position, capacity) = (state.anchors, state.options.restart.map_or(0, |r| r.replay));
    let Some(tiles) = &mut state.restarts else {
        return Ok(());
    };
    let entry = &mut tiles[tile];
    entry.signal.observe_input(input, history)?;
    if capacity > 0 && entry.recent.back() != Some(&position) {
        entry.recent.push_back(position);
        if entry.recent.len() > capacity as usize {
            entry.recent.pop_front();
        }
    }
    Ok(())
}

/// Folds the pre-rate norm of `tile`'s mean local gradient into its signal.
pub(super) fn observe_gradient(state: &mut State, tile: usize, mean: &TileGradients) {
    if let Some(tiles) = &mut state.restarts {
        let squares = mean.values.iter().flatten().map(|&v| f64::from(v).powi(2));
        tiles[tile]
            .signal
            .observe_gradient(squares.sum::<f64>().sqrt());
    }
}

/// Warm restarts after Loshchilov & Hutter 2017, SGDR (https://arxiv.org/abs/1608.03983), here
/// triggered by per-tile drift instead of a fixed period; runs after `anchors` advanced.
pub(super) fn finish<S: TrainingSource>(
    state: &mut State,
    store: &mut Store,
    source: &S,
    mark: &Mark,
    loss: f32,
) -> Result<()> {
    let Some(config) = state.options.restart else {
        return Ok(());
    };
    let change = state.counters.change_sum - mark.change;
    let anchors = state.anchors;
    let counters = &state.counters;
    let tiles = state
        .restarts
        .as_mut()
        .context("missing connected restart state")?;
    for (tile, entry) in tiles.iter_mut().enumerate() {
        let visits = counters.local_visits[tile] + counters.anchor_visits[tile];
        if !std::mem::take(&mut entry.stepped) && visits == mark.visits[tile] {
            continue;
        }
        entry.cycle_loss += f64::from(loss);
        entry.cycle_anchors += 1;
        entry.signal.observe_loss(f64::from(loss));
        match (config.law, config.signal, config.threshold) {
            (Law::Lambert, _, Some(bar)) => super::connected_plasticity::visit(entry, bar)?,
            (_, Signal::Tile, _) => entry.drift = entry.signal.distance(),
            _ => entry.drift += change,
        }
    }
    let chosen = due::due(tiles, config, anchors);
    let Some(threshold) = config.threshold else {
        return Ok(());
    };
    for &tile in &chosen {
        restart(&mut tiles[tile], config, threshold, anchors);
    }
    due::mark_waiting(tiles, config, anchors);
    if config.forget {
        // Resample: stale corrections from a restarted tile are no longer retrieved or replayed.
        for &tile in &chosen {
            tiles[tile].forgotten_records += state.memory.forget(tile) as u64;
        }
        if !chosen.is_empty() && state.options.replay.is_some() {
            state.replay.clear();
        }
    }
    for &tile in &chosen {
        super::connected_restart_replay::run(state, store, source, tile)?;
    }
    Ok(())
}

/// Restarts the tile's schedule at `anchors`; with `adapt`, a cycle that lowered the mean loss
/// shortens the next one and any other lengthens it, within `clamp` of the threshold.
pub(super) fn restart(entry: &mut TileRestart, config: Restart, threshold: f64, anchors: u64) {
    let mean = entry.cycle_loss / entry.cycle_anchors.max(1) as f64;
    if let (true, Some(previous)) = (config.adapt > 0.0, entry.previous_mean) {
        let scale = if mean < previous {
            1.0 - config.adapt
        } else {
            1.0 + config.adapt
        };
        entry.bar = (entry.bar * scale).clamp(threshold / config.clamp, threshold * config.clamp);
    }
    entry.previous_mean = Some(mean);
    entry.origin = anchors;
    entry.drift = 0.0;
    entry.signal.snapshot();
    entry.cycle_loss = 0.0;
    entry.cycle_anchors = 0;
    entry.restarts += 1;
}
