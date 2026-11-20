//! Anchor waves: tiles frozen over `anchor_batch` anchors, then one step each; see CONNECTED.md
//! §Anchor waves.

use super::connected_anchor;
use super::connected_config::{CreditMode, Options};
use super::connected_lane_apply::{route_gain, routed, Accumulator};
use super::connected_restart::{self as restart, Mark};
use super::connected_state::State;
use super::connected_step::{self, Tally};
use super::connected_store::Store;
use super::core_config::Config;
use super::execution::TrainingSource;
use super::geometry::TILES;
use super::tuning;
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};

/// Largest admitted `connected.anchor_batch`: the reference path's batch.
pub(super) const MAX_ANCHOR_BATCH: u32 = 32;

/// Waves completed and the tiles their boundaries stepped.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Waves {
    pub completed: u64,
    pub tiles_stepped: u64,
}

impl Waves {
    /// Whether no wave has run; such counters are omitted from checkpoints and reports.
    pub fn is_unused(&self) -> bool {
        *self == Self::default()
    }
}

/// Serialization skip for the default `anchor_batch`.
pub(super) fn is_one(value: &u32) -> bool {
    *value == 1
}

/// Admission: 1..=32; above 1 only under `anchor_exact` and without either replay.
pub(super) fn validate(options: &Options) -> Result<()> {
    ensure!(
        (1..=MAX_ANCHOR_BATCH).contains(&options.anchor_batch),
        "connected anchor_batch must be in 1..={MAX_ANCHOR_BATCH}"
    );
    if options.anchor_batch == 1 {
        return Ok(());
    }
    ensure!(
        options.credit_mode == CreditMode::AnchorExact,
        "connected anchor_batch above 1 requires credit_mode anchor_exact"
    );
    ensure!(
        options.replay.is_none() && options.restart.is_none_or(|restart| restart.replay == 0),
        "connected anchor_batch above 1 refuses connected.replay and restart.replay: \
         both assume one tile update per anchor"
    );
    Ok(())
}

/// Run admission: the anchor limit and every checkpoint and evaluation fall on a wave end (the
/// shared wave when `connected.shared` is set; it is a multiple of `anchor_batch`).
pub(super) fn validate_run(options: &Options, config: &Config) -> Result<()> {
    let batch = u64::from(
        options
            .shared
            .map_or(options.anchor_batch, |shared| shared.wave),
    );
    ensure!(
        [config.anchors, config.checkpoint_every, config.eval_every]
            .iter()
            .all(|value| value.is_multiple_of(batch)),
        "connected anchors, checkpoint_every and eval_every must be multiples of anchor_batch \
         and of shared.wave"
    );
    Ok(())
}

/// Wave counters: none at batch 1, waves within anchors, at most every tile per wave.
pub(super) fn validate_state(state: &State) -> Result<()> {
    let waves = state.counters.waves;
    ensure!(
        (state.options.anchor_batch > 1 || waves.is_unused())
            && waves.completed <= state.anchors
            && waves.tiles_stepped <= waves.completed.saturating_mul(TILES as u64),
        "connected wave accounting mismatch"
    );
    Ok(())
}

/// The training step: the rest of the aligned shared wave when `connected.shared` is set (its
/// write included), else one `unit`.
pub(super) fn step<S: TrainingSource>(
    state: &mut State,
    store: &mut Store,
    source: &S,
    options: Options,
    rate: f64,
    threaded: bool,
) -> Result<f32> {
    if options.shared.is_some() {
        return super::connected_sections_write::step(
            state, store, source, options, rate, threaded,
        );
    }
    unit(state, store, source, options, rate, threaded)
}

/// `connected_step::step` at batch 1, else the rest of the aligned anchor wave.
pub(super) fn unit<S: TrainingSource>(
    state: &mut State,
    store: &mut Store,
    source: &S,
    options: Options,
    rate: f64,
    threaded: bool,
) -> Result<f32> {
    let batch = u64::from(options.anchor_batch);
    if batch == 1 {
        return connected_step::step(state, store, source, options, rate, threaded);
    }
    state.validate()?;
    ensure!(
        store.precision == options.precision && state.options == options,
        "connected wave precision/options mismatch"
    );
    store.quiescent()?;
    source.validate(&state.cursor)?;
    tuning::set_connected(&options);
    state.ready = false;
    // Waves cover anchors [k b, (k + 1) b), so numerics depend only on the anchor index.
    let result = wave(
        state,
        store,
        source,
        options,
        rate,
        batch - state.anchors % batch,
    );
    if result.is_ok() {
        state.ready = true;
        if let Err(error) = state.validate() {
            state.ready = false;
            return Err(error);
        }
    }
    result
}

/// Tile credit summed over a wave: route and exact local gradients, and each tile's route gain
/// as of its last routing anchor.
#[derive(Default)]
struct Pending {
    routes: Accumulator,
    locals: Accumulator,
    route_gains: [f32; TILES],
}

/// `length` anchors under frozen tiles, then the boundary; returns the wave's mean loss.
fn wave<S: TrainingSource>(
    state: &mut State,
    store: &mut Store,
    source: &S,
    options: Options,
    rate: f64,
    length: u64,
) -> Result<f32> {
    // Tiles commit only at the boundary, so one revision snapshot serves every anchor.
    let revisions = std::array::from_fn(|id| store.slots[id].meta.revision);
    let start = Mark::of(state);
    let mut pending = Pending::default();
    let (mut visited, mut loss) = ([0u64; TILES], 0.0f64);
    for index in 0..length {
        let mark = Mark::of(state);
        let tally = anchor(
            state,
            store,
            source,
            options,
            rate,
            &revisions,
            &mut pending,
        )?;
        let seen = restart::visited(state, &mark);
        if index + 1 == length {
            // Before the last anchor advances, as at batch 1: the rate's anchor is that anchor.
            boundary(state, store, &options, rate, std::mem::take(&mut pending))?;
        }
        connected_step::advance(state, source, &tally)?;
        restart::observe_loss(state, &seen, tally.loss)?;
        for (count, flag) in visited.iter_mut().zip(seen) {
            *count += u64::from(flag);
        }
        loss += f64::from(tally.loss);
    }
    restart::settle(state, store, source, &start, visited, length)?;
    Ok((loss / length as f64) as f32)
}

/// One anchor: E, bias and all per-anchor bookkeeping now; its tile credit joins `pending`.
fn anchor<S: TrainingSource>(
    state: &mut State,
    store: &mut Store,
    source: &S,
    options: Options,
    rate: f64,
    revisions: &[u64; TILES],
    pending: &mut Pending,
) -> Result<Tally> {
    let (_, prefix, trace) = connected_step::forward(state, store, source, options, revisions)?;
    let target = source.anchor_target(&state.cursor)?;
    let mut credit = connected_anchor::gather(&state.shared, store, &prefix, &trace, target.byte)?;
    super::connected_sections_read::observe(state, store, &prefix, &trace, &mut credit)?;
    let observed_target = state.shared.target_feature(target.byte)?;
    let tally = Tally::of(&trace, &credit)?;
    // The advantage uses this anchor's baseline; only the route step waits.
    let advantage = credit.loss - state.baseline;
    for tile in (0..TILES).filter(|&tile| routed(&trace, tile)) {
        let gradient =
            connected_anchor::route_gradient(store, state.geometry, &trace, tile, advantage)?;
        pending.routes.add(tile, gradient)?;
        // The gain batch 1 would use for this anchor's route step (before its observations).
        pending.route_gains[tile] = route_gain(state, store, tile, &options);
    }
    connected_step::step_shared(state, &credit, rate, &options)?;
    connected_step::observe(state, &trace, &credit, &observed_target, revisions, options)?;
    for (frame, gradient) in trace.frames.iter().zip(credit.gradients) {
        state.counters.local_visits[frame.tile] += 1;
        pending.locals.add_exact(frame.tile, gradient)?;
    }
    Ok(tally)
}

/// Route steps, then one mean local step per touched tile, each checked out alone.
fn boundary(
    state: &mut State,
    store: &mut Store,
    options: &Options,
    rate: f64,
    pending: Pending,
) -> Result<()> {
    let gains = pending.route_gains;
    pending
        .routes
        .apply_routes(state, store, options, rate, &gains)?;
    // Non-persisted guard: one mean local step per touched tile per wave.
    let (before, tiles) = (state.counters.local_updates, pending.locals.tiles());
    pending
        .locals
        .apply_with_trust(state, store, options, false)?;
    let stepped = state.counters.local_updates - before;
    ensure!(
        stepped <= tiles,
        "connected wave stepped a tile more than once"
    );
    state.counters.waves.completed += 1;
    state.counters.waves.tiles_stepped += stepped;
    store.quiescent()
}

#[cfg(test)]
#[path = "connected_wave_contracts.rs"]
mod contracts;
