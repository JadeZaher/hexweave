//! Legacy single-anchor/local execution retained for numerical reference contracts.

use super::credit;
use super::execution::{
    checked_tile, destination, forward_path, Step, Trace, TrainingSource, START_TILE,
};
use super::geometry::VISITS;
use super::shared::Prefix;
use super::state::{count, Boundary, State};
use super::store_api::TileStore;
use super::telemetry::{self, Phase};
use super::tile_gradients::TileGradients;
use anyhow::{ensure, Context, Result};

/// STOP is observed before entry; an entered opportunity completes or poisons.
pub(super) fn step<S: TileStore, D: TrainingSource>(
    state: &mut State,
    store: &mut S,
    source: &D,
    stop_requested: bool,
) -> Result<Step> {
    state.validate_boundary()?;
    source.validate(&state.cursor)?;
    ensure!(
        store.versions()? == state.tile_versions,
        "store/state version mismatch"
    );
    if stop_requested {
        return Ok(Step::Stopped);
    }
    let next = state
        .global_opportunity
        .checked_add(1)
        .context("opportunity overflow")?;
    let bytes = telemetry::time(Phase::Data, || source.pending_prefix(&state.cursor))?;
    let prefix = telemetry::time(Phase::Prefix, || state.shared.prefix(&bytes))?;
    drop(bytes);
    state.boundary = Boundary::InFlight;
    let result = complete(state, store, source, &prefix, next);
    if result.is_err() {
        state.boundary = Boundary::Poisoned;
    }
    result
}

fn complete<S: TileStore, D: TrainingSource>(
    state: &mut State,
    store: &mut S,
    source: &D,
    prefix: &Prefix,
    next: u64,
) -> Result<Step> {
    let opportunity = state.global_opportunity;
    let numerical_clock = state.numerical_clock()?;
    let knobs = super::tuning::current();
    let mut random = (!knobs.greedy_routes).then(|| {
        state
            .seed
            .wrapping_add(numerical_clock.wrapping_mul(7919))
            .wrapping_add(knobs.route_salt)
    });
    let previous_visits = state.counters.visits;
    prune(state)?;
    let loss = if state.anchors_only() || opportunity.is_multiple_of(4) {
        telemetry::history(state.history.records.len());
        let history = if knobs.history {
            state.history.mean(state.geometry)?
        } else {
            vec![0.0; state.geometry.width]
        };
        let versions = state.tile_versions;
        let trace = telemetry::time(Phase::Forward, || {
            forward_path(
                store,
                state.geometry,
                prefix,
                history,
                &mut random,
                versions,
            )
        })?;
        telemetry::route(&trace);
        record_trace(state, &trace)?;
        let target = telemetry::time(Phase::Data, || source.anchor_target(&state.cursor))?;
        let loss = credit::anchor(state, store, prefix, &trace, target.byte)?;
        count(&mut state.counters.anchors, 1)?;
        count(&mut state.counters.anchor_prefix_bytes, prefix.bytes.len())?;
        state.counters.anchor_loss_sum += f64::from(loss);
        ensure!(
            state.counters.anchor_loss_sum.is_finite(),
            "anchor loss metric overflow"
        );
        state.baseline = 0.9 * state.baseline + 0.1 * loss;
        source.advance_anchor(&mut state.cursor)?;
        Some(loss)
    } else {
        local(state, store, prefix, &mut random)?;
        count(&mut state.counters.local_opportunities, 1)?;
        count(&mut state.counters.local_prefix_bytes, prefix.bytes.len())?;
        None
    };
    state.counters.maximum_prefix_bytes =
        state.counters.maximum_prefix_bytes.max(prefix.bytes.len());
    state.tile_versions = store.versions()?;
    state.global_opportunity = next;
    state.boundary = Boundary::Ready;
    state.validate_boundary()?;
    source.validate(&state.cursor)?;
    Ok(Step::Committed {
        opportunity,
        loss,
        visits: usize::try_from(state.counters.visits - previous_visits)?,
        prefix_bytes: prefix.bytes.len(),
    })
}

pub(super) fn prune(state: &mut State) -> Result<()> {
    let removed = state.history.prune(
        state.numerical_clock()?,
        &state.tile_versions,
        state.shared.version,
    );
    count(&mut state.counters.history_pruned, removed)
}

fn local<S: TileStore>(
    state: &mut State,
    store: &mut S,
    prefix: &Prefix,
    random: &mut Option<u64>,
) -> Result<()> {
    if !state.history.records.is_empty() {
        let pairs: Vec<_> = state
            .history
            .replay()
            .iter()
            .map(|r| (r.features.as_slice(), r.correction.as_slice()))
            .collect();
        let update = state.predictor.train_joint(&pairs)?;
        state.counters.predictor(update, true)?;
    }
    let mesh = super::geometry::mesh();
    let (mut id, mut input, mut seen) = (START_TILE, prefix.initial(state.geometry).to_vec(), 0u32);
    for depth in 0..VISITS {
        prune(state)?;
        let history = state.history.mean(state.geometry)?;
        let point = mesh.coordinates()[id];
        let (output, action, transitions, version) = store.update(id, |tile| {
            checked_tile(tile, state.geometry, id, state.tile_versions[id])?;
            let forward = tile.forward(&input, &prefix.memory, &history)?;
            let action = if depth + 1 == VISITS {
                None
            } else {
                Some(tile.route(&forward, &history, random)?)
            };
            let features =
                state
                    .predictor
                    .features(&input, &forward.output, &history, point.q, point.r)?;
            let correction = state.predictor.correction(&features)?;
            let mut gradient = TileGradients::zeros(state.geometry, id)?;
            tile.backward(
                &input,
                &prefix.memory,
                &history,
                &correction,
                None,
                Some(&mut gradient),
                None,
            )?;
            let transitions = tile.apply(&gradient)?;
            Ok((forward.output, action, transitions, tile.version))
        })?;
        state.tile_versions[id] = version;
        state.counters.transitions(transitions)?;
        count(&mut state.counters.local_tile_updates, 1)?;
        state
            .counters
            .visit(state.geometry, id, action, depth, &mut seen)?;
        input = output;
        match destination(mesh, id, action)? {
            Some(next) => id = next,
            None => break,
        }
    }
    Ok(())
}

fn record_trace(state: &mut State, trace: &Trace) -> Result<()> {
    let mut seen = 0u32;
    for frame in &trace.frames {
        state.counters.visit(
            state.geometry,
            frame.tile_id,
            frame.action,
            frame.depth,
            &mut seen,
        )?;
    }
    Ok(())
}
