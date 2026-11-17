//! Loss-free local lanes, published between worker groups; see CONNECTED.md §Learning loop
//! (label feature: §v5).

use super::connected_anchor::Trace;
use super::connected_config::Options;
use super::connected_lane_apply::Accumulator;
use super::connected_state::State;
use super::connected_store::Store;
use super::execution::START_TILE;
use super::geometry::TILES;
use super::shared::Prefix;
use super::store_api::TileStore;
use super::tile_gradients::TileGradients;
use super::tuning;
use anyhow::{ensure, Result};

#[path = "connected_lane_job.rs"]
mod job;
use job::{compute, Job, Outcome};

struct Lane {
    tile: usize,
    input: Vec<f32>,
    depth: usize,
    done: bool,
}

/// Distinct tile leases run together; lane order determines complete-record publication.
/// `target` is the byte already read out at this prefix; training calls this after E updates.
/// Tile steps use `local_rate` under `local_decay`, never the anchor rate (CONNECTED.md §v5).
pub(super) fn run(
    state: &mut State,
    store: &mut Store,
    prefix: &Prefix,
    target: u8,
    options: Options,
    threaded: bool,
) -> Result<()> {
    run_with_accumulator(
        state,
        store,
        prefix,
        target,
        options,
        threaded,
        Accumulator::default(),
        true,
    )
}

/// Runs local lanes starting from an existing `Accumulator` (e.g. seeded with exact anchor credit).
#[allow(clippy::too_many_arguments)]
pub(super) fn run_with_accumulator(
    state: &mut State,
    store: &mut Store,
    prefix: &Prefix,
    target: u8,
    options: Options,
    threaded: bool,
    gradients: Accumulator,
    use_trust: bool,
) -> Result<()> {
    drive(
        state, store, prefix, target, options, threaded, true, gradients, use_trust,
    )
}

/// Evaluates and differentiates predicted credit only along the realized anchor route (`trace`).
pub(super) fn run_anchor_predicted(
    state: &mut State,
    store: &mut Store,
    prefix: &Prefix,
    trace: &Trace,
    label: &[f32],
    options: Options,
) -> Result<()> {
    let mut gradients = Accumulator::default();
    for frame in &trace.frames {
        let revision = store.slots[frame.tile].meta.revision;
        let gain = state.gains[frame.tile].value_at(state.epoch, revision, &options)
            * state.predictor.trust(frame.tile) as f32;
        state.counters.local_visits[frame.tile] += 1;
        if gain <= 0.0 {
            continue;
        }
        let features = state.predictor.features(
            &frame.input,
            &frame.output,
            &frame.history,
            label,
            frame.tile,
        )?;
        let correction = state.predictor.predict(&features, frame.tile)?;
        let mut gradient = TileGradients::zeros(state.geometry, frame.tile)?;
        store.read(frame.tile, |tile| {
            tile.backward(
                &frame.input,
                &prefix.memory,
                &frame.history,
                &correction,
                None,
                Some(&mut gradient),
                None,
            )
        })?;
        gradients.add(frame.tile, gradient)?;
    }
    gradients.apply_with_trust(state, store, &options, true)?;
    store.quiescent()
}

/// Evaluation replays local evidence under frozen weights without fitting or applying updates.
pub(super) fn replay(
    state: &mut State,
    store: &mut Store,
    prefix: &Prefix,
    target: u8,
    options: Options,
) -> Result<()> {
    drive(
        state,
        store,
        prefix,
        target,
        options,
        false,
        false,
        Accumulator::default(),
        true,
    )
}

#[allow(clippy::too_many_arguments)]
fn drive(
    state: &mut State,
    store: &mut Store,
    prefix: &Prefix,
    target: u8,
    options: Options,
    threaded: bool,
    train: bool,
    mut gradients: Accumulator,
    use_trust: bool,
) -> Result<()> {
    // One label per phase, from the active E view every lane predicts with (CONNECTED.md §v5).
    let label = state.shared.target_feature(target)?;
    let mut lanes: Vec<_> = (0..options.lanes)
        .map(|_| Lane {
            tile: START_TILE,
            input: prefix.initial(state.geometry).to_vec(),
            depth: 0,
            done: false,
        })
        .collect();
    // Jobs only read tiles; each tile then takes one mean step (CONNECTED.md §Scale control).
    while lanes.iter().any(|lane| !lane.done) {
        let selected = select(
            &lanes,
            options.workers,
            &mut state.counters.collision_deferrals,
        );
        ensure!(!selected.is_empty(), "connected lanes cannot progress");
        let jobs = match prepare(state, store, &lanes, &selected, &options) {
            Ok(jobs) => jobs,
            Err(error) => {
                for &index in &selected {
                    store.abort(lanes[index].tile);
                }
                return Err(error);
            }
        };
        let tiles: Vec<_> = jobs.iter().map(|job| job.lease.tile().id).collect();
        let knobs = tuning::current();
        let result = compute(
            jobs,
            prefix,
            &label,
            &state.predictor,
            options,
            knobs,
            threaded,
            train,
        );
        let outcomes = match result {
            Ok(outcomes) => outcomes,
            Err(error) => {
                for tile in tiles {
                    store.abort(tile);
                }
                return Err(error);
            }
        };
        state.counters.local_groups += 1;
        for outcome in outcomes {
            publish(state, store, &mut lanes, &mut gradients, outcome)?;
        }
    }
    gradients.apply_with_trust(state, store, &options, use_trust)?;
    store.quiescent()
}

/// Returns the read-only lease, publishes the record, banks any gradient and advances the lane.
fn publish(
    state: &mut State,
    store: &mut Store,
    lanes: &mut [Lane],
    gradients: &mut Accumulator,
    outcome: Outcome,
) -> Result<()> {
    let tile = outcome.lease.tile().id;
    store.return_read(outcome.lease)?;
    state.memory.publish(outcome.record)?;
    state.epoch += 1;
    state.counters.predicted_records += 1;
    state.counters.local_visits[tile] += 1;
    if let Some(gradient) = outcome.gradient {
        gradients.add(tile, gradient)?;
    }
    let lane = &mut lanes[outcome.lane];
    lane.input = outcome.output;
    lane.depth += 1;
    match outcome.next {
        Some(next) => lane.tile = next,
        None => lane.done = true,
    }
    Ok(())
}

fn select(lanes: &[Lane], workers: usize, deferred: &mut u64) -> Vec<usize> {
    let mut used = [false; TILES];
    let mut selected = Vec::new();
    for (index, lane) in lanes.iter().enumerate() {
        if lane.done {
            continue;
        }
        if used[lane.tile] {
            *deferred += 1;
            continue;
        }
        selected.push(index);
        used[lane.tile] = true;
        if selected.len() == workers {
            break;
        }
    }
    selected
}

fn prepare(
    state: &mut State,
    store: &mut Store,
    lanes: &[Lane],
    selected: &[usize],
    options: &Options,
) -> Result<Vec<Job>> {
    let mut jobs = Vec::with_capacity(selected.len());
    let revisions = std::array::from_fn(|id| store.slots[id].meta.revision);
    let shared_revision = state.shared_latent.as_ref().map_or(0, |latent| latent.step);
    for &index in selected {
        let lane = &lanes[index];
        let retrieval = state.memory.query_report(
            &lane.input,
            lane.tile,
            state.epoch,
            &revisions,
            shared_revision,
            Some(index),
        )?;
        let history = options.history_input(retrieval.values);
        state.counters.memory_reads += 1;
        state.counters.retrieved_records += retrieval.records as u64;
        if retrieval.cross_lane_records > 0 {
            state.counters.cross_lane_reads += 1;
        }
        let revision = store.slots[lane.tile].meta.revision;
        let lease = store.checkout(lane.tile)?;
        jobs.push(Job {
            lane: index,
            lease,
            input: lane.input.clone(),
            history,
            depth: lane.depth,
            epoch: state.epoch,
            revision,
            shared_revision,
            // Zero trust skips local differentiation; see CONNECTED.md §Calibrated predictor.
            gain: state.gains[lane.tile].value_at(state.epoch, revision, options)
                * state.predictor.trust(lane.tile) as f32,
            random: tuning::hash(
                state.seed,
                &[state.anchors, index as u64, lane.depth as u64, 0x4c4f4341],
            ),
        });
    }
    Ok(jobs)
}
