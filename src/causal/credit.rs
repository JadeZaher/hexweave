use super::execution::{checked_tile, Trace};
use super::geometry::TILES;
use super::history::Record;
use super::norm;
use super::shared::Prefix;
use super::state::{count, State};
use super::store_api::TileStore;
use super::telemetry::{self, Phase};
use super::tile_gradients::TileGradients;
use crate::topology::Mesh;
use anyhow::{ensure, Result};

pub(super) struct Credit {
    pub loss: f32,
    pub corrections: Vec<Vec<f32>>,
    pub embedding_gradient: Vec<f32>,
}

/// Gather all conditional corrections and tied E credit before any commit.
pub(super) fn gather<S: TileStore>(
    state: &State,
    store: &mut S,
    prefix: &Prefix,
    trace: &Trace,
    target: u8,
) -> Result<Credit> {
    ensure!(
        store.versions()? == trace.tile_versions && trace.tile_versions == state.tile_versions,
        "anchor versions changed before gather"
    );
    ensure!(
        !trace.frames.is_empty() && trace.frames.len() <= 4,
        "invalid anchor tape length"
    );
    let task = state.shared.task(&trace.output, target)?;
    let mut upstream = task.input_correction;
    let mut corrections = Vec::with_capacity(trace.frames.len());
    let mut prefix_adjoint = vec![0.0; prefix.memory.len()];
    for frame in trace.frames.iter().rev() {
        corrections.push(upstream.clone());
        upstream = store.read(frame.tile_id, |tile| {
            checked_tile(tile, state.geometry, frame.tile_id, frame.version)?;
            tile.backward(
                &frame.input,
                &prefix.memory,
                &trace.history,
                &upstream,
                Some(&mut prefix_adjoint),
                None,
                None,
            )
        })?;
    }
    corrections.reverse();
    let final_row = prefix_adjoint.len() - state.geometry.width;
    norm::add(&mut prefix_adjoint[final_row..], &upstream)?;
    let mut embedding_gradient = task.embedding_gradient;
    state
        .shared
        .add_prefix_gradient(prefix, &prefix_adjoint, &mut embedding_gradient)?;
    Ok(Credit {
        loss: task.loss,
        corrections,
        embedding_gradient,
    })
}

pub(super) fn anchor<S: TileStore>(
    state: &mut State,
    store: &mut S,
    prefix: &Prefix,
    trace: &Trace,
    target: u8,
) -> Result<f32> {
    let credit = telemetry::time(Phase::Gather, || {
        gather(state, store, prefix, trace, target)
    })?;
    let advantage = credit.loss - state.baseline;
    ensure!(advantage.is_finite(), "invalid route advantage");
    commit_tiles(state, store, prefix, trace, &credit.corrections, advantage)?;
    let transitions = telemetry::time(Phase::SharedApply, || {
        apply_shared(state, &credit.embedding_gradient)
    })?;
    count(&mut state.counters.shared_transitions, transitions)?;
    drop(credit.embedding_gradient);
    telemetry::time(Phase::Records, || {
        train_records(state, trace, &credit.corrections, prefix.shared_version)
    })?;
    Ok(credit.loss)
}

/// Apply tied-E credit under the assay ledger, tuned numerics or the pinned rule.
fn apply_shared(state: &mut State, gradient: &[f32]) -> Result<usize> {
    if let Some(assay) = &mut state.assay {
        let metrics = super::assay_update::apply(
            &mut state.shared,
            gradient,
            assay.rule,
            state.seed,
            state.counters.anchors,
        )?;
        let transitions = usize::try_from(metrics.transitions)?;
        assay.embedding.add(&metrics)?;
        Ok(transitions)
    } else if let Some(moved) = super::assay_update::tuned(&mut state.shared, gradient)? {
        Ok(usize::try_from(moved)?)
    } else {
        state.shared.apply(gradient)
    }
}

/// Aggregate repeated visits in forward order, then commit each unique tile once.
fn commit_tiles<S: TileStore>(
    state: &mut State,
    store: &mut S,
    prefix: &Prefix,
    trace: &Trace,
    corrections: &[Vec<f32>],
    advantage: f32,
) -> Result<()> {
    for id in 0..TILES {
        if !trace.frames.iter().any(|frame| frame.tile_id == id) {
            continue;
        }
        let (transitions, version) = store.update(id, |tile| {
            checked_tile(tile, state.geometry, id, trace.tile_versions[id])?;
            let mut gradient = TileGradients::zeros(state.geometry, id)?;
            telemetry::time(Phase::TileBackward, || -> Result<()> {
                for (frame, correction) in trace.frames.iter().zip(corrections) {
                    if frame.tile_id != id {
                        continue;
                    }
                    ensure!(
                        frame.version == tile.version,
                        "repeated visit changed frozen version"
                    );
                    tile.backward(
                        &frame.input,
                        &prefix.memory,
                        &trace.history,
                        correction,
                        None,
                        Some(&mut gradient),
                        frame.action.map(|action| (action, advantage)),
                    )?;
                }
                Ok(())
            })?;
            let transitions = telemetry::time(Phase::TileApply, || tile.apply(&gradient))?;
            Ok((transitions, tile.version))
        })?;
        state.tile_versions[id] = version;
        state.counters.transitions(transitions)?;
        count(&mut state.counters.anchor_tile_commits, 1)?;
    }
    Ok(())
}

/// One history record per visited frame: its predictor features and unit-clipped correction,
/// stamped with the trace's versions and the clock `opportunity`. Shared by training and replay.
pub(super) fn records(
    state: &State,
    trace: &Trace,
    corrections: &[Vec<f32>],
    shared_version: u64,
    opportunity: u64,
) -> Result<Vec<Record>> {
    let mesh = Mesh::new(1)?;
    let mut records = Vec::with_capacity(trace.frames.len());
    for (index, frame) in trace.frames.iter().enumerate() {
        let output = trace
            .frames
            .get(index + 1)
            .map_or(trace.output.as_slice(), |next| next.input.as_slice());
        let point = mesh.coordinates()[frame.tile_id];
        let features =
            state
                .predictor
                .features(&frame.input, output, &trace.history, point.q, point.r)?;
        records.push(Record {
            features,
            correction: norm::unit_clip(&corrections[index])?,
            tile_versions: trace.tile_versions,
            shared_version,
            opportunity,
            tile_id: frame.tile_id,
        });
    }
    Ok(records)
}

fn train_records(
    state: &mut State,
    trace: &Trace,
    corrections: &[Vec<f32>],
    shared_version: u64,
) -> Result<()> {
    let records = records(
        state,
        trace,
        corrections,
        shared_version,
        state.numerical_clock()?,
    )?;
    let pairs: Vec<_> = records
        .iter()
        .map(|r| (r.features.as_slice(), r.correction.as_slice()))
        .collect();
    let update = state.predictor.train_joint(&pairs)?;
    state.counters.predictor(update, false)?;
    for record in records {
        state.history.push(record, state.geometry)?;
        state.counters.history_peak = state.counters.history_peak.max(state.history.records.len());
    }
    Ok(())
}
