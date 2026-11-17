//! Fixed-weight core steps with sequential correction history; see AGENTS.md §Core v2.

use super::credit::{self, Credit};
use super::cursor::Cursor;
use super::execution::{Step, Trace, TrainingSource};
use super::geometry::TILES;
use super::shared::Prefix;
use super::state::{count, Boundary, State};
use super::store_api::TileStore;
use super::telemetry::{self, Phase};
use super::{core_gradients, core_route, core_schedule, tuning};
use anyhow::{ensure, Context, Result};

/// One retained prefix, route tape and correction bundle per batch position; see STUDY.md.
pub(super) struct Position {
    pub prefix: Prefix,
    pub trace: Trace,
    pub credit: Credit,
}

/// Complete exactly one configured optimizer step, or poison the entered state on error.
pub(super) fn step<S: TrainingSource, T: TileStore>(
    state: &mut State,
    store: &mut T,
    source: &S,
    threaded: bool,
) -> Result<Vec<Step>> {
    state.validate_boundary()?;
    source.validate(&state.cursor)?;
    let knobs = tuning::current();
    ensure!(
        state.anchor_only && state.assay.is_none() && !knobs.fit_predictor,
        "batch step requires the v2 core profile"
    );
    ensure!(
        store.versions()? == state.tile_versions,
        "store/state version mismatch"
    );
    ensure!(
        (1..=32).contains(&knobs.batch) && (!knobs.float_forward || knobs.batch == 1),
        "invalid batch or float mode"
    );
    ensure!(
        state
            .global_opportunity
            .is_multiple_of(u64::from(knobs.batch)),
        "resume must start at a complete batch boundary"
    );
    state.boundary = Boundary::InFlight;
    let result = complete(state, store, source, threaded);
    if result.is_err() {
        state.boundary = Boundary::Poisoned;
    }
    result
}

/// Build positions sequentially under fixed weights, feeding each correction into history.
pub(super) fn prepare<S: TrainingSource, T: TileStore>(
    state: &mut State,
    store: &mut T,
    source: &S,
) -> Result<(Vec<Position>, Cursor)> {
    let knobs = tuning::current();
    let counts = store.optimizer_steps()?;
    let mut cursor = state.cursor.clone();
    let mut positions = Vec::with_capacity(knobs.batch as usize);
    // History couples adjacent positions even though no parameter changes inside this loop.
    for offset in 0..u64::from(knobs.batch) {
        let anchor = state
            .global_opportunity
            .checked_add(offset)
            .context("anchor overflow")?;
        let clock = anchor.checked_mul(4).context("clock overflow")?;
        let removed = state
            .history
            .prune(clock, &state.tile_versions, state.shared.version);
        count(&mut state.counters.history_pruned, removed)?;
        telemetry::history(state.history.records.len());
        let bytes = telemetry::time(Phase::Data, || source.pending_prefix(&cursor))?;
        let prefix = telemetry::time(Phase::Prefix, || state.shared.prefix(&bytes))?;
        let history = if knobs.history {
            state.history.mean(state.geometry)?
        } else {
            vec![0.0; state.geometry.width]
        };
        let trace = telemetry::time(Phase::Forward, || {
            core_route::forward(state, store, &prefix, history, anchor, &counts)
        })?;
        telemetry::route(&trace);
        let target = telemetry::time(Phase::Data, || source.anchor_target(&cursor))?;
        // Phase 1 returns input/prefix corrections; tile gradients are recomputed at commit.
        let credit = telemetry::time(Phase::Gather, || {
            credit::gather(state, store, &prefix, &trace, target.byte)
        })?;
        telemetry::time(Phase::Records, || -> Result<()> {
            if knobs.history {
                for record in credit::records(
                    state,
                    &trace,
                    &credit.corrections,
                    prefix.shared_version,
                    clock,
                )? {
                    state.history.push(record, state.geometry)?;
                }
                state.counters.history_peak =
                    state.counters.history_peak.max(state.history.records.len());
            }
            Ok(())
        })?;
        source.advance_anchor(&mut cursor)?;
        // Overlapping byte windows remain separate embedded prefixes until the batch commits.
        positions.push(Position {
            prefix,
            trace,
            credit,
        });
    }
    Ok((positions, cursor))
}

/// Recompute and commit visited tiles in ID order, then apply the shared embedding once.
fn commit<T: TileStore>(
    state: &mut State,
    store: &mut T,
    positions: &[Position],
    threaded: bool,
) -> Result<()> {
    for id in 0..TILES {
        if !positions
            .iter()
            .any(|p| p.trace.frames.iter().any(|frame| frame.tile_id == id))
        {
            continue;
        }
        let (transitions, version) = store.update(id, |tile| {
            let gradient = telemetry::time(Phase::TileBackward, || {
                core_gradients::mean(tile, positions, state.baseline, tuning::current(), threaded)
            })?;
            let transitions = telemetry::time(Phase::TileApply, || tile.apply(&gradient))?;
            Ok((transitions, tile.version))
        })?;
        state.tile_versions[id] = version;
        state.counters.transitions(transitions)?;
        count(&mut state.counters.anchor_tile_commits, 1)?;
    }
    // E is tied: every position contributes both readout and prefix-embedding credit.
    let mut embedding = positions[0].credit.embedding_gradient.clone();
    for position in &positions[1..] {
        for (sum, value) in embedding
            .iter_mut()
            .zip(&position.credit.embedding_gradient)
        {
            *sum += value;
        }
    }
    if positions.len() > 1 {
        for value in &mut embedding {
            *value /= positions.len() as f32;
        }
    }
    let transitions = telemetry::time(Phase::SharedApply, || {
        credit::apply_shared(state, &embedding)
    })?;
    count(&mut state.counters.shared_transitions, transitions)?;
    count(&mut state.counters.optimizer_steps, 1)
}

fn complete<S: TrainingSource, T: TileStore>(
    state: &mut State,
    store: &mut T,
    source: &S,
    threaded: bool,
) -> Result<Vec<Step>> {
    let (positions, cursor) = prepare(state, store, source)?;
    let original = tuning::current();
    tuning::anneal(core_schedule::at(original, state.global_opportunity));
    let committed = commit(state, store, &positions, threaded);
    tuning::anneal(original);
    committed?;
    let mut steps = Vec::with_capacity(positions.len());
    // All gradients used the step-start baseline; update its moving mean only after commit.
    for position in positions {
        let mut seen = 0;
        for frame in &position.trace.frames {
            state.counters.visit(
                state.geometry,
                frame.tile_id,
                frame.action,
                frame.depth,
                &mut seen,
            )?;
        }
        count(&mut state.counters.anchors, 1)?;
        count(
            &mut state.counters.anchor_prefix_bytes,
            position.prefix.bytes.len(),
        )?;
        state.counters.maximum_prefix_bytes = state
            .counters
            .maximum_prefix_bytes
            .max(position.prefix.bytes.len());
        state.counters.anchor_loss_sum += f64::from(position.credit.loss);
        state.baseline = 0.9 * state.baseline + 0.1 * position.credit.loss;
        steps.push(Step::Committed {
            opportunity: state.global_opportunity,
            loss: Some(position.credit.loss),
            visits: position.trace.frames.len(),
            prefix_bytes: position.prefix.bytes.len(),
        });
        count(&mut state.global_opportunity, 1)?;
    }
    state.cursor = cursor;
    state.boundary = Boundary::Ready;
    state.validate_boundary()?;
    source.validate(&state.cursor)?;
    Ok(steps)
}
