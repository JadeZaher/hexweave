use super::{
    credit, forward_path, prune, record_trace, Boundary, State, TileStore, TrainingSource,
};
use crate::causal::state::count;
use anyhow::{ensure, Result};

/// Execute one real anchor at its virtual clock; see AGENTS.md.
pub(in crate::causal) fn step<S: TileStore, D: TrainingSource>(
    state: &mut State,
    store: &mut S,
    source: &D,
) -> Result<f32> {
    ensure!(
        state.boundary == Boundary::Ready
            && state.counters.anchors < 8
            && state.global_opportunity == 4 * state.counters.anchors
            && state.counters.local_opportunities == 0
            && state.counters.predictor_replay_updates == 0,
        "anchor-only fixture ledger mismatch"
    );
    source.validate(&state.cursor)?;
    ensure!(store.versions()? == state.tile_versions, "fixture versions");
    let prefix = state
        .shared
        .prefix(&source.pending_prefix(&state.cursor)?)?;
    state.boundary = Boundary::InFlight;
    let result = complete(state, store, source, &prefix);
    if result.is_err() {
        state.boundary = Boundary::Poisoned;
    }
    result
}

fn complete<S: TileStore, D: TrainingSource>(
    state: &mut State,
    store: &mut S,
    source: &D,
    prefix: &crate::causal::shared::Prefix,
) -> Result<f32> {
    let mut random = Some(
        state
            .seed
            .wrapping_add(state.global_opportunity.wrapping_mul(7919)),
    );
    prune(state)?;
    let trace = forward_path(
        store,
        state.geometry,
        prefix,
        state.history.mean(state.geometry)?,
        &mut random,
        state.tile_versions,
    )?;
    record_trace(state, &trace)?;
    let target = source.anchor_target(&state.cursor)?;
    let loss = credit::anchor(state, store, prefix, &trace, target.byte)?;
    count(&mut state.counters.anchors, 1)?;
    count(&mut state.counters.anchor_prefix_bytes, prefix.bytes.len())?;
    state.counters.anchor_loss_sum += f64::from(loss);
    ensure!(state.counters.anchor_loss_sum.is_finite(), "fixture loss");
    state.baseline = 0.9 * state.baseline + 0.1 * loss;
    source.advance_anchor(&mut state.cursor)?;
    state.counters.maximum_prefix_bytes =
        state.counters.maximum_prefix_bytes.max(prefix.bytes.len());
    state.tile_versions = store.versions()?;
    state.global_opportunity = state.counters.anchors * 4;
    state.boundary = Boundary::Ready;
    source.validate(&state.cursor)?;
    state.history.validate(
        state.geometry,
        state.global_opportunity,
        &state.tile_versions,
        state.shared.version,
    )?;
    Ok(loss)
}
