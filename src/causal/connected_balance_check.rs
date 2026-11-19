//! Visit-balance state validation and per-tile reporting; see CONNECTED.md §Visit balance.

use super::{bonus, multiplier, State, MAX_BONUS, TILES};
use anyhow::{bail, ensure, Result};
use serde_json::{json, Value};

/// Estimates exist exactly when configured, track every anchor and stay finite and bounded.
pub(in super::super) fn validate(state: &State) -> Result<()> {
    let wanted = state.options.balance.is_some_and(|balance| balance.memory);
    ensure!(
        state.memory.balanced == wanted,
        "connected memory eviction differs from balance.memory"
    );
    let estimate = match (state.options.balance, &state.balance) {
        (None, None) => return Ok(()),
        (Some(_), Some(estimate)) => estimate,
        _ => bail!("connected balance state/options mismatch"),
    };
    ensure!(
        estimate.samples == state.anchors,
        "connected balance sample count"
    );
    for tile in 0..TILES {
        let (visits, unvisited) = (
            estimate.visits[tile],
            estimate.samples.checked_sub(estimate.visits[tile]),
        );
        let (seen, missed) = (
            estimate.advantage_visited[tile],
            estimate.advantage_unvisited[tile],
        );
        let share = estimate.share[tile];
        let bonus = estimate.bonus_sum[tile];
        ensure!(
            unvisited.is_some()
                && share.is_finite()
                && (0.0..=1.0).contains(&share)
                && seen.is_finite()
                && missed.is_finite()
                && (visits > 0 || seen == 0.0)
                && (unvisited != Some(0) || missed == 0.0)
                && estimate.changed[tile] <= estimate.decisions[tile]
                && bonus.is_finite()
                && (0.0..=MAX_BONUS * estimate.decisions[tile] as f64).contains(&bonus),
            "connected balance estimate of tile {tile} is inconsistent"
        );
    }
    Ok(())
}

/// Per-tile balance report for log rows and the final report; None when off.
pub(in super::super) fn report(state: &State) -> Result<Option<Value>> {
    let Some(estimate) = state.balance.as_ref() else {
        return Ok(None);
    };
    let multipliers = (0..TILES)
        .map(|tile| multiplier(&state.options, Some(estimate), tile))
        .collect::<Result<Vec<f64>>>()?;
    let mut held = [0u64; TILES];
    state
        .memory
        .records
        .iter()
        .for_each(|record| held[record.tile_id] += 1);
    // Worth is defined once both sides have a sample; positive means visits help.
    let worth = (0..TILES).map(|tile| {
        let unvisited = estimate.samples - estimate.visits[tile];
        (estimate.visits[tile] > 0 && unvisited > 0)
            .then(|| estimate.advantage_unvisited[tile] - estimate.advantage_visited[tile])
    });
    let bonus_mean = (0..TILES).map(|tile| {
        let decisions = estimate.decisions[tile];
        (decisions > 0).then(|| estimate.bonus_sum[tile] / decisions as f64)
    });
    Ok(Some(json!({
        "visit_share": estimate.share,
        "advantage_visited": estimate.advantage_visited,
        "advantage_unvisited": estimate.advantage_unvisited,
        "worth": worth.collect::<Vec<_>>(),
        "rate_multiplier": multipliers,
        "route_bonus": bonus(&state.options, Some(estimate)),
        "route_decisions": estimate.decisions,
        "route_changed": estimate.changed,
        "route_bonus_mean": bonus_mean.collect::<Vec<_>>(),
        "memory_records": held,
    })))
}
