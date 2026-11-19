//! Restart-state validation and per-tile reporting; see CONNECTED.md §Warm restarts.

use super::super::connected_restart_config::{Law, MAX_PLASTICITY};
use super::{Restart, State, TileRestart, TILES};
use anyhow::{bail, ensure, Result};
use serde_json::{json, Value};

/// Restart state exists exactly when configured and is consistent at a quiescent boundary.
pub(in super::super) fn validate(state: &State) -> Result<()> {
    let (config, tiles) = match (state.options.restart, &state.restarts) {
        (None, None) => return Ok(()),
        (Some(config), Some(tiles)) => (config, tiles),
        _ => bail!("connected restart state/options mismatch"),
    };
    ensure!(tiles.len() == TILES, "connected restart tile count");
    for (tile, entry) in tiles.iter().enumerate() {
        check(entry, config, state.anchors)?;
        // Under discrete restarts nothing moves the dial off its configured value yet.
        ensure!(
            config.law == Law::Lambert || entry.plasticity == config.dial(tile),
            "connected restart plasticity differs from its configured dial"
        );
        entry.signal.validate(2 * state.geometry.width)?;
    }
    Ok(())
}

/// Local steps taken by restart replays, which the local-update bound admits on top.
pub(in super::super) fn replay_steps(state: &State) -> u64 {
    let tiles = state.restarts.as_deref().unwrap_or_default();
    tiles.iter().map(|entry| entry.replay_steps).sum()
}

/// One tile's restart state against its configuration at a boundary of `anchors`.
fn check(entry: &TileRestart, config: Restart, anchors: u64) -> Result<()> {
    ensure!(
        entry.origin <= anchors
            && entry.restarts <= anchors
            && entry.cycle_anchors <= anchors - entry.origin
            && !entry.stepped,
        "connected restart clock out of range"
    );
    ensure!(
        entry.drift.is_finite()
            && entry.drift >= 0.0
            && entry.cycle_loss.is_finite()
            && entry.cycle_loss >= 0.0
            && entry.previous_mean.is_none_or(f64::is_finite),
        "nonfinite connected restart accounting"
    );
    let history = if entry.restarts == 0 {
        entry.origin == 0
            && entry.previous_mean.is_none()
            && entry.forgotten_records == 0
            && entry.replays == 0
    } else {
        entry.origin > 0 && entry.previous_mean.is_some() && config.law == Law::Restart
    };
    ensure!(history, "connected restart history mismatch");
    let bar = match config.threshold {
        None => entry.restarts == 0 && entry.bar == 0.0,
        Some(threshold) if config.adapt == 0.0 => entry.bar == threshold,
        Some(threshold) => {
            (threshold / config.clamp..=threshold * config.clamp).contains(&entry.bar)
        }
    };
    // Under `stagger` a due tile may wait; under the Lambert law nothing restarts.
    let waits = config.stagger > 0 || config.law == Law::Lambert;
    ensure!(
        bar && (config.threshold.is_none() || waits || entry.drift < entry.bar),
        "connected restart threshold out of range"
    );
    ensure!(
        super::super::connected_restart_due::consistent(entry, config, anchors),
        "connected restart due-since mismatch"
    );
    check_dial_and_replay(entry, config, anchors)
}

/// The plasticity dial and restart-replay accounting.
fn check_dial_and_replay(entry: &TileRestart, config: Restart, anchors: u64) -> Result<()> {
    ensure!(
        entry.plasticity.is_finite()
            && (0.0..=MAX_PLASTICITY).contains(&entry.plasticity)
            && entry.plasticity_sum.is_finite()
            && entry.plasticity_sum >= 0.0
            && (config.law == Law::Lambert || entry.plasticity_sum == 0.0),
        "connected restart plasticity out of range"
    );
    let capacity = u64::from(config.replay);
    ensure!(
        entry.replay_steps <= entry.replays
            && entry.replays <= entry.restarts.saturating_mul(capacity)
            && entry.replay_loss_sum.is_finite()
            && entry.replay_loss_sum >= 0.0
            && entry.recent.len() as u64 <= capacity
            && entry.recent.iter().all(|&position| position < anchors),
        "connected restart replay accounting"
    );
    Ok(())
}

/// Per-tile restart, signal, plasticity and restart-replay report; None when off.
pub(in super::super) fn report(state: &State) -> Result<Option<Value>> {
    let Some(tiles) = state.restarts.as_deref() else {
        return Ok(None);
    };
    let factors = (0..TILES)
        .map(|tile| super::super::connected_balance::rate_factor(&state.options, state, tile))
        .collect::<Result<Vec<f64>>>()?;
    let field = |read: fn(&TileRestart) -> Value| tiles.iter().map(read).collect::<Vec<_>>();
    let mean = |sum: f64, count: u64| (count > 0).then(|| sum / count as f64);
    Ok(Some(json!({
        "restarts": field(|entry| json!(entry.restarts)),
        "bar": field(|entry| json!(entry.bar)),
        "drift": field(|entry| json!(entry.drift)),
        "forgotten_records": field(|entry| json!(entry.forgotten_records)),
        "rate_factor": factors,
        "distance": field(|entry| json!(entry.signal.distance())),
        "gradient_norm_ema": field(|entry| json!(entry.signal.gradient_norm_ema)),
        "loss_ema": field(|entry| json!(entry.signal.loss_ema)),
        "plasticity": field(|entry| json!(entry.plasticity)),
        "plasticity_mean": tiles.iter()
            .map(|entry| mean(entry.plasticity_sum, entry.signal.anchors)).collect::<Vec<_>>(),
        "restart_replays": field(|entry| json!(entry.replays)),
        "restart_replay_steps": field(|entry| json!(entry.replay_steps)),
        "restart_replay_loss_mean": tiles.iter()
            .map(|entry| mean(entry.replay_loss_sum, entry.replays)).collect::<Vec<_>>(),
        "due_since": field(|entry| json!(entry.due_since)),
    })))
}
