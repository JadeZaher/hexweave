//! Visit balance: per-tile visit-share and advantage estimates, the tile-rate multiplier, the
//! training route bonus and per-tile memory eviction; see CONNECTED.md §Visit balance.

use super::connected_anchor::Trace;
use super::connected_config::{CreditMode, Options};
use super::connected_signal::{Memory, Record};
use super::connected_state::State;
use super::geometry::{Geometry, START_TILE, TILES};
use anyhow::{bail, ensure, Result};
use serde::{Deserialize, Serialize};

/// Bounds of the rate multiplier [1/4, 4] and of the route bonus [-2, 2].
pub(super) const MAX_MULTIPLIER: f64 = 4.0;
const MAX_BONUS: f64 = 2.0;

/// Share floor of the ratio `mean / max(share, TINY)`.
const TINY: f64 = 1e-6;

/// `connected.balance`: estimate horizon and the three "learn faster where visited less" uses.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(super) struct Balance {
    #[serde(
        default = "default_horizon",
        skip_serializing_if = "is_default_horizon"
    )]
    pub horizon: u64,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub rate: f64,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub route: f64,
    #[serde(default, skip_serializing_if = "is_false")]
    pub memory: bool,
}

/// Default estimate horizon in anchors.
fn default_horizon() -> u64 {
    1024
}

/// Serialization skip for the default horizon.
fn is_default_horizon(value: &u64) -> bool {
    *value == default_horizon()
}

/// Serialization skip for a disabled use.
fn is_zero(value: &f64) -> bool {
    *value == 0.0
}

/// Serialization skip for an unset flag.
pub(super) fn is_false(value: &bool) -> bool {
    !*value
}

impl Balance {
    /// Admission: horizon 64..=65536, rate [0, 1], route [0, 2], at least one use enabled, and
    /// rate/route only under `anchor_exact`.
    pub fn validate(&self, credit: CreditMode) -> Result<()> {
        ensure!(
            (64..=65536).contains(&self.horizon),
            "connected balance horizon must be 64..=65536"
        );
        ensure!(
            self.rate.is_finite() && (0.0..=1.0).contains(&self.rate),
            "connected balance rate must be in [0, 1]"
        );
        ensure!(
            self.route.is_finite() && (0.0..=2.0).contains(&self.route),
            "connected balance route must be in [0, 2]"
        );
        ensure!(
            self.rate > 0.0 || self.route > 0.0 || self.memory,
            "connected balance enables nothing: set rate, route or memory"
        );
        ensure!(
            (self.rate == 0.0 && self.route == 0.0) || credit == CreditMode::AnchorExact,
            "connected balance rate and route require credit_mode anchor_exact: the estimate \
             counts live-route visits, which equal local steps only there"
        );
        Ok(())
    }
}

/// The largest multiplier a configured `rate` can apply, for the decay-safety bound.
pub(super) fn max_multiplier(options: &Options) -> f64 {
    if options.balance.is_some_and(|balance| balance.rate > 0.0) {
        MAX_MULTIPLIER
    } else {
        1.0
    }
}

/// Per-tile training-side estimates (never in tile bundles, never read by evaluation).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Estimate {
    /// Anchors folded; equals the completed-anchor count.
    pub samples: u64,
    /// Anchors whose live route visited each tile.
    pub visits: [u64; TILES],
    /// EMA of "the live route visited the tile on this anchor".
    pub share: [f64; TILES],
    /// EMA of the anchor advantage (loss - baseline) over visited and over unvisited anchors.
    pub advantage_visited: [f64; TILES],
    pub advantage_unvisited: [f64; TILES],
    /// Bonus-steered live-anchor route decisions per deciding tile, those whose argmax the bonus
    /// changed, and their summed mean absolute neighbour bonus.
    pub decisions: [u64; TILES],
    pub changed: [u64; TILES],
    pub bonus_sum: [f64; TILES],
}

/// A bonus-steered route decision: whether the bonus moved the argmax, and its mean |bonus|.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Steer {
    pub changed: bool,
    pub bonus: f32,
}

/// One live anchor's route as the estimate folds it.
pub(super) struct Visit {
    visited: [bool; TILES],
    steers: Vec<(usize, Steer)>,
}

impl Visit {
    /// The tiles `trace` visited and its bonus-steered decisions.
    pub fn of(trace: &Trace) -> Self {
        let mut visited = [false; TILES];
        trace
            .frames
            .iter()
            .for_each(|frame| visited[frame.tile] = true);
        let steers = trace.frames.iter();
        let steers = steers.filter_map(|frame| frame.steer.map(|steer| (frame.tile, steer)));
        Self {
            visited,
            steers: steers.collect(),
        }
    }
}

/// Fresh estimates when `options.balance` is configured.
pub(super) fn fresh(options: &Options) -> Option<Estimate> {
    options.balance.map(|_| Estimate::default())
}

/// Shared memory with the eviction policy `options` selects (training and evaluation alike).
pub(super) fn memory(geometry: Geometry, options: &Options) -> Result<Memory> {
    let mut memory = Memory::new(geometry, options.memory_capacity)?;
    memory.balanced = options.balance.is_some_and(|balance| balance.memory);
    memory.lag = options.memory_lag;
    Ok(memory)
}

/// Index to evict from a full memory: the oldest record, or under `balanced` the oldest record
/// of the tile holding the most (records are in publication order, so ties go to the oldest).
pub(super) fn evict(records: &[Record], balanced: bool) -> usize {
    if !balanced {
        return 0;
    }
    let mut held = [0usize; TILES];
    records.iter().for_each(|record| held[record.tile_id] += 1);
    let most = held.iter().copied().max().unwrap_or(0);
    records
        .iter()
        .position(|record| held[record.tile_id] == most)
        .unwrap_or(0)
}

/// Per-tile ratio `mean share / max(share, TINY)` over the outer tiles; 1 for the start tile
/// (visited every anchor by construction), before `horizon` anchors and while the mean is tiny.
fn ratios(estimate: &Estimate, horizon: u64) -> [f64; TILES] {
    let outer = (0..TILES).filter(|&tile| tile != START_TILE);
    let mean = outer.map(|tile| estimate.share[tile]).sum::<f64>() / (TILES - 1) as f64;
    if estimate.samples < horizon || mean <= TINY {
        return [1.0; TILES];
    }
    std::array::from_fn(|tile| {
        if tile == START_TILE {
            1.0
        } else {
            mean / estimate.share[tile].max(TINY)
        }
    })
}

/// `tile`'s balance rate multiplier `clamp(r ^ rate, 1/4, 4)`; exactly 1 when `rate` is 0.
pub(super) fn multiplier(
    options: &Options,
    estimate: Option<&Estimate>,
    tile: usize,
) -> Result<f64> {
    match (options.balance, estimate) {
        (Some(balance), Some(estimate)) if balance.rate > 0.0 => {
            Ok(ratios(estimate, balance.horizon)[tile]
                .powf(balance.rate)
                .clamp(MAX_MULTIPLIER.recip(), MAX_MULTIPLIER))
        }
        (Some(_), Some(_)) | (None, None) => Ok(1.0),
        _ => bail!("connected balance state/options mismatch"),
    }
}

/// The single tile-rate helper: the restart/Lambert factor times the balance multiplier.
pub(super) fn rate_factor(options: &Options, state: &State, tile: usize) -> Result<f64> {
    let restarts = state.restarts.as_deref();
    let base = super::connected_restart::factor(options, restarts, state.anchors, tile)?;
    Ok(base * multiplier(options, state.balance.as_ref(), tile)?)
}

/// Training route bonus per destination tile, `clamp(route ln r, -2, 2)`; None when off.
pub(super) fn bonus(options: &Options, estimate: Option<&Estimate>) -> Option<[f32; TILES]> {
    let balance = options.balance?;
    let (route, estimate) = (balance.route, estimate.filter(|_| balance.route > 0.0)?);
    Some(
        ratios(estimate, balance.horizon)
            .map(|ratio| (route * ratio.ln()).clamp(-MAX_BONUS, MAX_BONUS) as f32),
    )
}

/// Folds one live anchor of `loss` into the estimates, before the baseline moves.
pub(super) fn observe(state: &mut State, visit: &Visit, loss: f32) -> Result<()> {
    let Some(balance) = state.options.balance else {
        return Ok(());
    };
    let advantage = f64::from(loss) - f64::from(state.baseline);
    let Some(estimate) = state.balance.as_mut() else {
        bail!("missing connected balance state");
    };
    // Running mean for the first `horizon` samples, then an EMA of that horizon (no zero bias).
    let weight = |count: u64| (count as f64).recip().max((balance.horizon as f64).recip());
    estimate.samples += 1;
    let share_weight = weight(estimate.samples);
    for tile in 0..TILES {
        let hit = visit.visited[tile];
        estimate.share[tile] += share_weight * (f64::from(u8::from(hit)) - estimate.share[tile]);
        let (count, average) = if hit {
            estimate.visits[tile] += 1;
            (estimate.visits[tile], &mut estimate.advantage_visited[tile])
        } else {
            let unvisited = estimate.samples - estimate.visits[tile];
            (unvisited, &mut estimate.advantage_unvisited[tile])
        };
        *average += weight(count) * (advantage - *average);
    }
    for &(tile, steer) in &visit.steers {
        estimate.decisions[tile] += 1;
        estimate.changed[tile] += u64::from(steer.changed);
        estimate.bonus_sum[tile] += f64::from(steer.bonus);
    }
    Ok(())
}

#[path = "connected_balance_check.rs"]
mod check;
pub(super) use check::{report, validate};

#[cfg(test)]
#[path = "connected_balance_contracts.rs"]
mod contracts;

#[cfg(test)]
#[path = "connected_balance_rules_contracts.rs"]
mod rules;
