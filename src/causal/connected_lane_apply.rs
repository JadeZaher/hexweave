//! One mean local Adam step per tile after all lanes finish; see CONNECTED.md §Scale control.

use super::connected_anchor::{self, Trace};
use super::connected_config::Options;
use super::connected_restart as restart;
use super::connected_state::{Counters, State};
use super::connected_store::Store;
use super::connected_update::{self, Mode, UpdateMetrics};
use super::geometry::TILES;
use super::norm;
use super::tile_gradients::TileGradients;
use anyhow::Result;

/// Per-tile sums of read-only local gradients and their contribution counts for one phase.
#[derive(Default)]
pub(super) struct Accumulator {
    sums: [Option<(TileGradients, u32)>; TILES],
    exact: [bool; TILES],
}

impl Accumulator {
    /// Adds one lane visit's local gradient to `tile`'s running sum.
    pub fn add(&mut self, tile: usize, gradient: TileGradients) -> Result<()> {
        let slot = &mut self.sums[tile];
        if let Some((sum, count)) = slot {
            for (total, values) in sum.values.iter_mut().zip(&gradient.values) {
                norm::add(total, values)?;
            }
            *count += 1;
        } else {
            *slot = Some((gradient, 1));
        }
        Ok(())
    }

    /// Adds an exact anchor-route visit gradient to `tile`'s running sum and marks `tile` exact.
    pub fn add_exact(&mut self, tile: usize, gradient: TileGradients) -> Result<()> {
        self.exact[tile] = true;
        self.add(tile, gradient)
    }

    /// Number of tiles holding a gradient sum: the most mean steps one apply may take.
    pub fn tiles(&self) -> u64 {
        self.sums.iter().filter(|sum| sum.is_some()).count() as u64
    }

    /// One route step per tile from its mean summed route gradient at `gains[tile]` (waves).
    pub fn apply_routes(
        mut self,
        state: &mut State,
        store: &mut Store,
        options: &Options,
        rate: f64,
        gains: &[f32; TILES],
    ) -> Result<()> {
        for (tile, &gain) in gains.iter().enumerate() {
            if let Some(mean) = self.take_mean(tile) {
                route_step(state, store, tile, &mean, gain, options, rate)?;
            }
        }
        Ok(())
    }

    /// `tile`'s mean gradient, if any contribution reached it.
    pub fn into_mean(mut self, tile: usize) -> Option<TileGradients> {
        self.take_mean(tile)
    }

    /// Removes `tile`'s sum and divides it by its contribution count: the one mean divisor.
    pub fn take_mean(&mut self, tile: usize) -> Option<TileGradients> {
        let (mut mean, count) = self.sums[tile].take()?;
        for values in &mut mean.values {
            values.iter_mut().for_each(|value| *value /= count as f32);
        }
        Some(mean)
    }

    /// Applies each tile's mean gradient; `use_trust = false` uses unit gain (`AnchorExact`),
    /// while `use_trust = true` uses `Lambert(gain)` on exact-seeded tiles and `Lambert(gain) * trust`
    /// on predicted-only tiles.
    pub fn apply_with_trust(
        mut self,
        state: &mut State,
        store: &mut Store,
        options: &Options,
        use_trust: bool,
    ) -> Result<()> {
        for (tile, exact) in self.exact.into_iter().enumerate() {
            if self.sums[tile].is_none() {
                continue;
            }
            let revision = store.slots[tile].meta.revision;
            let lambert = state.gains[tile].value_at(state.epoch, revision, options);
            let gain = if !use_trust {
                1.0
            } else if exact {
                lambert
            } else {
                lambert * state.predictor.trust(tile) as f32
            };
            if gain <= 0.0 {
                continue;
            }
            let Some(mean) = self.take_mean(tile) else {
                continue;
            };
            // `anchors` counts completed anchors, so during this lane phase it is the anchor's index.
            restart::observe_gradient(state, tile, &mean);
            let factor = super::connected_balance::rate_factor(options, state, tile)?;
            let rate = options.local_rate * factor;
            let update = update_tile(store, tile, &mean, rate, gain, options)?;
            record(&mut state.counters, tile, &update);
            restart::mark_stepped(state, tile);
        }
        Ok(())
    }
}

/// One checked-out `Mode::Local` update; a failed update aborts the lease and poisons the store.
pub(super) fn update_tile(
    store: &mut Store,
    tile: usize,
    mean: &TileGradients,
    rate: f64,
    gain: f32,
    options: &Options,
) -> Result<UpdateMetrics> {
    let mut lease = store.checkout(tile)?;
    let update = (|| {
        let (tile, latents) = lease.parts_mut()?;
        connected_update::apply(tile, latents, mean, rate, gain, options, Mode::Local)
    })();
    match update {
        Ok(update) => {
            store.commit(lease)?;
            Ok(update)
        }
        Err(error) => {
            store.abort(tile);
            Err(error)
        }
    }
}

/// Accounting for one applied tile: `local_updates` now counts tiles, not lane visits.
pub(super) fn record(counters: &mut Counters, tile: usize, update: &UpdateMetrics) {
    counters.local_updates += 1;
    counters.tile_transitions += update.transitions as u64;
    counters.opposed_momentum += update.opposed_momentum as u64;
    counters.local_base_norm_sum += update.base_norm;
    counters.local_applied_norm_sum += update.applied_norm;
    counters.change_sum += restart::relative(update);
    counters.local_boosted += u64::from(update.gain > 1.0);
    counters.local_damped += u64::from(update.gain < 1.0);
    track_rms(counters, tile, update.max_rms);
}

/// Raises `tile`'s running maximum master RMS; a NaN stays visible to state validation.
pub(super) fn track_rms(counters: &mut Counters, tile: usize, max_rms: f64) {
    // f64 like every other float counter: connected.json and report.json must print it identically.
    let maximum = &mut counters.master_rms_max[tile];
    if max_rms.is_nan() || max_rms > *maximum {
        *maximum = max_rms;
    }
}

/// REINFORCE route steps at `rate` for every tile that took a routing action this anchor.
pub(super) fn update_routes(
    state: &mut State,
    store: &mut Store,
    trace: &Trace,
    loss: f32,
    options: Options,
    rate: f64,
) -> Result<()> {
    for tile in (0..TILES).filter(|&tile| routed(trace, tile)) {
        let gradient = connected_anchor::route_gradient(
            store,
            state.geometry,
            trace,
            tile,
            loss - state.baseline,
        )?;
        let gain = route_gain(state, store, tile, &options);
        route_step(state, store, tile, &gradient, gain, &options, rate)?;
    }
    Ok(())
}

/// Whether `tile` took a routing action on `trace`.
pub(super) fn routed(trace: &Trace, tile: usize) -> bool {
    trace
        .frames
        .iter()
        .any(|frame| frame.tile == tile && frame.action.is_some())
}

/// `tile`'s route gain now: its gain at the current epoch and store revision.
pub(super) fn route_gain(state: &State, store: &Store, tile: usize, options: &Options) -> f32 {
    let revision = store.slots[tile].meta.revision;
    state.gains[tile].value_at(state.epoch, revision, options)
}

/// One checked-out `Mode::RouteOnly` step at `gain`.
pub(super) fn route_step(
    state: &mut State,
    store: &mut Store,
    tile: usize,
    gradient: &TileGradients,
    gain: f32,
    options: &Options,
    rate: f64,
) -> Result<()> {
    let mut lease = store.checkout(tile)?;
    let update = (|| {
        let (tile, latents) = lease.parts_mut()?;
        connected_update::apply(
            tile,
            latents,
            gradient,
            rate,
            gain,
            options,
            Mode::RouteOnly,
        )
    })();
    let update = match update {
        Ok(update) => update,
        Err(error) => {
            store.abort(tile);
            return Err(error);
        }
    };
    store.commit(lease)?;
    state.counters.route_updates += 1;
    state.counters.tile_transitions += update.transitions as u64;
    state.counters.opposed_momentum += update.opposed_momentum as u64;
    track_rms(&mut state.counters, tile, update.max_rms);
    Ok(())
}
