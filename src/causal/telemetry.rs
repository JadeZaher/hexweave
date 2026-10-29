//! Interval training telemetry: phase wall time and realized routes; never persisted.
//! See AGENTS.md §Telemetry.

use super::execution::Trace;
use super::geometry::{TILES, VISITS};
use serde_json::{json, Map, Value};
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::time::Instant;

/// Disjoint core-path phases; `Evaluate` runs outside the training wall interval.
#[derive(Clone, Copy)]
pub(super) enum Phase {
    Data,
    Prefix,
    Forward,
    Gather,
    TileBackward,
    TileApply,
    SharedApply,
    Records,
    Checkpoint,
    Evaluate,
}

const NAMES: [&str; 10] = [
    "data",
    "prefix",
    "forward",
    "gather",
    "tile_backward",
    "tile_apply",
    "shared_apply",
    "records",
    "checkpoint",
    "evaluate",
];
const TOP_ROUTES: usize = 8;

#[derive(Default)]
struct Interval {
    seconds: [f64; NAMES.len()],
    depth_visits: [[u64; TILES]; VISITS],
    anchors_per_tile: [u64; TILES],
    lengths: [u64; VISITS],
    routes: BTreeMap<Vec<usize>, u64>,
}

thread_local! {
    static INTERVAL: RefCell<Interval> = RefCell::new(Interval::default());
}

/// Run `work`, adding its wall time to `phase`; callers must not nest phases.
pub(super) fn time<T>(phase: Phase, work: impl FnOnce() -> T) -> T {
    let started = Instant::now();
    let result = work();
    let seconds = started.elapsed().as_secs_f64();
    INTERVAL.with(|cell| cell.borrow_mut().seconds[phase as usize] += seconds);
    result
}

/// Count one training anchor's realized route.
pub(super) fn route(trace: &Trace) {
    let tiles: Vec<usize> = trace.frames.iter().map(|frame| frame.tile_id).collect();
    INTERVAL.with(|cell| {
        let interval = &mut *cell.borrow_mut();
        for (depth, &tile) in tiles.iter().enumerate() {
            interval.depth_visits[depth][tile] += 1;
        }
        for (tile, anchors) in interval.anchors_per_tile.iter_mut().enumerate() {
            *anchors += u64::from(tiles.contains(&tile));
        }
        if let Some(slot) = tiles.len().checked_sub(1) {
            interval.lengths[slot] += 1;
        }
        *interval.routes.entry(tiles).or_default() += 1;
    });
}

/// Report and reset the interval; `wall` is the training wall time it covers.
pub(super) fn drain(wall: f64) -> Value {
    let interval = INTERVAL.with(RefCell::take);
    let mut seconds: Map<String, Value> = NAMES
        .iter()
        .zip(interval.seconds)
        .map(|(name, value)| ((*name).to_owned(), json!(value)))
        .collect();
    let timed: f64 = interval.seconds[..Phase::Evaluate as usize].iter().sum();
    seconds.insert("unattributed".to_owned(), json!(wall - timed));
    let mut routes: Vec<_> = interval.routes.iter().collect();
    routes.sort_by(|a, b| b.1.cmp(a.1).then_with(|| a.0.cmp(b.0)));
    let top: Vec<Value> = routes
        .iter()
        .take(TOP_ROUTES)
        .map(|(tiles, anchors)| json!({"tiles": tiles, "anchors": anchors}))
        .collect();
    json!({
        "seconds": seconds,
        "anchors": interval.lengths.iter().sum::<u64>(),
        "tile_visits_by_depth": interval.depth_visits,
        "anchors_per_tile": interval.anchors_per_tile,
        "route_lengths": interval.lengths,
        "distinct_routes": interval.routes.len(),
        "top_routes": top,
    })
}

#[cfg(test)]
#[path = "execution_fixture.rs"]
#[allow(clippy::duplicate_mod, dead_code)]
mod fixture;

#[cfg(test)]
mod tests {
    use super::super::execution::{self, START_TILE};
    use super::super::geometry::Geometry;
    use super::super::state::State;
    use super::fixture::{Source, Store};
    use super::*;
    use anyhow::Result;

    #[test]
    fn anchor_steps_report_every_route_visit_and_disjoint_phase_time() -> Result<()> {
        let source = Source::new();
        let mut state = State::with_cursor(Geometry::fixture(), 29, source.cursor())?;
        state.anchor_only = true;
        let mut store = Store::new(29)?;
        drain(0.0);
        let started = Instant::now();
        for _ in 0..6 {
            execution::step(&mut state, &mut store, &source, false)?;
        }
        let report = drain(started.elapsed().as_secs_f64());
        let by_depth: Vec<Vec<u64>> =
            serde_json::from_value(report["tile_visits_by_depth"].clone())?;
        assert_eq!(report["anchors"], 6);
        assert_eq!(
            by_depth.iter().flatten().sum::<u64>(),
            state.counters.visits
        );
        assert_eq!(by_depth[0][START_TILE], 6);
        let seconds = &report["seconds"];
        for phase in ["forward", "tile_backward", "tile_apply"] {
            assert!(
                seconds[phase].as_f64().is_some_and(|value| value > 0.0),
                "{phase}"
            );
        }
        assert!(seconds["unattributed"]
            .as_f64()
            .is_some_and(|value| value >= 0.0));
        Ok(())
    }
}
