//! Per-tile rate floor and change-driven warm restarts through the real step path.

use super::connected_bias_contracts::corpus;
use super::connected_checkpoint as checkpoint;
use super::connected_config::{CreditMode, Options, Replay, Restart};
use super::connected_precision::Precision;
use super::connected_restart::{factor, restart, TileRestart};
use super::connected_state::State;
use super::connected_step;
use super::connected_storage_contracts::fresh;
use super::connected_store::Store;
use super::core_corpus::Corpus;
use super::core_schedule::Decay;
use super::geometry::{Geometry, TILES};
use super::tuning::{self, Knobs};
use anyhow::{Context, Result};
use std::path::Path;

/// About two fixture anchors of change (each ~2.3e-3, mostly E's relative step).
const THRESHOLD: f64 = 0.005;

/// Anchor-exact FP32 options with replay and a fast inverse-sqrt local schedule.
pub(super) fn options(restart: Option<Restart>) -> Options {
    Options {
        precision: Precision::Fp32,
        lanes: 2,
        workers: 2,
        visits: 2,
        credit_mode: CreditMode::AnchorExact,
        replay: Some(Replay {
            capacity: 4,
            per_anchor: 1,
        }),
        local_decay: Decay::InverseSqrt { anchors: 1 },
        restart,
        ..Options::default()
    }
}

pub(super) fn start(out: &Path, corpus: &Corpus, options: &Options) -> Result<(State, Store)> {
    let geometry = Geometry::fixture();
    let state = State::new(geometry, 7, corpus.cursor(0)?, options)?;
    let tiles = out.join("connected-tiles");
    let store = Store::new_precision(&tiles, geometry, 7, 2, Precision::Fp32)?;
    Ok((state, store))
}

pub(super) fn steps(
    state: &mut State,
    store: &mut Store,
    corpus: &Corpus,
    count: usize,
) -> Result<Vec<f32>> {
    let options = state.options;
    (0..count)
        .map(|_| connected_step::step(state, store, corpus, options, 0.003, false))
        .collect()
}

/// Per tile: restarts, drift, live visits and store revision.
fn snapshot(state: &State, store: &Store) -> Result<Vec<(u64, f64, u64, u64)>> {
    let tiles = state.restarts.as_deref().context("missing restart state")?;
    let counters = &state.counters;
    Ok((0..TILES)
        .map(|t| {
            let visits = counters.local_visits[t] + counters.anchor_visits[t];
            let revision = store.slots[t].meta.revision;
            (tiles[t].restarts, tiles[t].drift, visits, revision)
        })
        .collect())
}

/// Per-anchor losses of a fresh `options` run in its own directory.
fn losses(out: &Path, corpus: &Corpus, options: Options) -> Result<Vec<f32>> {
    let (mut state, mut store) = start(out, corpus, &options)?;
    steps(&mut state, &mut store, corpus, 6)
}

#[test]
fn restarts_reset_the_tile_clock_memory_and_bar_and_survive_a_checkpoint() -> Result<()> {
    let _guard = tuning::scoped(Knobs {
        connected: true,
        connected_fp32: true,
        ..Knobs::core(0.003, 7)
    });
    let out = fresh("restart")?;
    let train: Vec<u8> = b"abacabad".iter().cycle().take(512).copied().collect();
    let corpus = corpus(&out.join("corpus"), &train)?;
    // A floor of 1 must hold every tile step at `local_rate`: the constant schedule exactly.
    let held = Restart {
        floor: 1.0,
        threshold: None,
        adapt: 0.0,
        ..Restart::default()
    };
    let constant = Options {
        local_decay: Decay::Constant,
        ..options(None)
    };
    let reference = losses(&out.join("constant"), &corpus, constant)?;
    assert_eq!(
        losses(&out.join("held"), &corpus, options(Some(held)))?,
        reference
    );
    assert_ne!(
        losses(&out.join("decayed"), &corpus, options(None))?,
        reference
    );
    let restart = Restart {
        floor: 0.25,
        threshold: Some(THRESHOLD),
        adapt: 0.5,
        ..Restart::default()
    };
    let options = options(Some(restart));
    let run = out.join("restart");
    let (mut state, mut store) = start(&run, &corpus, &options)?;
    let (mut observed, mut idle) = (0, 0);
    for _ in 0..16 {
        let before = snapshot(&state, &store)?;
        steps(&mut state, &mut store, &corpus, 1)?;
        let after = snapshot(&state, &store)?;
        let tiles = state.restarts.as_deref().context("missing restart state")?;
        // An untouched tile (no visit, no commit, no restart) keeps its drift exactly.
        for (old, new) in before.iter().zip(&after) {
            if (old.0, old.2, old.3) == (new.0, new.2, new.3) {
                assert_eq!(old.1, new.1);
                idle += 1;
            }
        }
        for tile in (0..TILES).filter(|&tile| after[tile].0 > before[tile].0) {
            observed += 1;
            let entry = &tiles[tile];
            assert!(entry.origin == state.anchors && entry.drift == 0.0);
            // Back to the full local rate at the next anchor, then decayed down to the floor.
            assert_eq!(factor(&options, Some(tiles), state.anchors, tile)?, 1.0);
            let late = factor(&options, Some(tiles), state.anchors + 1000, tile)?;
            assert_eq!(late, restart.floor);
            assert!(state
                .memory
                .records
                .iter()
                .all(|record| record.tile_id != tile));
            assert_eq!(state.replay.report()["buffer_size"], 0);
        }
    }
    let tiles = state.restarts.clone().context("missing restart state")?;
    assert!(
        observed >= 2 && idle > 0,
        "{observed} restarts, {idle} idle tiles"
    );
    assert!(tiles.iter().any(|entry| entry.forgotten_records > 0));
    // Later cycles moved some tile's bar, always within eight times the threshold either way.
    assert!(tiles.iter().any(|entry| entry.bar != THRESHOLD));
    let range = THRESHOLD / 8.0..=THRESHOLD * 8.0;
    assert!(tiles.iter().all(|entry| range.contains(&entry.bar)));
    state.validate()?;
    // The checkpoint carries the restart state; the session-only replay buffer restarts empty.
    let anchors = state.anchors;
    checkpoint::save(&run, &mut store, "identity", anchors, &state.payloads()?)?;
    state.replay.clear();
    let continued = steps(&mut state, &mut store, &corpus, 4)?;
    drop(store);
    let loaded = checkpoint::resume(&run, Geometry::fixture(), "identity", 2)?;
    let mut loaded = loaded.context("no checkpoint")?;
    let mut restored = State::restore(&loaded.payloads, Geometry::fixture(), 7)?;
    assert_eq!(restored.restarts, Some(tiles));
    let resumed = steps(&mut restored, &mut loaded.store, &corpus, 4)?;
    assert_eq!(resumed, continued);
    assert_eq!(restored.restarts, state.restarts);
    Ok(())
}

#[test]
fn adaptation_moves_the_bar_by_cycle_loss_within_the_clamp() {
    let config = |adapt, clamp| Restart {
        floor: 0.0,
        threshold: Some(1.0),
        adapt,
        clamp,
        ..Restart::default()
    };
    // (adapt, clamp, bar, previous cycle mean, this cycle's mean, expected bar)
    for (adapt, clamp, bar, previous, mean, expected) in [
        (0.5, 8.0, 1.0, None, 1.0, 1.0),
        (0.5, 8.0, 1.0, Some(2.0), 1.0, 0.5),
        (0.5, 8.0, 1.0, Some(1.0), 2.0, 1.5),
        (0.5, 8.0, 1.0, Some(1.0), 1.0, 1.5),
        (0.5, 8.0, 0.2, Some(2.0), 1.0, 0.125),
        (0.5, 8.0, 6.0, Some(1.0), 2.0, 8.0),
        (0.0, 8.0, 1.0, Some(2.0), 1.0, 1.0),
        // `clamp` sets both bounds: [1/2, 2], a wide 1024 and a pinned 1.
        (0.5, 2.0, 0.6, Some(2.0), 1.0, 0.5),
        (0.5, 2.0, 1.5, Some(1.0), 2.0, 2.0),
        (0.5, 1024.0, 6.0, Some(1.0), 2.0, 9.0),
        (0.5, 1.0, 1.0, Some(2.0), 1.0, 1.0),
    ] {
        let mut entry = TileRestart {
            origin: 3,
            drift: 2.0,
            bar,
            restarts: 1,
            forgotten_records: 0,
            cycle_loss: 4.0 * mean,
            cycle_anchors: 4,
            previous_mean: previous,
            ..TileRestart::new(bar, 1.0)
        };
        restart(&mut entry, config(adapt, clamp), 1.0, 9);
        assert_eq!(
            entry.bar, expected,
            "{adapt} {clamp} {bar} {previous:?} {mean}"
        );
        assert_eq!(entry.previous_mean, Some(mean));
        assert_eq!((entry.origin, entry.drift, entry.restarts), (9, 0.0, 2));
        assert_eq!((entry.cycle_loss, entry.cycle_anchors), (0.0, 0));
    }
}

#[test]
fn an_earlier_format_tile_record_loads_and_validates() -> Result<()> {
    let _guard = tuning::scoped(Knobs {
        connected: true,
        connected_fp32: true,
        ..Knobs::core(0.003, 7)
    });
    let out = fresh("restart-earlier")?;
    let train: Vec<u8> = b"abacabad".iter().cycle().take(512).copied().collect();
    let corpus = corpus(&out.join("corpus"), &train)?;
    let live = Restart {
        floor: 0.25,
        threshold: Some(2.0),
        adapt: 0.25,
        ..Restart::default()
    };
    let (mut state, mut store) = start(&out.join("run"), &corpus, &options(Some(live)))?;
    steps(&mut state, &mut store, &corpus, 6)?;
    // The eight keys a pre-option checkpoint carries, after two adapted restarts.
    let record: TileRestart = serde_json::from_value(serde_json::json!({"origin":4,"drift":0.5,
        "bar":1.0,"restarts":2,"forgotten_records":3,"cycle_loss":2.0,"cycle_anchors":2,
        "previous_mean":1.5}))?;
    state.restarts = Some(vec![record; TILES]);
    state.validate()
}
