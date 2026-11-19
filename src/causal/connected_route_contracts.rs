//! Greedy training routes, their exploration share and the run's tile-count admission.

use super::connected_anchor;
use super::connected_bias_contracts::corpus;
use super::connected_checkpoint as checkpoint;
use super::connected_config::{Options, RouteTraining};
use super::connected_state::State;
use super::connected_step;
use super::connected_storage_contracts::fresh;
use super::connected_store::Store;
use super::core_config::Config;
use super::core_corpus::Corpus;
use super::execution::TrainingSource;
use super::geometry::{Geometry, TILES};
use super::tuning::{self, Knobs};
use anyhow::Result;
use serde_json::json;
use std::path::Path;

#[test]
fn connected_identity_route_and_tile_count_admission() -> Result<()> {
    let base = json!({"schema":1,"out":"work/core-runs/connected-route-test",
        "data":{"kind":"corpus","root":"work/corpora/test","manifest_sha256":"test","window":32},
        "seed":7,"rate":0.003,"anchors":8,"wall_seconds":60,"checkpoint_every":8,
        "eval_every":8,"eval_cases":1,"resume":false,"connected":{},"tiles":TILES});
    let identity = serde_json::from_value::<Config>(base.clone())?.identity()?;
    let greedy = |explore: f64| json!({"route_training":"greedy","route_explore":explore});
    // Absent `tiles` means 7; only this build's count is admitted, and never as identity.
    for (key, value, admitted) in [
        ("connected", greedy(0.25), true),
        ("connected", greedy(1.0), false),
        ("connected", greedy(-0.1), false),
        ("connected", json!({"route_explore":0.25}), false),
        ("tiles", json!(null), TILES == 7),
        ("tiles", json!(7), TILES == 7),
        ("tiles", json!(19), TILES == 19),
    ] {
        let mut changed = base.clone();
        changed[key] = value.clone();
        let config: Config = serde_json::from_value(changed)?;
        assert_eq!(config.validate().is_ok(), admitted, "{key} {value}");
        if key == "tiles" && admitted {
            assert_eq!(config.identity()?, identity);
        }
    }
    Ok(())
}

/// The route taken from `state` with argmax actions and the step's own memory reads.
fn greedy_route(state: &State, store: &mut Store, corpus: &Corpus) -> Result<[u64; TILES]> {
    let prefix = state
        .shared
        .prefix(&corpus.pending_prefix(&state.cursor)?)?;
    let revisions = std::array::from_fn(|id| store.slots[id].meta.revision);
    let trace = connected_anchor::forward(
        store,
        state.geometry,
        &prefix,
        state.options.visits,
        super::connected_route::Draw::GREEDY,
        |tile, input| {
            let retrieved =
                state
                    .memory
                    .query(input, tile, state.epoch, &revisions, state.anchors)?;
            Ok(state.options.history_input(retrieved))
        },
    )?;
    let mut visits = [0; TILES];
    trace
        .frames
        .iter()
        .for_each(|frame| visits[frame.tile] += 1);
    Ok(visits)
}

/// Trains `count` anchors; returns how many took a route other than the greedy one.
fn train(state: &mut State, store: &mut Store, corpus: &Corpus, count: usize) -> Result<usize> {
    let mut departures = 0;
    for _ in 0..count {
        let expected = greedy_route(state, store, corpus)?;
        let before = state.counters.anchor_visits;
        connected_step::step(state, store, corpus, state.options, 0.003, false)?;
        let taken: [u64; TILES] =
            std::array::from_fn(|tile| state.counters.anchor_visits[tile] - before[tile]);
        departures += usize::from(taken != expected);
    }
    Ok(departures)
}

fn arm(
    root: &Path,
    corpus: &Corpus,
    training: RouteTraining,
    explore: f64,
) -> Result<(State, Store)> {
    let options = Options {
        lanes: 2,
        workers: 2,
        route_training: training,
        route_explore: explore,
        ..Options::default()
    };
    let geometry = Geometry::fixture();
    let store = Store::new(&root.join("connected-tiles"), geometry, 7, options.workers)?;
    Ok((State::new(geometry, 7, corpus.cursor(0)?, &options)?, store))
}

#[test]
fn greedy_training_routes_follow_argmax_explore_departs_and_resume_is_exact() -> Result<()> {
    let _guard = tuning::scoped(Knobs {
        connected: true,
        ..Knobs::core(0.003, 7)
    });
    let out = fresh("greedy-routes")?;
    let text: Vec<u8> = b"the cat sat on the mat. "
        .iter()
        .cycle()
        .take(512)
        .copied()
        .collect();
    let corpus = corpus(&out.join("corpus"), &text)?;
    // Sampled routes depart from the argmax, so the greedy check below can fail.
    let (mut sampled, mut store) = arm(&out.join("sampled"), &corpus, RouteTraining::Sampled, 0.0)?;
    assert!(train(&mut sampled, &mut store, &corpus, 8)? > 0);
    let (mut greedy, mut store) = arm(&out.join("greedy"), &corpus, RouteTraining::Greedy, 0.0)?;
    assert_eq!(train(&mut greedy, &mut store, &corpus, 8)?, 0);
    // With exploration some anchors sample; a checkpoint resume reproduces the straight run.
    let (mut straight, mut straight_store) =
        arm(&out.join("straight"), &corpus, RouteTraining::Greedy, 0.5)?;
    let departures = train(&mut straight, &mut straight_store, &corpus, 8)?;
    assert!((1..8).contains(&departures), "{departures} departures");
    let run = out.join("run");
    let (mut first, mut first_store) = arm(&run, &corpus, RouteTraining::Greedy, 0.5)?;
    let mut resumed_departures = train(&mut first, &mut first_store, &corpus, 4)?;
    checkpoint::save(&run, &mut first_store, "identity", 4, &first.payloads()?)?;
    drop((first, first_store));
    let loaded = checkpoint::resume(&run, Geometry::fixture(), "identity", 2)?.unwrap();
    let mut resumed = State::restore(&loaded.payloads, Geometry::fixture(), 7)?;
    let mut store = loaded.store;
    assert_eq!(resumed.options.route_explore, 0.5);
    resumed_departures += train(&mut resumed, &mut store, &corpus, 4)?;
    assert_eq!(resumed_departures, departures);
    assert_eq!(resumed.payloads()?, straight.payloads()?);
    store.flush()?;
    straight_store.flush()?;
    let meta = |store: &Store| {
        store
            .slots
            .iter()
            .map(|s| s.meta.clone())
            .collect::<Vec<_>>()
    };
    assert_eq!(meta(&store), meta(&straight_store));
    Ok(())
}
