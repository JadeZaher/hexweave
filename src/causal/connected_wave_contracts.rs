//! Anchor-wave admission and the wave flow: frozen tiles, bounded paging, exact resume.

use super::super::connected_bias_contracts::corpus;
use super::super::connected_checkpoint as checkpoint;
use super::super::connected_restart as restart;
use super::super::connected_storage_contracts::fresh;
use super::super::core_config::Config;
use super::super::core_corpus::Corpus;
use super::super::geometry::Geometry;
use super::super::tile_gradients::TileGradients;
use super::super::tuning::Knobs;
use super::*;
use serde_json::{json, Value};
use std::path::Path;

#[test]
fn anchor_batch_admission_identity_and_refusals() -> Result<()> {
    let base = json!({"schema":1,"out":"work/core-runs/connected-wave-test",
        "data":{"kind":"corpus","root":"work/corpora/test","manifest_sha256":"test","window":32},
        "seed":7,"rate":0.003,"anchors":8,"wall_seconds":60,"checkpoint_every":8,
        "eval_every":8,"eval_cases":1,"resume":false,"tiles":TILES,
        "connected":{"credit_mode":"anchor_exact"}});
    let config = |connected: Value, top: Value| -> Result<Config> {
        let mut changed = base.clone();
        for (key, value) in connected.as_object().into_iter().flatten() {
            changed["connected"][key] = value.clone();
        }
        for (key, value) in top.as_object().into_iter().flatten() {
            changed[key] = value.clone();
        }
        Ok(serde_json::from_value(changed)?)
    };
    let identity = config(json!({}), json!({}))?.identity()?;
    let restart_replay = json!({"floor":0.1,"threshold":0.5,"replay":2});
    // (connected keys, top-level keys, admitted, identity unchanged)
    for (connected, top, admitted, same) in [
        (json!({"anchor_batch":1}), json!({}), true, true),
        (json!({"anchor_batch":4}), json!({}), true, false),
        (
            json!({"anchor_batch":32}),
            json!({"anchors":32,"checkpoint_every":32,
            "eval_every":32}),
            true,
            false,
        ),
        (json!({"anchor_batch":0}), json!({}), false, false),
        (json!({"anchor_batch":33}), json!({}), false, false),
        (
            json!({"anchor_batch":4,"credit_mode":"lanes"}),
            json!({}),
            false,
            false,
        ),
        (
            json!({"anchor_batch":4,"credit_mode":"hybrid"}),
            json!({}),
            false,
            false,
        ),
        (
            json!({"anchor_batch":4,"replay":{"capacity":64,"per_anchor":2}}),
            json!({}),
            false,
            false,
        ),
        (
            json!({"anchor_batch":4,"restart":restart_replay}),
            json!({}),
            false,
            false,
        ),
        // Boundaries must fall on wave ends.
        (json!({"anchor_batch":3}), json!({}), false, false),
        (
            json!({"anchor_batch":4}),
            json!({"anchors":12,"checkpoint_every":6}),
            false,
            false,
        ),
        (
            json!({"anchor_batch":4}),
            json!({"eval_every":2}),
            false,
            false,
        ),
    ] {
        let shown = format!("{connected} {top}");
        let config = config(connected, top)?;
        assert_eq!(config.validate().is_ok(), admitted, "{shown}");
        if admitted {
            assert_eq!(config.identity()? == identity, same, "{shown}");
        }
    }
    let options = config(json!({"anchor_batch":1}), json!({}))?
        .connected
        .unwrap();
    assert!(serde_json::to_value(options)?["anchor_batch"].is_null());
    Ok(())
}

/// Anchor-exact FP32 options with a Lambert restart block, `batch` anchors per wave, cap 2.
fn options(batch: u32) -> Result<Options> {
    Ok(serde_json::from_value(json!({
        "precision":"fp32","lanes":2,"workers":2,"visits":4,"credit_mode":"anchor_exact",
        "anchor_batch":batch,
        "restart":{"floor":0.1,"threshold":0.5,"signal":"tile","law":"lambert"},
    }))?)
}

fn arm(root: &Path, corpus: &Corpus, options: Options) -> Result<(State, Store)> {
    let geometry = Geometry::fixture();
    let state = State::new(geometry, 7, corpus.cursor(0)?, &options)?;
    let mut store = Store::new_precision(
        &root.join("connected-tiles"),
        geometry,
        7,
        options.workers,
        options.precision,
    )?;
    store.set_paging(options.resident(), 0)?;
    Ok((state, store))
}

fn revisions(store: &Store) -> [u64; TILES] {
    std::array::from_fn(|tile| store.slots[tile].meta.revision)
}

/// Runs `waves` waves, checking each boundary's commits; returns the most tiles one wave touched.
fn train(state: &mut State, store: &mut Store, corpus: &Corpus, waves: usize) -> Result<usize> {
    let mut widest = 0;
    for _ in 0..waves {
        let (before, counters) = (revisions(store), state.counters.clone());
        step(state, store, corpus, state.options, 0.003, false)?;
        state.validate()?;
        let after = revisions(store);
        let mut touched = 0;
        for tile in 0..TILES {
            let visits = state.counters.local_visits[tile] - counters.local_visits[tile];
            let commits = after[tile] - before[tile];
            // A visited tile commits once locally and at most once for its routes.
            assert!(if visits > 0 {
                (1..=2).contains(&commits)
            } else {
                commits == 0
            });
            touched += usize::from(visits > 0);
        }
        let routes = state.counters.route_updates - counters.route_updates;
        let locals = state.counters.local_updates - counters.local_updates;
        let commits: u64 = (0..TILES).map(|tile| after[tile] - before[tile]).sum();
        assert_eq!((locals, commits), (touched as u64, locals + routes));
        // E and the bias step every anchor: their Adam clocks equal `anchors`.
        let clocks = (
            state.shared_latent.as_ref().map(|latent| latent.step),
            state.bias_latent.as_ref().map(|latent| latent.step),
        );
        assert_eq!(clocks, (Some(state.anchors), Some(state.anchors)));
        widest = widest.max(touched);
    }
    Ok(widest)
}

#[test]
fn waves_freeze_tiles_page_within_the_cap_and_resume_exactly() -> Result<()> {
    let _guard = tuning::scoped(Knobs {
        connected: true,
        connected_fp32: true,
        ..Knobs::core(0.003, 7)
    });
    let out = fresh("anchor-waves")?;
    let text: Vec<u8> = b"the cat sat on the mat. "
        .iter()
        .cycle()
        .take(512)
        .copied()
        .collect();
    let corpus = corpus(&out.join("corpus"), &text)?;
    let (mut straight, mut straight_store) = arm(&out.join("straight"), &corpus, options(4)?)?;
    let widest = train(&mut straight, &mut straight_store, &corpus, 3)?;
    let stats = straight_store.stats();
    assert!(widest > 2 && stats.peak_resident <= 2, "{widest} {stats:?}");
    assert_eq!(straight.anchors, 12);
    assert_eq!(straight.counters.waves.completed, 3);
    // Lambert dials hold at 1 before warm-up; per-anchor weighting keeps their mean exactly 1.
    let report = restart::report(&straight)?.expect("restart report");
    let means = report["plasticity_mean"].as_array().expect("means").clone();
    assert!(means
        .iter()
        .all(|mean| mean.is_null() || *mean == json!(1.0)));
    // A checkpoint at a wave boundary, resumed, reproduces the uninterrupted run.
    let run = out.join("run");
    let (mut first, mut first_store) = arm(&run, &corpus, options(4)?)?;
    train(&mut first, &mut first_store, &corpus, 2)?;
    checkpoint::save(&run, &mut first_store, "identity", 8, &first.payloads()?)?;
    drop((first, first_store));
    let loaded = checkpoint::resume(&run, Geometry::fixture(), "identity", 2)?.unwrap();
    let mut resumed = State::restore(&loaded.payloads, Geometry::fixture(), 7)?;
    let mut store = loaded.store;
    store.set_paging(2, 0)?;
    train(&mut resumed, &mut store, &corpus, 1)?;
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
    // Four single-anchor steps let later anchors see earlier tile steps, so E ends elsewhere.
    let (mut wave, mut wave_store) = arm(&out.join("wave"), &corpus, options(4)?)?;
    train(&mut wave, &mut wave_store, &corpus, 1)?;
    let (mut single, mut single_store) = arm(&out.join("single"), &corpus, options(1)?)?;
    for _ in 0..4 {
        let options = single.options;
        step(
            &mut single,
            &mut single_store,
            &corpus,
            options,
            0.003,
            false,
        )?;
    }
    let embedding = |state: &State| state.shared_latent.as_ref().map(|l| l.weights.clone());
    assert_ne!(embedding(&wave), embedding(&single));
    assert!(single.counters.waves.is_unused());
    Ok(())
}

#[test]
fn route_and_local_accumulators_apply_the_mean_of_their_contributions() -> Result<()> {
    // `take_mean` is the one divisor of `apply_routes` (via `add`) and `apply_with_trust`
    // (via `add_exact`); a sum, or a divisor other than the count, fails here.
    for (exact, first, second) in [(false, 1.0f32, 3.0f32), (true, -2.0, 6.0)] {
        let mut accumulator = Accumulator::default();
        for value in [first, second] {
            let mut gradient = TileGradients::zeros(Geometry::fixture(), 0)?;
            gradient
                .values
                .iter_mut()
                .flatten()
                .for_each(|v| *v = value);
            if exact {
                accumulator.add_exact(0, gradient)?;
            } else {
                accumulator.add(0, gradient)?;
            }
        }
        let mean = accumulator.take_mean(0).expect("tile 0 mean");
        let expected = (first + second) / 2.0;
        assert!(
            mean.values.iter().flatten().all(|&v| v == expected),
            "{exact}"
        );
        assert!(accumulator.take_mean(1).is_none());
    }
    Ok(())
}
