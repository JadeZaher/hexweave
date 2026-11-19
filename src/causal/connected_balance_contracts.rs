//! Visit-balance admission, its rate/route/memory flow and exact resume.

use super::super::connected_bias_contracts::corpus;
use super::super::connected_checkpoint as checkpoint;
use super::super::connected_storage_contracts::fresh;
use super::super::connected_store::Store;
use super::super::connected_wave::step;
use super::super::core_config::Config;
use super::super::core_corpus::Corpus;
use super::super::geometry::START_TILE;
use super::super::tuning::{self, Knobs};
use super::*;
use serde_json::{json, Value};
use std::path::Path;

#[test]
fn balance_admission_bounds_refusals_and_identity() -> Result<()> {
    let base = json!({"schema":1,"out":"work/core-runs/connected-balance-test",
        "data":{"kind":"corpus","root":"work/corpora/test","manifest_sha256":"test","window":32},
        "seed":7,"rate":0.003,"anchors":8,"wall_seconds":60,"checkpoint_every":8,
        "eval_every":8,"eval_cases":1,"resume":false,"tiles":TILES,
        "connected":{"credit_mode":"anchor_exact","weight_decay":1.0,"local_rate":0.05}});
    let config = |connected: Value| -> Result<Config> {
        let mut changed = base.clone();
        for (key, value) in connected.as_object().into_iter().flatten() {
            changed["connected"][key] = value.clone();
        }
        Ok(serde_json::from_value(changed)?)
    };
    let identity = config(json!({}))?.identity()?;
    // (connected keys, admitted); every admitted block changes the identity.
    for (connected, admitted) in [
        (json!({"rate":0.5}), true),
        (json!({"route":2.0}), true),
        (json!({"memory":true}), true),
        (json!({"horizon":64,"route":0.5}), true),
        (json!({"horizon":65536,"memory":true}), true),
        // Nothing enabled: the block would silently do nothing.
        (json!({}), false),
        (
            json!({"horizon":64,"rate":0.0,"route":0.0,"memory":false}),
            false,
        ),
        (json!({"horizon":63,"memory":true}), false),
        (json!({"horizon":65537,"memory":true}), false),
        (json!({"rate":1.01}), false),
        (json!({"rate":-0.1}), false),
        (json!({"route":2.01}), false),
        (json!({"memory":true,"extra":1}), false),
    ]
    .map(|(balance, admitted)| (json!({ "balance": balance }), admitted))
    .into_iter()
    .chain([
        // weight_decay 1 x local_rate 0.1 x gain_max 4 = 0.4 is admitted; 4x of it is not.
        (json!({"local_rate":0.1,"balance":{"route":1.0}}), true),
        (json!({"local_rate":0.1,"balance":{"rate":0.1}}), false),
        // Rate and route need anchor-exact credit; memory is admitted in any mode.
        (json!({"credit_mode":"lanes","balance":{"rate":0.5}}), false),
        (
            json!({"credit_mode":"hybrid","balance":{"route":1.0}}),
            false,
        ),
        (
            json!({"credit_mode":"lanes","balance":{"memory":true}}),
            true,
        ),
    ]) {
        let shown = connected.to_string();
        let parsed = config(connected);
        let admitted_now = parsed
            .as_ref()
            .is_ok_and(|config| config.validate().is_ok());
        assert_eq!(admitted_now, admitted, "{shown}");
        if admitted {
            assert_ne!(parsed?.identity()?, identity, "{shown}");
        }
    }
    // Defaults inside the block are omitted, so spelling one out keeps the identity.
    let with_default = config(json!({"balance":{"route":1.0,"horizon":1024}}))?;
    assert_eq!(
        with_default.identity()?,
        config(json!({"balance":{"route":1.0}}))?.identity()?
    );
    let absent = config(json!({}))?.connected.expect("connected options");
    assert!(serde_json::to_value(absent)?["balance"].is_null());
    Ok(())
}

/// Anchor-exact FP32 options with `balance` and `extra` keys, cap 2.
fn options(balance: Value, extra: Value) -> Result<Options> {
    let mut value = json!({"precision":"fp32","lanes":2,"workers":2,"visits":4,
        "credit_mode":"anchor_exact"});
    if !balance.is_null() {
        value["balance"] = balance;
    }
    for (key, item) in extra.as_object().into_iter().flatten() {
        value[key] = item.clone();
    }
    Ok(serde_json::from_value(value)?)
}

fn arm(root: &Path, corpus: &Corpus, options: Options) -> Result<(State, Store)> {
    let geometry = Geometry::fixture();
    let state = State::new(geometry, 7, corpus.cursor(0)?, &options)?;
    let tiles = root.join("connected-tiles");
    let mut store = Store::new_precision(&tiles, geometry, 7, options.workers, options.precision)?;
    store.set_paging(options.resident(), 0)?;
    Ok((state, store))
}

fn train(state: &mut State, store: &mut Store, corpus: &Corpus, anchors: u64) -> Result<()> {
    let target = state.anchors + anchors;
    while state.anchors < target {
        step(state, store, corpus, state.options, 0.003, false)?;
    }
    state.validate()
}

/// A run of `anchors` anchors from scratch under `options`.
fn run(root: &Path, corpus: &Corpus, options: Options, anchors: u64) -> Result<(State, Store)> {
    let (mut state, mut store) = arm(root, corpus, options)?;
    train(&mut state, &mut store, corpus, anchors)?;
    Ok((state, store))
}

/// Largest share of off-centre live visits held by one tile.
fn concentration(state: &State) -> f64 {
    let visits = state.counters.anchor_visits;
    let off: Vec<u64> = (0..TILES)
        .filter(|&t| t != START_TILE)
        .map(|t| visits[t])
        .collect();
    *off.iter().max().unwrap() as f64 / off.iter().sum::<u64>().max(1) as f64
}

/// Checks the report's per-tile estimates are finite; returns a never-visited tile, if any,
/// after checking its visited advantage stayed 0 and its worth is undefined.
fn check_report(state: &State) -> Result<Option<usize>> {
    let report = report(state)?.expect("balance report");
    for key in ["visit_share", "advantage_visited", "advantage_unvisited"] {
        let values = report[key].as_array().expect(key);
        let finite = |v: &Value| v.as_f64().is_some_and(f64::is_finite);
        assert!(values.iter().all(finite), "{key}");
    }
    let estimate = state.balance.as_ref().expect("balance state");
    let unseen = (0..TILES).find(|&tile| estimate.visits[tile] == 0);
    if let Some(tile) = unseen {
        assert_eq!(estimate.advantage_visited[tile], 0.0);
        assert!(report["worth"][tile].is_null());
    }
    Ok(unseen)
}

#[test]
fn balance_rate_route_and_resume_flow() -> Result<()> {
    let _guard = tuning::scoped(Knobs {
        connected: true,
        connected_fp32: true,
        ..Knobs::core(0.003, 7)
    });
    let out = fresh("balance-flow")?;
    let text: Vec<u8> = b"the cat sat on the mat. "
        .iter()
        .cycle()
        .take(512)
        .copied()
        .collect();
    let corpus = corpus(&out.join("corpus"), &text)?;
    // Past the 64-anchor warm-up, so multipliers and bonuses are live.
    let anchors = 80;
    // Rate: a below-mean outer tile learns faster, an above-mean one slower; the always-visited
    // start tile is outside the ratio.
    let (base, _) = run(
        &out.join("base"),
        &corpus,
        options(Value::Null, json!({}))?,
        anchors,
    )?;
    let rated = options(json!({"horizon":64,"rate":1.0}), json!({}))?;
    let (rate, _) = run(&out.join("rate"), &corpus, rated, anchors)?;
    let estimate = rate.balance.as_ref().expect("balance state");
    let outer: Vec<usize> = (0..TILES).filter(|&tile| tile != START_TILE).collect();
    let mean = outer.iter().map(|&tile| estimate.share[tile]).sum::<f64>() / outer.len() as f64;
    let factors: Vec<f64> = (0..TILES)
        .map(|tile| multiplier(&rate.options, Some(estimate), tile))
        .collect::<Result<_>>()?;
    assert!(
        factors.iter().all(|f| (0.25..=4.0).contains(f)),
        "{factors:?}"
    );
    assert_eq!(estimate.share[START_TILE], 1.0);
    assert_eq!(factors[START_TILE], 1.0);
    let quiet = outer.iter().find(|&&tile| estimate.share[tile] < mean);
    let busy = outer.iter().find(|&&tile| estimate.share[tile] > mean);
    let (quiet, busy) = (
        *quiet.expect("below-mean tile"),
        *busy.expect("above-mean tile"),
    );
    assert!(factors[quiet] > 1.0 && factors[busy] < 1.0, "{factors:?}");
    // The multiplied tile steps change training (the fixture's per-anchor losses here do not move).
    let embedding = |state: &State| state.shared_latent.as_ref().map(|l| l.weights.clone());
    assert_ne!(embedding(&rate), embedding(&base));
    // Route: under greedy training routes the bonus spreads off-centre visits.
    let greedy = json!({"route_training":"greedy"});
    let plain = options(Value::Null, greedy.clone())?;
    let (plain, _) = run(&out.join("greedy"), &corpus, plain, anchors)?;
    let routed = options(json!({"horizon":64,"route":2.0}), greedy.clone())?;
    let (routed, _) = run(&out.join("routed"), &corpus, routed, anchors)?;
    assert!(concentration(&routed) < concentration(&plain));
    let steered = routed.balance.as_ref().expect("balance state");
    assert!(steered.changed.iter().sum::<u64>() > 0);
    assert_eq!(
        bonus(&routed.options, Some(steered)).unwrap()[START_TILE],
        0.0
    );
    // Lane draws (also used by evaluation replay) are never tilted.
    assert!(
        super::super::connected_route::lane(&routed.options, &routed, 0, 0)
            .bonus
            .is_none()
    );
    // Memory alone: the same routes keep different records, yet only the last two anchors'
    // records are eligible, so anchor-exact training is unchanged (CONNECTED.md).
    let kept = options(json!({"memory":true}), greedy)?;
    let (mut kept, mut kept_store) = run(&out.join("kept"), &corpus, kept, 1)?;
    // After one anchor of at most four visits some tile is unvisited; its visited mean stays 0.
    check_report(&kept)?.expect("an unvisited tile");
    train(&mut kept, &mut kept_store, &corpus, anchors - 1)?;
    let held = |state: &State| {
        state
            .memory
            .records
            .iter()
            .map(|r| r.epoch)
            .collect::<Vec<_>>()
    };
    assert_eq!(kept.counters.anchor_visits, plain.counters.anchor_visits);
    assert_ne!(held(&kept), held(&plain));
    assert_eq!(kept.counters.loss_sum, plain.counters.loss_sum);
    check_report(&kept)?;
    // Resume: all three uses under waves of 4, resumed after warm-up, reproduce the straight run.
    let all = json!({"horizon":64,"rate":1.0,"route":2.0,"memory":true});
    let all = options(all, json!({"anchor_batch":4}))?;
    let (straight, _) = run(&out.join("straight"), &corpus, all, 72)?;
    let path = out.join("resumed");
    let (first, mut first_store) = run(&path, &corpus, all, 64)?;
    checkpoint::save(&path, &mut first_store, "identity", 64, &first.payloads()?)?;
    drop((first, first_store));
    let loaded = checkpoint::resume(&path, Geometry::fixture(), "identity", 2)?.unwrap();
    let mut resumed = State::restore(&loaded.payloads, Geometry::fixture(), 7)?;
    let mut store = loaded.store;
    store.set_paging(2, 0)?;
    train(&mut resumed, &mut store, &corpus, 8)?;
    assert_eq!(resumed.payloads()?, straight.payloads()?);
    assert!(resumed.balance.is_some() && resumed.memory.balanced);
    Ok(())
}
