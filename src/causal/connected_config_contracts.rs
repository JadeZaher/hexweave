//! Connected run identity and admission tables.

use super::*;
use anyhow::Context;
use serde_json::{json, Value};

/// A valid connected corpus config with default connected options.
fn base() -> Value {
    json!({"schema":1,"out":"work/core-runs/connected-config-test",
        "data":{"kind":"corpus","root":"work/corpora/test","manifest_sha256":"test","window":32},
        "seed":7,"rate":0.003,"anchors":8,"wall_seconds":60,"checkpoint_every":8,
        "eval_every":8,"eval_cases":1,"resume":false,"connected":{},"tiles":TILES})
}

/// A plasticity block whose last tile's dial is `last`, every other tile at 1.
fn dial(last: f64) -> Value {
    let mut values = vec![1.0; TILES];
    values[TILES - 1] = last;
    json!({"floor":0.1,"plasticity":values})
}

/// Whether admission refuses `base()` with `connected[key]` set to `value`.
fn refused(key: &str, value: Value) -> Result<bool> {
    let mut invalid = base();
    invalid["connected"][key] = value;
    Ok(serde_json::from_value::<Config>(invalid)?
        .validate()
        .is_err())
}

/// A valid Lambert-law restart block with `key` added.
fn lambert(key: &str, value: Value) -> Value {
    let mut block = json!({"floor":0.1,"threshold":0.5,"signal":"tile","law":"lambert"});
    block[key] = value;
    block
}

#[test]
fn connected_identity_and_admission_cover_execution_policy() -> Result<()> {
    let base = base();
    let config: Config = serde_json::from_value(base.clone())?;
    config.validate()?;
    let identity = config.identity()?;
    assert!(serde_json::to_value(config.connected.unwrap())?["precision"].is_null());
    assert!(serde_json::to_value(config.knobs())?["connected_fp32"].is_null());
    for (key, value) in [
        ("precision", json!("ternary")),
        ("write_behind_tiles", json!(0)),
        ("history", json!(true)),
        // Paging controls change no numerics, so they never enter run identity.
        ("write_behind_tiles", json!(8)),
        ("resident_tiles", json!(TILES)),
        ("route_training", json!("sampled")),
        ("route_explore", json!(0.0)),
        ("memory_lag", json!(1)),
        ("visit_norm", json!(true)),
        ("context_rows", json!(false)),
    ] {
        let mut same = base.clone();
        same["connected"][key] = value;
        let config: Config = serde_json::from_value(same)?;
        config.validate()?;
        assert_eq!(identity, config.identity()?);
    }
    let mut measured = base.clone();
    measured["connected_eval"] =
        json!({"budget_seconds":180,"retention_spans":6,"sample_bytes":128});
    let measurement_config: Config = serde_json::from_value(measured.clone())?;
    measurement_config.validate()?;
    assert_eq!(identity, measurement_config.identity()?);
    measured["connected"] = serde_json::Value::Null;
    assert!(serde_json::from_value::<Config>(measured)?
        .validate()
        .is_err());
    for (key, value) in [
        ("lanes", json!(4)),
        ("workers", json!(2)),
        ("gain_max", json!(2.0)),
        ("predictor_rate", json!(0.001)),
        ("precision", json!("fp32")),
        ("weight_decay", json!(0.02)),
        ("master_rms_cap", json!(2.0)),
        ("local_rate", json!(0.001)),
        ("local_decay", json!({"kind":"inverse_sqrt","anchors":50})),
        ("credit_mode", json!("anchor_exact")),
        ("slice_mode", json!("windowed")),
        ("history", json!(false)),
        ("restart", json!({"floor":0.1})),
        ("restart", json!({"floor":0.0,"threshold":0.5,"adapt":0.25})),
        (
            "restart",
            json!({"floor":0.1,"threshold":0.5,"forget":false}),
        ),
        (
            "restart",
            json!({"floor":0.1,"threshold":0.5,"signal":"tile"}),
        ),
        ("restart", json!({"floor":0.1,"threshold":0.5,"stagger":16})),
        (
            "restart",
            json!({"floor":0.1,"threshold":0.5,"adapt":0.25,"clamp":1024.0}),
        ),
        ("restart", dial(4.0)),
        ("route_training", json!("greedy")),
        (
            "restart",
            json!({"floor":0.1,"threshold":0.5,"signal":"tile","law":"lambert"}),
        ),
        ("memory_lag", json!(64)),
        ("visit_norm", json!(false)),
    ] {
        let mut changed = base.clone();
        changed["connected"][key] = value;
        let config: Config = serde_json::from_value(changed)?;
        config.validate()?;
        assert_ne!(identity, config.identity()?);
    }
    for (key, value) in [
        ("float_forward", json!(true)),
        ("history", json!(false)),
        ("batch", json!(2)),
        ("explore", json!(0.1)),
        ("eval_spans", json!(129)),
        ("eval_cases", json!(5)),
    ] {
        let mut invalid = base.clone();
        invalid[key] = value;
        assert!(serde_json::from_value::<Config>(invalid)?
            .validate()
            .is_err());
    }
    for (key, value) in [
        ("lanes", json!(9)),
        ("workers", json!(5)),
        ("gain_min", json!(0.0)),
        ("memory_capacity", json!(65)),
        ("weight_decay", json!(0.0)),
        ("weight_decay", json!(1.5)),
        ("master_rms_cap", json!(0.5)),
        ("master_rms_cap", json!(9.0)),
        ("write_behind_tiles", json!(65)),
        ("resident_tiles", json!(3)),
        ("resident_tiles", json!(TILES + 1)),
        ("route_explore", json!(0.25)),
        ("memory_lag", json!(0)),
        ("memory_lag", json!(65)),
        ("replay", json!({"capacity":64,"per_anchor":2})),
        ("context_rows", json!(true)),
        ("restart", json!({"floor":-0.1})),
        ("restart", json!({"floor":1.5})),
        ("restart", json!({"floor":0.1,"threshold":0.0})),
        ("restart", json!({"floor":0.1,"threshold":-1.0})),
        ("restart", json!({"floor":0.1,"threshold":0.5,"adapt":1.0})),
        ("restart", json!({"floor":0.1,"threshold":0.5,"adapt":-0.1})),
        // Adaptation without a threshold would silently do nothing.
        ("restart", json!({"floor":0.1,"adapt":0.5})),
        // So would each per-restart option.
        ("restart", json!({"floor":0.1,"forget":false})),
        ("restart", json!({"floor":0.1,"signal":"tile"})),
        ("restart", json!({"floor":0.1,"stagger":4})),
        ("restart", json!({"floor":0.1,"threshold":0.5,"clamp":0.5})),
        (
            "restart",
            json!({"floor":0.1,"threshold":0.5,"clamp":1025.0}),
        ),
        ("restart", dial(4.5)),
        ("restart", dial(-0.1)),
        // A zero dial would hand Adam rate 0; so would a Lambert dial at 0 over floor 0.
        ("restart", dial(0.0)),
        ("restart", lambert("floor", json!(0.0))),
        // Restart replay needs anchor_exact (the base is lanes).
        ("restart", json!({"floor":0.1,"threshold":0.5,"replay":4})),
        // The Lambert law needs a tile signal and a bar and refuses discrete-restart options.
        (
            "restart",
            json!({"floor":0.1,"threshold":0.5,"law":"lambert"}),
        ),
        (
            "restart",
            json!({"floor":0.1,"signal":"tile","law":"lambert"}),
        ),
        ("restart", lambert("adapt", json!(0.25))),
        ("restart", lambert("stagger", json!(2))),
        ("restart", lambert("clamp", json!(4.0))),
        ("restart", lambert("forget", json!(false))),
    ] {
        assert!(refused(key, value)?, "accepted {key}");
    }
    // Replay: in-range under anchor_exact changes identity; bounds are refused there too.
    let mut exact = base.clone();
    exact["connected"]["credit_mode"] = json!("anchor_exact");
    let exact_identity = serde_json::from_value::<Config>(exact.clone())?.identity()?;
    // Restart replay: admitted 0..=6 under anchor_exact, never under the Lambert law; seven
    // steps plus the anchor's route and local commits would pass the revision lag of 8.
    for (restart, admitted) in [
        (json!({"floor":0.1,"threshold":0.5,"replay":6}), true),
        (json!({"floor":0.1,"threshold":0.5,"replay":7}), false),
        (lambert("replay", json!(1)), false),
    ] {
        let mut changed = exact.clone();
        changed["connected"]["restart"] = restart;
        let config: Config = serde_json::from_value(changed)?;
        assert_eq!(config.validate().is_ok(), admitted);
    }
    // The Lambert law ignores local_decay, so only the constant schedule is admitted with it.
    for (decay, admitted) in [
        (json!({"kind":"constant"}), true),
        (json!({"kind":"inverse_sqrt","anchors":50}), false),
    ] {
        let mut changed = base.clone();
        changed["connected"]["restart"] = lambert("floor", json!(0.1));
        changed["connected"]["local_decay"] = decay;
        let config: Config = serde_json::from_value(changed)?;
        assert_eq!(config.validate().is_ok(), admitted);
    }
    for (replay, admitted) in [
        (json!({"capacity":64,"per_anchor":2}), true),
        (json!({"capacity":0,"per_anchor":2}), false),
        (json!({"capacity":4097,"per_anchor":2}), false),
        (json!({"capacity":64,"per_anchor":0}), false),
        (json!({"capacity":64,"per_anchor":9}), false),
    ] {
        let mut changed = exact.clone();
        changed["connected"]["replay"] = replay;
        let config: Config = serde_json::from_value(changed)?;
        assert_eq!(config.validate().is_ok(), admitted);
        if admitted {
            assert_ne!(exact_identity, config.identity()?);
        }
    }
    // Each value is in range, but a step realized at gain_max 4 would keep 1 - 1.0 * 0.5 * 4 < 0.
    let mut decaying = base.clone();
    decaying["rate"] = json!(0.5);
    decaying["connected"]["weight_decay"] = json!(1.0);
    assert!(serde_json::from_value::<Config>(decaying)?
        .validate()
        .is_err());
    Ok(())
}

#[test]
fn local_rate_and_schedule_admission_refuse_unusable_values() -> Result<()> {
    // Out-of-range rates; the shared schedule rules (zero anchors; a cosine floor above 1,
    // which only `Decay::validate` refuses); and a schedule reaching zero (floor 0, which
    // only the positive-factor guard refuses).
    for (key, value) in [
        ("local_rate", json!(0.0)),
        ("local_rate", json!(0.2)),
        ("local_decay", json!({"kind":"inverse_sqrt","anchors":0})),
        (
            "local_decay",
            json!({"kind":"cosine","anchors":9,"floor":1.5}),
        ),
        (
            "local_decay",
            json!({"kind":"cosine","anchors":9,"floor":0.0}),
        ),
    ] {
        let shown = value.to_string();
        assert!(refused(key, value)?, "accepted {key} {shown}");
    }
    Ok(())
}

#[test]
fn default_restart_options_and_state_keep_the_earlier_identity_and_record() -> Result<()> {
    // The live run's block, bare and with every new key at its default, has one identity.
    let live = json!({"floor":0.25,"threshold":2.0,"adapt":0.25});
    let mut bare = base();
    bare["connected"]["restart"] = live.clone();
    let bare: Config = serde_json::from_value(bare)?;
    let mut spelled = base();
    spelled["connected"]["restart"] = json!({"floor":0.25,"threshold":2.0,"adapt":0.25,
        "forget":true,"signal":"shared","stagger":0,"clamp":8.0,"replay":0,"law":"restart"});
    let spelled: Config = serde_json::from_value(spelled)?;
    assert_eq!(bare.identity()?, spelled.identity()?);
    let options = bare.connected.context("connected")?;
    assert_eq!(serde_json::to_value(options.restart)?, live);
    // A fresh tile serializes exactly the earlier TileRestart keys, so its record is unchanged.
    let tiles = super::super::connected_restart::fresh(&options).context("restart")?;
    let record = serde_json::to_value(&tiles[0])?;
    let mut keys: Vec<&String> = record.as_object().context("object")?.keys().collect();
    keys.sort();
    let earlier = [
        "bar",
        "cycle_anchors",
        "cycle_loss",
        "drift",
        "forgotten_records",
        "origin",
        "previous_mean",
        "restarts",
    ];
    assert_eq!(keys, earlier);
    Ok(())
}
