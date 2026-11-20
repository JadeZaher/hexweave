//! Shared-section admission, queue and write-weight rules (pure or config-only).

use super::super::connected_restart;
use super::super::connected_sections_read::{surprise, weights};
use super::super::connected_sections_write::{ratio, select, settle};
use super::super::connected_tile_signal::WARMUP;
use super::super::core_config::Config;
use super::*;
use serde_json::{json, Value};

/// A run config with `connected` keys over an anchor-exact base of eight anchors.
pub(super) fn config(connected: Value) -> Result<Config> {
    let mut value = json!({"schema":1,"out":"work/core-runs/connected-shared-test",
        "data":{"kind":"corpus","root":"work/corpora/test","manifest_sha256":"test","window":32},
        "seed":7,"rate":0.003,"anchors":8,"wall_seconds":60,"checkpoint_every":8,
        "eval_every":8,"eval_cases":1,"resume":false,"tiles":TILES,
        "connected":{"credit_mode":"anchor_exact"}});
    for (key, item) in connected.as_object().into_iter().flatten() {
        value["connected"][key] = item.clone();
    }
    Ok(serde_json::from_value(value)?)
}

#[test]
fn shared_admission_rows_and_identity() -> Result<()> {
    let identity = config(json!({}))?.identity()?;
    let minimal = config(json!({"shared":{"sections":2}}))?.identity()?;
    let (none, lanes) = (json!({}), json!({"credit_mode":"lanes"}));
    let batch = json!({"anchor_batch":4});
    let replay = json!({"replay":{"capacity":8,"per_anchor":1}});
    // (other connected keys, shared block, admitted); every admitted block changes the identity.
    for (connected, admitted) in [
        (&none, json!({"sections":1}), true),
        (
            &none,
            json!({"sections":4,"wave":4,"queue":false,"teach":0.5}),
            true,
        ),
        (&batch, json!({"sections":1,"wave":8}), true),
        (&none, json!({"sections":0}), false),
        (&none, json!({"sections":5}), false),
        (&none, json!({"wave":8}), false),
        (&none, json!({"sections":1,"wave":0}), false),
        (&none, json!({"sections":1,"wave":65}), false),
        // The wave must be a multiple of anchor_batch, and run boundaries of the wave.
        (&batch, json!({"sections":1,"wave":6}), false),
        (&none, json!({"sections":1,"wave":16}), false),
        (&none, json!({"sections":1,"teach":-0.1}), false),
        (&none, json!({"sections":1,"teach":0.51}), false),
        (&none, json!({"sections":1,"credit":"entry"}), true),
        (&none, json!({"sections":1,"credit":"exact"}), false),
        (&lanes, json!({"sections":1}), false),
        (&replay, json!({"sections":1}), false),
        (&none, json!({"sections":1,"extra":1}), false),
    ]
    .map(|(other, shared, admitted)| {
        let mut connected = other.clone();
        connected["shared"] = shared;
        (connected, admitted)
    }) {
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
    // Defaults spelled out keep the identity; absent is omitted from it.
    let spelled = json!({"shared":{"sections":2,"wave":8,"queue":true,"teach":0.0,"credit":"mix"}});
    assert_eq!(config(spelled)?.identity()?, minimal);
    let absent = config(json!({}))?.connected.expect("connected options");
    assert!(serde_json::to_value(absent)?["shared"].is_null());
    Ok(())
}

#[test]
fn queue_reproduces_the_priority_trace_and_greedy_match() {
    // Constant matches 0.6/0.3/0.1, r = 1: wave 4 ties A and C; C was written longest ago.
    let matches = [0.6, 0.3, 0.1];
    for (queue, expected) in [(true, [0, 1, 0, 2, 0, 1]), (false, [0; 6])] {
        let (mut priorities, mut last, mut winners) = ([0.0; 3], [None; 3], Vec::new());
        for wave in 0..6 {
            let winner = select(&mut priorities, &last, &matches, queue);
            if queue {
                settle(&mut priorities[winner], &matches, ratio(1.0, None));
            }
            last[winner] = Some(wave);
            winners.push(winner);
        }
        assert_eq!(winners, expected, "queue {queue}");
    }
    // r = clamp(size / EMA, 0.5, 2), 1 on a first write.
    let (first, high, low) = (
        ratio(3.0, None),
        ratio(5.0, Some(1.0)),
        ratio(0.1, Some(1.0)),
    );
    assert_eq!([first, high, low], [1.0, 2.0, 0.5]);
}

#[test]
fn write_weights_sum_to_one_and_follow_lambert_surprise() -> Result<()> {
    for visits in 1..=4usize {
        let rising: Vec<f64> = (0..visits).map(|index| 0.5 * index as f64).collect();
        // (surprise, weights equal, Lambert used)
        for (surprise, equal, lambert) in [
            (None, true, false),
            (Some(vec![0.0; visits]), true, false),
            (Some(vec![0.7; visits]), true, true),
            (Some(rising), visits == 1, visits > 1),
        ] {
            let (weights, used) = weights(surprise.as_deref(), visits)?;
            assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);
            assert_eq!(used, lambert, "{visits} {surprise:?}");
            let uniform = weights
                .iter()
                .all(|w| (w - 1.0 / visits as f64).abs() < 1e-15);
            assert_eq!(uniform, equal, "{visits} {surprise:?}");
            if !equal {
                assert!(weights.windows(2).all(|pair| pair[0] < pair[1]));
            }
        }
    }
    Ok(())
}

#[test]
fn lambert_weights_follow_warm_tile_surprise_and_fall_back_when_one_is_cold() -> Result<()> {
    let options: Options = serde_json::from_value(json!({"credit_mode":"anchor_exact",
        "restart":{"floor":0.1,"threshold":0.5,"signal":"tile","law":"lambert"}}))?;
    let mut tiles = connected_restart::fresh(&options).expect("restart state");
    // Warm signals at distances 0.1, 0.4 and 0.2 from the reference [1, 0]; bar 0.5.
    for (tile, ema) in [(0, 1.1), (1, 1.4), (2, 0.8)] {
        let signal = &mut tiles[tile].signal;
        (signal.input_ema, signal.reference) = (vec![ema, 0.0], vec![1.0, 0.0]);
        signal.visits = WARMUP;
    }
    let warm = surprise(Some(tiles.as_slice()), &[0, 1, 2]).expect("warm surprise");
    let (lambert, used) = weights(Some(&warm), 3)?;
    assert!(used && (lambert.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    assert!(
        lambert[1] > lambert[2] && lambert[2] > lambert[0],
        "{lambert:?}"
    );
    // A cold visited tile (3 never warmed) withdraws the surprise: equal weights.
    let cold = surprise(Some(tiles.as_slice()), &[0, 3]);
    assert!(cold.is_none() && weights(cold.as_deref(), 2)? == (vec![0.5, 0.5], false));
    Ok(())
}
