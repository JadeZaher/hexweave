//! Core v2 parity, deterministic workers and continuation contracts; see AGENTS.md §Core v2.

use super::checkpoint_state::StateRecord;
use super::core_contracts::{assay, fingerprint, fresh_temp};
use super::core_drill::Drill;
use super::core_schedule::Decay;
use super::execution::{self, TrainingSource};
use super::geometry::{Geometry, TILES};
use super::resident::ResidentStore;
use super::state::State;
use super::tuning::{self, Knobs};
use super::{core_batch, core_checkpoint};
use anyhow::Result;

#[path = "execution_fixture.rs"]
#[allow(clippy::duplicate_mod, dead_code)]
mod fixture;

fn v2(batch: u32) -> Knobs {
    Knobs {
        batch,
        history_ticks: 8,
        fit_predictor: false,
        ..Knobs::core(0.003, 7)
    }
}

fn fresh(drill: &Drill) -> Result<(State, ResidentStore)> {
    let geometry = Geometry::candidate();
    let mut state = State::with_cursor(geometry, 7, drill.cursor(0)?)?;
    state.anchor_only = true;
    Ok((state, ResidentStore::new(geometry, 7)?))
}

/// Compare tiles, E, latents, baseline, loss and correction history; exclude predictor bytes/features.
fn comparable(state: &State, store: &ResidentStore) -> Result<Vec<Vec<u8>>> {
    let mut parts = fingerprint(state, store)?;
    // Parts are the tiles, then shared, predictor, latents, calls, history, baseline, loss.
    parts.remove(TILES + 1);
    let mut history = state.history.clone();
    for record in &mut history.records {
        record.features.clear();
    }
    parts[TILES + 3] = serde_json::to_vec(&history)?;
    Ok(parts)
}

#[test]
fn batch_one_matches_v1_over_twelve_anchors() -> Result<()> {
    let source = assay()?;
    let drill = Drill::new(&source, 7)?;
    for (history, float_forward) in [(true, false), (false, false), (true, true), (false, true)] {
        let legacy = Knobs {
            history,
            float_forward,
            ..Knobs::core(0.003, 7)
        };
        let _guard = tuning::scoped(legacy);
        let (mut state, mut store) = fresh(&drill)?;
        let mut reference = Vec::new();
        for _ in 0..12 {
            execution::step(&mut state, &mut store, &drill, false)?;
            reference.push(comparable(&state, &store)?);
        }
        tuning::install(Knobs {
            history,
            float_forward,
            ..v2(1)
        });
        let (mut state, mut store) = fresh(&drill)?;
        for expected in reference {
            core_batch::step(&mut state, &mut store, &drill, true)?;
            let actual = comparable(&state, &store)?;
            assert!(
                actual == expected,
                "history={history}, float={float_forward}, anchor={}, part={:?}",
                state.global_opportunity,
                actual.iter().zip(&expected).position(|(a, b)| a != b)
            );
        }
    }
    Ok(())
}

#[test]
fn threaded_batch_matches_sequential_and_resumes_exactly() -> Result<()> {
    let source = assay()?;
    let drill = Drill::new(&source, 7)?;
    for (batch, history) in [(4, true), (4, false), (8, true)] {
        let knobs = Knobs {
            history,
            temperature: 2.0,
            attention_scale: 3.0,
            attention_mix: 0.75,
            explore: 0.4,
            decay: Decay::Cosine {
                anchors: 12,
                floor: 0.1,
            },
            ..v2(batch)
        };
        let _guard = tuning::scoped(knobs);
        let (mut sequential, mut store) = fresh(&drill)?;
        for _ in 0..3 {
            core_batch::step(&mut sequential, &mut store, &drill, false)?;
        }
        let expected = fingerprint(&sequential, &store)?;
        let expected_state = serde_json::to_vec(&StateRecord::capture(&sequential)?)?;
        tuning::install(knobs);
        let (mut parallel, mut store) = fresh(&drill)?;
        let out = fresh_temp(if history { "batch-on" } else { "batch-off" })?;
        core_batch::step(&mut parallel, &mut store, &drill, true)?;
        let checkpoint = core_checkpoint::save(&out, &parallel, &store, "v2-test")?;
        assert!(!checkpoint.join("predictor.bin").exists());
        tuning::install(knobs);
        let (mut parallel, mut store) =
            core_checkpoint::load(&checkpoint, Geometry::candidate(), "v2-test")?;
        for _ in 0..2 {
            core_batch::step(&mut parallel, &mut store, &drill, true)?;
        }
        assert_eq!(fingerprint(&parallel, &store)?, expected);
        assert_eq!(
            serde_json::to_vec(&StateRecord::capture(&parallel)?)?,
            expected_state
        );
        std::fs::remove_dir_all(out)?;
    }
    Ok(())
}

#[test]
fn chunked_gradients_match_full_batch_reduction() -> Result<()> {
    let source = assay()?;
    let drill = Drill::new(&source, 7)?;
    let knobs = Knobs {
        explore: 0.4,
        ..v2(8)
    };
    let _guard = tuning::scoped(knobs);
    let (mut state, mut store) = fresh(&drill)?;
    let (positions, _) = core_batch::prepare(&mut state, &mut store, &drill)?;
    for count in [5, 8] {
        let positions = &positions[..count];
        for tile in &store.tiles {
            let reference = super::core_gradients::reference(tile, positions, state.baseline)?;
            for threaded in [false, true] {
                if threaded {
                    tuning::install(Knobs {
                        attention_scale: 3.0,
                        ..knobs
                    });
                }
                let actual =
                    super::core_gradients::mean(tile, positions, state.baseline, knobs, threaded)?;
                tuning::install(knobs);
                for (a, b) in actual.values.iter().zip(&reference.values) {
                    assert!(
                        a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits()),
                        "gradient mismatch: tile {}, positions {count}, threaded {threaded}",
                        tile.id
                    );
                }
            }
        }
    }
    Ok(())
}

#[test]
fn batch_uses_store_updates_and_poison_on_store_failure() -> Result<()> {
    let source = assay()?;
    let drill = Drill::new(&source, 7)?;
    let knobs = v2(4);
    let _guard = tuning::scoped(knobs);
    let (mut state, mut resident) = fresh(&drill)?;
    core_batch::step(&mut state, &mut resident, &drill, false)?;
    let expected = fingerprint(&state, &resident)?;

    tuning::install(knobs);
    let (mut state, _) = fresh(&drill)?;
    let mut store = fixture::Store::with_geometry(Geometry::candidate(), 7)?;
    core_batch::step(&mut state, &mut store, &drill, false)?;
    assert_eq!(
        store.reads.iter().sum::<usize>() as u64,
        2 * state.counters.visits
    );
    assert!(store.updates.iter().any(|count| *count > 0));
    let resident = ResidentStore::from_tiles(store.tiles, Geometry::candidate())?;
    assert_eq!(fingerprint(&state, &resident)?, expected);

    for failure in ["steps", "read", "update"] {
        tuning::install(knobs);
        let (mut state, _) = fresh(&drill)?;
        let mut store = fixture::Store::with_geometry(Geometry::candidate(), 7)?;
        store.fail_steps = failure == "steps";
        store.fail_read = failure == "read";
        store.fail_update = failure == "update";
        assert!(core_batch::step(&mut state, &mut store, &drill, false).is_err());
        assert_eq!(state.boundary, super::state::Boundary::Poisoned);
    }
    Ok(())
}

#[test]
fn v2_history_retains_original_two_anchor_window_at_frozen_weights() -> Result<()> {
    let _guard = tuning::scoped(v2(1));
    let source = assay()?;
    let drill = Drill::new(&source, 7)?;
    let (state, mut store) = fresh(&drill)?;
    let prefix = state.shared.prefix(&drill.pending_prefix(&state.cursor)?)?;
    let trace = execution::forward_path(
        &mut store,
        state.geometry,
        &prefix,
        vec![0.0; state.geometry.width],
        &mut None,
        state.tile_versions,
    )?;
    let mut all = super::replay::Replay::default();
    for byte in b"abc" {
        all.observe(&state, &mut store, &prefix, &trace, *byte)?;
    }
    let mut last = super::replay::Replay::default();
    for byte in b"bc" {
        last.observe(&state, &mut store, &prefix, &trace, *byte)?;
    }
    assert_eq!(all.input(&state)?, last.input(&state)?);
    assert!(all.input(&state)?.iter().any(|value| *value != 0.0));
    Ok(())
}

#[test]
fn batch_config_rejects_invalid_resource_and_resume_boundaries() -> Result<()> {
    use serde_json::json;
    let base = json!({"schema":1,"out":"work/core-runs/config-test",
        "data":{"kind":"drill","root":"datasets/retained-text-assay"},
        "seed":7,"rate":0.003,"anchors":32,"wall_seconds":1,
        "checkpoint_every":5,"eval_every":7,"eval_cases":1,"resume":false,"batch":4,
        "tiles":super::geometry::TILES});
    for (field, value) in [
        ("batch", json!(0)),
        ("batch", json!(33)),
        ("anchors", json!(31)),
        ("float_forward", json!(true)),
        ("explore", json!(1.0)),
        ("decay", json!({"kind":"cosine","anchors":0,"floor":0.1})),
        ("decay", json!({"kind":"cosine","anchors":32,"floor":-0.1})),
        ("decay", json!({"kind":"inverse_sqrt","anchors":0})),
    ] {
        let mut invalid = base.clone();
        invalid[field] = value;
        let config: super::core_config::Config = serde_json::from_value(invalid)?;
        assert!(config.validate().is_err(), "accepted invalid {field}");
    }
    let mut scheduled = base.clone();
    scheduled["decay"] = json!({"kind":"inverse_sqrt","anchors":32});
    serde_json::from_value::<super::core_config::Config>(scheduled)?.validate()?;
    let valid: super::core_config::Config = serde_json::from_value(base)?;
    valid.validate()?;
    Ok(())
}

#[test]
fn exploration_updates_all_tiles_and_schedule_crosses_nondivisible_intervals() -> Result<()> {
    let source = assay()?;
    let drill = Drill::new(&source, 7)?;
    let _guard = tuning::scoped(Knobs {
        explore: 0.9,
        ..v2(4)
    });
    let (mut state, mut store) = fresh(&drill)?;
    for _ in 0..32 {
        core_batch::step(&mut state, &mut store, &drill, false)?;
    }
    assert!(super::core_schedule::tile_steps()
        .iter()
        .all(|step| *step > 0));
    for (before, after, interval, expected) in [
        (0, 4, 5, false),
        (4, 8, 5, true),
        (8, 12, 5, true),
        (12, 16, 5, true),
        (16, 20, 5, true),
    ] {
        assert_eq!(
            super::core_schedule::crossed(before, after, interval),
            expected
        );
    }
    Ok(())
}
