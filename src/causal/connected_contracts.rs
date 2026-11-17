//! Connected-engine causal, concurrency and continuation regressions; see AGENTS.md.

use super::connected_config::Options;
use super::connected_signal::{Kind, Record};
use super::connected_state::State;
use super::connected_store::Store;
use super::core_schedule::Decay;
use super::geometry::{Geometry, TILES};
use super::scratch_dir::ScratchDir;
use super::{connected_checkpoint, connected_lanes, connected_step, tuning};
use anyhow::Result;
use std::path::Path;

#[path = "execution_fixture.rs"]
#[allow(clippy::duplicate_mod, dead_code)]
mod fixture;

fn fresh(path: &Path, source: &fixture::Source, options: Options) -> Result<(State, Store)> {
    let geometry = Geometry::fixture();
    Ok((
        State::new(geometry, 7, source.cursor(), &options)?,
        Store::new_precision(path, geometry, 7, options.workers, options.precision)?,
    ))
}

#[test]
fn connected_threading_and_checkpoint_continuation_preserve_the_complete_state() -> Result<()> {
    use super::connected_precision::Precision;
    // The FP32 variant schedules its local rate, so resume must restore the anchor index too.
    for (precision, local_decay) in [
        (Precision::Ternary, Decay::Constant),
        (Precision::Fp32, Decay::InverseSqrt { anchors: 2 }),
    ] {
        continuation(Options {
            precision,
            local_decay,
            ..Options::default()
        })?;
    }
    Ok(())
}

fn continuation(options: Options) -> Result<()> {
    let _guard = tuning::scoped(tuning::Knobs {
        connected: true,
        connected_fp32: !options.precision.is_ternary(),
        ..tuning::Knobs::core(0.003, 7)
    });
    let path = ScratchDir::new("connected-engine-continuation")?;
    let source = fixture::Source::new();
    let (mut sequential, mut seq_store) = fresh(&path.join("sequential"), &source, options)?;
    seq_store.set_eager(true)?;
    for _ in 0..3 {
        connected_step::step(
            &mut sequential,
            &mut seq_store,
            &source,
            options,
            0.003,
            false,
        )?;
    }
    assert_eq!(source.target_reads.get(), 3);
    assert_eq!(source.advances.get(), 3);
    // Parity covers trusted mean steps (at most one per tile per anchor) and zero-trust skips.
    assert!((1..=3 * TILES as u64).contains(&sequential.counters.local_updates));
    assert!(sequential.counters.cross_lane_reads > 0);
    assert!(sequential.counters.collision_deferrals > 0);
    let source = fixture::Source::new();
    let (mut threaded, mut thread_store) = fresh(&path.join("threaded"), &source, options)?;
    for _ in 0..2 {
        connected_step::step(
            &mut threaded,
            &mut thread_store,
            &source,
            options,
            0.003,
            true,
        )?;
    }
    let out = path.join("run");
    connected_checkpoint::save(
        &out,
        &mut thread_store,
        "fixture-connected",
        threaded.anchors,
        &threaded.payloads()?,
    )?;
    drop(thread_store);
    drop(threaded);
    let loaded = connected_checkpoint::resume(
        &out,
        Geometry::fixture(),
        "fixture-connected",
        options.workers,
    )?
    .unwrap();
    let mut continued = State::restore(&loaded.payloads, Geometry::fixture(), 7)?;
    let mut continued_store = loaded.store;
    let before = continued.counters.local_updates;
    connected_step::step(
        &mut continued,
        &mut continued_store,
        &source,
        options,
        0.003,
        true,
    )?;
    // The resumed anchor (index 2) takes local steps, so its scheduled rate is really exercised.
    assert!(
        continued.counters.local_updates > before,
        "no local step at the resumed anchor"
    );
    assert_eq!(sequential.payloads()?, continued.payloads()?);
    compare_stores(&mut seq_store, &mut continued_store, options)?;
    drop(continued_store);
    drop(seq_store);
    Ok(())
}

fn compare_stores(
    seq_store: &mut Store,
    continued_store: &mut Store,
    options: Options,
) -> Result<()> {
    seq_store.flush()?;
    continued_store.flush()?;
    let meta = |store: &Store| {
        store
            .slots
            .iter()
            .map(|slot| slot.meta.clone())
            .collect::<Vec<_>>()
    };
    assert_eq!(meta(seq_store), meta(continued_store));
    assert!(seq_store.stats().peak_live <= options.workers);
    assert!(continued_store.stats().peak_live <= options.workers);
    assert!(continued_store.stats().peak_resident <= options.workers);
    assert!(continued_store.stats().cache_hits > 0);
    let view_bound = if options.precision.is_ternary() {
        0
    } else {
        options.workers * Geometry::fixture().tile_coefficients(3)? * 4
    };
    assert!(continued_store.stats().peak_view_bytes <= view_bound);
    assert!(
        seq_store.stats().peak_live > 1,
        "fixture must exercise multiple simultaneous tile leases"
    );
    continued_store.quiescent()?;
    Ok(())
}

/// Consistent observed corrections under one label give every tile positive trust.
fn prefit(state: &mut State, label: &[f32]) -> Result<()> {
    let d = state.geometry.width;
    let mut correction = vec![0.0; d];
    correction[0] = 0.2;
    for _ in 0..24 {
        for tile_id in 0..TILES {
            let record = Record {
                input: vec![0.25; d],
                output: vec![0.5; d],
                history: vec![0.0; d],
                correction: correction.clone(),
                tile_id,
                revision: 0,
                shared_revision: 0,
                epoch: 0,
                kind: Kind::Observed,
                lane: None,
            };
            state.predictor.fit(&record, label, 0.003)?;
        }
    }
    assert!((0..TILES).all(|tile| state.predictor.trust(tile) > 0.0));
    Ok(())
}

#[test]
fn predicted_credit_changes_real_local_weights_and_shared_memory_changes_tile_behavior(
) -> Result<()> {
    let _guard = tuning::scoped(tuning::Knobs {
        connected: true,
        ..tuning::Knobs::core(0.003, 7)
    });
    let path = ScratchDir::new("connected-engine-connections")?;
    let options = Options::default();
    let source = fixture::Source::new();
    let (mut predicted, mut predicted_store) = fresh(&path.join("predicted"), &source, options)?;
    let (mut zero, mut zero_store) = fresh(&path.join("zero"), &source, options)?;
    // An already-read-out byte's label feature: lanes get the byte, never a task target read.
    let label = predicted.shared.target_feature(b'!')?;
    prefit(&mut predicted, &label)?;
    zero.predictor = predicted.predictor.clone();
    zero.predictor.weights.fill(0.0);
    let prefix = predicted.shared.prefix(b"?")?;
    for (state, store) in [
        (&mut predicted, &mut predicted_store),
        (&mut zero, &mut zero_store),
    ] {
        connected_lanes::run(state, store, &prefix, b'!', options, true)?;
    }
    assert_eq!(
        source.target_reads.get(),
        0,
        "local work must not obtain a task target"
    );
    // Colliding lanes read fixed tiles (revision 0); then each visited tile takes one mean step.
    let counters = &predicted.counters;
    let visited = counters.local_visits.map(|count| u64::from(count > 0));
    let revisions = std::array::from_fn(|id| predicted_store.slots[id].meta.revision);
    let visits: u64 = counters.local_visits.iter().sum();
    assert_eq!(revisions, visited, "visited tiles commit once");
    assert_eq!(counters.local_updates, visited.iter().sum::<u64>());
    assert!(visits > counters.local_updates, "lanes must share tiles");
    let records = &predicted.memory.records;
    assert!(records.iter().all(|record| record.revision == 0));
    let predicted_lease = predicted_store.checkout(3)?;
    let zero_lease = zero_store.checkout(3)?;
    let weights = |lease: &super::connected_store::Lease| {
        lease
            .latents()
            .iter()
            .map(|(key, latent)| (*key, latent.weights.clone()))
            .collect::<Vec<_>>()
    };
    assert_ne!(
        weights(&predicted_lease),
        weights(&zero_lease),
        "predictor must change the persistent tile master weights"
    );
    let initial = super::tile::Tile::new(Geometry::fixture(), 3, 7)?;
    for ((_, matrix), latent) in zero_lease.latents() {
        let expected = (0..initial.matrices[*matrix].len())
            .map(|index| initial.matrices[*matrix].value(index).map(f32::from))
            .collect::<Result<Vec<_>>>()?;
        // Decoupled decay may shrink a master toward zero; any credit would move it elsewhere.
        let decayed = |(w, e): (&f32, &f32)| w * e >= 0.0 && w.abs() <= e.abs();
        assert!(
            latent.weights.iter().zip(&expected).all(decayed),
            "zero credit moved master weights"
        );
    }
    predicted_store.return_read(predicted_lease)?;
    zero_store.return_read(zero_lease)?;
    memory_changes_tile_behavior(&predicted, &mut predicted_store, &prefix, &revisions)?;
    drop(predicted_store);
    drop(zero_store);
    Ok(())
}

/// Published memory is retrieved and changes the center tile's output and route policy.
fn memory_changes_tile_behavior(
    state: &State,
    store: &mut Store,
    prefix: &super::shared::Prefix,
    revisions: &[u64; TILES],
) -> Result<()> {
    let input = prefix.initial(Geometry::fixture());
    let memory = state.memory.query(input, 3, state.epoch, revisions, 0)?;
    assert!(memory.iter().any(|value| *value != 0.0));
    let lease = store.checkout(3)?;
    let with_memory = lease.tile().forward(input, &prefix.memory, &memory)?;
    let without_memory = lease
        .tile()
        .forward(input, &prefix.memory, &vec![0.0; input.len()])?;
    assert_ne!(with_memory.output, without_memory.output);
    assert_ne!(
        super::routing::probabilities(
            lease.tile(),
            &with_memory.output,
            &with_memory.heads,
            &memory
        )?,
        super::routing::probabilities(
            lease.tile(),
            &without_memory.output,
            &without_memory.heads,
            &vec![0.0; input.len()]
        )?,
    );
    store.return_read(lease)
}
