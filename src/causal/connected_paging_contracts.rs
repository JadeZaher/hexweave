//! Write-behind paging parity and prioritized replay / history-ablation flows.

use super::connected_bias_contracts::corpus;
use super::connected_checkpoint as checkpoint;
use super::connected_config::{CreditMode, Options, Replay};
use super::connected_eval::frozen_histories;
use super::connected_precision::Precision;
use super::connected_replay::replay_one;
use super::connected_state::State;
use super::connected_step;
use super::connected_storage_contracts::fresh;
use super::connected_store::{Store, TileMeta};
use super::core_corpus::Corpus;
use super::geometry::Geometry;
use super::tile_gradients::TileGradients;
use super::tuning::{self, Knobs};
use anyhow::Result;
use std::path::Path;

/// One trained arm: per-anchor live losses, final state and its store.
struct Arm {
    losses: Vec<f32>,
    state: State,
    store: Store,
}

fn train(out: &Path, corpus: &Corpus, options: Options, anchors: usize) -> Result<Arm> {
    let geometry = Geometry::fixture();
    let mut state = State::new(geometry, 7, corpus.cursor(0)?, &options)?;
    let mut store = Store::new_precision(
        &out.join("connected-tiles"),
        geometry,
        7,
        options.workers,
        options.precision,
    )?;
    store.set_paging(options.resident(), options.write_behind_tiles)?;
    let mut losses = Vec::new();
    for _ in 0..anchors {
        losses.push(connected_step::step(
            &mut state, &mut store, corpus, options, 0.003, false,
        )?);
    }
    Ok(Arm {
        losses,
        state,
        store,
    })
}

fn metadata(store: &Store) -> Vec<TileMeta> {
    store.slots.iter().map(|slot| slot.meta.clone()).collect()
}

fn base() -> Options {
    Options {
        precision: Precision::Fp32,
        lanes: 2,
        workers: 2,
        visits: 2,
        ..Options::default()
    }
}

#[test]
fn write_behind_matches_eager_eviction_with_fewer_disk_writes() -> Result<()> {
    let _guard = tuning::scoped(Knobs {
        connected: true,
        connected_fp32: true,
        ..Knobs::core(0.003, 7)
    });
    let out = fresh("write-behind")?;
    let train_bytes: Vec<u8> = b"the cat sat. ".iter().cycle().take(512).copied().collect();
    let corpus = corpus(&out.join("corpus"), &train_bytes)?;
    let mut reference = train(&out.join("reference"), &corpus, base(), 12)?;
    reference.store.flush()?;
    // Three pending bundles force full-buffer spills; a wider cap trades RAM for fewer evictions.
    for (name, resident) in [("deferred", None), ("wide", Some(4))] {
        let options = Options {
            write_behind_tiles: 3,
            resident_tiles: resident,
            ..base()
        };
        let arm_out = out.join(name);
        let mut arm = train(&arm_out, &corpus, options, 12)?;
        let pending = arm.store.stats();
        assert!(pending.writes_deferred > 0, "{name}: nothing deferred");
        if resident.is_none() {
            assert!(
                pending.pending_hits > 0,
                "{name}: no checkout decoded from RAM"
            );
        }
        // Checkpoint publication drains every pending bundle before copying and hashing tiles.
        checkpoint::save(
            &arm_out,
            &mut arm.store,
            "identity",
            12,
            &arm.state.payloads()?,
        )?;
        let stats = arm.store.stats();
        assert_eq!(arm.losses, reference.losses, "{name}");
        assert_eq!(metadata(&arm.store), metadata(&reference.store), "{name}");
        assert!(stats.writes < reference.store.stats().writes, "{name}");
        assert!(
            stats.peak_resident <= options.resident() && stats.resident_cap == options.resident()
        );
    }
    assert!(reference.store.stats().peak_resident <= 2);
    Ok(())
}

#[test]
fn replay_adds_tile_credit_only_and_history_ablation_zeroes_the_input() -> Result<()> {
    let _guard = tuning::scoped(Knobs {
        connected: true,
        connected_fp32: true,
        ..Knobs::core(0.003, 7)
    });
    let out = fresh("replay")?;
    let train_bytes: Vec<u8> = b"abacabad".iter().cycle().take(512).copied().collect();
    let corpus = corpus(&out.join("corpus"), &train_bytes)?;
    let exact = Options {
        credit_mode: CreditMode::AnchorExact,
        ..base()
    };
    let replay = Options {
        replay: Some(Replay {
            capacity: 4,
            per_anchor: 2,
        }),
        ..exact
    };
    let ablated = Options {
        history: false,
        ..exact
    };
    let mut plain = train(&out.join("plain"), &corpus, exact, 8)?;
    let mut replayed = train(&out.join("replayed"), &corpus, replay, 8)?;
    let blind = train(&out.join("blind"), &corpus, ablated, 8)?;
    // An empty buffer cannot change the first anchor; after it, replayed tile credit does.
    assert_eq!(plain.losses[0], replayed.losses[0]);
    assert_ne!(plain.losses, replayed.losses);
    plain.store.flush()?;
    replayed.store.flush()?;
    assert_ne!(metadata(&plain.store), metadata(&replayed.store));
    // Replays never advance the cursor, publish, fit, step E/bias or count as training loss.
    let (live, rerun) = (&plain.state, &replayed.state);
    rerun.validate()?;
    assert_eq!((live.anchors, &live.cursor), (rerun.anchors, &rerun.cursor));
    assert_eq!(
        live.counters.observed_records,
        rerun.counters.observed_records
    );
    assert_eq!(live.memory.publications, rerun.memory.publications);
    assert_eq!(live.predictor.fits, rerun.predictor.fits);
    let live_sum: f64 = replayed.losses.iter().map(|loss| f64::from(*loss)).sum();
    assert_eq!(rerun.counters.loss_sum, live_sum);
    // Anchor 1 replays its single entry, anchors 2..8 replay two each; capacity bounds the buffer.
    let report = rerun.replay.report();
    assert_eq!(report["replay_anchors"], 13);
    assert_eq!(report["buffer_size"], 4);
    // The ablation still queries and publishes, but every forward history input is zero.
    assert!(blind.state.counters.memory_reads > 0 && !blind.state.memory.records.is_empty());
    let nonzero = |state: &State| {
        state
            .memory
            .records
            .iter()
            .any(|record| record.history.iter().any(|value| *value != 0.0))
    };
    assert!(nonzero(live) && !nonzero(&blind.state));
    // Frozen evaluation/generation replay zeroes the same input after observing two bytes.
    let observed: [(&[u8], u8); 2] = [(b"abac", b'a'), (b"abaca", b'b')];
    let frozen = |arm: &mut Arm| {
        frozen_histories(&arm.state, &mut arm.store, &observed, b"abacab").map(|histories| {
            histories
                .iter()
                .any(|history| history.iter().any(|value| *value != 0.0))
        })
    };
    let mut blind = blind;
    assert!(frozen(&mut plain)? && !frozen(&mut blind)?);
    Ok(())
}

#[test]
fn replay_never_retrieves_records_from_the_replayed_anchor_onward() -> Result<()> {
    let _guard = tuning::scoped(Knobs {
        connected: true,
        connected_fp32: true,
        ..Knobs::core(0.003, 7)
    });
    let out = fresh("replay-cutoff")?;
    let train_bytes: Vec<u8> = b"abacabad".iter().cycle().take(512).copied().collect();
    let corpus = corpus(&out.join("corpus"), &train_bytes)?;
    let options = Options {
        credit_mode: CreditMode::AnchorExact,
        ..base()
    };
    let mut arm = train(&out.join("cutoff"), &corpus, options, 6)?;
    // Anchor 5's records hold corrections from its own target: replaying 5 must not see them.
    let position = arm.state.anchors - 1;
    let gradients = |frames: Vec<(usize, TileGradients)>| -> Vec<(usize, Vec<Vec<f32>>)> {
        let values = |gradient: TileGradients| gradient.values.to_vec();
        frames
            .into_iter()
            .map(|(tile, gradient)| (tile, values(gradient)))
            .collect()
    };
    let (loss, frames) = replay_one(&arm.state, &mut arm.store, &corpus, position)?;
    let records = arm.state.memory.records.len();
    arm.state
        .memory
        .records
        .retain(|record| record.shared_revision < position);
    assert!(
        arm.state.memory.records.len() < records,
        "no records to cut off"
    );
    let (cut_loss, cut_frames) = replay_one(&arm.state, &mut arm.store, &corpus, position)?;
    assert_eq!(loss, cut_loss);
    assert_eq!(gradients(frames), gradients(cut_frames));
    Ok(())
}
