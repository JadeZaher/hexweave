//! Restart options (forget, signal, stagger, replay, Lambert law) through the real step path.

use super::connected_bias_contracts::corpus;
use super::connected_checkpoint as checkpoint;
use super::connected_config::{Options, Restart};
use super::connected_gain::lambert_w;
use super::connected_replay::replay_salted;
use super::connected_restart::{factor, TileRestart};
use super::connected_restart_config::{Law, Signal};
use super::connected_restart_contracts::{options, start, steps};
use super::connected_restart_replay::RESTART_SALT;
use super::connected_signal::eligible;
use super::connected_state::State;
use super::connected_storage_contracts::fresh;
use super::connected_store::Store;
use super::core_corpus::Corpus;
use super::core_schedule::Decay;
use super::geometry::{Geometry, TILES};
use super::tuning::{self, Knobs};
use anyhow::{Context, Result};
use std::path::Path;

/// About two fixture anchors of shared change, as in `connected_restart_contracts`.
const SHARED_BAR: f64 = 0.005;

fn knobs() -> tuning::Scoped {
    tuning::scoped(Knobs {
        connected: true,
        connected_fp32: true,
        ..Knobs::core(0.003, 7)
    })
}

fn fixture(out: &Path) -> Result<Corpus> {
    let train: Vec<u8> = b"abacabad".iter().cycle().take(512).copied().collect();
    corpus(&out.join("corpus"), &train)
}

/// Anchor-exact FP32 restart options without the prioritized replay buffer.
fn configured(restart: Restart) -> Options {
    Options {
        replay: None,
        ..options(Some(restart))
    }
}

fn tiles(state: &State) -> Result<&[TileRestart]> {
    state.restarts.as_deref().context("missing restart state")
}

fn visits(state: &State) -> [u64; TILES] {
    let counters = &state.counters;
    std::array::from_fn(|t| counters.local_visits[t] + counters.anchor_visits[t])
}

/// ||ema - reference|| / ||reference||, recomputed independently of `TileSignal::distance`.
fn distance(entry: &TileRestart) -> f64 {
    let norm = |v: &mut dyn Iterator<Item = f64>| v.map(|x| x * x).sum::<f64>().sqrt();
    let signal = &entry.signal;
    let gap = norm(
        &mut signal
            .input_ema
            .iter()
            .zip(&signal.reference)
            .map(|(a, b)| a - b),
    );
    gap / norm(&mut signal.reference.iter().copied()).max(f64::MIN_POSITIVE.sqrt())
}

/// Checkpoints, continues four anchors, resumes and requires identical losses and restart state.
fn round_trip(run: &Path, state: &mut State, mut store: Store, corpus: &Corpus) -> Result<()> {
    checkpoint::save(
        run,
        &mut store,
        "identity",
        state.anchors,
        &state.payloads()?,
    )?;
    let continued = steps(state, &mut store, corpus, 4)?;
    drop(store);
    let mut loaded =
        checkpoint::resume(run, Geometry::fixture(), "identity", 2)?.context("none")?;
    let mut restored = State::restore(&loaded.payloads, Geometry::fixture(), 7)?;
    assert_eq!(
        steps(&mut restored, &mut loaded.store, corpus, 4)?,
        continued
    );
    assert_eq!(restored.restarts, state.restarts);
    Ok(())
}

#[test]
fn stagger_spaces_restarts_and_forget_false_keeps_records() -> Result<()> {
    let _guard = knobs();
    let out = fresh("restart-stagger")?;
    let corpus = fixture(&out)?;
    let restart = Restart {
        floor: 0.25,
        threshold: Some(SHARED_BAR),
        forget: false,
        stagger: 3,
        ..Restart::default()
    };
    // Keep the prioritized buffer: forget = false must also leave it filled.
    let (mut state, mut store) = start(&out.join("run"), &corpus, &options(Some(restart)))?;
    let (mut last, mut kept, mut waited, mut served) = (None, 0, [false; TILES], 0);
    for _ in 0..30 {
        let before: Vec<u64> = tiles(&state)?.iter().map(|e| e.restarts).collect();
        steps(&mut state, &mut store, &corpus, 1)?;
        let entries = tiles(&state)?;
        let restarted: Vec<usize> = (0..TILES)
            .filter(|&t| entries[t].restarts > before[t])
            .collect();
        assert!(restarted.len() <= 1, "two restarts on one anchor");
        if let [tile] = restarted[..] {
            if let Some(previous) = last {
                assert!(
                    state.anchors - previous >= 3,
                    "restarts closer than stagger"
                );
            }
            last = Some(state.anchors);
            served += usize::from(waited[tile]);
            kept += usize::from(state.memory.records.iter().any(|r| r.tile_id == tile));
        }
        for (tile, entry) in entries.iter().enumerate() {
            waited[tile] |= entry.drift >= entry.bar;
        }
    }
    assert!(served > 0, "no tile that waited while due ever restarted");
    assert!(
        kept > 0,
        "forget = false dropped a restarted tile's records"
    );
    assert!(tiles(&state)?.iter().all(|e| e.forgotten_records == 0));
    assert_eq!(state.replay.report()["buffer_clears"], 0);
    Ok(())
}

#[test]
fn tile_signal_drift_is_its_input_distance_and_resumes_exactly() -> Result<()> {
    let _guard = knobs();
    let out = fresh("restart-signal")?;
    let corpus = fixture(&out)?;
    let restart = Restart {
        floor: 0.25,
        threshold: Some(0.005),
        signal: Signal::Tile,
        ..Restart::default()
    };
    let run = out.join("run");
    let (mut state, mut store) = start(&run, &corpus, &configured(restart))?;
    // The start tile, visited every anchor, warms up at anchor 256; then it drifts and restarts.
    for _ in 0..300 {
        let (seen, old) = (visits(&state), tiles(&state)?.to_vec());
        steps(&mut state, &mut store, &corpus, 1)?;
        let now = visits(&state);
        for (tile, (entry, old)) in tiles(&state)?.iter().zip(&old).enumerate() {
            assert!(entry.signal.is_warm() || (entry.restarts, entry.drift) == (0, 0.0));
            if entry.restarts > old.restarts {
                assert_eq!((entry.drift, distance(entry)), (0.0, 0.0));
            } else if now[tile] > seen[tile] {
                assert_eq!(entry.drift, distance(entry));
            } else {
                assert_eq!(entry.drift, old.drift);
            }
        }
    }
    let counts: Vec<u64> = tiles(&state)?.iter().map(|e| e.restarts).collect();
    assert!(counts.iter().sum::<u64>() > 0 && counts.iter().any(|&c| c != counts[0]));
    round_trip(&run, &mut state, store, &corpus)
}

#[test]
fn restart_replay_steps_only_the_restarted_tile_below_its_position() -> Result<()> {
    let _guard = knobs();
    let out = fresh("restart-replay")?;
    let corpus = fixture(&out)?;
    let restart = |replay| Restart {
        floor: 0.25,
        threshold: Some(SHARED_BAR),
        forget: false,
        stagger: 1,
        replay,
        ..Restart::default()
    };
    let (mut state, mut store) = start(&out.join("replay"), &corpus, &configured(restart(1)))?;
    let (mut base, mut base_store) = start(&out.join("base"), &corpus, &configured(restart(0)))?;
    let tile = loop {
        assert!(state.anchors < 32, "no restart");
        let loss = steps(&mut state, &mut store, &corpus, 1)?;
        assert_eq!(loss, steps(&mut base, &mut base_store, &corpus, 1)?);
        let entries = tiles(&state)?;
        if let Some(tile) = (0..TILES).find(|&t| entries[t].restarts > 0) {
            break tile;
        }
    };
    let entry = &tiles(&state)?[tile];
    let replays = entry.replay_steps;
    // The replay samples its own route, so the tile may not be revisited (0 or 1 steps).
    assert!(entry.replays == 1 && replays <= 1);
    // The base arm holds the replay arm's state just before its replay; replay there.
    let position = *entry.recent.back().context("empty ring")?;
    let revisions = std::array::from_fn(|t| base_store.slots[t].meta.revision);
    assert!(base
        .memory
        .records
        .iter()
        .any(|r| r.shared_revision >= position
            && eligible(r, base.epoch, &revisions, position, base.memory.lag)));
    base.memory.records.retain(|r| r.shared_revision < position);
    let (loss, _) = replay_salted(&base, &mut base_store, &corpus, position, RESTART_SALT)?;
    assert_eq!(entry.replay_loss_sum, f64::from(loss));
    for t in 0..TILES {
        let (a, b) = (&store.slots[t].meta, &base_store.slots[t].meta);
        let extra = if t == tile { replays } else { 0 };
        assert_eq!(
            (a.revision, a.steps),
            (b.revision + extra, b.steps + extra),
            "tile {t}"
        );
    }
    assert_eq!(
        state.counters.local_updates,
        base.counters.local_updates + replays
    );
    let (embedding, bias) = (state.shared_latent.as_ref(), state.bias_latent.as_ref());
    assert_eq!(
        embedding.map(|l| (&l.weights, l.step)),
        base.shared_latent.as_ref().map(|l| (&l.weights, l.step))
    );
    assert_eq!(
        bias.map(|l| (&l.weights, l.step)),
        base.bias_latent.as_ref().map(|l| (&l.weights, l.step))
    );
    assert_eq!(embedding.map(|l| l.step), Some(state.anchors));
    assert_eq!(
        (&state.cursor, &state.gains, state.baseline),
        (&base.cursor, &base.gains, base.baseline)
    );
    // Every later anchor still validates the extended local-update bound.
    steps(&mut state, &mut store, &corpus, 16)?;
    let entries = tiles(&state)?;
    assert!(entries.iter().map(|e| e.replays).sum::<u64>() > 1);
    assert!(entries.iter().map(|e| e.replay_steps).sum::<u64>() > 0);
    Ok(())
}

#[test]
fn lambert_law_sets_plasticity_from_distance_and_chases_the_reference() -> Result<()> {
    let bar = 0.2;
    let _guard = knobs();
    let out = fresh("restart-lambert")?;
    let corpus = fixture(&out)?;
    let restart = Restart {
        floor: 0.25,
        threshold: Some(bar),
        signal: Signal::Tile,
        law: Law::Lambert,
        ..Restart::default()
    };
    let options = Options {
        local_decay: Decay::Constant,
        ..configured(restart)
    };
    let run = out.join("run");
    let (mut state, mut store) = start(&run, &corpus, &options)?;
    let mut chased = 0;
    // Past anchor 256 the start tile is warm; the others still hold the dial at 1.
    for _ in 0..280 {
        let seen = visits(&state);
        steps(&mut state, &mut store, &corpus, 1)?;
        let now = visits(&state);
        let entries = tiles(&state)?;
        for (tile, entry) in entries
            .iter()
            .enumerate()
            .filter(|(t, _)| now[*t] > seen[*t])
        {
            let expected = match entry.signal.is_warm() {
                true => lambert_w(std::f64::consts::E * entry.drift / bar)?.min(4.0),
                false => 1.0,
            };
            assert_eq!(entry.plasticity, expected);
            let rate = factor(&options, Some(entries), state.anchors, tile)?;
            assert_eq!(rate, entry.plasticity.max(restart.floor));
            if entry.signal.is_warm() && entry.plasticity > 0.0 {
                assert!(distance(entry) < entry.drift);
                chased += 1;
            }
        }
        assert!(entries
            .iter()
            .all(|e| (e.restarts, e.origin, e.forgotten_records) == (0, 0, 0)));
    }
    assert!(chased > 0);
    round_trip(&run, &mut state, store, &corpus)
}
