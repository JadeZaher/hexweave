//! `memory_lag` and `visit_norm` flows: reach, causality, exact resume and the evaluation setting.

use super::connected_bias_contracts::corpus;
use super::connected_checkpoint as checkpoint;
use super::connected_config::{CreditMode, Options};
use super::connected_eval::frozen_histories;
use super::connected_precision::Precision;
use super::connected_replay::replay_one;
use super::connected_signal::eligible;
use super::connected_state::State;
use super::connected_step;
use super::connected_storage_contracts::fresh;
use super::connected_store::Store;
use super::core_corpus::Corpus;
use super::geometry::Geometry;
use super::tuning::{self, Knobs};
use anyhow::{Context, Result};
use std::path::Path;

/// Anchor-exact FP32 options with two visits, so a record's shared revision is its anchor.
fn options(memory_lag: u64, visit_norm: bool) -> Options {
    Options {
        precision: Precision::Fp32,
        lanes: 2,
        workers: 2,
        visits: 2,
        credit_mode: CreditMode::AnchorExact,
        memory_lag,
        visit_norm,
        ..Options::default()
    }
}

/// Connected FP32 knobs with post-visit normalization `visit_norm`.
fn knobs(visit_norm: bool) -> Knobs {
    Knobs {
        connected: true,
        connected_fp32: true,
        visit_norm,
        ..Knobs::core(0.003, 7)
    }
}

fn cycled(text: &[u8]) -> Vec<u8> {
    text.iter().cycle().take(512).copied().collect()
}

/// Steps `anchors` anchors; before each, memory holds only records of earlier anchors.
fn proceed(
    state: &mut State,
    store: &mut Store,
    corpus: &Corpus,
    anchors: usize,
) -> Result<Vec<f32>> {
    let mut losses = Vec::new();
    for _ in 0..anchors {
        let anchor = state.anchors;
        assert!(state
            .memory
            .records
            .iter()
            .all(|r| r.shared_revision < anchor));
        let options = state.options;
        losses.push(connected_step::step(
            state, store, corpus, options, 0.003, false,
        )?);
    }
    Ok(losses)
}

/// A fresh run under `out` trained for `anchors` anchors, with its per-anchor losses.
fn train(
    out: &Path,
    corpus: &Corpus,
    options: Options,
    anchors: usize,
) -> Result<(State, Store, Vec<f32>)> {
    let geometry = Geometry::fixture();
    let mut state = State::new(geometry, 7, corpus.cursor(0)?, &options)?;
    let mut store = Store::new_precision(&out.join("tiles"), geometry, 7, 2, options.precision)?;
    store.set_paging(options.resident(), options.write_behind_tiles)?;
    let losses = proceed(&mut state, &mut store, corpus, anchors)?;
    Ok((state, store, losses))
}

/// Checkpoints after `split` anchors, resumes and finishes; the state must equal `straight`.
fn assert_resume_exact(
    out: &Path,
    corpus: &Corpus,
    straight: &State,
    split: usize,
) -> Result<State> {
    let (first, mut store, _) = train(out, corpus, straight.options, split)?;
    checkpoint::save(
        out,
        &mut store,
        "identity",
        first.anchors,
        &first.payloads()?,
    )?;
    drop((first, store));
    let loaded = checkpoint::resume(out, Geometry::fixture(), "identity", 2)?.context("resume")?;
    let mut resumed = State::restore(&loaded.payloads, Geometry::fixture(), 7)?;
    let mut store = loaded.store;
    store.set_paging(2, 0)?;
    let rest = (straight.anchors - resumed.anchors) as usize;
    proceed(&mut resumed, &mut store, corpus, rest)?;
    assert_eq!(resumed.payloads()?, straight.payloads()?);
    Ok(resumed)
}

#[test]
fn memory_lag_reaches_older_records_never_its_own_anchor_and_resumes_exactly() -> Result<()> {
    let _guard = tuning::scoped(knobs(true));
    let out = fresh("memory-lag")?;
    let corpus = corpus(&out.join("corpus"), &cycled(b"abacabad"))?;
    let (one, _, one_losses) = train(&out.join("one"), &corpus, options(1, true), 12)?;
    let (mut four, mut store, four_losses) =
        train(&out.join("four"), &corpus, options(4, true), 12)?;
    // Anchors 0 and 1 hold no record two anchors old, so both limits read the same records.
    assert_eq!(one_losses[..2], four_losses[..2]);
    assert_ne!(one_losses, four_losses);
    assert_eq!(one.reach.lag_max(), 1);
    assert!(
        (2..=4).contains(&four.reach.lag_max()),
        "{}",
        four.reach.lag_max()
    );
    let resumed = assert_resume_exact(&out.join("resumed"), &corpus, &four, 8)?;
    assert_eq!(resumed.memory.lag, 4);
    // Replay at `position`: the wider window reaches records below it ...
    let position = four.anchors - 3;
    let (wide, frames) = replay_one(&four, &mut store, &corpus, position)?;
    four.memory.lag = 1;
    let (narrow, _) = replay_one(&four, &mut store, &corpus, position)?;
    four.memory.lag = 4;
    assert_ne!(wide, narrow);
    // ... but never one published at or after it, though such records are lag-eligible.
    let revisions = std::array::from_fn(|t| store.slots[t].meta.revision);
    assert!(four
        .memory
        .records
        .iter()
        .any(|r| r.shared_revision == position
            && eligible(r, four.epoch, &revisions, position, four.memory.lag)));
    four.memory.records.retain(|r| r.shared_revision < position);
    let (cut, cut_frames) = replay_one(&four, &mut store, &corpus, position)?;
    assert_eq!(wide, cut);
    let values = |frames: Vec<(usize, super::tile_gradients::TileGradients)>| {
        let rows = frames
            .into_iter()
            .map(|(tile, gradient)| (tile, gradient.values.to_vec()));
        rows.collect::<Vec<_>>()
    };
    assert_eq!(values(frames), values(cut_frames));
    Ok(())
}

#[test]
fn visit_norm_off_changes_training_resumes_exactly_and_evaluates_under_its_setting() -> Result<()> {
    let _guard = tuning::scoped(knobs(true));
    let out = fresh("visit-norm")?;
    let corpus = corpus(&out.join("corpus"), &cycled(b"the cat sat on the mat. "))?;
    // Fixture visits stay near raw RMS 1 and its loss does not fall within 48 anchors in either
    // arm, so forward/reverse consistency is the central-difference row in connected_curve_contracts.
    let (_, _, normed) = train(&out.join("normed"), &corpus, options(1, true), 12)?;
    let (mut raw, mut store, losses) = train(&out.join("raw"), &corpus, options(1, false), 12)?;
    assert_ne!(normed, losses);
    assert_resume_exact(&out.join("resumed"), &corpus, &raw, 8)?;
    // Frozen replay (evaluation and completion) takes the state's setting, not the thread's.
    let observed: [(&[u8], u8); 2] = [(b"the c", b'a'), (b"the ca", b't')];
    let mut probe = |state: &State, ambient: bool| {
        tuning::anneal(knobs(ambient));
        frozen_histories(state, &mut store, &observed, b"the cat")
    };
    let under_on = probe(&raw, true)?;
    assert_eq!(under_on, probe(&raw, false)?);
    raw.options.visit_norm = true;
    assert_ne!(under_on, probe(&raw, false)?);
    Ok(())
}
