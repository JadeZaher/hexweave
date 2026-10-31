//! `Replay` contract: the two-anchor window and frozen weights; see AGENTS.md §core.

use super::core_contracts::{assay, fresh_temp, train};
use super::core_corpus::Corpus;
use super::core_drill::Drill;
use super::core_eval::score_spans;
use super::execution::{self, forward_path, Step};
use super::geometry::Geometry;
use super::immutable;
use super::replay::{Replay, SPAN_WARMUP};
use super::resident::ResidentStore;
use super::state::State;
use super::storage_paths::digest;
use super::tuning::{self, Knobs};
use anyhow::Result;
use serde_json::json;

/// `score_spans` f64 bits on the fixed probe below; moving it means evaluation numerics changed.
const GOLDEN_SPAN_NLL: u64 = 0x401b_5985_3000_0000;

/// Seed-7 candidate state trained three production-drill anchors; caller installs core knobs.
fn trained(geometry: Geometry) -> Result<(State, ResidentStore)> {
    let source = assay()?;
    let drill = Drill::new(&source, 7)?;
    let mut state = State::with_cursor(geometry, 7, drill.cursor(0)?)?;
    state.anchor_only = true;
    let mut store = ResidentStore::new(geometry, 7)?;
    train(&mut state, &mut store, &drill, 3)?;
    Ok((state, store))
}

#[test]
fn replay_window_is_two_anchors_and_freezes_weights() -> Result<()> {
    let geometry = Geometry::candidate();
    let _core = tuning::scoped(Knobs::core(0.003, 7));
    let (state, mut store) = trained(geometry)?;

    let zero = vec![0.0; geometry.width];
    assert_eq!(Replay::default().input(&state)?, zero, "replay starts zero");
    let prefix = state.shared.prefix(b"replay contract probe")?;
    let trace = forward_path(
        &mut store,
        geometry,
        &prefix,
        zero,
        &mut None,
        state.tile_versions,
    )?;

    let mut three = Replay::default();
    for target in [b'a', b'b', b'c'] {
        three.observe(&state, &mut store, &prefix, &trace, target)?;
    }
    let windowed = three.input(&state)?;

    let mut last_two = Replay::default();
    for target in [b'b', b'c'] {
        last_two.observe(&state, &mut store, &prefix, &trace, target)?;
    }
    assert_eq!(
        windowed,
        last_two.input(&state)?,
        "Replay's window keeps up to two anchors; training keeps the older anchor only when no \
         trit flipped in between"
    );

    let mut last_one = Replay::default();
    last_one.observe(&state, &mut store, &prefix, &trace, b'c')?;
    assert_ne!(
        windowed,
        last_one.input(&state)?,
        "a one-anchor window would drop one record"
    );

    let before = immutable::identity(&state, &mut store)?;
    three.observe(&state, &mut store, &prefix, &trace, b'z')?;
    assert_eq!(
        before,
        immutable::identity(&state, &mut store)?,
        "observe must not touch weights"
    );
    Ok(())
}

#[test]
fn span_scoring_numerics_are_pinned() -> Result<()> {
    let geometry = Geometry::candidate();
    let _core = tuning::scoped(Knobs::core(0.003, 7));
    let (state, mut store) = trained(geometry)?;
    let text: Vec<u8> = (0..64u8)
        .map(|i| b"once upon a time, "[usize::from(i) % 18])
        .collect();
    let window = 32;
    let unclipped = &text[10..10 + window + SPAN_WARMUP + 8];
    let clipped = &text[..9];
    let (nll, targets) = score_spans(&state, &mut store, &[(unclipped, 8), (clipped, 8)], window)?;
    assert_eq!(targets, 16);
    assert_eq!(
        nll.to_bits(),
        GOLDEN_SPAN_NLL,
        "evaluation numerics changed ({nll:?}, bits {:#x}): this also moves with training \
         numerics and the platform's libm (exp/tanh/ln), so re-score before trusting comparisons \
         across commits, and update the constant only when the change is intended",
        nll.to_bits()
    );
    Ok(())
}

/// A fresh frozen probe's byte-0 loss must equal training's own anchor-0 loss bit for bit: both
/// predict the corpus' first train byte from the same `[START]` prefix with zero history on the
/// same initial weights, through the same `credit::gather` seam (`score_spans`' `Replay::observe`
/// and training's `credit::anchor`, which reads its loss before any commit).
#[test]
fn frozen_probe_byte_zero_matches_training_anchor_zero() -> Result<()> {
    let geometry = Geometry::candidate();
    let _core = tuning::scoped(Knobs::core(0.003, 7));
    let root = fresh_temp("frozen-alignment")?;
    std::fs::create_dir_all(&root)?;
    let train_bytes = b"hexweave frozen probe alignment fixture bytes".to_vec();
    let heldout = b"held out alignment fixture bytes".to_vec();
    let entry = |bytes: &[u8]| json!({ "bytes": bytes.len(), "sha256": digest(bytes) });
    let manifest = serde_json::to_vec(&json!({ "schema": 1, "kind": "byte-corpus",
        "files": { "train.txt": entry(&train_bytes), "heldout.txt": entry(&heldout) } }))?;
    std::fs::write(root.join("train.txt"), &train_bytes)?;
    std::fs::write(root.join("heldout.txt"), &heldout)?;
    std::fs::write(root.join("manifest.json"), &manifest)?;
    let corpus = Corpus::open(&root, &digest(&manifest), 32)?;

    let mut live = State::with_cursor(geometry, 7, corpus.cursor(0)?)?;
    live.anchor_only = true;
    let mut live_store = ResidentStore::new(geometry, 7)?;
    let training_loss = match execution::step(&mut live, &mut live_store, &corpus, false)? {
        Step::Committed {
            loss: Some(loss), ..
        } => loss,
        other => panic!("expected a committed anchor loss, got {other:?}"),
    };

    let frozen = State::with_cursor(geometry, 7, corpus.cursor(0)?)?;
    let mut frozen_store = ResidentStore::new(geometry, 7)?;
    let (nll, targets) = score_spans(&frozen, &mut frozen_store, &[(&train_bytes[..1], 1)], 32)?;
    assert_eq!(targets, 1);
    assert_eq!(
        nll,
        f64::from(training_loss),
        "a fresh frozen probe on byte 0 must match training's own anchor-0 loss bit for bit"
    );
    std::fs::remove_dir_all(&root)?;
    Ok(())
}
