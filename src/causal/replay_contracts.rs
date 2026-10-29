//! `Replay` contract: the two-anchor window and frozen weights; see AGENTS.md §core.

use super::core_contracts::{assay, train};
use super::core_corpus::SPAN_WARMUP;
use super::core_drill::Drill;
use super::core_eval::score_spans;
use super::execution::forward_path;
use super::geometry::Geometry;
use super::immutable;
use super::replay::Replay;
use super::resident::ResidentStore;
use super::state::State;
use super::tuning::{self, Knobs};
use anyhow::Result;

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
        "window keeps the last two anchors"
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
        "evaluation numerics changed ({nll:?}, bits {:#x}): re-score before trusting comparisons across commits",
        nll.to_bits()
    );
    Ok(())
}
