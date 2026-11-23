//! Context gate flow: starts at 0, trains, checkpoints and is read by frozen replays; see
//! CONNECTED.md §Contextual rows (Context gate).

use super::super::connected_config::Options;
use super::super::connected_context::{self, Context};
use super::super::connected_context_gate;
use super::super::connected_step;
use super::super::core_corpus::window_before;
use super::super::geometry::Geometry;
use super::super::scratch_dir::ScratchDir;
use super::super::tuning::{self, Knobs};
use super::context_contracts::{corpus, options, TEXT};
use super::*;

#[test]
fn gate_trains_from_zero_survives_a_checkpoint_and_is_read_by_frozen_replays() -> Result<()> {
    let _guard = tuning::scoped(Knobs {
        connected: true,
        ..Knobs::core(0.003, 7)
    });
    let path = ScratchDir::new("connected-context-gate")?;
    let corpus = corpus(&path)?;
    let gated = Options {
        context_gate: true,
        ..options()
    };
    let geometry = Geometry::fixture();
    let mut state = State::new(geometry, 7, corpus.cursor(0)?, &gated)?;
    let mut store = Store::new(&path.join("tiles"), geometry, 7, 2)?;
    // A fresh gate is exactly 0, which leaves every row raw even where states are buffered.
    let mut buffer = Context::new(8);
    (1..3u64).for_each(|key| buffer.push(key, vec![0.5; geometry.width]));
    let gate = connected_context_gate::value(&state.gate);
    assert_eq!(gate, Some(0.0));
    let (zero, _) = connected_context::rows(&state.shared, b"abab", &buffer, 3, 0, gate)?;
    assert_eq!(zero.memory, state.shared.prefix(b"abab")?.memory);
    for _ in 0..6 {
        connected_step::step(&mut state, &mut store, &corpus, gated, 0.003, false)?;
    }
    let restored = State::restore(&state.payloads()?, geometry, 7)?.gate;
    assert!(restored == state.gate && restored.is_some_and(|g| g.weights[0] != 0.0 && g.step == 6));
    let plain = State::new(geometry, 7, corpus.cursor(0)?, &options())?;
    assert!(!String::from_utf8(plain.payloads()?["connected.json"].clone())?.contains("\"gate\""));
    // A frozen replay builds its rows with the checkpoint's trained g, not 0 and not raw.
    let trained = connected_context_gate::value(&state.gate);
    let mut replay = replay::Replay::new(&state, &mut store)?;
    for (t, &target) in TEXT.iter().enumerate().take(6) {
        replay.score_at(&window_before(TEXT, t, 8), target, t as u64, 0)?;
    }
    let mut held = Context::new(8);
    for key in 0..6u64 {
        held.push(
            key,
            replay
                .buffered(key)
                .ok_or_else(|| anyhow::anyhow!("state"))?,
        );
    }
    let bytes = window_before(TEXT, 6, 8);
    let read = replay.prefix(&bytes, 6, 0)?;
    let rows = |gate| connected_context::rows(&state.shared, &bytes, &held, 6, 0, gate);
    assert_eq!(read.memory, rows(trained)?.0.memory);
    assert_ne!(read.memory, rows(None)?.0.memory);
    Ok(())
}
