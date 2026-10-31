//! `tuning::Knobs::history` ablation: off skips records/predictor training and moves both the
//! trained loss sequence and the run identity; see AGENTS.md §Core path and §State encoding.

use super::assay_source::Source;
use super::core_contracts::assay;
use super::core_drill::Drill;
use super::execution::{self, forward_path};
use super::geometry::Geometry;
use super::replay::Replay;
use super::resident::ResidentStore;
use super::state::State;
use super::telemetry;
use super::tuning::{self, Knobs};
use anyhow::{Context, Result};
use serde_json::Value;

/// Train `anchors` production-drill anchors under `knobs`, returning the final state, the
/// per-anchor loss sequence, and the telemetry report drained right after training.
fn drill_run(
    source: &Source,
    geometry: Geometry,
    knobs: Knobs,
    anchors: u64,
) -> Result<(State, Vec<f32>, Value)> {
    let _scope = tuning::scoped(knobs);
    telemetry::drain(0.0);
    let drill = Drill::new(source, 7)?;
    let mut state = State::with_cursor(geometry, 7, drill.cursor(0)?)?;
    state.anchor_only = true;
    let mut store = ResidentStore::new(geometry, 7)?;
    let mut losses = Vec::with_capacity(usize::try_from(anchors)?);
    for _ in 0..anchors {
        match execution::step(&mut state, &mut store, &drill, false)? {
            execution::Step::Committed { loss, .. } => {
                losses.push(loss.context("anchor-only step must report a loss")?);
            }
            execution::Step::Stopped => anyhow::bail!("unexpected stop mid-drill"),
        }
    }
    let report = telemetry::drain(0.0);
    Ok((state, losses, report))
}

/// `history: false` skips `train_records` and `Replay`'s push entirely: the predictor never
/// trains, `History` stays empty, no anchor ever logs a history record, and the trained loss
/// sequence (hence the run identity, pinned separately in `core_contracts`) differs from the
/// same seed with history on.
#[test]
fn history_off_skips_records_and_changes_loss_and_identity() -> Result<()> {
    let source = assay()?;
    let geometry = Geometry::candidate();
    let on = Knobs::core(0.003, 7);
    let off = Knobs {
        history: false,
        ..on
    };
    assert_ne!(
        serde_json::to_vec(&on)?,
        serde_json::to_vec(&off)?,
        "history knob must move the serialized run identity"
    );
    let (_, loss_on, _) = drill_run(&source, geometry, on, 12)?;
    let (state_off, loss_off, report_off) = drill_run(&source, geometry, off, 12)?;
    {
        // `validate_boundary` reads the installed knob, same as a real resume re-validating a
        // restored state under the knob it was trained with (AGENTS.md §State encoding).
        let _scope = tuning::scoped(off);
        state_off.validate_boundary()?;
    }
    assert_eq!(
        state_off.counters.predictor_anchor_updates, 0,
        "predictor must not train while history is off"
    );
    assert!(
        state_off.history.records.is_empty(),
        "history must stay empty while the knob is off"
    );
    assert_ne!(
        loss_on, loss_off,
        "the history knob must change the trained loss sequence"
    );
    assert_eq!(
        report_off["history_len"].as_f64(),
        Some(0.0),
        "no anchor may see a history record while the knob is off: {}",
        report_off["history_len"]
    );
    Ok(())
}

/// With history off, `Replay::observe` must skip `credit::gather`'s backward pass entirely and
/// return exactly `state.shared.readout(&trace.output)?.loss(target)` (the same value `gather`
/// would have computed via `Shared::task`), bit for bit.
#[test]
fn history_off_observe_returns_readout_loss_bit_for_bit() -> Result<()> {
    let source = assay()?;
    let geometry = Geometry::candidate();
    let off = Knobs {
        history: false,
        ..Knobs::core(0.003, 7)
    };
    let _scope = tuning::scoped(off);
    let drill = Drill::new(&source, 7)?;
    let state = State::with_cursor(geometry, 7, drill.cursor(0)?)?;
    let mut store = ResidentStore::new(geometry, 7)?;
    let prefix = state.shared.prefix(b"history off replay probe")?;
    let trace = forward_path(
        &mut store,
        geometry,
        &prefix,
        vec![0.0; geometry.width],
        &mut None,
        state.tile_versions,
    )?;
    let expected = state.shared.readout(&trace.output)?.loss(b'z')?;
    let observed = Replay::default().observe(&state, &mut store, &prefix, &trace, b'z')?;
    assert_eq!(
        observed.to_bits(),
        expected.to_bits(),
        "history-off observe must return exactly the readout loss, with no backward pass"
    );
    Ok(())
}
