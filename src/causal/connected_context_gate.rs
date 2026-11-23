//! EXPERIMENT context gate: one learned scalar scaling the buffered states in the rows; see
//! CONNECTED.md §Contextual rows (Context gate).

use super::connected_adam as adam;
use super::connected_config::Options;
use super::connected_state::State;
use super::norm;
use super::shared::Prefix;
use super::tuning::Latent;
use anyhow::{ensure, Result};

/// A fresh gate master (g = 0, zero Adam moments) exactly when `options.context_gate`.
pub(super) fn fresh(options: &Options) -> Option<Latent> {
    options.context_gate.then(|| Latent {
        weights: vec![0.0],
        moments: vec![0.0; 2],
        step: 0,
    })
}

/// The current gate value `g`; None when the option is off.
pub(super) fn value(gate: &Option<Latent>) -> Option<f32> {
    gate.as_ref().map(|latent| latent.weights[0])
}

/// Admission: the gate scales contextual rows, so it needs them.
pub(super) fn validate_options(options: &Options) -> Result<()> {
    ensure!(
        !options.context_gate || options.context_rows,
        "connected context_gate requires context_rows"
    );
    Ok(())
}

/// dL/dg = sum over gated rows of dot(u_i, s_i), u_i the row's pre-normalization adjoint.
pub(super) fn gradient(
    prefix: &Prefix,
    prefix_adjoint: &[f32],
    width: usize,
) -> Result<Option<f32>> {
    let Some(states) = &prefix.gated_states else {
        return Ok(None);
    };
    ensure!(
        states.len() == prefix.memory.len(),
        "context gate state shape"
    );
    let mut total = 0.0f64;
    for (i, &inverse) in prefix.inverse.iter().enumerate() {
        let span = i * width..(i + 1) * width;
        if states[span.clone()].iter().all(|s| *s == 0.0) {
            continue;
        }
        let row = norm::backward(
            &prefix.memory[span.clone()],
            inverse,
            &prefix_adjoint[span.clone()],
        )?;
        total += row
            .iter()
            .zip(&states[span])
            .map(|(u, s)| f64::from(*u) * f64::from(*s))
            .sum::<f64>();
    }
    ensure!(total.is_finite(), "nonfinite context gate gradient");
    Ok(Some(total as f32))
}

/// The gate steps at this fraction of the anchor rate (CONNECTED.md §Contextual rows).
const RATE_SCALE: f64 = 0.1;

/// One Adam step of `g` at `RATE_SCALE * rate` on the bias's anchor clock (no-op when off).
pub(super) fn step(
    gate: &mut Option<Latent>,
    gradient: Option<f32>,
    rate: f64,
    options: &Options,
) -> Result<()> {
    if let (Some(latent), Some(gradient)) = (gate, gradient) {
        super::connected_update::apply_gate(latent, gradient, RATE_SCALE * rate, options)?;
    }
    Ok(())
}

/// Present exactly when the option is; one finite master whose Adam clock tracks `anchors`.
pub(super) fn validate(state: &State) -> Result<()> {
    ensure!(
        state.gate.is_some() == state.options.context_gate,
        "connected context gate/options mismatch"
    );
    if let Some(latent) = &state.gate {
        adam::validate(latent, 1)?;
        ensure!(
            latent.step == state.anchors,
            "connected context gate step mismatch"
        );
    }
    Ok(())
}

/// The log's effective state-to-row ratio scale: |g| when gated, else 1.
pub(super) fn strength(gate: Option<f32>) -> f64 {
    gate.map_or(1.0, |g| f64::from(g.abs()))
}
