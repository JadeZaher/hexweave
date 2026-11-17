//! EXPERIMENT: continuous Lambert-W plasticity law; see CONNECTED.md §Lambert plasticity.

use super::connected_gain::lambert_w;
use super::connected_restart::TileRestart;
use super::connected_restart_config::MAX_PLASTICITY;
use anyhow::Result;
use std::f64::consts::E;

/// Reference chase speed per unit plasticity: reference += min(1, p * BETA) (ema - reference).
pub(super) const BETA: f64 = 1.0 / 64.0;

/// W(e d / bar), capped at the dial's maximum; W(4 e^4) = 4, so larger inputs take the cap.
pub(super) fn dial(distance: f64, bar: f64) -> Result<f64> {
    let input = E * distance / bar;
    if input >= MAX_PLASTICITY * MAX_PLASTICITY.exp() {
        return Ok(MAX_PLASTICITY);
    }
    Ok(lambert_w(input)?.min(MAX_PLASTICITY))
}

/// One visited anchor under the law: drift is the distance, it sets the dial, the reference chases;
/// before the signal's warm-up the dial holds at 1 (full rate).
pub(super) fn visit(entry: &mut TileRestart, bar: f64) -> Result<()> {
    entry.drift = entry.signal.distance();
    entry.plasticity = if entry.signal.is_warm() {
        dial(entry.drift, bar)?
    } else {
        1.0
    };
    entry.plasticity_sum += entry.plasticity;
    entry.signal.chase((entry.plasticity * BETA).min(1.0));
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn dial_is_one_at_the_bar_monotone_and_capped() {
        // d = bar gives W(e) = 1; the dial rises monotonically and caps at 4 past 4 e^3 bar.
        let bar = 0.2;
        let dials = [0.0, 0.5, 1.0, 2.0, 10.0, 81.0].map(|d| super::dial(d * bar, bar).unwrap());
        assert!((dials[2] - 1.0).abs() < 1e-12 && dials[0] == 0.0 && dials[5] == 4.0);
        assert!(dials.windows(2).all(|pair| pair[0] < pair[1]));
    }
}
