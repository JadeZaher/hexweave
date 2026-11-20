//! Connected-state consistency at quiescent boundaries; see CONNECTED.md.

use super::{State, TILES, VOCAB};
use anyhow::{ensure, Result};

impl State {
    pub fn validate(&self) -> Result<()> {
        self.geometry.validate()?;
        self.options.validate()?;
        ensure!(
            self.options.precision.is_ternary() == self.shared.weights.is_none(),
            "connected shared precision/view mismatch"
        );
        ensure!(self.ready, "connected state is in flight or poisoned");
        ensure!(
            self.cursor.supervised_targets == self.anchors,
            "connected cursor exposure mismatch"
        );
        ensure!(self.gains.len() == TILES, "connected gain count");
        ensure!(
            self.baseline.is_finite() && self.counters.loss_sum.is_finite(),
            "nonfinite connected accounting"
        );
        self.predictor.validate(self.geometry)?;
        self.memory.validate()?;
        ensure!(
            self.memory.geometry == self.geometry
                && self.memory.capacity == self.options.memory_capacity
                && self.memory.lag == self.options.memory_lag,
            "connected memory geometry/options mismatch"
        );
        for gain in &self.gains {
            gain.validate(&self.options)?;
        }
        ensure!(
            self.predictor.fits == self.counters.predictor_fits
                && self.counters.predictor_fits == self.counters.observed_records,
            "connected predictor accounting mismatch"
        );
        ensure!(
            self.memory.observed == self.counters.observed_records
                && self.memory.predicted == self.counters.predicted_records
                && self.memory.self_generated == 0
                && self.epoch == self.memory.publications,
            "connected publication accounting mismatch"
        );
        // At most one mean update per visited trusted tile, and every visit publishes a record.
        // Replay can step tiles the live route skipped, but still at most once per tile per anchor.
        let max_updates = if self.options.replay.is_some() {
            (TILES as u64).saturating_mul(self.anchors)
        } else if self.options.credit_mode == super::super::connected_config::CreditMode::Lanes {
            self.counters.predicted_records
        } else {
            self.counters.observed_records + self.counters.predicted_records
        };
        // Restart replays step only their restarted tile, counted per tile (CONNECTED.md).
        let max_updates =
            max_updates.saturating_add(super::super::connected_restart::replay_steps(self));
        ensure!(
            self.counters.local_updates <= max_updates,
            "connected local-update accounting mismatch"
        );
        let cap = self.options.master_rms_cap * (1.0 + 1e-5);
        for (tile, rms) in self.counters.master_rms_max.iter().enumerate() {
            ensure!(
                *rms <= cap,
                "connected tile {tile} master RMS {rms} exceeded the cap {}",
                self.options.master_rms_cap
            );
        }
        ensure!(
            [
                self.counters.predictor_mse_sum,
                self.counters.predictor_zero_mse_sum,
                self.counters.predictor_cosine_sum,
                self.counters.local_base_norm_sum,
                self.counters.local_applied_norm_sum,
                self.counters.readout_rms_sum,
                self.counters.visit_rms_sum,
                self.counters.visit_rms_max,
                self.counters.knot_overflow_sum,
                self.counters.change_sum,
            ]
            .iter()
            .all(|value| value.is_finite()),
            "nonfinite predictor metrics"
        );
        super::super::connected_restart::validate(self)?;
        super::super::connected_wave::validate_state(self)?;
        super::super::connected_balance::validate(self)?;
        self.validate_bias()?;
        self.validate_shared()
    }

    fn validate_shared(&self) -> Result<()> {
        if let Some(latent) = &self.shared_latent {
            if let Some(weights) = &self.shared.weights {
                ensure!(
                    weights.version == self.shared.version
                        && weights.mode == super::super::connected_precision::View::Fp32
                        && weights.matrices.len() == 1
                        && weights.matrices[0] == latent.weights,
                    "stale shared FP32 view"
                );
            }
            ensure!(
                latent.weights.len() == self.geometry.width * VOCAB,
                "connected embedding latent shape"
            );
            ensure!(
                latent.moments.len() == 2 * latent.weights.len(),
                "connected embedding moment shape"
            );
            ensure!(
                latent
                    .weights
                    .iter()
                    .chain(&latent.moments)
                    .all(|v| v.is_finite()),
                "nonfinite connected embedding state"
            );
            ensure!(
                latent
                    .moments
                    .iter()
                    .skip(1)
                    .step_by(2)
                    .all(|value| *value >= 0.0),
                "negative connected embedding second moment"
            );
            ensure!(
                latent.step == self.anchors,
                "connected embedding step mismatch"
            );
            // E has no tile slot in `master_rms_max`; its master obeys the same cap.
            let rms = super::super::connected_update_scale::rms(&latent.weights);
            ensure!(
                rms <= self.options.master_rms_cap * (1.0 + 1e-5),
                "connected embedding master RMS {rms} exceeded the cap {}",
                self.options.master_rms_cap
            );
        } else {
            ensure!(
                self.anchors == 0 && self.options.precision.is_ternary(),
                "missing connected embedding master state"
            );
        }
        Ok(())
    }

    /// The FP32 bias equals its Adam master, which steps once per anchor; before that, any prior.
    fn validate_bias(&self) -> Result<()> {
        let bias = self
            .shared
            .bias
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("missing connected readout bias"))?;
        super::super::norm::vector(bias, VOCAB)?;
        match &self.bias_latent {
            Some(latent) => {
                super::super::connected_adam::validate(latent, VOCAB)?;
                ensure!(
                    latent.weights == *bias && latent.step == self.anchors,
                    "connected readout bias master mismatch"
                );
            }
            // Finite VOCAB length is checked above; a fresh run's bias may hold its prior.
            None => ensure!(
                self.anchors == 0,
                "missing connected readout bias optimizer"
            ),
        }
        Ok(())
    }
}
