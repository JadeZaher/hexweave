//! Connected learning controls and admission; see AGENTS.md §Connected training.

use super::core_config::{Config, Data};
use super::core_schedule::Decay;
use super::geometry::TILES;
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};

pub(super) use super::connected_restart_config::Restart;
pub(super) use super::connected_route::RouteTraining;

/// How tile transformations receive training credit on each connected step.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(super) enum CreditMode {
    #[default]
    Lanes,
    AnchorExact,
    AnchorPredicted,
    Hybrid,
}

impl CreditMode {
    pub const fn is_default(&self) -> bool {
        matches!(self, Self::Lanes)
    }

    pub const fn uses_lanes(self) -> bool {
        matches!(self, Self::Lanes | Self::Hybrid)
    }
}

/// How each tile visit selects its active FFN channel window and projection perspective.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(super) enum SliceMode {
    #[default]
    Full,
    Windowed,
    Hybrid,
}

impl SliceMode {
    pub const fn is_default(&self) -> bool {
        matches!(self, Self::Full)
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub(super) struct Options {
    #[serde(
        default,
        skip_serializing_if = "super::connected_precision::Precision::is_ternary"
    )]
    pub precision: super::connected_precision::Precision,
    pub lanes: usize,
    pub workers: usize,
    pub visits: usize,
    pub memory_capacity: usize,
    pub predictor_rate: f64,
    pub gain_min: f32,
    pub gain_max: f32,
    pub max_update_norm: f64,
    /// Decoupled decay: a touched master keeps `1 - weight_decay * step rate * realized multiplier`.
    pub weight_decay: f64,
    /// Per-matrix master RMS bound, restored by uniform projection after each commit.
    pub master_rms_cap: f64,
    /// Adam rate of predicted-credit (local) tile steps and their decay; E, bias and routes keep
    /// `rate`, the predictor `predictor_rate`.
    pub local_rate: f64,
    /// Schedule on `local_rate` by anchor index (v5.2); the other learners stay constant.
    pub local_decay: Decay,
    #[serde(default, skip_serializing_if = "CreditMode::is_default")]
    pub credit_mode: CreditMode,
    #[serde(default, skip_serializing_if = "SliceMode::is_default")]
    pub slice_mode: SliceMode,
    /// Training route choice (CONNECTED.md §Greedy training routes); default sampled.
    #[serde(default, skip_serializing_if = "RouteTraining::is_default")]
    pub route_training: RouteTraining,
    /// Per-anchor share of greedy-mode passes that sample their route instead; default 0.
    #[serde(default, skip_serializing_if = "super::connected_route::is_zero")]
    pub route_explore: f64,
    /// Anchors per tile update under `anchor_exact` (CONNECTED.md §Anchor waves); default 1.
    #[serde(skip_serializing_if = "super::connected_wave::is_one")]
    pub anchor_batch: u32,
    /// Evicted dirty bundles held encoded in RAM before a disk write; paging only, never identity.
    #[serde(default, skip_serializing)]
    pub write_behind_tiles: usize,
    /// Decoded-bundle residency cap in `workers..=TILES` (absent: `workers`); never identity.
    #[serde(default, skip_serializing)]
    pub resident_tiles: Option<usize>,
    /// Whether tiles read retrieved shared memory; false feeds zeros (forward-input ablation).
    #[serde(skip_serializing_if = "is_true")]
    pub history: bool,
    /// Prioritized replay of past anchors under `anchor_exact`; absent is off.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub replay: Option<Replay>,
    /// Per-tile local-rate floor and change-driven warm restarts; absent is off (CONNECTED.md).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub restart: Option<Restart>,
}

/// Bounded prioritized replay: buffer `capacity` and replayed anchors `per_anchor`.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct Replay {
    pub capacity: usize,
    pub per_anchor: usize,
}

fn is_true(value: &bool) -> bool {
    *value
}

impl Default for Options {
    fn default() -> Self {
        Self {
            precision: super::connected_precision::Precision::Ternary,
            lanes: 8,
            workers: 4,
            visits: 4,
            memory_capacity: 32,
            predictor_rate: 0.003,
            gain_min: 0.25,
            gain_max: 4.0,
            max_update_norm: 1.0,
            weight_decay: 0.01,
            master_rms_cap: 1.5,
            local_rate: 3e-4,
            local_decay: Decay::Constant,
            credit_mode: CreditMode::Lanes,
            slice_mode: SliceMode::Full,
            route_training: RouteTraining::Sampled,
            route_explore: 0.0,
            anchor_batch: 1,
            write_behind_tiles: 0,
            resident_tiles: None,
            history: true,
            replay: None,
            restart: None,
        }
    }
}

impl Options {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            (1..=8).contains(&self.lanes)
                && (1..=4).contains(&self.workers)
                && (1..=4).contains(&self.visits)
                && (8..=64).contains(&self.memory_capacity),
            "connected lanes, workers, visits or memory capacity out of range"
        );
        ensure!(
            self.predictor_rate.is_finite()
                && self.predictor_rate > 0.0
                && self.predictor_rate <= 0.1,
            "invalid connected predictor rate"
        );
        ensure!(
            self.gain_min.is_finite()
                && self.gain_max.is_finite()
                && (0.0..=1.0).contains(&self.gain_min)
                && self.gain_min > 0.0
                && (1.0..=8.0).contains(&self.gain_max),
            "invalid connected gain bounds"
        );
        ensure!(
            self.max_update_norm.is_finite()
                && self.max_update_norm > 0.0
                && self.max_update_norm <= 4.0,
            "invalid connected update norm bound"
        );
        ensure!(
            self.weight_decay.is_finite() && self.weight_decay > 0.0 && self.weight_decay <= 1.0,
            "connected weight decay must be in (0, 1]"
        );
        ensure!(
            self.master_rms_cap.is_finite() && (1.0..=8.0).contains(&self.master_rms_cap),
            "connected master RMS cap must be in [1, 8]"
        );
        ensure!(
            self.local_rate.is_finite() && self.local_rate > 0.0 && self.local_rate <= 0.1,
            "connected local rate must be in (0, 0.1]"
        );
        self.local_decay.validate()?;
        super::connected_route::validate(self)?;
        super::connected_wave::validate(self)?;
        // Its long-run factor must stay positive: cosine with floor 0 would hand Adam rate 0.
        ensure!(
            self.local_decay.factor(u64::MAX) > 0.0,
            "connected local_decay must keep the local rate positive"
        );
        if let Some(restart) = self.restart {
            restart.validate()?;
        }
        self.validate_paging_and_replay()
    }

    /// Store paging bounds and replay's anchor-exact-only admission.
    fn validate_paging_and_replay(&self) -> Result<()> {
        ensure!(
            self.write_behind_tiles <= super::connected_store::MAX_WRITE_BEHIND
                && (self.workers..=TILES).contains(&self.resident()),
            "connected write_behind_tiles must be 0..=64 and resident_tiles workers..={TILES}"
        );
        if let Some(replay) = self.replay {
            ensure!(
                (1..=4096).contains(&replay.capacity) && (1..=8).contains(&replay.per_anchor),
                "connected replay capacity must be 1..=4096 and per_anchor 1..=8"
            );
            ensure!(
                self.credit_mode == CreditMode::AnchorExact,
                "connected replay requires credit_mode anchor_exact"
            );
        }
        ensure!(
            self.restart.is_none_or(|restart| restart.replay == 0)
                || self.credit_mode == CreditMode::AnchorExact,
            "connected restart replay requires credit_mode anchor_exact"
        );
        // The Lambert law ignores the local schedule, so a non-constant one would be silent.
        ensure!(
            self.restart
                .is_none_or(|restart| restart.law != super::connected_restart_config::Law::Lambert)
                || self.local_decay == Decay::Constant,
            "connected restart law lambert requires a constant local_decay"
        );
        Ok(())
    }

    /// These options with `paging`'s store-only controls, which a resume may change.
    pub fn with_paging(self, paging: &Self) -> Self {
        Self {
            write_behind_tiles: paging.write_behind_tiles,
            resident_tiles: paging.resident_tiles,
            ..self
        }
    }

    /// The decoded-bundle residency cap the store enforces.
    pub fn resident(&self) -> usize {
        self.resident_tiles.unwrap_or(self.workers)
    }

    /// The tile history input: retrieved memory, or zeros of its width under the ablation.
    pub fn history_input(&self, retrieved: Vec<f32>) -> Vec<f32> {
        if self.history {
            retrieved
        } else {
            vec![0.0; retrieved.len()]
        }
    }

    pub fn validate_run(&self, config: &Config) -> Result<()> {
        self.validate()?;
        super::connected_wave::validate_run(self, config)?;
        // A tile step's realized multiplier never exceeds gain_max, so every decay keeps a master.
        // max() is future-proofing: within current ranges only `rate` can trip this bound.
        let plasticity = self.restart.map_or(1.0, |restart| restart.max_plasticity());
        let rate = config.rate.max(self.local_rate * plasticity.max(1.0));
        ensure!(
            self.weight_decay * rate * f64::from(self.gain_max) < 1.0,
            "connected weight_decay * max(rate, local_rate) * gain_max must be below 1"
        );
        ensure!(
            matches!(config.data, Data::Corpus { .. }),
            "connected training requires corpus data"
        );
        ensure!(
            !config.float_forward
                && config.history
                && config.batch == 1
                && config.decay == Decay::Constant
                && config.explore == 0.0,
            "connected training requires ternary, history on, batch 1, constant rate and explore 0"
        );
        Ok(())
    }
}

#[cfg(test)]
#[path = "connected_config_contracts.rs"]
mod tests;
