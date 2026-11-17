//! `connected.restart` controls and admission; see CONNECTED.md §Warm restarts.

use super::geometry::TILES;
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};

/// Revision lag past which a tile's gain and memory records expire (`Gain::value_at`, `eligible`).
const REVISION_LAG: u32 = 8;

/// Commits a visited tile takes per anchor: its route step and its one local step.
const ANCHOR_COMMITS: u32 = 2;

/// Restart replays keep the tile's newest records and gain inside the lag: 8 - 2 = 6.
pub(super) const MAX_RESTART_REPLAY: u32 = REVISION_LAG - ANCHOR_COMMITS;

/// Largest per-tile plasticity the dial admits.
pub(super) const MAX_PLASTICITY: f64 = 4.0;

/// What a tile's `drift` measures: the shared change stream or its own input distance.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(super) enum Signal {
    #[default]
    Shared,
    Tile,
}

/// Plasticity law: discrete warm restarts, or the experimental continuous Lambert dial.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(super) enum Law {
    #[default]
    Restart,
    Lambert,
}

/// Tile rate factor floor, restart threshold, adaptation and the per-tile restart options.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(super) struct Restart {
    pub floor: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub threshold: Option<f64>,
    #[serde(default)]
    pub adapt: f64,
    /// A restart drops the tile's memory records and the replay buffer.
    #[serde(default = "yes", skip_serializing_if = "is_true")]
    pub forget: bool,
    #[serde(default, skip_serializing_if = "is_shared")]
    pub signal: Signal,
    /// Minimum anchors between any two tile restarts; 0 is off.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub stagger: u64,
    /// The adaptive bar stays within [threshold / clamp, threshold * clamp].
    #[serde(default = "default_clamp", skip_serializing_if = "is_default_clamp")]
    pub clamp: f64,
    /// Recent visited positions re-run for a restarted tile only; 0 is off.
    #[serde(default, skip_serializing_if = "is_zero_u32")]
    pub replay: u32,
    /// Initial per-tile plasticity dial; absent is all 1.0.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plasticity: Option<[f64; TILES]>,
    #[serde(default, skip_serializing_if = "is_restart_law")]
    pub law: Law,
}

fn yes() -> bool {
    true
}

fn is_true(value: &bool) -> bool {
    *value
}

fn is_shared(signal: &Signal) -> bool {
    *signal == Signal::Shared
}

fn is_zero(value: &u64) -> bool {
    *value == 0
}

fn is_zero_u32(value: &u32) -> bool {
    *value == 0
}

fn default_clamp() -> f64 {
    8.0
}

fn is_default_clamp(value: &f64) -> bool {
    *value == default_clamp()
}

fn is_restart_law(law: &Law) -> bool {
    *law == Law::Restart
}

impl Default for Restart {
    fn default() -> Self {
        Self {
            floor: 0.0,
            threshold: None,
            adapt: 0.0,
            forget: true,
            signal: Signal::Shared,
            stagger: 0,
            clamp: default_clamp(),
            replay: 0,
            plasticity: None,
            law: Law::Restart,
        }
    }
}

impl Restart {
    /// Admission of each field, and of discrete-restart options only where they act.
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.floor.is_finite() && (0.0..=1.0).contains(&self.floor),
            "connected restart floor must be in [0, 1]"
        );
        ensure!(
            self.threshold
                .is_none_or(|bar| bar.is_finite() && bar > 0.0),
            "connected restart threshold must be positive and finite"
        );
        ensure!(
            self.adapt.is_finite()
                && (0.0..1.0).contains(&self.adapt)
                && (self.adapt == 0.0 || self.threshold.is_some()),
            "connected restart adapt must be in [0, 1) and needs a threshold"
        );
        ensure!(
            self.clamp.is_finite() && (1.0..=1024.0).contains(&self.clamp),
            "connected restart clamp must be in [1, 1024]"
        );
        ensure!(
            self.replay <= MAX_RESTART_REPLAY,
            "connected restart replay must be 0..=6"
        );
        ensure!(
            self.plasticity.is_none_or(|dial| dial
                .iter()
                .all(|p| p.is_finite() && *p > 0.0 && *p <= MAX_PLASTICITY)),
            "connected restart plasticity must be in (0, 4]"
        );
        // Each of these acts only on a restart, so without a threshold it would silently do nothing.
        let inert =
            !self.forget || self.signal == Signal::Tile || self.stagger > 0 || self.replay > 0;
        ensure!(
            self.threshold.is_some() || !inert,
            "connected restart forget, signal, stagger and replay need a threshold"
        );
        if self.law == Law::Lambert {
            ensure!(
                self.signal == Signal::Tile && self.threshold.is_some() && self.floor > 0.0,
                "connected restart law lambert needs signal tile, a threshold and floor > 0"
            );
            ensure!(
                self.adapt == 0.0
                    && self.stagger == 0
                    && self.clamp == default_clamp()
                    && self.replay == 0
                    && self.forget,
                "connected restart law lambert refuses adapt, stagger, clamp, replay and forget"
            );
        }
        Ok(())
    }

    /// Largest initial plasticity, or the Lambert law's cap, for the decay-safety bound.
    pub fn max_plasticity(&self) -> f64 {
        match (self.law, self.plasticity) {
            (Law::Lambert, _) => MAX_PLASTICITY,
            (Law::Restart, Some(dial)) => dial.iter().copied().fold(0.0, f64::max),
            (Law::Restart, None) => 1.0,
        }
    }

    /// The configured dial of `tile`: its `plasticity` entry, or 1.0.
    pub fn dial(&self, tile: usize) -> f64 {
        self.plasticity.map_or(1.0, |dial| dial[tile])
    }
}
