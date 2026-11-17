//! Batch boundaries, rate schedules and deterministic exploration; see AGENTS.md §Core v2.

use super::core_config::Config;
use super::geometry::TILES;
use super::tuning::{self, Knobs};
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};

/// A rate multiplier by anchor index (serde `kind`: `constant`, `cosine`, `inverse_sqrt`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(super) enum Decay {
    #[default]
    Constant,
    Cosine {
        anchors: u64,
        floor: f64,
    },
    /// Inverse square-root decay `1 / sqrt(1 + anchor / anchors)`, after Vaswani et al. 2017
    /// (https://arxiv.org/abs/1706.03762), whose warmup form is `min(step^-0.5, step * w^-1.5)`.
    InverseSqrt {
        anchors: u64,
    },
}

impl Decay {
    pub fn factor(self, anchor: u64) -> f64 {
        match self {
            Self::Constant => 1.0,
            Self::Cosine { anchors, floor } => {
                let phase = std::f64::consts::PI * anchor.min(anchors) as f64 / anchors as f64;
                floor + (1.0 - floor) * 0.5 * (1.0 + phase.cos())
            }
            Self::InverseSqrt { anchors } => (1.0 + anchor as f64 / anchors as f64).sqrt().recip(),
        }
    }

    /// Shared admission: positive anchors, and a cosine floor in [0, 1].
    pub fn validate(self) -> Result<()> {
        match self {
            Self::Constant => {}
            Self::Cosine { anchors, floor } => ensure!(
                anchors > 0 && floor.is_finite() && (0.0..=1.0).contains(&floor),
                "cosine decay needs positive anchors and floor in [0, 1]"
            ),
            Self::InverseSqrt { anchors } => {
                ensure!(anchors > 0, "inverse_sqrt decay needs positive anchors")
            }
        }
        Ok(())
    }
}

pub(super) fn validate(config: &Config) -> Result<()> {
    ensure!((1..=32).contains(&config.batch), "batch must be in 1..=32");
    ensure!(
        config.anchors.is_multiple_of(u64::from(config.batch)),
        "anchors must end on a complete batch boundary"
    );
    ensure!(
        !config.float_forward || config.batch == 1,
        "float_forward requires batch 1 while latents are thread-local"
    );
    ensure!(
        config.explore.is_finite() && (0.0..1.0).contains(&config.explore),
        "explore must be in [0, 1)"
    );
    config.decay.validate()
}

pub(super) fn at(mut knobs: Knobs, anchor: u64) -> Knobs {
    let factor = knobs.decay.factor(anchor);
    knobs.tile_rate = knobs.tile_rate.map(|rate| rate * factor);
    knobs.router_rate = knobs.router_rate.map(|rate| rate * factor);
    knobs.embed_rate = knobs.embed_rate.map(|rate| rate * factor);
    knobs
}

pub(super) fn crossed(before: u64, after: u64, interval: u64) -> bool {
    before / interval != after / interval
}

pub(super) fn tile_steps() -> [u64; TILES] {
    let mut counts = [0; TILES];
    for ((id, _), steps) in tuning::steps() {
        if id < TILES {
            counts[id] = counts[id].max(steps);
        }
    }
    counts
}

pub(super) fn explore(
    knobs: Knobs,
    anchor: u64,
    depth: usize,
    neighbors: &[usize],
    counts: &[u64; TILES],
) -> Option<usize> {
    let draw = (tuning::hash(knobs.seed, &[anchor, depth as u64]) >> 11) as f64
        * (1.0 / ((1u64 << 53) as f64));
    if depth == 0 || draw >= knobs.explore {
        return None;
    }
    neighbors
        .iter()
        .enumerate()
        .min_by_key(|(_, id)| (counts[**id], **id))
        .map(|(action, _)| action)
}

#[cfg(test)]
mod tests {
    use super::Decay;

    #[test]
    fn decay_factors_follow_their_schedules() {
        let cosine = Decay::Cosine {
            anchors: 100,
            floor: 0.1,
        };
        let inverse = Decay::InverseSqrt { anchors: 8 };
        // (schedule, anchor, factor): existing kinds unchanged; t / anchors = 0, 1, 3 for inverse.
        for (decay, anchor, expected) in [
            (Decay::Constant, 1_000, 1.0),
            (cosine, 0, 1.0),
            (cosine, 50, 0.55),
            (cosine, 1_000, 0.1),
            (inverse, 0, 1.0),
            (inverse, 8, std::f64::consts::FRAC_1_SQRT_2),
            (inverse, 24, 0.5),
        ] {
            let factor = decay.factor(anchor);
            assert!(
                (factor - expected).abs() < 1e-12,
                "{decay:?} at {anchor}: {factor}"
            );
        }
    }
}
