//! Batch boundaries, cosine rates and deterministic exploration; see AGENTS.md §Core v2.

use super::core_config::Config;
use super::geometry::TILES;
use super::tuning::{self, Knobs};
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(super) enum Decay {
    #[default]
    Constant,
    Cosine {
        anchors: u64,
        floor: f64,
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
        }
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
    if let Decay::Cosine { anchors, floor } = config.decay {
        ensure!(
            anchors > 0 && floor.is_finite() && (0.0..=1.0).contains(&floor),
            "cosine decay needs positive anchors and floor in [0, 1]"
        );
    }
    Ok(())
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
