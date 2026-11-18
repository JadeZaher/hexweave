//! Training-time route selection: sampled, or greedy with per-anchor exploration; see
//! CONNECTED.md §Greedy training routes.

use super::connected_config::Options;
use super::connected_state::State;
use super::tuning;
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};

/// Hash salt of the per-anchor exploration draw (`EXPL`).
const EXPLORE_SALT: u64 = 0x4558_504c;

/// How training passes choose route actions; evaluation and completion are always greedy.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(super) enum RouteTraining {
    #[default]
    Sampled,
    Greedy,
}

impl RouteTraining {
    pub const fn is_default(&self) -> bool {
        matches!(self, Self::Sampled)
    }
}

/// Serialization skip for the default exploration share.
pub(super) fn is_zero(value: &f64) -> bool {
    *value == 0.0
}

/// Admission: `route_explore` in [0, 1), and nonzero only with greedy training routes.
pub(super) fn validate(options: &Options) -> Result<()> {
    let explore = options.route_explore;
    ensure!(
        explore.is_finite() && (0.0..1.0).contains(&explore),
        "connected route_explore must be in [0, 1)"
    );
    ensure!(
        explore == 0.0 || options.route_training == RouteTraining::Greedy,
        "connected route_explore requires route_training greedy"
    );
    Ok(())
}

/// The route RNG for one training pass keyed by `key`: `Some(random)` samples, `None` is argmax.
pub(super) fn random(options: &Options, seed: u64, key: &[u64], random: u64) -> Option<u64> {
    if options.route_training == RouteTraining::Sampled {
        return Some(random);
    }
    let words: Vec<u64> = key.iter().copied().chain([EXPLORE_SALT]).collect();
    // The top 53 bits as a uniform draw in [0, 1).
    let draw = (tuning::hash(seed, &words) >> 11) as f64 / (1u64 << 53) as f64;
    (draw < options.route_explore).then_some(random)
}

/// A lane job's route RNG; lanes share the live anchor's exploration decision (key `[anchors]`).
pub(super) fn lane(options: &Options, state: &State, index: usize, depth: usize) -> Option<u64> {
    let words = [state.anchors, index as u64, depth as u64, 0x4c4f4341];
    random(
        options,
        state.seed,
        &[state.anchors],
        tuning::hash(state.seed, &words),
    )
}
