//! Training-time route selection: sampled, or greedy with per-anchor exploration; see
//! CONNECTED.md §Greedy training routes.

use super::connected_balance::{self as balance, Steer};
use super::connected_config::Options;
use super::connected_state::State;
use super::execution::destination;
use super::geometry::TILES;
use super::routing;
use super::tile::{Forward, Tile};
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

/// A lane job's route draw, never tilted by a balance bonus (lane draws also serve evaluation);
/// lanes share the live anchor's exploration decision (key `[anchors]`).
pub(super) fn lane(options: &Options, state: &State, index: usize, depth: usize) -> Draw {
    let words = [state.anchors, index as u64, depth as u64, 0x4c4f4341];
    let seed = tuning::hash(state.seed, &words);
    Draw {
        random: random(options, state.seed, &[state.anchors], seed),
        bonus: None,
    }
}

/// The live anchor's route draw (salt "ANCH"), also taken by the contextual-row resume refill.
pub(super) fn anchor(options: &Options, state: &State, anchor: u64) -> Draw {
    draw(
        options,
        state,
        &[anchor],
        tuning::hash(state.seed, &[anchor, 0x414e_4348]),
    )
}

/// One training pass's route draw: `random`'s choice plus any visit-balance bonus.
pub(super) fn draw(options: &Options, state: &State, key: &[u64], seed: u64) -> Draw {
    Draw {
        random: random(options, state.seed, key, seed),
        bonus: balance::bonus(options, state.balance.as_ref()),
    }
}

/// A pass's route choice: RNG (`None` is argmax) and per-destination-tile logit bonus.
#[derive(Clone, Copy, Debug)]
pub(super) struct Draw {
    pub random: Option<u64>,
    pub bonus: Option<[f32; TILES]>,
}

impl Draw {
    /// Evaluation and completion: argmax of the unadjusted policy.
    pub const GREEDY: Self = Self {
        random: None,
        bonus: None,
    };

    /// Chooses an action, each neighbour's logit raised by its destination's bonus when present.
    pub fn route(
        &mut self,
        tile: &Tile,
        forward: &Forward,
        history: &[f32],
    ) -> Result<(usize, Option<Steer>)> {
        // An all-zero bonus (warm-up, neutral estimate) takes the untilted path bit for bit.
        let Some(bonus) = self
            .bonus
            .filter(|bonus| bonus.iter().any(|&value| value != 0.0))
        else {
            return Ok((tile.route(forward, history, &mut self.random)?, None));
        };
        let policy = tile.route_probabilities(forward, history)?;
        let mesh = super::geometry::mesh();
        let (mut adjusted, mut magnitude, mut neighbours) = (policy.clone(), 0.0f32, 0u16);
        for (action, probability) in adjusted.iter_mut().enumerate() {
            if let Some(next) = destination(mesh, tile.id, Some(action))? {
                // p e^b is softmax(logit + b) before renormalization.
                *probability *= bonus[next].exp();
                magnitude += bonus[next].abs();
                neighbours += 1;
            }
        }
        let total: f32 = adjusted.iter().sum();
        adjusted.iter_mut().for_each(|p| *p /= total);
        let plain = routing::choose(&policy, &mut None)?;
        let changed = routing::choose(&adjusted, &mut None)? != plain;
        let action = routing::choose(&adjusted, &mut self.random)?;
        let bonus = magnitude / f32::from(neighbours.max(1));
        Ok((action, Some(Steer { changed, bonus })))
    }
}
