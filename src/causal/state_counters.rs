use super::geometry::{Geometry, MATRIX_COUNT, TILES};
#[cfg(test)]
use super::predictor_joint::Update;
use super::state::count;
use super::tuning;
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Counters {
    pub anchors: u64,
    #[serde(default)]
    pub optimizer_steps: u64,
    pub local_opportunities: u64,
    pub visits: u64,
    pub repeated_visits: u64,
    pub sampled_routes: u64,
    pub halts: u64,
    pub forced_exits: u64,
    pub anchor_prefix_bytes: u64,
    pub local_prefix_bytes: u64,
    pub anchor_tile_commits: u64,
    pub local_tile_updates: u64,
    pub tile_transitions: [u64; MATRIX_COUNT],
    pub shared_transitions: u64,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub predictor_anchor_updates: u64,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub predictor_replay_updates: u64,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub predictor_anchor_pairs: u64,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub predictor_replay_pairs: u64,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub predictor_transitions: u64,
    #[serde(default, skip_serializing_if = "is_zero_float")]
    pub predictor_mse_sum: f64,
    pub anchor_loss_sum: f64,
    pub history_pruned: u64,
    pub history_peak: usize,
    pub maximum_prefix_bytes: usize,
}

fn is_zero(value: &u64) -> bool {
    *value == 0
}

fn is_zero_float(value: &f64) -> bool {
    value.to_bits() == 0
}

impl Counters {
    pub fn visit(
        &mut self,
        geometry: Geometry,
        id: usize,
        action: Option<usize>,
        depth: usize,
        seen: &mut u32,
    ) -> Result<()> {
        ensure!(id < TILES && depth < 4, "invalid visit metadata");
        count(&mut self.visits, 1)?;
        if *seen & (1 << id) != 0 {
            count(&mut self.repeated_visits, 1)?;
        }
        *seen |= 1 << id;
        if depth == 3 {
            ensure!(action.is_none(), "forced exit sampled a route");
            count(&mut self.forced_exits, 1)?;
        } else {
            let action = action.context("early visit missing route")?;
            ensure!(
                action < geometry.routes(id)?,
                "route action outside legal set"
            );
            count(&mut self.sampled_routes, 1)?;
            if action + 1 == geometry.routes(id)? {
                count(&mut self.halts, 1)?;
            }
        }
        Ok(())
    }

    pub fn transitions(&mut self, transitions: [usize; MATRIX_COUNT]) -> Result<()> {
        for (counter, amount) in self.tile_transitions.iter_mut().zip(transitions) {
            count(counter, amount)?;
        }
        Ok(())
    }

    #[cfg(test)]
    pub fn predictor(&mut self, update: Update, replay: bool) -> Result<()> {
        ensure!(
            (1..=4).contains(&update.pair_mse.len())
                && update
                    .pair_mse
                    .iter()
                    .all(|mse| mse.is_finite() && *mse >= 0.0),
            "invalid predictor metric"
        );
        count(&mut self.predictor_transitions, update.transitions)?;
        count(
            if replay {
                &mut self.predictor_replay_updates
            } else {
                &mut self.predictor_anchor_updates
            },
            1,
        )?;
        count(
            if replay {
                &mut self.predictor_replay_pairs
            } else {
                &mut self.predictor_anchor_pairs
            },
            update.pair_mse.len(),
        )?;
        for mse in update.pair_mse {
            self.predictor_mse_sum += f64::from(mse);
        }
        ensure!(
            self.predictor_mse_sum.is_finite(),
            "predictor metric overflow"
        );
        Ok(())
    }

    pub fn validate_schedule(
        &self,
        opportunity: u64,
        geometry: Geometry,
        assay: bool,
        history_empty: bool,
    ) -> Result<()> {
        let anchors = if assay {
            opportunity
        } else {
            opportunity / 4 + u64::from(!opportunity.is_multiple_of(4))
        };
        ensure!(
            self.anchors == anchors && self.local_opportunities == opportunity - anchors,
            "opportunity schedule counters mismatch"
        );
        let exits = self
            .halts
            .checked_add(self.forced_exits)
            .context("exit count overflow")?;
        ensure!(
            exits == opportunity
                && self.visits >= opportunity
                && self.visits <= opportunity.saturating_mul(4),
            "visit/exit counters mismatch"
        );
        ensure!(
            self.sampled_routes == self.visits - self.forced_exits
                && self.repeated_visits <= self.visits,
            "route/revisit counters mismatch"
        );
        ensure!(
            self.local_tile_updates >= self.local_opportunities
                && self.local_tile_updates <= self.local_opportunities.saturating_mul(4),
            "local update counters mismatch"
        );
        ensure!(
            self.predictor_replay_updates <= self.local_opportunities
                && self.predictor_replay_pairs >= self.predictor_replay_updates
                && self.predictor_replay_pairs <= self.predictor_replay_updates.saturating_mul(4),
            "predictor replay counters mismatch"
        );
        // `tuning::Knobs::history` off: `credit::anchor` never calls `train_records`, so no
        // anchor pair/update is ever produced and `History` stays empty; see AGENTS.md §State
        // encoding. On, this is the original anchors-equal-updates invariant unchanged.
        if !tuning::current().fit_predictor {
            let batch = u64::from(tuning::current().batch);
            ensure!(
                assay
                    && batch > 0
                    && self.optimizer_steps.checked_mul(batch) == Some(self.anchors)
                    && self.anchor_tile_commits >= self.optimizer_steps
                    && self.anchor_tile_commits
                        <= self.optimizer_steps.saturating_mul(TILES as u64)
                    && self.anchor_tile_commits <= self.visits - self.repeated_visits
                    && self.predictor_anchor_updates == 0
                    && self.predictor_anchor_pairs == 0
                    && self.predictor_replay_updates == 0
                    && self.predictor_replay_pairs == 0
                    && self.predictor_transitions == 0
                    && self.predictor_mse_sum == 0.0
                    && (tuning::current().history || history_empty),
                "v2 optimizer/history counters mismatch"
            );
            if batch == 1 {
                ensure!(
                    self.anchor_tile_commits.checked_add(self.repeated_visits) == Some(self.visits),
                    "v2 single-anchor visit accounting mismatch"
                );
            }
        } else if tuning::current().history {
            ensure!(
                self.predictor_anchor_updates == self.anchors
                    && self.predictor_anchor_pairs >= self.anchors
                    && self.predictor_anchor_pairs <= self.anchors.saturating_mul(4)
                    && self
                        .predictor_anchor_pairs
                        .checked_add(self.local_tile_updates)
                        == Some(self.visits)
                    && self.anchor_tile_commits >= self.anchors
                    && self.anchor_tile_commits <= self.predictor_anchor_pairs,
                "predictor anchor counters mismatch"
            );
        } else {
            ensure!(
                self.predictor_anchor_updates == 0
                    && self.predictor_anchor_pairs == 0
                    && history_empty
                    && self.anchor_tile_commits >= self.anchors
                    && self.anchor_tile_commits <= self.visits
                    && (!assay
                        || self.anchor_tile_commits.checked_add(self.repeated_visits)
                            == Some(self.visits)),
                "history-off predictor/history counters mismatch"
            );
        }
        ensure!(
            self.predictor_transitions
                <= self
                    .predictor_anchor_updates
                    .saturating_add(self.predictor_replay_updates)
                    .saturating_mul(geometry.width as u64),
            "update/visit accounting mismatch"
        );
        ensure!(
            self.history_peak <= 8 && self.maximum_prefix_bytes <= geometry.context,
            "bounded allocation counters mismatch"
        );
        ensure!(
            self.predictor_mse_sum.is_finite()
                && self.predictor_mse_sum >= 0.0
                && self.anchor_loss_sum.is_finite()
                && self.anchor_loss_sum >= 0.0,
            "invalid aggregate metric"
        );
        Ok(())
    }
}
