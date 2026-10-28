use super::assay_update::{Metrics, Rule};
use super::state::State;
use super::storage_paths::hash_valid;
use anyhow::{ensure, Result};
use serde::{Deserialize, Deserializer, Serialize};

#[cfg(test)]
pub(super) const PROTOCOL: &str = "causal-retained-text-assay-v1";
pub(super) const MANIFEST_SHA: &str =
    "0a77b8fe038c8139314e7331417ced5aa853805ec26562680c3d770a01ade6db";
pub(super) const ROWS_SHA: &str =
    "0ad6de82f45fb0cff217ad34ac2ebe60eaff2ef1ce32399ef892b36ecbfc40b9";
pub(super) const SCHEDULE_SHA: &str =
    "7ceefd7f72531f01c3336af6779d89e4e1ac5cfd649ba2b5f94f62004501ad3c";
pub(super) const PASSES: [u64; 8] = [0, 1, 2, 4, 8, 16, 17, 20];
pub(super) const GENERATIONS: [u64; 6] = [0, 4, 8, 16, 17, 20];
pub(super) const ANCHORS: u64 = 1600;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ObservationRef {
    pub pass: u64,
    pub sha256: String,
    pub bytes: u64,
    pub complete: bool,
    pub teacher_readouts: u64,
    pub generated_readouts: u64,
}

#[cfg(test)]
impl ObservationRef {
    pub fn name(&self) -> String {
        format!("assay-observation-{}.json", self.sha256)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Progress {
    pub schema: u32,
    pub rule: Rule,
    pub embedding: Metrics,
    pub observations: Vec<ObservationRef>,
}

// Present null is rejected; absence alone retains exact v3 acceptance.
pub(super) fn present<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<Progress>, D::Error> {
    Progress::deserialize(deserializer).map(Some)
}

impl Progress {
    #[cfg(test)]
    pub fn new(rule: Rule) -> Self {
        Self {
            schema: 1,
            rule,
            embedding: Metrics::default(),
            observations: Vec::new(),
        }
    }

    pub fn validate(&self, state: &State) -> Result<()> {
        let n = state.global_opportunity;
        ensure!(
            self.schema == 1 && n <= ANCHORS,
            "invalid assay schema/exposure"
        );
        ensure!(
            state.cursor.data == super::assay_source::identity(),
            "assay data identity mismatch"
        );
        ensure!(
            Some(self.embedding.attempts)
                == self
                    .embedding
                    .clamped
                    .checked_add(self.embedding.transitions)
                && self.embedding.transitions == state.counters.shared_transitions
                && self.embedding.attempts <= n * state.shared.embedding.len() as u64
                && self.embedding.probability_sum.is_finite()
                && self.embedding.probability_sum >= 0.0
                && self.embedding.probability_sum
                    <= (n * state.shared.embedding.len() as u64) as f64,
            "assay E ledger mismatch"
        );
        ensure!(
            self.rule != Rule::Dense
                || self.embedding.probability_sum == self.embedding.attempts as f64,
            "dense E probability ledger mismatch"
        );
        ensure!(
            self.observations.len() <= PASSES.len(),
            "assay observation count"
        );
        for (index, entry) in self.observations.iter().enumerate() {
            ensure!(
                entry.pass == PASSES[index]
                    && entry.pass * 80 <= n
                    && hash_valid(&entry.sha256)
                    && (1..=65_536).contains(&entry.bytes)
                    && entry.teacher_readouts <= 240
                    && entry.generated_readouts <= 384
                    && (GENERATIONS.contains(&entry.pass) || entry.generated_readouts == 0)
                    && (!entry.complete || entry.teacher_readouts == 240)
                    && (entry.complete || index + 1 == self.observations.len()),
                "assay observation ledger"
            );
        }
        let due = PASSES.iter().filter(|pass| **pass * 80 < n).count();
        ensure!(
            self.observations.len() >= due
                && self
                    .observations
                    .iter()
                    .take(due)
                    .all(|entry| entry.complete),
            "missing prior observation"
        );
        ensure!(
            state.counters.local_prefix_bytes == 0
                && state.counters.local_tile_updates == 0
                && state.counters.predictor_replay_updates == 0
                && state.counters.predictor_replay_pairs == 0,
            "AO ledger contains local work"
        );
        super::assay_source::validate_cursor(&state.cursor)
    }
}
