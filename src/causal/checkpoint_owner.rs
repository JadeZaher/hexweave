use super::checkpoint_state::CheckpointRef;
use super::storage_paths::{hash_valid, identifier, DISK_CAP};
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Budget {
    pub maximum_opportunities: u64,
    pub maximum_wall_millis: u64,
    pub maximum_attempts: u32,
    pub disk_bytes_per_attempt: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Consumed {
    pub opportunities: u64,
    pub wall_millis: u64,
    pub attempts: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Parent {
    pub attempt_id: String,
    pub checkpoint_sha256: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[allow(clippy::enum_variant_names)]
pub(super) enum Stop {
    Running,
    UserStop,
    ResourceStop,
    BudgetStop,
    Failed,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct OwnerEnvelope {
    pub schema: u32,
    pub run_id: String,
    pub attempt_id: String,
    pub parent: Option<Parent>,
    pub budget: Budget,
    pub consumed: Consumed,
    pub attempt_start_opportunity: u64,
    pub attempt_start_wall_millis: u64,
    pub observer_sequence: u64,
    pub stop: Stop,
    pub physical_work_unknown: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum Terminal {
    Completed,
    UserStop,
    ResourceStop,
    BudgetStop,
    Failed,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ProcessIdentity {
    pub pid: u32,
    pub creation_filetime: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Release {
    pub schema: u32,
    pub run_id: String,
    pub attempt_id: String,
    pub checkpoint: CheckpointRef,
    pub status: Terminal,
    pub consumed: Consumed,
    pub observer_sequence: u64,
    pub child_reaped: bool,
    pub observer_acknowledged: bool,
    pub physical_work_unknown: bool,
    pub owner_identity: ProcessIdentity,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ResumeRequest {
    pub release_sha256: String,
    pub release_seal_sha256: String,
    pub explicit_user_renewal: bool,
    pub new_owner: OwnerEnvelope,
}

impl Budget {
    pub fn validate(&self, consumed: &Consumed) -> Result<()> {
        ensure!(
            self.maximum_opportunities > 0
                && self.maximum_wall_millis > 0
                && (1..=64).contains(&self.maximum_attempts)
                && (1_048_576..=DISK_CAP).contains(&self.disk_bytes_per_attempt),
            "invalid cumulative budget"
        );
        ensure!(
            consumed.attempts > 0 && consumed.attempts <= self.maximum_attempts,
            "attempt budget exceeded"
        );
        Ok(())
    }
}

impl OwnerEnvelope {
    pub fn validate(&self, opportunity: u64) -> Result<()> {
        ensure!(
            self.schema == 1 && identifier(&self.run_id) && identifier(&self.attempt_id),
            "invalid owner identity"
        );
        self.budget.validate(&self.consumed)?;
        ensure!(
            self.stop == Stop::BudgetStop
                || (self.consumed.opportunities <= self.budget.maximum_opportunities
                    && self.consumed.wall_millis <= self.budget.maximum_wall_millis),
            "cumulative budget exceeded without terminal budget STOP"
        );
        ensure!(
            self.attempt_start_opportunity <= opportunity
                && opportunity <= self.consumed.opportunities
                && self.attempt_start_wall_millis <= self.consumed.wall_millis,
            "owner work accounting mismatch"
        );
        if let Some(parent) = &self.parent {
            ensure!(
                identifier(&parent.attempt_id)
                    && parent.attempt_id != self.attempt_id
                    && hash_valid(&parent.checkpoint_sha256)
                    && self.consumed.attempts >= 2,
                "invalid parent attempt"
            );
        } else {
            ensure!(
                self.consumed.attempts == 1
                    && self.attempt_start_opportunity == 0
                    && self.attempt_start_wall_millis == 0,
                "fresh attempt reset mismatch"
            );
        }
        ensure!(
            !self.physical_work_unknown && self.stop != Stop::Failed,
            "unknown/failed attempt cannot publish a resumable state"
        );
        Ok(())
    }

    pub fn follows(&self, old: &Self) -> Result<()> {
        ensure!(
            self.run_id == old.run_id
                && self.attempt_id == old.attempt_id
                && self.parent == old.parent
                && self.budget == old.budget
                && self.attempt_start_opportunity == old.attempt_start_opportunity
                && self.attempt_start_wall_millis == old.attempt_start_wall_millis
                && self.consumed.attempts == old.consumed.attempts
                && self.consumed.opportunities >= old.consumed.opportunities
                && self.consumed.wall_millis >= old.consumed.wall_millis
                && self.observer_sequence >= old.observer_sequence,
            "owner envelope regressed or changed"
        );
        ensure!(
            old.stop == Stop::Running || self.stop == old.stop,
            "terminal STOP category changed"
        );
        Ok(())
    }
}

impl ResumeRequest {
    fn validate_release(
        &self,
        old: &OwnerEnvelope,
        release: &Release,
        checkpoint: &CheckpointRef,
    ) -> Result<()> {
        ensure!(
            hash_valid(&self.release_sha256) && hash_valid(&self.release_seal_sha256),
            "missing terminal release/seal SHA256"
        );
        ensure!(
            release.schema == 1
                && release.run_id == old.run_id
                && release.attempt_id == old.attempt_id
                && &release.checkpoint == checkpoint,
            "terminal release lineage mismatch"
        );
        ensure!(
            release.child_reaped && release.observer_acknowledged && !release.physical_work_unknown,
            "parent process/observer/physical-work release evidence missing"
        );
        ensure!(
            release.owner_identity.pid > 0 && release.owner_identity.creation_filetime > 0,
            "missing exact owner process identity"
        );
        ensure!(
            !matches!(release.status, Terminal::Failed | Terminal::BudgetStop),
            "terminal failure/budget stop cannot resume"
        );
        ensure!(
            !(release.status == Terminal::UserStop || old.stop == Stop::UserStop)
                || self.explicit_user_renewal,
            "user STOP requires explicit renewal"
        );
        ensure!(
            old.stop != Stop::BudgetStop,
            "checkpoint exhausted its budget"
        );
        old.budget.validate(&release.consumed)?;
        ensure!(
            release.consumed.opportunities < old.budget.maximum_opportunities
                && release.consumed.wall_millis < old.budget.maximum_wall_millis,
            "terminal release exhausted cumulative budget"
        );
        ensure!(
            release.consumed.opportunities >= old.consumed.opportunities
                && release.consumed.wall_millis >= old.consumed.wall_millis
                && release.consumed.attempts == old.consumed.attempts
                && release.observer_sequence >= old.observer_sequence,
            "terminal cumulative evidence regressed"
        );
        Ok(())
    }

    pub fn validate(
        &self,
        old: &OwnerEnvelope,
        release: &Release,
        checkpoint: &CheckpointRef,
        opportunity: u64,
    ) -> Result<()> {
        self.validate_release(old, release, checkpoint)?;
        let new = &self.new_owner;
        new.validate(opportunity)?;
        ensure!(
            new.run_id == old.run_id
                && new.attempt_id != old.attempt_id
                && new.parent
                    == Some(Parent {
                        attempt_id: old.attempt_id.clone(),
                        checkpoint_sha256: checkpoint.manifest_sha256.clone()
                    })
                && new.budget == old.budget
                && new.stop == Stop::Running,
            "new attempt identity/budget mismatch"
        );
        ensure!(
            release.consumed.attempts.checked_add(1) == Some(new.consumed.attempts)
                && new.consumed.opportunities >= release.consumed.opportunities
                && new.consumed.wall_millis >= release.consumed.wall_millis
                && new.attempt_start_wall_millis == release.consumed.wall_millis
                && new.attempt_start_opportunity == opportunity
                && new.observer_sequence > release.observer_sequence,
            "resume reset cumulative budgets or observer sequence"
        );
        ensure!(
            new.consumed.opportunities < new.budget.maximum_opportunities
                && new.consumed.wall_millis < new.budget.maximum_wall_millis,
            "no cumulative budget remains"
        );
        Ok(())
    }
}
