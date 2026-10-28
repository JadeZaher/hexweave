use super::checkpoint_owner::{OwnerEnvelope, Stop, Terminal};
use super::checkpoint_state::{CheckpointRef, Identity};
use super::config::Config;
use super::geometry::Geometry;
use super::storage_paths::hash_valid;
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Limits {
    pub opportunities: u64,
    pub wall_millis: u64,
    pub stop_reserve_millis: u64,
    pub checkpoint_every: u64,
    pub evaluation_targets: u64,
    pub generation_cases: usize,
    pub generation_bytes: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct SourceManifest {
    pub path: PathBuf,
    pub sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", deny_unknown_fields)]
pub(super) enum Operation {
    Train,
    RetainedAssay { study: SourceManifest, arm: usize },
    PromptEvaluation { prompts: SourceManifest },
}

impl Operation {
    pub fn is_prompt(&self) -> bool {
        matches!(self, Self::PromptEvaluation { .. })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Resume {
    pub parent_directory: PathBuf,
    pub release_path: PathBuf,
    pub release_sha256: String,
    pub release_seal_sha256: String,
    pub checkpoint: CheckpointRef,
    pub explicit_user_renewal: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Request {
    pub schema: u32,
    pub protocol: String,
    pub workspace: PathBuf,
    pub attempt_directory: PathBuf,
    pub owner: OwnerEnvelope,
    pub identity: Identity,
    pub config: Config,
    pub data_root: PathBuf,
    pub data_manifest_sha256: String,
    pub kernel: String,
    pub limits: Limits,
    pub stop_path: PathBuf,
    pub source_manifest: SourceManifest,
    pub launch_receipt: PathBuf,
    pub resume: Option<Resume>,
    pub operation: Operation,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct StopFile {
    pub schema: u32,
    pub reason: Terminal,
    pub observer_sequence: u64,
}

impl Request {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == 1 && super::config::protocol_hash(&self.protocol).is_some(),
            "worker request protocol mismatch"
        );
        self.config.validate()?;
        self.identity.validate()?;
        ensure!(
            self.identity.geometry == Geometry::candidate()
                && self.identity.config_sha256 == self.config.sha256()?
                && self.identity.protocol == self.protocol
                && self.config.protocol == self.protocol
                && (matches!(self.operation, Operation::RetainedAssay { .. })
                    == self.config.assay_rule().is_some())
                && self.identity.kernel == self.kernel,
            "worker model/config/kernel pin mismatch"
        );
        ensure!(
            self.data_manifest_sha256 == self.identity.data.manifest_sha256
                && self.source_manifest.sha256 == self.identity.source_sha256
                && hash_valid(&self.source_manifest.sha256),
            "worker data/source pin mismatch"
        );
        self.owner.validate(self.owner.attempt_start_opportunity)?;
        ensure!(
            self.owner.stop == Stop::Running && !self.owner.physical_work_unknown,
            "worker owner is terminal/unknown"
        );
        self.validate_limits()?;
        if let Operation::PromptEvaluation { prompts } = &self.operation {
            ensure!(
                self.resume.is_some(),
                "prompt evaluation requires completed parent"
            );
            ensure!(
                prompts.path.is_absolute() && hash_valid(&prompts.sha256),
                "prompt input must have absolute path and SHA256"
            );
        }
        ensure!(
            self.workspace.is_absolute()
                && self.attempt_directory.is_absolute()
                && self.data_root.is_absolute(),
            "worker paths must be absolute"
        );
        ensure!(
            self.stop_path == self.attempt_directory.join("worker-stop.json")
                && self.launch_receipt == self.attempt_directory.join("launch.json"),
            "worker control path mismatch"
        );
        if let Some(resume) = &self.resume {
            let parent = resume
                .parent_directory
                .parent()
                .context("missing parent attempt")?;
            ensure!(
                resume.parent_directory.file_name().and_then(|v| v.to_str()) == Some("persistence")
                    && resume.release_path == parent.join("release.json")
                    && hash_valid(&resume.release_sha256)
                    && hash_valid(&resume.release_seal_sha256),
                "resume path/release identity mismatch"
            );
            resume.checkpoint.validate()?;
        } else {
            ensure!(
                self.owner.parent.is_none(),
                "fresh worker has parent lineage"
            );
        }
        Ok(())
    }

    fn validate_limits(&self) -> Result<()> {
        let limits = &self.limits;
        let remaining_ops = self
            .owner
            .budget
            .maximum_opportunities
            .checked_sub(self.owner.consumed.opportunities)
            .context("opportunity budget underflow")?;
        let remaining_wall = self
            .owner
            .budget
            .maximum_wall_millis
            .checked_sub(self.owner.consumed.wall_millis)
            .context("wall budget underflow")?;
        ensure!(
            limits.opportunities <= remaining_ops
                && limits.wall_millis <= remaining_wall
                && limits.stop_reserve_millis >= 5000
                && limits.wall_millis > limits.stop_reserve_millis,
            "invalid attempt/cumulative budgets"
        );
        ensure!(
            (1..=192).contains(&limits.generation_bytes),
            "invalid generation byte limit"
        );
        if matches!(self.operation, Operation::RetainedAssay { .. }) {
            super::assay_study::validate(self)?;
            ensure!(
                self.identity.data == super::assay_source::identity()
                    && self.owner.attempt_start_opportunity <= 1600
                    && limits.opportunities == 1600 - self.owner.attempt_start_opportunity
                    && limits.opportunities > 0
                    && (1..=64).contains(&limits.checkpoint_every)
                    && limits.evaluation_targets == 1920
                    && limits.generation_cases == 48
                    && limits.generation_bytes == 8
                    && self.owner.budget.maximum_opportunities == 1601,
                "invalid retained assay course/limits"
            );
        } else if self.operation.is_prompt() {
            ensure!(
                remaining_ops > 0
                    && limits.opportunities == 0
                    && limits.checkpoint_every == 0
                    && limits.evaluation_targets == 0
                    && (1..=40).contains(&limits.generation_cases),
                "invalid prompt-only limits"
            );
        } else {
            ensure!(
                (1..=1_000_000).contains(&limits.opportunities)
                    && (1..=limits.opportunities).contains(&limits.checkpoint_every)
                    && (1..=65536).contains(&limits.evaluation_targets)
                    && (1..=32).contains(&limits.generation_cases),
                "invalid training/evaluation limits"
            );
        }
        Ok(())
    }
}
