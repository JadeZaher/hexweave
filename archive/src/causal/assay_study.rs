use super::child_protocol::{Operation, Request};
use anyhow::{ensure, Result};

pub(super) fn validate(request: &Request) -> Result<()> {
    let Operation::RetainedAssay { study, arm } = &request.operation else {
        anyhow::bail!("assay operation required");
    };
    ensure!(
        *arm < 6 && study.path.is_absolute() && super::storage_paths::hash_valid(&study.sha256),
        "assay study pin/ordinal invalid"
    );
    let bytes = super::child_io::read(&study.path, 16_384)?;
    ensure!(
        super::storage_paths::digest(&bytes) == study.sha256,
        "assay study declaration changed"
    );
    let value: serde_json::Value = serde_json::from_slice(&bytes)?;
    let rule = if arm.is_multiple_of(2) {
        "dense"
    } else {
        "magnitude"
    };
    let seed = [7, 17, 29][arm / 2];
    let run = value["run_id"].as_str().unwrap_or("");
    ensure!(
        super::storage_paths::identifier(run)
            && study.path
                == request
                    .workspace
                    .join("work/causal-studies")
                    .join(run)
                    .join("study.json")
            && value["schema"] == 1
            && value["protocol"] == super::assay_state::PROTOCOL
            && value["whole_wall_millis"] == 7_200_000
            && value["disk_cap_bytes"] == 536_870_912
            && value["arms"][*arm]["seed"] == seed
            && value["arms"][*arm]["rule"] == rule
            && request.config.seed == seed
            && request.config.profile == format!("retained-{rule}")
            && request.resume.is_none()
            && request.owner.parent.is_none()
            && request.owner.budget.maximum_attempts == 1
            && request.owner.budget.maximum_wall_millis == 1_100_000
            && request.owner.budget.disk_bytes_per_attempt == 67_108_864,
        "assay escaped closed study account"
    );
    let mut identity = serde_json::to_value(&request.identity)?;
    identity.as_object_mut().unwrap().remove("config_sha256");
    ensure!(
        value["common_identity"] == identity,
        "assay study immutable identity changed"
    );
    Ok(())
}
