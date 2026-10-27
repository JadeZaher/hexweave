use super::checkpoint_owner::{OwnerEnvelope, ProcessIdentity, Release, ResumeRequest};
use super::checkpoint_state::{Identity, MANIFEST_CAP};
use super::storage_paths as paths;
use anyhow::{ensure, Result};
use serde::Deserialize;
use std::path::Path;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Ready {
    schema: u32,
    release_sha256: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Claim {
    schema: u32,
    release_sha256: String,
    new_attempt_id: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Seal {
    schema: u32,
    release_sha256: String,
    owner_identity: ProcessIdentity,
    finalizer_identity: ProcessIdentity,
    owner_reaped: bool,
    owner_exit_code: u32,
    external_wall_millis: u64,
    source_sha256: String,
    runtime_sha256: String,
}

pub(super) fn read(
    workspace: &Path,
    parent: &Path,
    request: &ResumeRequest,
    identity: &Identity,
    old: &OwnerEnvelope,
) -> Result<Release> {
    ensure!(
        paths::identifier(&request.new_owner.run_id)
            && paths::hash_valid(&request.release_sha256)
            && paths::hash_valid(&request.release_seal_sha256),
        "invalid release/claim pins"
    );
    ensure!(
        !parent.join("release-invalid.json").try_exists()?,
        "terminal release was revoked"
    );
    let ready: Ready = serde_json::from_slice(&paths::read(parent, "release-ready.json", 4096)?)?;
    ensure!(
        ready.schema == 1 && ready.release_sha256 == request.release_sha256,
        "terminal release ready marker mismatch"
    );
    let claims = workspace
        .join("work/causal")
        .join(&request.new_owner.run_id)
        .join("claims");
    let claims = paths::attempt(workspace, &claims)?;
    let claim: Claim = serde_json::from_slice(&paths::read(
        &claims,
        &format!("{}.json", request.release_sha256),
        4096,
    )?)?;
    ensure!(
        claim.schema == 1
            && claim.release_sha256 == request.release_sha256
            && claim.new_attempt_id == request.new_owner.attempt_id,
        "external one-use resume claim mismatch"
    );
    let bytes = paths::read(parent, "release.json", MANIFEST_CAP)?;
    ensure!(
        paths::digest(&bytes) == request.release_sha256,
        "terminal release hash mismatch"
    );
    let release: Release = serde_json::from_slice(&bytes)?;
    let seal_bytes = paths::read(parent, "release-seal.json", 4096)?;
    ensure!(
        paths::digest(&seal_bytes) == request.release_seal_sha256,
        "external release seal pin mismatch"
    );
    let seal: Seal = serde_json::from_slice(&seal_bytes)?;
    validate_seal(&seal, &release, request, identity, old)?;
    Ok(release)
}

fn validate_seal(
    seal: &Seal,
    release: &Release,
    request: &ResumeRequest,
    identity: &Identity,
    old: &OwnerEnvelope,
) -> Result<()> {
    ensure!(
        seal.schema == 1
            && seal.release_sha256 == request.release_sha256
            && seal.owner_identity == release.owner_identity
            && seal.owner_reaped
            && seal.owner_exit_code == 0,
        "owner was not externally reaped successfully"
    );
    ensure!(
        seal.finalizer_identity.pid > 0
            && seal.finalizer_identity.creation_filetime > 0
            && seal.finalizer_identity != seal.owner_identity,
        "invalid external finalizer identity"
    );
    ensure!(
        seal.source_sha256 == identity.source_sha256
            && seal.runtime_sha256 == identity.runtime_sha256,
        "external seal source/runtime mismatch"
    );
    let charged = release
        .consumed
        .wall_millis
        .checked_sub(old.attempt_start_wall_millis)
        .ok_or_else(|| anyhow::anyhow!("external wall charge underflow"))?;
    ensure!(
        seal.external_wall_millis > 0 && seal.external_wall_millis <= charged,
        "external whole-owner wall time exceeds conserved charge"
    );
    Ok(())
}
