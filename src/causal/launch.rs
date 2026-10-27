use super::child_protocol::Request;
use super::{child_io, storage_paths};
use anyhow::{ensure, Context, Result};
use serde::Deserialize;
use std::path::Path;

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ProcessIdentity {
    pub pid: u32,
    pub creation_filetime: u64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Receipt {
    schema: u32,
    protocol: String,
    attempt_id: String,
    nonce: String,
    worker_request_sha256: String,
    owner: ProcessIdentity,
    child: ProcessIdentity,
    job_assigned: bool,
    job_active_process_limit: u32,
    job_commit_limit_bytes: u64,
    threads: usize,
}

#[cfg(windows)]
#[path = "launch_windows.rs"]
mod native;
#[cfg(windows)]
pub(super) use native::OwnerProcess;

#[cfg(not(windows))]
pub(super) struct OwnerProcess;
#[cfg(not(windows))]
impl OwnerProcess {
    pub fn alive(&self) -> Result<()> {
        anyhow::bail!("owned causal worker requires Windows Job admission")
    }
}

pub(super) fn verify(path: &Path, bytes: &[u8], request: &Request) -> Result<OwnerProcess> {
    ensure!(
        path == request.attempt_directory.join("worker-request.json"),
        "noncanonical worker request path"
    );
    ensure!(
        std::env::current_dir()?.canonicalize()? == request.workspace.canonicalize()?,
        "worker cwd/workspace mismatch"
    );
    storage_paths::attempt(&request.workspace, &request.attempt_directory)?;
    let expected_hash = std::env::var("HEXWEAVE_CAUSAL_REQUEST_SHA256")
        .context("missing owned launch request hash")?;
    let nonce = std::env::var("HEXWEAVE_CAUSAL_NONCE").context("missing owned launch nonce")?;
    let owner_pid: u32 = std::env::var("HEXWEAVE_CAUSAL_OWNER_PID")
        .context("missing owned launcher PID")?
        .parse()?;
    ensure!(
        storage_paths::hash_valid(&expected_hash)
            && storage_paths::digest(bytes) == expected_hash
            && (32..=128).contains(&nonce.len())
            && nonce.bytes().all(|b| b.is_ascii_hexdigit()),
        "launch nonce/request pin mismatch"
    );
    let receipt: Receipt = serde_json::from_slice(&child_io::read(&request.launch_receipt, 4096)?)?;
    ensure!(
        receipt.schema == 1
            && receipt.protocol == request.protocol
            && receipt.attempt_id == request.owner.attempt_id
            && receipt.nonce == nonce
            && receipt.worker_request_sha256 == expected_hash
            && receipt.owner.pid == owner_pid
            && receipt.child.pid == std::process::id(),
        "launch receipt identity mismatch"
    );
    ensure!(
        receipt.job_assigned
            && receipt.job_active_process_limit == 1
            && receipt.job_commit_limit_bytes == 536_870_912
            && receipt.threads == 1,
        "launch admission contract mismatch"
    );
    #[cfg(windows)]
    {
        native::verify(receipt.owner, receipt.child)
    }
    #[cfg(not(windows))]
    {
        anyhow::bail!("owned causal worker requires Windows Job admission")
    }
}
