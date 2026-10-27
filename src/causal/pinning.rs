use super::child_protocol::Request;
use super::{child_io, storage_paths};
use anyhow::{ensure, Context, Result};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::Read;
use std::path::{Component, Path};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    schema: u32,
    files: BTreeMap<String, String>,
}

pub(super) fn hash_file(path: &Path, maximum: u64) -> Result<String> {
    child_io::regular(path)?;
    let mut file = File::open(path)?;
    let expected = file.metadata()?.len();
    ensure!(expected <= maximum, "pinned file exceeds bound");
    let (mut buffer, mut length, mut hasher) = ([0u8; 8192], 0u64, Sha256::new());
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        length = length
            .checked_add(count as u64)
            .context("pinned file length overflow")?;
        ensure!(length <= maximum, "pinned file grew beyond bound");
        hasher.update(&buffer[..count]);
    }
    ensure!(length == expected, "pinned file length changed");
    Ok(format!("{:x}", hasher.finalize()))
}

pub(super) fn verify(request: &Request) -> Result<()> {
    ensure!(
        hash_file(&std::env::current_exe()?, 536_870_912)? == request.identity.runtime_sha256,
        "running executable pin mismatch"
    );
    let bytes = child_io::read(&request.source_manifest.path, 65_536)?;
    ensure!(
        storage_paths::digest(&bytes) == request.source_manifest.sha256,
        "source manifest pin mismatch"
    );
    let manifest: Manifest = serde_json::from_slice(&bytes)?;
    ensure!(
        manifest.schema == 1 && !manifest.files.is_empty() && manifest.files.len() <= 1024,
        "source manifest bounds mismatch"
    );
    let workspace = request.workspace.canonicalize()?;
    for (relative, expected) in &manifest.files {
        ensure!(
            storage_paths::hash_valid(expected),
            "invalid source file hash"
        );
        let path = Path::new(relative);
        ensure!(
            !relative.contains('\\')
                && !path.is_absolute()
                && path
                    .components()
                    .all(|component| matches!(component, Component::Normal(_))),
            "invalid source manifest path"
        );
        let mut current = workspace.clone();
        for component in path.components() {
            current.push(component.as_os_str());
            let metadata = fs::symlink_metadata(&current)?;
            ensure!(
                !metadata.file_type().is_symlink(),
                "source symlink is not admitted"
            );
            #[cfg(windows)]
            {
                use std::os::windows::fs::MetadataExt;
                ensure!(
                    metadata.file_attributes() & 0x400 == 0,
                    "source reparse point is not admitted"
                );
            }
        }
        ensure!(
            current.canonicalize()?.starts_with(&workspace),
            "source path escaped workspace"
        );
        ensure!(
            hash_file(&current, 1_048_576)? == *expected,
            "source file pin mismatch: {relative}"
        );
    }
    for required in [
        "Cargo.toml",
        "Cargo.lock",
        "src/lib.rs",
        "src/main.rs",
        "src/causal/mod.rs",
        "docs/CAUSAL-PREFIX-CONTRACT.md",
        "docs/CAUSAL-PREFIX-V2-CONTRACT.md",
        "docs/CAUSAL-PREFIX-V3-CONTRACT.md",
    ] {
        ensure!(
            manifest.files.contains_key(required),
            "source manifest missing required entry"
        );
    }
    ensure!(
        manifest.files["docs/CAUSAL-PREFIX-V3-CONTRACT.md"] == super::config::PROTOCOL_SHA256
            && manifest.files["docs/CAUSAL-PREFIX-V2-CONTRACT.md"]
                == super::config::V2_PROTOCOL_SHA256
            && manifest.files["docs/CAUSAL-PREFIX-CONTRACT.md"]
                == super::config::BASE_PROTOCOL_SHA256,
        "durable protocol document pin mismatch"
    );
    if request.protocol == super::assay_state::PROTOCOL {
        ensure!(
            manifest
                .files
                .get("docs/CAUSAL-RETAINED-TEXT-ASSAY-V1-CONTRACT.md")
                .map(String::as_str)
                == Some(super::assay_state::CONTRACT_SHA),
            "assay contract source pin"
        );
    }
    Ok(())
}
