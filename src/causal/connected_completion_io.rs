//! Owned inference scratch and read-only source hashes; see CONNECTED.md.

use super::super::connected_checkpoint_io as io;
use super::super::connected_sections::{payload, MAX_SECTIONS};
use super::super::connected_store_codec;
use super::super::geometry::{Geometry, TILES};
use super::super::storage_paths::digest;
use anyhow::{ensure, Context, Result};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT: AtomicU64 = AtomicU64::new(0);

pub(super) struct Scratch {
    pub path: PathBuf,
    base: PathBuf,
    removed: bool,
}

impl Scratch {
    pub fn new() -> Result<Self> {
        let workspace = std::env::current_dir()?.canonicalize()?;
        let base = workspace.join("work/inference");
        std::fs::create_dir_all(&base)?;
        let base = base.canonicalize()?;
        ensure!(
            base.parent().and_then(Path::parent) == Some(workspace.as_path()),
            "inference scratch escaped workspace work directory"
        );
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos();
        let name = format!(
            "connected-{}-{stamp}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        );
        let path = base.join(name);
        std::fs::create_dir(&path)?;
        Ok(Self {
            path,
            base,
            removed: false,
        })
    }

    pub fn remove(&mut self) -> Result<()> {
        if self.removed {
            return Ok(());
        }
        let resolved = self.path.canonicalize()?;
        ensure!(
            resolved.parent() == Some(self.base.as_path()),
            "inference cleanup escaped owned scratch"
        );
        io::retry(|| std::fs::remove_dir_all(&resolved))?;
        self.removed = true;
        Ok(())
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        if let Err(error) = self.remove() {
            eprintln!(
                "inference scratch cleanup failed for {}: {error:#}",
                self.path.display()
            );
        }
    }
}

/// The hashed payload set: manifest, common state, tiles and any shared-section bundles present.
fn payloads(path: &Path, geometry: Geometry) -> Result<Vec<(String, u64)>> {
    let mut payloads = vec![
        ("manifest.json".to_string(), 1024 * 1024),
        ("connected.json".to_string(), io::COMMON_CAP),
        ("shared.bin".to_string(), io::COMMON_CAP),
    ];
    for id in 0..TILES {
        payloads.push((
            format!("tiles/{id}.bin"),
            connected_store_codec::cap(geometry, id)?,
        ));
    }
    // Section files are bound by the manifest, so a dropped one fails the verifying load.
    let sections = (0..MAX_SECTIONS as usize).map(payload);
    payloads.extend(
        sections
            .filter(|name| path.join(name).is_file())
            .map(|name| (name, io::COMMON_CAP)),
    );
    Ok(payloads)
}

pub(super) fn checkpoint_hash(path: &Path, geometry: Geometry) -> Result<String> {
    let mut hashes = BTreeMap::new();
    for (name, cap) in payloads(path, geometry)? {
        let hash = io::hash(&path.join(&name), cap)?;
        hashes.insert(name, hash);
    }
    Ok(digest(&serde_json::to_vec(&hashes)?))
}

/// Copies the hashed checkpoint payload set into owned scratch and returns its checkpoint hash.
pub(super) fn snapshot(source: &Path, target: &Path, geometry: Geometry) -> Result<String> {
    std::fs::create_dir_all(target.join("tiles"))?;
    let mut hashes = BTreeMap::new();
    for (name, cap) in payloads(source, geometry)? {
        let hash = io::copy(&source.join(&name), &target.join(&name), cap)?;
        hashes.insert(name, hash);
    }
    Ok(digest(&serde_json::to_vec(&hashes)?))
}

pub(super) fn checkpoint_path(out: &Path, checkpoint: &str) -> Result<PathBuf> {
    ensure!(
        checkpoint.len() == 13
            && checkpoint.starts_with('a')
            && checkpoint[1..].bytes().all(|byte| byte.is_ascii_digit()),
        "invalid connected checkpoint name"
    );
    let base = out.join("checkpoints").canonicalize()?;
    let path = base
        .join(checkpoint)
        .canonicalize()
        .context("named connected checkpoint does not exist")?;
    ensure!(
        path.parent() == Some(base.as_path()),
        "connected checkpoint escaped run directory"
    );
    Ok(path)
}
