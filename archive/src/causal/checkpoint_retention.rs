use super::checkpoint::{committed, inspect};
use super::checkpoint_state::Manifest;
use super::storage::{read_blob, DiskStore};
use super::storage_paths as paths;
use anyhow::{ensure, Result};
use serde::Serialize;
use std::collections::BTreeSet;

pub(super) const RECEIPT_GENERATIONS: usize = 3;

#[derive(Debug, Serialize)]
pub(super) struct Retention {
    pub retained_checkpoints: usize,
    pub removed_files: usize,
    pub removed_bytes: u64,
}

fn referenced(manifest: &Manifest) -> Vec<(&str, &super::storage::Blob)> {
    let mut blobs = vec![
        ("state", &manifest.state),
        ("shared", &manifest.shared),
        ("predictor", &manifest.predictor),
    ];
    blobs.extend(manifest.tiles.iter().map(|tile| ("tile", &tile.blob)));
    blobs
}

fn object_hash(name: &str) -> Option<&str> {
    for prefix in ["tile-", "shared-", "predictor-", "state-", "manifest-"] {
        if let Some(value) = name
            .strip_prefix(prefix)
            .and_then(|s| s.strip_suffix(".bin"))
        {
            return paths::hash_valid(value).then_some(value);
        }
    }
    None
}

/// Retain complete recent generations and current working tiles, within this attempt only.
pub(super) fn retain(store: &mut DiskStore, generations: usize) -> Result<Retention> {
    ensure!(
        (2..=8).contains(&generations),
        "checkpoint retention must keep two to eight generations"
    );
    store.reconcile()?;
    let expected = format!(
        "causal-persistence-v1\n{}\n",
        store.directory.parent().unwrap().display()
    );
    ensure!(
        paths::read(&store.directory, "OWNED", 4096)? == expected.as_bytes(),
        "owned persistence marker mismatch"
    );
    let checkpoints = committed(&store.directory)?;
    let retained = checkpoints.len().min(generations);
    let split = checkpoints.len() - retained;
    let mut keep = BTreeSet::from([String::from("OWNED")]);
    for tile in store.references()? {
        read_blob(&store.directory, &tile.blob, "tile")?;
        keep.insert(tile.blob.file);
    }
    for reference in &checkpoints[split..] {
        let manifest = inspect(&store.directory, reference)?;
        keep.insert(reference.commit_name());
        keep.insert(reference.manifest_name());
        for (prefix, blob) in referenced(&manifest) {
            read_blob(&store.directory, blob, prefix)?;
            keep.insert(blob.file.clone());
        }
    }
    let mut retired_markers = Vec::new();
    for reference in &checkpoints[..split] {
        inspect(&store.directory, reference)?;
        let name = reference.commit_name();
        let bytes = paths::read(&store.directory, &name, 512)?;
        retired_markers.push((name, paths::digest(&bytes)));
    }
    let inventory = paths::inventory(&store.directory)?;
    let mut objects = Vec::new();
    for (name, _) in inventory {
        if keep.contains(&name) || name.starts_with("commit-") || name.starts_with(".pending-") {
            continue;
        }
        let hash = object_hash(&name)
            .ok_or_else(|| anyhow::anyhow!("unknown persistence entry; retention fails closed"))?;
        let bytes = paths::read(&store.directory, &name, paths::FILE_CAP)?;
        ensure!(
            paths::digest(&bytes) == hash,
            "changed orphan object; retention fails closed"
        );
        objects.push((name.clone(), hash.to_owned()));
    }
    let mut result = Retention {
        retained_checkpoints: retained,
        removed_files: 0,
        removed_bytes: 0,
    };
    for (name, hash) in retired_markers.into_iter().chain(objects) {
        let removed = paths::remove(&store.directory, &name, &hash).and_then(|bytes| {
            store.removed(bytes)?;
            Ok(bytes)
        });
        let bytes = match removed {
            Ok(bytes) => bytes,
            Err(error) => {
                store.invalidate_ledger();
                return Err(error);
            }
        };
        result.removed_bytes += bytes;
        result.removed_files += 1;
    }
    store.reconcile()?;
    Ok(result)
}
