//! Complete connected-state checkpoints with paged restore; see AGENTS.md.

use super::connected_checkpoint_io as io;
use super::connected_store::{Slot, Store, TileMeta};
use super::geometry::{Geometry, TILES};
use super::storage_paths::digest;
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

const KEEP: usize = 3;

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    schema: u32,
    protocol: String,
    identity: String,
    geometry: Geometry,
    anchors: u64,
    tiles: Vec<TileMeta>,
    common: BTreeMap<String, String>,
}

pub(super) struct Loaded {
    pub store: Store,
    pub anchors: u64,
    pub payloads: BTreeMap<String, Vec<u8>>,
    #[cfg(test)]
    pub path: PathBuf,
}

fn checkpoints(out: &Path) -> Result<Vec<PathBuf>> {
    let root = out.join("checkpoints");
    if !root.exists() {
        return Ok(Vec::new());
    }
    let mut paths = Vec::new();
    for entry in std::fs::read_dir(root)? {
        let entry = entry?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if entry.file_type()?.is_dir()
            && name.len() == 13
            && name.starts_with('a')
            && name[1..].bytes().all(|c| c.is_ascii_digit())
        {
            paths.push(entry.path());
        }
    }
    paths.sort();
    Ok(paths)
}

pub(super) fn latest(out: &Path) -> Result<Option<PathBuf>> {
    Ok(checkpoints(out)?.pop())
}

fn manifest(path: &Path) -> Result<Manifest> {
    Ok(serde_json::from_slice(&io::read(
        &path.join("manifest.json"),
        1024 * 1024,
    )?)?)
}

#[derive(Debug)]
struct ForeignConnected;

impl std::fmt::Display for ForeignConnected {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            out,
            "foreign connected checkpoint identity; refusing to resume"
        )
    }
}

impl std::error::Error for ForeignConnected {}

fn validate_identity(value: &Manifest, geometry: Geometry, identity: &str) -> Result<()> {
    if super::connected_precision::Precision::from_protocol(&value.protocol).is_err()
        || value.identity != identity
        || value.geometry != geometry
    {
        return Err(ForeignConnected.into());
    }
    Ok(())
}

pub(super) fn save(
    out: &Path,
    store: &mut Store,
    identity: &str,
    anchors: u64,
    payloads: &BTreeMap<String, Vec<u8>>,
) -> Result<PathBuf> {
    store.flush()?;
    ensure!(
        anchors < 1_000_000_000_000,
        "connected checkpoint counter exceeds name width"
    );
    ensure!(
        !payloads.is_empty() && payloads.len() <= 16,
        "connected common payload count"
    );
    ensure!(
        payloads.values().map(Vec::len).sum::<usize>() <= io::TOTAL_CAP,
        "connected common payload total bound"
    );
    for (name, bytes) in payloads {
        ensure!(
            io::name(name) && bytes.len() as u64 <= io::COMMON_CAP,
            "invalid connected common payload"
        );
    }
    let root = out.join("checkpoints");
    std::fs::create_dir_all(&root)?;
    let name = format!("a{anchors:012}");
    let target = root.join(&name);
    ensure!(!target.exists(), "connected checkpoint already exists");
    let partial = root.join(format!("{name}.partial"));
    if partial.exists() {
        let stale = root.join(format!("{name}.partial-abandoned-{}", std::process::id()));
        ensure!(
            !stale.exists(),
            "connected abandoned checkpoint already exists"
        );
        io::retry(|| std::fs::rename(&partial, &stale))?;
    }
    std::fs::create_dir(&partial)?;
    std::fs::create_dir(partial.join("tiles"))?;
    for (id, slot) in store.slots.iter().enumerate() {
        let hash = io::copy(
            &slot.path,
            &partial.join("tiles").join(format!("{id}.bin")),
            super::connected_store_codec::cap(store.geometry, id)?,
        )?;
        ensure!(
            hash == slot.meta.hash,
            "tile changed before connected checkpoint publication"
        );
    }
    let mut common = BTreeMap::new();
    for (name, bytes) in payloads {
        io::write(&partial.join(name), bytes)?;
        common.insert(name.clone(), digest(bytes));
    }
    let manifest = Manifest {
        schema: 1,
        protocol: store.precision.protocol().into(),
        identity: identity.into(),
        geometry: store.geometry,
        anchors,
        tiles: store.slots.iter().map(|slot| slot.meta.clone()).collect(),
        common,
    };
    io::write(
        &partial.join("manifest.json"),
        &serde_json::to_vec_pretty(&manifest)?,
    )?;
    io::retry(|| std::fs::rename(&partial, &target))?;
    prune(out);
    Ok(target)
}

pub(super) fn load(
    path: &Path,
    out: &Path,
    geometry: Geometry,
    identity: &str,
    workers: usize,
) -> Result<Loaded> {
    let value = manifest(path)?;
    validate_identity(&value, geometry, identity)?;
    ensure!(
        value.schema == 1 && value.tiles.len() == TILES,
        "connected checkpoint schema or tile count"
    );
    ensure!(
        path.file_name().and_then(|name| name.to_str()) == Some(&format!("a{:012}", value.anchors)),
        "connected checkpoint name/count mismatch"
    );
    ensure!(
        !value.common.is_empty() && value.common.len() <= 16,
        "connected common payload count"
    );
    let mut payloads = BTreeMap::new();
    let mut size = 0;
    for (name, hash) in value.common {
        ensure!(io::name(&name), "unsafe connected common payload name");
        let bytes = io::read(&path.join(&name), io::COMMON_CAP)?;
        size += bytes.len();
        ensure!(
            size <= io::TOTAL_CAP && digest(&bytes) == hash,
            "connected common payload hash or size mismatch"
        );
        payloads.insert(name, bytes);
    }
    let slots = value
        .tiles
        .into_iter()
        .enumerate()
        .map(|(id, meta)| Slot {
            path: path.join("tiles").join(format!("{id}.bin")),
            meta,
        })
        .collect();
    let store = Store::restore(
        &out.join("connected-tiles"),
        geometry,
        workers,
        slots,
        super::connected_precision::Precision::from_protocol(&value.protocol)?,
    )?;
    Ok(Loaded {
        store,
        anchors: value.anchors,
        payloads,
        #[cfg(test)]
        path: path.into(),
    })
}

pub(super) fn resume(
    out: &Path,
    geometry: Geometry,
    identity: &str,
    workers: usize,
) -> Result<Option<Loaded>> {
    let mut paths = checkpoints(out)?;
    for path in &paths {
        if let Ok(value) = manifest(path) {
            validate_identity(&value, geometry, identity)?;
        }
    }
    while let Some(path) = paths.pop() {
        match load(&path, out, geometry, identity, workers) {
            Ok(loaded) => return Ok(Some(loaded)),
            Err(error)
                if !error
                    .chain()
                    .any(|cause| cause.downcast_ref::<ForeignConnected>().is_some())
                    && super::core_checkpoint::integrity_failure(&error) =>
            {
                let rejected = path.with_extension(format!("rejected-{}", std::process::id()));
                io::retry(|| std::fs::rename(&path, &rejected))?;
                eprintln!(
                    "connected checkpoint {} rejected: {error:#}",
                    path.display()
                );
            }
            Err(error) => return Err(error.context("connected checkpoint left in place")),
        }
    }
    let root = out.join("checkpoints");
    if root.exists() {
        for entry in std::fs::read_dir(root)? {
            let name = entry?.file_name();
            let name = name.to_string_lossy();
            ensure!(
                !name.contains(".rejected-") && !name.contains(".partial"),
                "no complete connected checkpoint remains; refusing fresh restart"
            );
        }
    }
    ensure!(
        !out.join("connected-tiles").exists(),
        "uncheckpointed connected state exists; refusing fresh restart"
    );
    Ok(None)
}

fn prune(out: &Path) {
    let Ok(mut paths) = checkpoints(out) else {
        return;
    };
    while paths.len() > KEEP {
        let path = paths.remove(0);
        if let Err(error) = io::remove_checkpoint(out, &path) {
            eprintln!(
                "connected checkpoint pruning skipped {}: {error:#}",
                path.display()
            );
        }
    }
}
