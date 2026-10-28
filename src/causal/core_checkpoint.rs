//! Core-path checkpoints: resident model plus latent state, hash-manifested and fsynced; see AGENTS.md §core.

use super::checkpoint_state::StateRecord;
use super::core_codec::{decode_latents, decode_tiles, encode_latents, encode_tiles};
use super::geometry::Geometry;
use super::resident::ResidentStore;
use super::state::State;
use super::state_codec::{decode_predictor, decode_shared, encode_predictor, encode_shared};
use super::storage_paths::digest;
use super::tuning;
use anyhow::{bail, ensure, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Duration;

pub(super) const PROTOCOL: &str = "causal-latent-v1";
const KEEP: usize = 3;

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    schema: u32,
    protocol: String,
    anchors: u64,
    config_sha256: String,
    calls: u64,
    files: BTreeMap<String, String>,
}

/// Retry transient Windows access/sharing errors (os 5, 32), e.g. a scanner holding a just-closed file.
fn retry<T>(mut operation: impl FnMut() -> std::io::Result<T>) -> std::io::Result<T> {
    let mut delay = Duration::from_millis(50);
    for _ in 0..8 {
        match operation() {
            Err(error) if matches!(error.raw_os_error(), Some(5 | 32)) => {
                std::thread::sleep(delay);
                delay *= 2;
            }
            other => return other,
        }
    }
    operation()
}

fn write_synced(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = std::fs::File::create(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

/// Complete checkpoint directories (`a` + 12 digits), oldest first.
fn checkpoints(out: &Path) -> Result<Vec<PathBuf>> {
    let root = out.join("checkpoints");
    if !root.exists() {
        return Ok(Vec::new());
    }
    let mut names: Vec<String> = std::fs::read_dir(&root)?
        .filter_map(|entry| {
            let entry = entry.ok()?;
            entry.file_type().ok()?.is_dir().then_some(())?;
            entry.file_name().into_string().ok()
        })
        .filter(|name| {
            name.len() == 13
                && name.starts_with('a')
                && name[1..].bytes().all(|b| b.is_ascii_digit())
        })
        .collect();
    names.sort();
    Ok(names.into_iter().map(|name| root.join(name)).collect())
}

/// Most recent complete checkpoint directory, if any.
pub(super) fn latest(out: &Path) -> Result<Option<PathBuf>> {
    Ok(checkpoints(out)?.pop())
}

/// Write `checkpoints/a<anchors>`: fsynced files in a partial directory, then an atomic rename.
pub(super) fn save(
    out: &Path,
    state: &State,
    store: &ResidentStore,
    identity: &str,
) -> Result<PathBuf> {
    let root = out.join("checkpoints");
    std::fs::create_dir_all(&root)?;
    let name = format!("a{:012}", state.global_opportunity);
    let partial = root.join(format!("{name}.partial"));
    if partial.exists() {
        retry(|| std::fs::remove_dir_all(&partial))?;
    }
    std::fs::create_dir(&partial)?;
    let (latents, calls) = tuning::export();
    let payloads = [
        (
            "state.json",
            serde_json::to_vec(&StateRecord::capture(state)?)?,
        ),
        ("shared.bin", encode_shared(&state.shared)?),
        ("predictor.bin", encode_predictor(&state.predictor)?),
        ("tiles.bin", encode_tiles(store)?),
        ("latent.bin", encode_latents(&latents)?),
    ];
    let mut files = BTreeMap::new();
    for (file, bytes) in &payloads {
        write_synced(&partial.join(file), bytes)?;
        files.insert((*file).to_string(), digest(bytes));
    }
    let manifest = Manifest {
        schema: 1,
        protocol: PROTOCOL.into(),
        anchors: state.global_opportunity,
        config_sha256: identity.into(),
        calls,
        files,
    };
    write_synced(
        &partial.join("manifest.json"),
        &serde_json::to_vec_pretty(&manifest)?,
    )?;
    let target = root.join(&name);
    retry(|| std::fs::rename(&partial, &target))?;
    prune(out);
    Ok(target)
}

/// Keep the newest checkpoints; a failed removal is logged, never fatal (the new one is published).
fn prune(out: &Path) {
    let Ok(mut existing) = checkpoints(out) else {
        return;
    };
    while existing.len() > KEEP {
        let path = existing.remove(0);
        if let Err(error) = retry(|| std::fs::remove_dir_all(&path)) {
            eprintln!("checkpoint prune skipped {}: {error}", path.display());
        }
    }
}

fn manifest(path: &Path) -> Result<Manifest> {
    Ok(serde_json::from_slice(&std::fs::read(
        path.join("manifest.json"),
    )?)?)
}

/// Verify every file hash, rebuild state and tiles, then install the latent state.
pub(super) fn load(
    path: &Path,
    geometry: Geometry,
    identity: &str,
) -> Result<(State, ResidentStore)> {
    let manifest = manifest(path)?;
    ensure!(
        manifest.schema == 1 && manifest.protocol == PROTOCOL,
        "not a {PROTOCOL} checkpoint"
    );
    ensure!(
        manifest.config_sha256 == identity,
        "checkpoint belongs to a different run identity"
    );
    ensure!(
        path.file_name().and_then(|name| name.to_str())
            == Some(&format!("a{:012}", manifest.anchors)),
        "checkpoint directory name does not match its anchors"
    );
    let read = |file: &str| -> Result<Vec<u8>> {
        let bytes = std::fs::read(path.join(file))?;
        ensure!(
            manifest.files.get(file) == Some(&digest(&bytes)),
            "checkpoint file {file} hash mismatch"
        );
        Ok(bytes)
    };
    let record: StateRecord = serde_json::from_slice(&read("state.json")?)?;
    ensure!(
        record.anchor_only && record.assay.is_none(),
        "core checkpoint must be anchor-only without an assay ledger"
    );
    let shared = decode_shared(&read("shared.bin")?, geometry, record.shared_version)?;
    let predictor = decode_predictor(&read("predictor.bin")?, geometry, record.predictor_version)?;
    let versions = record.tile_versions;
    let state = record.restore(shared, predictor)?;
    ensure!(
        state.global_opportunity == manifest.anchors,
        "checkpoint anchor count mismatch"
    );
    let store = decode_tiles(&read("tiles.bin")?, geometry, versions)?;
    tuning::import(decode_latents(&read("latent.bin")?)?, manifest.calls);
    Ok((state, store))
}

/// Newest checkpoint that verifies. A different run identity aborts untouched; a checkpoint
/// that fails integrity is renamed `<name>.rejected` and the next-newest is tried.
pub(super) fn resume(
    out: &Path,
    geometry: Geometry,
    identity: &str,
) -> Result<Option<(State, ResidentStore, PathBuf)>> {
    let mut candidates = checkpoints(out)?;
    while let Some(path) = candidates.pop() {
        if let Ok(found) = manifest(&path) {
            if found.protocol != PROTOCOL || found.config_sha256 != identity {
                bail!(
                    "{} belongs to a different run identity; refusing to resume",
                    path.display()
                );
            }
        }
        match load(&path, geometry, identity) {
            Ok((state, store)) => return Ok(Some((state, store, path))),
            Err(error) => {
                eprintln!("checkpoint {} rejected: {error:#}", path.display());
                let rejected = path.with_extension(format!("rejected-{}", std::process::id()));
                retry(|| std::fs::rename(&path, &rejected))?;
            }
        }
    }
    Ok(None)
}
