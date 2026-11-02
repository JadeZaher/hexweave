//! Core-path checkpoints: resident model plus latent state, hash-manifested and fsynced; see AGENTS.md §core.

use super::checkpoint_state::StateRecord;
use super::core_codec::{decode_latents, decode_tiles, encode_latents, encode_tiles};
use super::geometry::Geometry;
use super::resident::ResidentStore;
use super::state::State;
use super::state_codec::{decode_predictor, decode_shared, encode_predictor, encode_shared};
use super::storage_paths::digest;
use super::tuning;
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Duration;

pub(super) const PROTOCOL: &str = "causal-latent-v2";
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

/// Replace a small file atomically: fsynced `<name>.partial`, then a retried rename.
pub(super) fn publish(path: &Path, bytes: &[u8]) -> Result<()> {
    let partial = path.with_extension("partial");
    write_synced(&partial, bytes)?;
    retry(|| std::fs::rename(&partial, path))?;
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
    let mut payloads = vec![
        (
            "state.json",
            serde_json::to_vec(&StateRecord::capture(state)?)?,
        ),
        ("shared.bin", encode_shared(&state.shared)?),
        ("tiles.bin", encode_tiles(store)?),
        ("latent.bin", encode_latents(&latents)?),
    ];
    if tuning::current().fit_predictor {
        payloads.push(("predictor.bin", encode_predictor(&state.predictor)?));
    }
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
    let bytes = retry(|| std::fs::read(path.join("manifest.json")))?;
    Ok(serde_json::from_slice(&bytes)?)
}

/// A checkpoint of another run identity (protocol or config); resume aborts with nothing touched.
#[derive(Debug)]
struct ForeignIdentity(PathBuf);

impl std::fmt::Display for ForeignIdentity {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let path = self.0.display();
        write!(
            formatter,
            "{path} belongs to a different run identity; refusing to resume"
        )
    }
}

impl std::error::Error for ForeignIdentity {}

fn check_identity(path: &Path, manifest: &Manifest, identity: &str) -> Result<()> {
    if manifest.protocol != PROTOCOL || manifest.config_sha256 != identity {
        return Err(ForeignIdentity(path.to_path_buf()).into());
    }
    Ok(())
}

/// Quarantinable: hash, decode/serde, missing expected file or name/anchor mismatch; never other I/O.
pub(super) fn integrity_failure(error: &anyhow::Error) -> bool {
    error.chain().all(|cause| {
        cause.downcast_ref::<ForeignIdentity>().is_none()
            && cause
                .downcast_ref::<std::io::Error>()
                .is_none_or(|io| io.kind() == std::io::ErrorKind::NotFound)
    })
}

/// Quarantined checkpoint directories (`a<12 digits>.rejected-<pid>`).
fn quarantined(out: &Path) -> Result<usize> {
    let root = out.join("checkpoints");
    if !root.exists() {
        return Ok(0);
    }
    let mut count = 0;
    for entry in std::fs::read_dir(&root)? {
        let name = entry?.file_name();
        let name = name.to_string_lossy();
        if name.starts_with('a')
            && name
                .get(13..)
                .is_some_and(|tail| tail.starts_with(".rejected-"))
        {
            count += 1;
        }
    }
    Ok(count)
}

/// Verify every file hash, rebuild state and tiles, then install the latent state.
pub(super) fn load(
    path: &Path,
    geometry: Geometry,
    identity: &str,
) -> Result<(State, ResidentStore)> {
    let manifest = manifest(path)?;
    check_identity(path, &manifest, identity)?;
    ensure!(
        manifest.schema == 1,
        "unsupported checkpoint manifest schema"
    );
    ensure!(
        path.file_name().and_then(|name| name.to_str())
            == Some(&format!("a{:012}", manifest.anchors)),
        "checkpoint directory name does not match its anchors"
    );
    let read = |file: &str| -> Result<Vec<u8>> {
        let bytes = retry(|| std::fs::read(path.join(file)))?;
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
    let predictor = if tuning::current().fit_predictor {
        decode_predictor(&read("predictor.bin")?, geometry, record.predictor_version)?
    } else {
        super::predictor::Predictor::new(geometry, record.seed)?
    };
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

/// Newest checkpoint that verifies. Every readable manifest is identity-checked first, so a
/// foreign run aborts with nothing touched; an integrity failure is renamed
/// `<name>.rejected-<pid>` and the next-newest is tried; any other I/O error aborts in place.
/// Ok(None) only when no checkpoint (complete or quarantined) exists.
pub(super) fn resume(
    out: &Path,
    geometry: Geometry,
    identity: &str,
) -> Result<Option<(State, ResidentStore, PathBuf)>> {
    let mut candidates = checkpoints(out)?;
    for path in &candidates {
        if let Ok(found) = manifest(path) {
            check_identity(path, &found, identity)?;
        }
    }
    while let Some(path) = candidates.pop() {
        match load(&path, geometry, identity) {
            Ok((state, store)) => return Ok(Some((state, store, path))),
            Err(error) if integrity_failure(&error) => {
                eprintln!("checkpoint {} rejected: {error:#}", path.display());
                let target = path.with_extension(format!("rejected-{}", std::process::id()));
                retry(|| std::fs::rename(&path, &target))?;
            }
            Err(error) => {
                return Err(error.context(format!("{} left in place", path.display())));
            }
        }
    }
    let count = quarantined(out)?;
    ensure!(
        count == 0,
        "all {count} checkpoints failed verification; see .rejected-*; refusing to restart from anchor 0"
    );
    Ok(None)
}
