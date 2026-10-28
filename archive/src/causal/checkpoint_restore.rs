use super::checkpoint::{inspect, latest, load_state};
use super::checkpoint_owner::ResumeRequest;
use super::checkpoint_state::{attempt_matches, CheckpointRef, Identity};
use super::execution::TrainingSource;
use super::state::State;
use super::state_codec;
use super::storage::{read_blob, DiskStore};
use super::storage_paths as paths;
use super::store_api::TileStore;
use anyhow::{ensure, Context, Result};
use std::path::Path;

pub(super) fn restore<S: TrainingSource>(
    workspace: &Path,
    parent_attempt: &Path,
    reference: &CheckpointRef,
    new_attempt: &Path,
    expected: &Identity,
    request: &ResumeRequest,
    source: &S,
) -> Result<(DiskStore, State)> {
    expected.validate()?;
    let parent = paths::attempt(workspace, parent_attempt)?;
    let directory = parent.join("persistence");
    ensure!(
        latest(&directory)?.as_ref() == Some(reference),
        "resume requires latest committed parent checkpoint"
    );
    let manifest = inspect(&directory, reference)?;
    ensure!(
        &manifest.identity == expected,
        "checkpoint protocol/config/data/kernel/source/runtime changed"
    );
    let release =
        super::checkpoint_release::read(workspace, &parent, request, expected, &manifest.owner)?;
    let state = load_state(&directory, &manifest)?;
    source.validate(&state.cursor)?;
    request.validate(
        &manifest.owner,
        &release,
        reference,
        state.global_opportunity,
    )?;
    ensure!(
        paths::attempt(workspace, new_attempt)? != parent,
        "resume attempt must be fresh"
    );
    let mut store = DiskStore::empty(
        workspace,
        new_attempt,
        state.geometry,
        state.seed,
        request.new_owner.budget.disk_bytes_per_attempt,
    )?;
    attempt_matches(&store.directory, &request.new_owner)?;
    for tile in &manifest.tiles {
        let bytes = read_blob(&directory, &tile.blob, "tile")?;
        let decoded = state_codec::decode_tile(&bytes, state.geometry, tile.id, tile.version)?;
        drop(decoded);
        store.io.peak_live_tiles = 1;
        store.io.loads = store
            .io
            .loads
            .checked_add(1)
            .context("restore I/O counter overflow")?;
        store.io.bytes_read = store
            .io
            .bytes_read
            .checked_add(bytes.len() as u64)
            .context("restore byte counter overflow")?;
        let blob = store.write_blob("tile", &bytes)?;
        ensure!(blob == tile.blob, "import changed immutable tile identity");
        store.tiles.push(tile.clone());
    }
    ensure!(
        store.versions()? == state.tile_versions,
        "restored store versions mismatch"
    );
    ensure!(
        super::checkpoint_release::read(workspace, &parent, request, expected, &manifest.owner)?
            == release,
        "terminal release changed during restore"
    );
    if let Some(assay) = &state.assay {
        super::assay_observation::copy_references(&parent, new_attempt, assay)?;
    }
    Ok((store, state))
}
