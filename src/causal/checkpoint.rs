use super::checkpoint_owner::OwnerEnvelope;
use super::checkpoint_state::{
    attempt_matches, CheckpointRef, Identity, Manifest, StateRecord, MANIFEST_CAP, STATE_CAP,
};
use super::execution::TrainingSource;
use super::state::State;
use super::state_codec;
use super::storage::{read_blob, DiskStore};
use super::storage_paths as paths;
use super::store_api::TileStore;
use anyhow::{ensure, Context, Result};
use std::path::Path;

pub(super) fn committed(directory: &Path) -> Result<Vec<CheckpointRef>> {
    let mut references = Vec::new();
    for (name, _) in paths::inventory(directory)? {
        if !name.starts_with("commit-") {
            continue;
        }
        let reference: CheckpointRef =
            serde_json::from_slice(&paths::read(directory, &name, 512)?)?;
        reference.validate()?;
        ensure!(
            name == reference.commit_name(),
            "noncanonical checkpoint commit marker"
        );
        references.push(reference);
    }
    references.sort_by_key(|r| r.sequence);
    ensure!(
        references
            .windows(2)
            .all(|pair| pair[0].sequence < pair[1].sequence),
        "ambiguous checkpoint head sequence"
    );
    Ok(references)
}

pub(super) fn latest(directory: &Path) -> Result<Option<CheckpointRef>> {
    Ok(committed(directory)?.pop())
}

pub(super) fn inspect(directory: &Path, reference: &CheckpointRef) -> Result<Manifest> {
    reference.validate()?;
    let committed: CheckpointRef =
        serde_json::from_slice(&paths::read(directory, &reference.commit_name(), 512)?)?;
    ensure!(
        &committed == reference,
        "checkpoint has no matching committed head"
    );
    let bytes = paths::read(directory, &reference.manifest_name(), MANIFEST_CAP)?;
    ensure!(
        paths::digest(&bytes) == reference.manifest_sha256,
        "checkpoint manifest SHA256 mismatch"
    );
    let manifest: Manifest = serde_json::from_slice(&bytes)?;
    manifest.validate(reference)?;
    attempt_matches(directory, &manifest.owner)?;
    Ok(manifest)
}

fn state_record(directory: &Path, manifest: &Manifest) -> Result<StateRecord> {
    let record: StateRecord =
        serde_json::from_slice(&read_blob(directory, &manifest.state, "state")?)?;
    ensure!(
        record.geometry == manifest.identity.geometry
            && record.cursor.data == manifest.identity.data,
        "checkpoint state/protocol/data mismatch"
    );
    ensure!(
        record.tile_versions == std::array::from_fn(|id| manifest.tiles[id].version),
        "checkpoint state/tile versions mismatch"
    );
    ensure!(
        record.assay.is_some() == (manifest.identity.protocol == super::assay_state::PROTOCOL),
        "checkpoint state protocol descriptor mismatch"
    );
    if let Some(assay) = &record.assay {
        ensure!(
            super::config::Config::assay(record.seed, assay.rule).sha256()?
                == manifest.identity.config_sha256,
            "checkpoint assay config hash mismatch"
        );
    }
    manifest.owner.validate(record.global_opportunity)?;
    Ok(record)
}

pub(super) fn load_state(directory: &Path, manifest: &Manifest) -> Result<State> {
    let record = state_record(directory, manifest)?;
    let shared = state_codec::decode_shared(
        &read_blob(directory, &manifest.shared, "shared")?,
        record.geometry,
        record.shared_version,
    )?;
    let predictor = state_codec::decode_predictor(
        &read_blob(directory, &manifest.predictor, "predictor")?,
        record.geometry,
        record.predictor_version,
    )?;
    let state = record.restore(shared, predictor)?;
    super::assay_observation::verify_references(
        directory.parent().context("missing attempt")?,
        &state,
    )?;
    Ok(state)
}

fn validate_publication<S: TrainingSource>(
    store: &mut DiskStore,
    state: &State,
    identity: &Identity,
    owner: &OwnerEnvelope,
    source: &S,
) -> Result<u64> {
    state.validate_boundary()?;
    store.reconcile()?;
    source.verify_boundary()?;
    super::assay_observation::verify_references(
        store.directory.parent().context("missing attempt")?,
        state,
    )?;
    source.validate(&state.cursor)?;
    identity.validate()?;
    ensure!(
        state.assay.is_some() == (identity.protocol == super::assay_state::PROTOCOL),
        "checkpoint state/identity protocol mismatch"
    );
    if let Some(assay) = &state.assay {
        ensure!(
            super::config::Config::assay(state.seed, assay.rule).sha256()?
                == identity.config_sha256,
            "published assay config hash mismatch"
        );
    }
    owner.validate(state.global_opportunity)?;
    attempt_matches(&store.directory, owner)?;
    ensure!(
        identity.geometry == state.geometry
            && identity.data == state.cursor.data
            && store.geometry == state.geometry
            && store.seed == state.seed
            && store.versions()? == state.tile_versions
            && store.disk_budget_bytes == owner.budget.disk_bytes_per_attempt,
        "checkpoint store/state/identity mismatch"
    );
    for id in 0..super::geometry::TILES {
        store.read(id, |tile| {
            ensure!(
                tile.geometry == state.geometry
                    && tile.id == id
                    && tile.version == state.tile_versions[id],
                "checkpoint tile integrity mismatch"
            );
            Ok(())
        })?;
    }
    let previous = latest(&store.directory)?;
    let sequence = previous
        .as_ref()
        .map_or(Some(1), |r| r.sequence.checked_add(1))
        .context("checkpoint sequence overflow")?;
    if let Some(reference) = &previous {
        let old = inspect(&store.directory, reference)?;
        ensure!(
            old.identity == *identity,
            "immutable checkpoint protocol changed"
        );
        owner.follows(&old.owner)?;
        let old_state = state_record(&store.directory, &old)?;
        ensure!(
            old_state.seed == state.seed
                && old_state.global_opportunity <= state.global_opportunity,
            "checkpoint scientific state regressed"
        );
    }
    Ok(sequence)
}

fn publish_inner<S: TrainingSource>(
    store: &mut DiskStore,
    state: &State,
    identity: &Identity,
    owner: &OwnerEnvelope,
    source: &S,
    before_commit: impl FnOnce() -> Result<()>,
) -> Result<CheckpointRef> {
    let sequence = validate_publication(store, state, identity, owner, source)?;
    let record = serde_json::to_vec(&StateRecord::capture(state)?)?;
    ensure!(
        record.len() as u64 <= STATE_CAP,
        "checkpoint continuation state exceeds bound"
    );
    let state_blob = store.write_blob("state", &record)?;
    drop(record);
    let shared = store.write_blob("shared", &state_codec::encode_shared(&state.shared)?)?;
    let predictor = store.write_blob(
        "predictor",
        &state_codec::encode_predictor(&state.predictor)?,
    )?;
    let manifest = Manifest {
        schema: 1,
        sequence,
        identity: identity.clone(),
        owner: owner.clone(),
        state: state_blob,
        shared,
        predictor,
        tiles: store.references()?,
        io: store.io.clone(),
    };
    let bytes = serde_json::to_vec(&manifest)?;
    ensure!(
        bytes.len() as u64 <= MANIFEST_CAP,
        "checkpoint manifest exceeds bound"
    );
    let reference = CheckpointRef {
        sequence,
        manifest_sha256: paths::digest(&bytes),
    };
    manifest.validate(&reference)?;
    store.write_named(&reference.manifest_name(), &bytes)?;
    before_commit()?;
    store.write_named(&reference.commit_name(), &serde_json::to_vec(&reference)?)?;
    Ok(reference)
}

pub(super) fn publish<S: TrainingSource>(
    store: &mut DiskStore,
    state: &State,
    identity: &Identity,
    owner: &OwnerEnvelope,
    source: &S,
) -> Result<CheckpointRef> {
    publish_inner(store, state, identity, owner, source, || Ok(()))
}

#[cfg(test)]
pub(super) fn interrupted_publish<S: TrainingSource>(
    store: &mut DiskStore,
    state: &State,
    identity: &Identity,
    owner: &OwnerEnvelope,
    source: &S,
) -> Result<CheckpointRef> {
    publish_inner(store, state, identity, owner, source, || {
        anyhow::bail!("injected interruption before committed head")
    })
}

pub(super) use super::checkpoint_restore::restore;
