use super::checkpoint;
use super::checkpoint_owner::Terminal;
use super::checkpoint_retention;
use super::checkpoint_state::StateRecord;
use super::execution;
use super::geometry::Geometry;
use super::state::{Boundary, State};
use super::storage::DiskStore;
use super::storage_paths;
use super::store_api::TileStore;
use anyhow::Result;
use std::fs;

#[path = "persistence_fixture.rs"]
#[allow(clippy::duplicate_mod)]
mod fixture;
#[path = "execution_fixture.rs"]
#[allow(clippy::duplicate_mod, dead_code)]
mod numerical;

fn same_state(left: &State, right: &State) -> Result<()> {
    assert_eq!(
        serde_json::to_vec(&StateRecord::capture(left)?)?,
        serde_json::to_vec(&StateRecord::capture(right)?)?
    );
    assert_eq!(left.shared.packed(), right.shared.packed());
    assert_eq!(left.predictor.packed(), right.predictor.packed());
    Ok(())
}

#[test]
fn resume_matches_uninterrupted_updates_and_leaves_parent_immutable() -> Result<()> {
    let f = fixture::Fixture::new()?;
    let source = numerical::Source::new();
    let first = f.attempt("first")?;
    let mut state = State::with_cursor(Geometry::fixture(), 29, source.cursor())?;
    let identity = fixture::identity(&state)?;
    let mut legacy = identity.clone();
    legacy.protocol = "causal-prefix-v1".into();
    legacy.protocol_sha256 = super::config::BASE_PROTOCOL_SHA256.into();
    assert!(legacy.validate().is_err());
    legacy.protocol = "causal-prefix-v2".into();
    legacy.protocol_sha256 = super::config::V2_PROTOCOL_SHA256.into();
    assert!(legacy.validate().is_err());
    let mut store = DiskStore::create(
        &f.workspace,
        &first,
        state.geometry,
        state.seed,
        8 * 1024 * 1024,
    )?;
    for _ in 0..5 {
        execution::step(&mut state, &mut store, &source, false)?;
    }
    let owner = fixture::owner(&state, "first");
    let reference = checkpoint::publish(&mut store, &state, &identity, &owner, &source)?;
    let request = fixture::release(
        &first,
        &state,
        &owner,
        &reference,
        "second",
        Terminal::Completed,
    )?;
    let before = fixture::parent_hashes(&first)?;
    let mut uninterrupted = numerical::Store::new(state.seed)?;
    for tile in store.references()? {
        let bytes = super::storage::read_blob(&store.directory, &tile.blob, "tile")?;
        uninterrupted.tiles[tile.id] =
            super::state_codec::decode_tile(&bytes, state.geometry, tile.id, tile.version)?;
    }
    let next = f.attempt("second")?;
    let (mut resumed_store, mut resumed) = checkpoint::restore(
        &f.workspace,
        &first,
        &reference,
        &next,
        &identity,
        &request,
        &source,
    )?;
    assert_eq!(fixture::parent_hashes(&first)?, before);
    same_state(&state, &resumed)?;
    for key in [
        "predictor_anchor_pairs",
        "predictor_replay_pairs",
        "predictor_anchor_updates",
        "predictor_transitions",
    ] {
        let mut altered = serde_json::to_value(StateRecord::capture(&state)?)?;
        altered["counters"][key] = serde_json::json!(u64::MAX);
        let altered: StateRecord = serde_json::from_value(altered)?;
        let copy = numerical::restore(&state)?;
        assert!(altered.restore(copy.shared, copy.predictor).is_err());
    }
    for _ in 0..5 {
        execution::step(&mut state, &mut uninterrupted, &source, false)?;
        execution::step(&mut resumed, &mut resumed_store, &source, false)?;
    }
    same_state(&state, &resumed)?;
    for (id, expected) in uninterrupted.tiles.iter().enumerate() {
        resumed_store.read(id, |actual| {
            assert_eq!(actual.packed(), expected.packed());
            assert_eq!(actual.version, expected.version);
            Ok(())
        })?;
    }
    assert_eq!(store.io.peak_live_tiles, 1);
    assert_eq!(resumed_store.io.peak_live_tiles, 1);
    assert_eq!(fixture::parent_hashes(&first)?, before);
    Ok(())
}

#[test]
fn interruption_leaves_previous_commit_and_rejects_inflight_or_stale_state() -> Result<()> {
    let f = fixture::Fixture::new()?;
    let source = numerical::Source::new();
    let attempt = f.attempt("first")?;
    let mut state = State::with_cursor(Geometry::fixture(), 7, source.cursor())?;
    let identity = fixture::identity(&state)?;
    let mut store = DiskStore::create(
        &f.workspace,
        &attempt,
        state.geometry,
        state.seed,
        8 * 1024 * 1024,
    )?;
    let initial = checkpoint::publish(
        &mut store,
        &state,
        &identity,
        &fixture::owner(&state, "first"),
        &source,
    )?;
    execution::step(&mut state, &mut store, &source, false)?;
    assert!(checkpoint::interrupted_publish(
        &mut store,
        &state,
        &identity,
        &fixture::owner(&state, "first"),
        &source
    )
    .is_err());
    assert_eq!(checkpoint::latest(&store.directory)?, Some(initial.clone()));
    let manifest = checkpoint::inspect(&store.directory, &initial)?;
    assert_eq!(
        checkpoint::load_state(&store.directory, &manifest)?.global_opportunity,
        0
    );
    state.boundary = Boundary::InFlight;
    assert!(checkpoint::publish(
        &mut store,
        &state,
        &identity,
        &fixture::owner(&state, "first"),
        &source
    )
    .is_err());
    state.boundary = Boundary::Ready;
    state.tile_versions[0] = state.tile_versions[0].saturating_add(1);
    assert!(checkpoint::publish(
        &mut store,
        &state,
        &identity,
        &fixture::owner(&state, "first"),
        &source
    )
    .is_err());
    assert_eq!(checkpoint::latest(&store.directory)?, Some(initial));
    Ok(())
}

#[test]
fn strict_resume_rejects_changed_identity_revocation_and_unrenewed_stop() -> Result<()> {
    let f = fixture::Fixture::new()?;
    let source = numerical::Source::new();
    let attempt = f.attempt("first")?;
    let state = State::with_cursor(Geometry::fixture(), 7, source.cursor())?;
    let identity = fixture::identity(&state)?;
    let owner = fixture::owner(&state, "first");
    let mut store = DiskStore::create(
        &f.workspace,
        &attempt,
        state.geometry,
        state.seed,
        8 * 1024 * 1024,
    )?;
    let reference = checkpoint::publish(&mut store, &state, &identity, &owner, &source)?;
    let mut request = fixture::release(
        &attempt,
        &state,
        &owner,
        &reference,
        "second",
        Terminal::UserStop,
    )?;
    let next = f.attempt("second")?;
    assert!(checkpoint::restore(
        &f.workspace,
        &attempt,
        &reference,
        &next,
        &identity,
        &request,
        &source
    )
    .is_err());
    request.explicit_user_renewal = true;
    for field in [0, 1, 2] {
        let mut changed = identity.clone();
        match field {
            0 => changed.config_sha256 = "e".repeat(64),
            1 => changed.data.artifact_sha256 = "e".repeat(64),
            _ => {
                changed.kernel = if identity.kernel == "scalar" {
                    "avx2"
                } else {
                    "scalar"
                }
                .into()
            }
        }
        assert!(checkpoint::restore(
            &f.workspace,
            &attempt,
            &reference,
            &next,
            &changed,
            &request,
            &source
        )
        .is_err());
    }
    fixture::write_json(
        &attempt.join("release-invalid.json"),
        &serde_json::json!({"reason":"late deadline"}),
    )?;
    assert!(checkpoint::restore(
        &f.workspace,
        &attempt,
        &reference,
        &next,
        &identity,
        &request,
        &source
    )
    .is_err());
    assert!(!next.join("persistence").exists());
    Ok(())
}

#[test]
fn retention_keeps_two_complete_generations_and_corruption_fails_closed() -> Result<()> {
    let f = fixture::Fixture::new()?;
    let source = numerical::Source::new();
    let attempt = f.attempt("first")?;
    let mut state = State::with_cursor(Geometry::fixture(), 17, source.cursor())?;
    let identity = fixture::identity(&state)?;
    let mut store = DiskStore::create(
        &f.workspace,
        &attempt,
        state.geometry,
        state.seed,
        8 * 1024 * 1024,
    )?;
    for _ in 0..4 {
        checkpoint::publish(
            &mut store,
            &state,
            &identity,
            &fixture::owner(&state, "first"),
            &source,
        )?;
        execution::step(&mut state, &mut store, &source, false)?;
    }
    let retained = checkpoint_retention::retain(&mut store, 2)?;
    assert_eq!(retained.retained_checkpoints, 2);
    assert!(retained.removed_files > 0);
    for reference in checkpoint::committed(&store.directory)? {
        let manifest = checkpoint::inspect(&store.directory, &reference)?;
        checkpoint::load_state(&store.directory, &manifest)?;
        for tile in manifest.tiles {
            super::storage::read_blob(&store.directory, &tile.blob, "tile")?;
        }
    }
    let reference = store.references()?[0].clone();
    let path = store.directory.join(&reference.blob.file);
    let mut bytes = fs::read(&path)?;
    bytes[48] ^= 1;
    fs::write(&path, bytes)?;
    assert!(store.read(0, |_| Ok(())).is_err());
    assert!(checkpoint_retention::retain(&mut store, 2).is_err());
    let used = storage_paths::usage(&store.directory)?;
    assert!(
        storage_paths::publish(&store.directory, "extra.bin", &[0u8; 1024], used + 1024).is_err()
    );
    assert!(!store.directory.join("extra.bin").exists());
    Ok(())
}
