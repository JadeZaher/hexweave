use super::checkpoint::{self, publish};
use super::checkpoint_owner::Terminal;
use super::config::Config;
use super::execution;
use super::geometry::Geometry;
use super::state::State;
use super::storage::DiskStore;
use anyhow::Result;

#[path = "persistence_fixture.rs"]
#[allow(clippy::duplicate_mod, dead_code)]
mod disk;
#[path = "execution_fixture.rs"]
#[allow(clippy::duplicate_mod, dead_code)]
mod source;

#[test]
fn checkpoint_segments_preserve_exposure_and_cumulative_owner_budget() -> Result<()> {
    let fixture = disk::Fixture::new()?;
    let source = source::Source::new();
    let parent = fixture.attempt("segment_a")?;
    let child = fixture.attempt("segment_b")?;
    let mut state = State::with_cursor(Geometry::fixture(), 97, source.cursor())?;
    let identity = disk::identity(&state)?;
    let mut owner = disk::owner(&state, "segment_a");
    let mut store = DiskStore::create(
        &fixture.workspace,
        &parent,
        state.geometry,
        state.seed,
        owner.budget.disk_bytes_per_attempt,
    )?;
    publish(&mut store, &state, &identity, &owner, &source)?;
    for _ in 0..5 {
        execution::step(&mut state, &mut store, &source, false)?;
    }
    owner.consumed.opportunities = 5;
    owner.consumed.wall_millis = 10;
    let checkpoint = publish(&mut store, &state, &identity, &owner, &source)?;
    let continuation = disk::release(
        &parent,
        &state,
        &owner,
        &checkpoint,
        "segment_b",
        Terminal::Completed,
    )?;
    let (mut resumed_store, mut resumed) = checkpoint::restore(
        &fixture.workspace,
        &parent,
        &checkpoint,
        &child,
        &identity,
        &continuation,
        &source,
    )?;
    assert_eq!(resumed.cursor, state.cursor);
    assert_eq!(resumed.counters, state.counters);
    let mut next_owner = continuation.new_owner;
    for _ in 0..4 {
        execution::step(&mut resumed, &mut resumed_store, &source, false)?;
    }
    next_owner.consumed.opportunities += 4;
    next_owner.consumed.wall_millis += 8;
    let final_checkpoint = publish(
        &mut resumed_store,
        &resumed,
        &identity,
        &next_owner,
        &source,
    )?;
    let manifest = checkpoint::inspect(&resumed_store.directory, &final_checkpoint)?;
    let loaded = checkpoint::load_state(&resumed_store.directory, &manifest)?;
    assert_eq!(loaded.global_opportunity, 9);
    assert_eq!(loaded.cursor.supervised_targets, 3);
    assert_eq!(loaded.cursor.epoch, 1);
    assert_eq!(loaded.cursor.completion_offset, 0);
    assert_eq!(loaded.counters.local_opportunities, 6);
    assert_eq!(manifest.owner.budget, owner.budget);
    assert_eq!(manifest.owner.consumed.opportunities, 9);
    assert_eq!(manifest.owner.consumed.attempts, 2);
    std::fs::write(resumed_store.directory.join("stray.bin"), [0u8; 16])?;
    let error = publish(
        &mut resumed_store,
        &resumed,
        &identity,
        &next_owner,
        &source,
    );
    assert!(error.unwrap_err().to_string().contains("ledger"));
    assert!(super::store_api::TileStore::versions(&resumed_store).is_err());
    Ok(())
}

#[test]
fn checkpoint_rehashes_sources_whose_guard_metadata_was_preserved() -> Result<()> {
    let data = super::data_tests::Fixture::new();
    let dataset = data.open()?;
    let fixture = disk::Fixture::new()?;
    let attempt = fixture.attempt("guarded")?;
    let state = State::new(Geometry::fixture(), 7, &dataset)?;
    let (identity, owner) = (disk::identity(&state)?, disk::owner(&state, "guarded"));
    let budget = owner.budget.disk_bytes_per_attempt;
    let mut store = DiskStore::create(&fixture.workspace, &attempt, state.geometry, 7, budget)?;
    publish(&mut store, &state, &identity, &owner, &dataset)?;
    let path = data.root.join("sources/docs/TRAINING-PULSE.md");
    let modified = std::fs::metadata(&path)?.modified()?;
    let mut bytes = std::fs::read(&path)?;
    *bytes.last_mut().unwrap() ^= 1;
    std::fs::write(&path, bytes)?;
    std::fs::OpenOptions::new()
        .write(true)
        .open(&path)?
        .set_modified(modified)?;
    let error = publish(&mut store, &state, &identity, &owner, &dataset).unwrap_err();
    assert!(error.to_string().contains("hash"), "{error}");
    Ok(())
}

#[test]
fn owned_configuration_rejects_fixture_and_experimental_drift() -> Result<()> {
    let original = Config::candidate(7);
    original.validate()?;
    let mut changed = original.clone();
    changed.profile = "fixture".into();
    assert!(changed.validate().is_err());
    changed = original.clone();
    changed.threads = 2;
    assert!(changed.validate().is_err());
    changed = original.clone();
    changed.anchor_interval = 1;
    assert!(changed.validate().is_err());
    assert_ne!(original.sha256()?, Config::candidate(8).sha256()?);
    Ok(())
}

#[cfg(windows)]
#[test]
fn ordinary_windows_owner_paths_match_canonical_owned_attempts() -> Result<()> {
    let fixture = disk::Fixture::new()?;
    let attempt = fixture.attempt("ordinary_windows_path")?;
    let text = attempt.to_string_lossy();
    let ordinary = std::path::PathBuf::from(text.strip_prefix(r"\\?\").unwrap_or(&text));
    assert_eq!(
        super::storage_paths::attempt(&fixture.workspace, &ordinary)?,
        attempt.canonicalize()?
    );
    Ok(())
}
