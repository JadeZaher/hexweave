use super::checkpoint;
use super::checkpoint_owner::Terminal;
use super::checkpoint_release;
use super::geometry::Geometry;
use super::state::State;
use super::storage::DiskStore;
use super::storage_paths::digest;
use anyhow::Result;
use serde_json::{json, Value};

#[path = "persistence_fixture.rs"]
#[allow(clippy::duplicate_mod, dead_code)]
mod fixture;
#[path = "execution_fixture.rs"]
#[allow(clippy::duplicate_mod, dead_code)]
mod numerical;

#[test]
fn external_owner_exit_wall_and_identity_seal_is_required_for_resume() -> Result<()> {
    let fixture = fixture::Fixture::new()?;
    let source = numerical::Source::new();
    let parent = fixture.attempt("sealed_parent")?;
    let state = State::with_cursor(Geometry::fixture(), 101, source.cursor())?;
    let identity = fixture::identity(&state)?;
    let owner = fixture::owner(&state, "sealed_parent");
    let mut store = DiskStore::create(
        &fixture.workspace,
        &parent,
        state.geometry,
        state.seed,
        owner.budget.disk_bytes_per_attempt,
    )?;
    let checkpoint = checkpoint::publish(&mut store, &state, &identity, &owner, &source)?;
    let request = fixture::release(
        &parent,
        &state,
        &owner,
        &checkpoint,
        "sealed_child",
        Terminal::Completed,
    )?;
    checkpoint_release::read(&fixture.workspace, &parent, &request, &identity, &owner)?;
    assert!(!parent.join("resume-claim.json").exists());
    let path = parent.join("release-seal.json");
    let original = std::fs::read(&path)?;
    let original_value: Value = serde_json::from_slice(&original)?;
    for (field, value) in [
        ("owner_exit_code", json!(1)),
        ("owner_reaped", json!(false)),
        ("external_wall_millis", json!(u64::MAX)),
        ("source_sha256", json!("0".repeat(64))),
        ("owner_identity", json!({"pid":99,"creation_filetime":1})),
        (
            "finalizer_identity",
            original_value["owner_identity"].clone(),
        ),
    ] {
        let mut changed = original_value.clone();
        changed[field] = value;
        let bytes = serde_json::to_vec(&changed)?;
        std::fs::write(&path, &bytes)?;
        let mut changed_request = request.clone();
        changed_request.release_seal_sha256 = digest(&bytes);
        assert!(checkpoint_release::read(
            &fixture.workspace,
            &parent,
            &changed_request,
            &identity,
            &owner
        )
        .is_err());
    }
    std::fs::write(&path, &original)?;
    let mut missing_pin = request.clone();
    missing_pin.release_seal_sha256 = "0".repeat(64);
    assert!(
        checkpoint_release::read(&fixture.workspace, &parent, &missing_pin, &identity, &owner)
            .is_err()
    );
    std::fs::remove_file(&path)?;
    assert!(
        checkpoint_release::read(&fixture.workspace, &parent, &request, &identity, &owner).is_err()
    );
    Ok(())
}
