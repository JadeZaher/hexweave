use super::checkpoint::{self, publish};
use super::checkpoint_owner::{Release, Stop, Terminal};
use super::checkpoint_retention::{retain, RECEIPT_GENERATIONS};
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
fn resumed_admission_keeps_prior_start_and_nonzero_setup_charge() -> Result<()> {
    let f = disk::Fixture::new()?;
    let source = source::Source::new();
    let parent = f.attempt("charged_parent")?;
    let child = f.attempt("charged_child")?;
    let state = State::with_cursor(Geometry::fixture(), 17, source.cursor())?;
    let identity = disk::identity(&state)?;
    let mut old = disk::owner(&state, "charged_parent");
    old.consumed.wall_millis = 1998;
    let mut store = DiskStore::create(
        &f.workspace,
        &parent,
        state.geometry,
        state.seed,
        old.budget.disk_bytes_per_attempt,
    )?;
    let reference = publish(&mut store, &state, &identity, &old, &source)?;
    let mut request = disk::release(
        &parent,
        &state,
        &old,
        &reference,
        "charged_child",
        Terminal::Completed,
    )?;
    request.new_owner.consumed.wall_millis += 125;
    assert_eq!(request.new_owner.attempt_start_wall_millis, 2000);
    assert_eq!(request.new_owner.consumed.wall_millis, 2125);
    let release: Release = serde_json::from_slice(&std::fs::read(parent.join("release.json"))?)?;
    for start in [0, 1999, 2125] {
        let mut changed = request.clone();
        changed.new_owner.attempt_start_wall_millis = start;
        assert!(changed.validate(&old, &release, &reference, 0).is_err());
    }
    let mut exhausted = request.clone();
    exhausted.new_owner.consumed.wall_millis = old.budget.maximum_wall_millis;
    assert!(exhausted.validate(&old, &release, &reference, 0).is_err());
    let (_, restored) = checkpoint::restore(
        &f.workspace,
        &parent,
        &reference,
        &child,
        &identity,
        &request,
        &source,
    )?;
    assert_eq!(restored.counters, state.counters);
    assert_eq!(request.new_owner.budget, old.budget);
    Ok(())
}

#[test]
fn late_stop_retains_evaluated_and_both_terminal_generations_for_resume() -> Result<()> {
    let f = disk::Fixture::new()?;
    let source = source::Source::new();
    let parent = f.attempt("late_stop_parent")?;
    let child = f.attempt("late_stop_child")?;
    let state = State::with_cursor(Geometry::fixture(), 23, source.cursor())?;
    let identity = disk::identity(&state)?;
    let mut owner = disk::owner(&state, "late_stop_parent");
    let mut store = DiskStore::create(
        &f.workspace,
        &parent,
        state.geometry,
        state.seed,
        owner.budget.disk_bytes_per_attempt,
    )?;
    let historical = publish(&mut store, &state, &identity, &owner, &source)?;
    let evaluated = publish(&mut store, &state, &identity, &owner, &source)?;
    retain(&mut store, RECEIPT_GENERATIONS)?;
    let terminal_before_stop = publish(&mut store, &state, &identity, &owner, &source)?;
    retain(&mut store, RECEIPT_GENERATIONS)?;
    owner.stop = Stop::UserStop;
    owner.observer_sequence += 1;
    owner.consumed.wall_millis += 1;
    let terminal_after_stop = publish(&mut store, &state, &identity, &owner, &source)?;
    assert_eq!(
        retain(&mut store, RECEIPT_GENERATIONS)?.retained_checkpoints,
        3
    );
    assert!(checkpoint::inspect(&store.directory, &historical).is_err());
    for reference in [&evaluated, &terminal_before_stop, &terminal_after_stop] {
        let manifest = checkpoint::inspect(&store.directory, reference)?;
        let loaded = checkpoint::load_state(&store.directory, &manifest)?;
        assert_eq!(loaded.cursor, state.cursor);
        assert_eq!(loaded.counters, state.counters);
    }
    let mut continuation = disk::release(
        &parent,
        &state,
        &owner,
        &terminal_after_stop,
        "late_stop_child",
        Terminal::UserStop,
    )?;
    continuation.explicit_user_renewal = true;
    checkpoint::restore(
        &f.workspace,
        &parent,
        &terminal_after_stop,
        &child,
        &identity,
        &continuation,
        &source,
    )?;
    Ok(())
}
