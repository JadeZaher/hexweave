use super::checkpoint;
use super::checkpoint_owner::Terminal;
use super::evaluation::Coverage;
use super::geometry::Geometry;
use super::prompt_evaluation::{self, Report};
use super::prompt_input::Input;
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
fn prompt_schema_bounds_reject_keys_duplicates_and_truncation() -> Result<()> {
    let good = br#"{"schema":1,"prompts":[{"id":"one","prompt":"a"}]}"#;
    Input::parse(good, 1, 2, 8)?;
    for bad in [
        r#"{"schema":1,"prompts":[{"id":"one","prompt":"a","answer":"x"}]}"#,
        r#"{"schema":1,"prompts":[{"id":"one","id":"two","prompt":"a"}]}"#,
        r#"{"schema":1,"prompts":[{"id":"one","prompt":"a"},{"id":"one","prompt":"b"}]}"#,
        r#"{"schema":1,"prompts":[{"id":"one","prompt":""}]}"#,
    ] {
        assert!(Input::parse(bad.as_bytes(), 1, 2, 8).is_err());
    }
    assert!(Input::parse(good, 2, 2, 8).is_err());
    assert!(Input::parse(good, 1, 8, 8).is_err());
    let forty = serde_json::json!({"schema":1,"prompts":(0..40).map(|i|
        serde_json::json!({"id":format!("case_{i}"),"prompt":"x".repeat(832)})).collect::<Vec<_>>()});
    Input::parse(&serde_json::to_vec(&forty)?, 40, 192, 1024)?;
    assert!(Input::parse(&vec![b' '; 65537], 1, 1, 8).is_err());
    Ok(())
}

#[test]
fn restored_prompt_evaluation_and_stop_leave_every_scientific_value_unchanged() -> Result<()> {
    let f = disk::Fixture::new()?;
    let source = source::Source::new();
    let parent = f.attempt("prompt_parent")?;
    let child = f.attempt("prompt_child")?;
    let state = State::with_cursor(Geometry::fixture(), 31, source.cursor())?;
    let identity = disk::identity(&state)?;
    let owner = disk::owner(&state, "prompt_parent");
    let mut store = DiskStore::create(
        &f.workspace,
        &parent,
        state.geometry,
        state.seed,
        owner.budget.disk_bytes_per_attempt,
    )?;
    let reference = checkpoint::publish(&mut store, &state, &identity, &owner, &source)?;
    let request = disk::release(
        &parent,
        &state,
        &owner,
        &reference,
        "prompt_child",
        Terminal::Completed,
    )?;
    let (mut restored_store, restored) = checkpoint::restore(
        &f.workspace,
        &parent,
        &reference,
        &child,
        &identity,
        &request,
        &source,
    )?;
    let evaluated = checkpoint::publish(
        &mut restored_store,
        &restored,
        &identity,
        &request.new_owner,
        &source,
    )?;
    let bytes = br#"{"schema":1,"prompts":[{"id":"one","prompt":"a"},{"id":"two","prompt":"b"}]}"#;
    let input = Input::parse(bytes, 2, 2, 8)?;
    let report = Report {
        input_sha256: super::storage_paths::digest(bytes),
        parent_checkpoint: reference,
        evaluation_checkpoint: evaluated,
        before_sha256: String::new(),
        after_sha256: String::new(),
        requested_cases: 2,
        cases: Vec::with_capacity(2),
        coverage: Coverage::Complete,
    };
    let result = prompt_evaluation::evaluate(
        &restored,
        &mut restored_store,
        &input,
        2,
        report.clone(),
        &mut || Ok(None),
    )?;
    assert_eq!(result.cases.len(), 2);
    assert_eq!(result.before_sha256, result.after_sha256);
    let stopped = prompt_evaluation::evaluate(
        &restored,
        &mut restored_store,
        &input,
        2,
        report,
        &mut || Ok(Some(Terminal::UserStop)),
    )?;
    assert!(matches!(
        stopped.coverage,
        Coverage::Interrupted(Terminal::UserStop)
    ));
    assert!(stopped.cases.is_empty());
    assert_eq!(stopped.before_sha256, result.before_sha256);
    assert_eq!(stopped.before_sha256, stopped.after_sha256);
    assert_eq!(restored.counters, state.counters);
    assert_eq!(restored.cursor, state.cursor);
    assert_eq!(source.target_reads.get(), 0);
    assert_eq!(source.advances.get(), 0);
    assert_eq!(
        request.new_owner.consumed.opportunities,
        owner.consumed.opportunities
    );
    Ok(())
}

#[test]
fn forty_maximum_outputs_leave_bounded_room_in_terminal_serializer() -> Result<()> {
    use super::generation::{Generation, Reason, Routes};
    let reference = super::checkpoint_state::CheckpointRef {
        sequence: u64::MAX,
        manifest_sha256: "f".repeat(64),
    };
    let report = Report {
        input_sha256: "f".repeat(64),
        parent_checkpoint: reference.clone(),
        evaluation_checkpoint: reference,
        before_sha256: "f".repeat(64),
        after_sha256: "f".repeat(64),
        requested_cases: 40,
        cases: (0..40)
            .map(|_| super::prompt_evaluation::Case {
                id: "x".repeat(64),
                prompt_sha256: "f".repeat(64),
                generation: Generation {
                    raw_bytes: vec![255; 192],
                    valid_utf8: false,
                    stop_reason: Reason::ByteBudget,
                    routes: Routes {
                        visits: [768, 0, 0, 0, 0, 0, 0],
                        route_calls: 576,
                        halts: 0,
                        forced_exits: 192,
                    },
                    wall_micros: 86_400_000_000,
                    before_sha256: "f".repeat(64),
                    after_sha256: "f".repeat(64),
                },
            })
            .collect(),
        coverage: Coverage::Complete,
    };
    let bytes = super::child_io::json(
        &serde_json::json!({"evaluation":report,"other_receipt_reserve":"x".repeat(65536)}),
    )?;
    assert!(bytes.len() <= super::child_io::RESULT_CAP as usize);
    Ok(())
}
