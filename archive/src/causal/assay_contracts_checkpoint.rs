//! Archived from src/causal/assay_contracts.rs: DiskStore checkpoint/observation test; does not build here (archive/README.md).

use super::assay_source::{cursor, Source};
use super::assay_state::{Progress, CONTRACT_SHA, MANIFEST_SHA, PROTOCOL};
use super::assay_update::{self, Rule};
use super::checkpoint;
use super::checkpoint_state::StateRecord;
use super::execution::{self, TrainingSource};
use super::geometry::Geometry;
use super::state::State;
use super::storage::DiskStore;
use anyhow::Result;

#[path = "persistence_fixture.rs"]
#[allow(clippy::duplicate_mod, dead_code)]
mod fixture;

fn source() -> Result<Source> {
    Source::open(
        &std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("datasets/retained-text-assay"),
        MANIFEST_SHA,
    )
}

// Fabricated metadata exercises persistence; it is never a measured quality result.
fn metadata_observation(
    state: &mut State,
    store: &mut DiskStore,
    source: &Source,
    attempt: &std::path::Path,
) -> Result<()> {
    let identity = super::immutable::identity(state, store)?;
    let cases = source
        .observations()?
        .into_iter()
        .map(|row| super::assay_measure::Case {
            id: row.id.clone(),
            content_nll: vec![0.0; 4],
            lf_nll: Some(0.0),
            first_byte_rank: Some(1),
            generated: Some(vec![b'\n']),
            generation_complete: true,
            exact: Some(false),
        })
        .collect();
    let observation = super::assay_observation::Observation {
        schema: 1,
        protocol: PROTOCOL.into(),
        manifest_sha256: MANIFEST_SHA.into(),
        seed: state.seed,
        rule: Rule::Magnitude,
        pass: 0,
        physical_anchors: 0,
        numerical_clock: 0,
        before_sha256: identity.clone(),
        after_sha256: identity,
        complete: true,
        interrupted: None,
        teacher_readouts: 240,
        generated_readouts: 48,
        cases,
    };
    let reference = super::assay_observation::publish(attempt, &observation)?;
    state.assay.as_mut().unwrap().observations.push(reference);
    Ok(())
}

#[test]
fn physical_ao_checkpoint_continuation_and_partial_observation_are_exact() -> Result<()> {
    let f = fixture::Fixture::new()?;
    let source = source()?;
    let first = f.attempt("first")?;
    let mut state = State::with_cursor(Geometry::candidate(), 7, cursor(0)?)?;
    state.assay = Some(Progress::new(Rule::Magnitude));
    let mut store = DiskStore::create(&f.workspace, &first, state.geometry, 7, 8 * 1024 * 1024)?;
    let mut polls = 0;
    let partial = super::assay_observation::observe(&state, &mut store, &source, &mut || {
        polls += 1;
        Ok((polls >= 5).then_some(super::checkpoint_owner::Terminal::UserStop))
    })?;
    assert!(!partial.complete);
    assert_eq!(partial.teacher_readouts, 3);
    assert_eq!(partial.generated_readouts, 0);
    assert_eq!(partial.before_sha256, partial.after_sha256);
    let partial_ref = super::assay_observation::publish(&first, &partial)?;
    let mut censored = Progress::new(Rule::Magnitude);
    censored.observations.push(partial_ref);
    assert!(
        super::assay_observation::copy_references(&first, &f.attempt("censored")?, &censored)
            .is_err()
    );
    metadata_observation(&mut state, &mut store, &source, &first)?;
    execution::step(&mut state, &mut store, &source, false)?;
    assert_eq!((state.global_opportunity, state.numerical_clock()?), (1, 4));
    assert_eq!(
        (state.counters.anchors, state.counters.local_opportunities),
        (1, 0)
    );
    let mut identity = fixture::identity(&state)?;
    identity.protocol = PROTOCOL.into();
    identity.protocol_sha256 = CONTRACT_SHA.into();
    identity.config_sha256 = super::config::Config::assay(7, Rule::Magnitude).sha256()?;
    let owner = fixture::owner(&state, "first");
    let reference = checkpoint::publish(&mut store, &state, &identity, &owner, &source)?;
    let continuation = fixture::release(
        &first,
        &state,
        &owner,
        &reference,
        "second",
        super::checkpoint_owner::Terminal::Completed,
    )?;
    let second = f.attempt("second")?;
    let (mut resumed_store, mut resumed) = checkpoint::restore(
        &f.workspace,
        &first,
        &reference,
        &second,
        &identity,
        &continuation,
        &source,
    )?;
    execution::step(&mut state, &mut store, &source, false)?;
    execution::step(&mut resumed, &mut resumed_store, &source, false)?;
    assert_eq!(
        super::immutable::identity(&state, &mut store)?,
        super::immutable::identity(&resumed, &mut resumed_store)?
    );
    assert_eq!(
        serde_json::to_vec(&StateRecord::capture(&state)?)?,
        serde_json::to_vec(&StateRecord::capture(&resumed)?)?
    );
    let mut broken = serde_json::to_value(StateRecord::capture(&state)?)?;
    broken["global_opportunity"] = serde_json::json!(8);
    let record: StateRecord = serde_json::from_value(broken)?;
    assert!(record.restore(resumed.shared, resumed.predictor).is_err());
    Ok(())
}
