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

#[test]
fn adjacent_sampling_endpoints_atomicity_and_route_independence() -> Result<()> {
    assert_eq!(assay_update::draw(0, 0), 0.8833108082136426);
    let mut shared = super::shared::Shared::new(Geometry::fixture(), 7)?;
    let before = shared.packed().to_vec();
    let zeros = vec![0.0; shared.embedding.len()];
    let result = assay_update::apply(&mut shared, &zeros, Rule::Magnitude, 7, 0)?;
    assert_eq!(result.attempts, 0);
    assert_eq!(shared.packed(), before);
    let ones = vec![1.0; shared.embedding.len()];
    let expected = (0..shared.embedding.len())
        .filter(|&i| shared.embedding.value(i).unwrap() > -1)
        .count();
    let result = assay_update::apply(&mut shared, &ones, Rule::Magnitude, 7, 0)?;
    assert_eq!(result.attempts, ones.len() as u64);
    assert_eq!(result.transitions, expected as u64);
    assert_eq!(result.attempts, result.transitions + result.clamped);
    assert_eq!(shared.version, 1);
    let before = shared.packed().to_vec();
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut values = ones.clone();
        *values.last_mut().unwrap() = bad;
        assert!(assay_update::apply(&mut shared, &values, Rule::Magnitude, 7, 0).is_err());
        assert_eq!(shared.packed(), before);
        assert_eq!(shared.version, 1);
    }
    assert!(assay_update::apply(&mut shared, &ones[..1], Rule::Magnitude, 7, 0).is_err());
    assert_ne!(assay_update::base(7, 0), assay_update::base(7, 1));
    assert_ne!(assay_update::base(7, 0), assay_update::base(17, 0));
    let mut tiny = zeros;
    tiny[..8].fill(2.0_f32.powi(-24));
    let result = assay_update::apply(&mut shared, &tiny, Rule::Magnitude, 7, 0)?;
    assert_eq!(
        (result.attempts, result.clamped, result.transitions),
        (0, 0, 0)
    );
    assert_eq!(result.probability_sum.to_bits(), 0x3ea0000000000000);
    assert_eq!(shared.packed(), before);
    assert_eq!(shared.version, 1);
    fractional_sampling_vectors()?;
    Ok(())
}

fn fractional_sampling_vectors() -> Result<()> {
    let initial = [-1, 1, 0, 0, 1, -1, 0, 0];
    let gradients = [0.75_f32, -0.5, 0.75, -0.25, 0.5, -0.75, 0.25, -0.5];
    let cases = [
        (
            7,
            0,
            0x380aa0f40d16c133,
            [
                0x3fe1de4c2c9ad2ba,
                0x3fd9b582c824f2ea,
                0x3fc2c9061d35e840,
                0x3fc65b31d024e298,
                0x3fe92638ffe8769a,
                0x3fbbd7ae09f20050,
                0x3fdb89314e927e12,
                0x3fe3806d148c2530,
            ],
            0x2f_u8,
            [0x88, 0x56],
        ),
        (
            17,
            3,
            0xb45ede8210b687e3,
            [
                0x3fd12ef895d1e3f4,
                0x3f6f0626cf3e3200,
                0x3fe83ce83202ee25,
                0x3fc0016f530c5a5c,
                0x3fec3c71b59fcc4f,
                0x3fe2883295457790,
                0x3feeb6c2b00ac438,
                0x3fc6a32a2023d6ec,
            ],
            0xab,
            [0x98, 0x96],
        ),
    ];
    for (seed, anchor, base, draw_bits, accepted, packed) in cases {
        assert_eq!(assay_update::base(seed, anchor), base);
        let mut sampled = 0_u8;
        for (index, g) in gradients.iter().enumerate() {
            let draw = assay_update::draw(base, index);
            assert_eq!(draw.to_bits(), draw_bits[index]);
            sampled |= u8::from(draw < f64::from(g.abs())) << index;
        }
        assert_eq!(sampled, accepted);
        let geometry = Geometry::fixture();
        let mut before = vec![0x55; geometry.width * super::geometry::VOCAB / 4];
        before[..2].copy_from_slice(&[0x58, 0x52]);
        let mut shared = super::shared::Shared::from_packed(
            geometry,
            9,
            &before,
            super::geometry::EVIDENCE_RULE,
        )?;
        let mut gradient = vec![0.0; shared.embedding.len()];
        gradient[..8].copy_from_slice(&gradients);
        let result = assay_update::apply(&mut shared, &gradient, Rule::Magnitude, seed, anchor)?;
        before[..2].copy_from_slice(&packed);
        assert_eq!(shared.packed(), before);
        assert_eq!(shared.version, 10);
        assert_eq!(
            (result.attempts, result.clamped, result.transitions),
            (5, 2, 3)
        );
        assert_eq!(result.probability_sum.to_bits(), 0x4011000000000000);
        let mut transitioned = 0_u8;
        for (index, old) in initial.iter().enumerate() {
            transitioned |= u8::from(shared.embedding.value(index)? != *old) << index;
        }
        assert_eq!(transitioned, accepted ^ 0x03);
        assert_eq!(sampled & !transitioned, 0x03);
    }
    Ok(())
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

#[test]
fn source_order_and_v3_state_acceptance_stay_separate() -> Result<()> {
    let source = source()?;
    assert_eq!(cursor(80)?.example_id, "foundation-00");
    assert_eq!(cursor(1280)?.example_id, "retention-stress-00");
    assert_eq!(cursor(1600)?.example_id, "complete");
    assert!(source.anchor_target(&cursor(1600)?).is_err());
    for n in 0..1600 {
        let value = cursor(n)?;
        source.validate(&value)?;
        assert_eq!(
            source.anchor_target(&value)?.kind == super::data_contracts::TargetKind::Format,
            n % 5 == 4
        );
    }
    let plain = State::with_cursor(Geometry::fixture(), 7, cursor(0)?)?;
    let mut value = serde_json::to_value(StateRecord::capture(&plain)?)?;
    assert!(value.get("assay").is_none());
    value["assay"] = serde_json::Value::Null;
    assert!(serde_json::from_value::<StateRecord>(value).is_err());
    let mut config = super::config::Config::assay(7, Rule::Magnitude);
    config.protocol = super::config::PROTOCOL.into();
    assert!(config.validate().is_err());
    Ok(())
}
