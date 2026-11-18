use super::credit;
use super::execution::{self, Frame, Step, Trace};
use super::geometry::{Geometry, TILES};
use super::state::{Boundary, State};
use super::store_api::TileStore;
use super::tile_gradients::TileGradients;
use anyhow::Result;

#[path = "execution_fixture.rs"]
#[allow(clippy::duplicate_mod)]
mod fixture;
use fixture::{Source, Store};

#[test]
fn local_targets_are_inaccessible_and_stop_does_not_begin_work() -> Result<()> {
    let source = Source::new();
    let mut state = State::with_cursor(Geometry::fixture(), 29, source.cursor())?;
    let mut store = Store::new(29)?;
    assert!(matches!(
        execution::step(&mut state, &mut store, &source, true)?,
        Step::Stopped
    ));
    assert_eq!(source.target_reads.get(), 0);
    execution::step(&mut state, &mut store, &source, false)?;
    let pending = state.cursor.clone();
    let embedding = state.shared.packed().to_vec();
    let shared_version = state.shared.version;
    for _ in 0..3 {
        assert!(matches!(
            execution::step(&mut state, &mut store, &source, false)?,
            Step::Committed { loss: None, .. }
        ));
        assert_eq!(state.cursor, pending);
        assert_eq!(source.target_reads.get(), 1);
        assert_eq!(source.advances.get(), 1);
        assert_eq!(state.shared.packed(), embedding);
        assert_eq!(state.shared.version, shared_version);
    }
    assert_eq!(state.global_opportunity, 4);
    execution::step(&mut state, &mut store, &source, false)?;
    assert_eq!(source.target_reads.get(), 2);
    assert_eq!(source.advances.get(), 2);
    assert_eq!(state.cursor.completion_offset, 2);
    Ok(())
}

#[test]
fn repeated_visits_aggregate_frozen_gradients_and_commit_tied_e_once() -> Result<()> {
    for count in 1..=4 {
        let source = Source::new();
        let mut state = State::with_cursor(Geometry::fixture(), 31, source.cursor())?;
        let mut store = Store::new(31)?;
        let prefix = state.shared.prefix(b"aba")?;
        let mut trace = repeated_trace(&mut store, &prefix, state.geometry)?;
        if count < 4 {
            trace.output = trace.frames[count].input.clone();
        }
        trace.frames.truncate(count);
        trace.frames.last_mut().unwrap().action = None;
        let credit = credit::gather(&state, &mut store, &prefix, &trace, b'a')?;
        assert_eq!(store.updates, [0; TILES]);
        let mut expected = store.copy()?;
        for id in [0, 3]
            .into_iter()
            .filter(|id| trace.frames.iter().any(|frame| frame.tile_id == *id))
        {
            let tile = &mut expected.tiles[id];
            let mut gradient = TileGradients::zeros(state.geometry, id)?;
            for (frame, correction) in trace.frames.iter().zip(&credit.corrections) {
                if frame.tile_id == id {
                    tile.backward(
                        &frame.input,
                        &prefix.memory,
                        &trace.history,
                        correction,
                        None,
                        Some(&mut gradient),
                        frame
                            .action
                            .map(|action| (action, credit.loss - state.baseline)),
                    )?;
                }
            }
            tile.apply(&gradient)?;
        }
        let mut expected_shared = fixture::restore(&state)?.shared;
        expected_shared.apply(&credit.embedding_gradient)?;
        credit::anchor(&mut state, &mut store, &prefix, &trace, b'a')?;
        // Seven-tile layout; tiles 3 and 0 are adjacent in both meshes, so the route still runs.
        #[cfg(not(feature = "tiles19"))]
        assert_eq!(store.updates, [usize::from(count > 1), 0, 0, 1, 0, 0, 0]);
        assert_eq!(state.shared.packed(), expected_shared.packed());
        assert_eq!(state.shared.version, expected_shared.version);
        for id in 0..7 {
            assert_eq!(store.tiles[id].packed(), expected.tiles[id].packed());
        }
        assert_eq!(state.history.records.len(), count);
        assert_eq!(state.counters.predictor_anchor_updates, 1);
        assert_eq!(state.counters.predictor_anchor_pairs, count as u64);
        assert_eq!(
            state
                .history
                .records
                .iter()
                .map(|r| r.tile_id)
                .collect::<Vec<_>>(),
            [3, 0, 3, 0][..count]
        );
        let mut expected_predictor = super::predictor::Predictor::new(state.geometry, state.seed)?;
        let pairs: Vec<_> = state
            .history
            .records
            .iter()
            .map(|r| (r.features.as_slice(), r.correction.as_slice()))
            .collect();
        let update = expected_predictor.train_joint(&pairs)?;
        assert_eq!(state.predictor.packed(), expected_predictor.packed());
        assert_eq!(state.predictor.version, expected_predictor.version);
        assert_eq!(
            state.counters.predictor_mse_sum.to_bits(),
            update
                .pair_mse
                .iter()
                .map(|&m| f64::from(m))
                .sum::<f64>()
                .to_bits()
        );
        assert!(state
            .history
            .records
            .iter()
            .all(|record| record.tile_versions == [0; TILES] && record.shared_version == 0));
    }
    Ok(())
}

fn repeated_trace(
    store: &mut Store,
    prefix: &super::shared::Prefix,
    geometry: Geometry,
) -> Result<Trace> {
    let mesh = super::geometry::mesh();
    let history = vec![0.0; geometry.width];
    let mut input = prefix.initial(geometry).to_vec();
    let mut frames = Vec::new();
    let ids = [3, 0, 3, 0];
    for (depth, &id) in ids.iter().enumerate() {
        let output = store.tiles[id]
            .forward(&input, &prefix.memory, &history)?
            .output;
        let action = ids.get(depth + 1).map(|next| {
            mesh.neighbors(id)
                .unwrap()
                .iter()
                .position(|id| id == next)
                .unwrap()
        });
        frames.push(Frame {
            input,
            tile_id: id,
            version: 0,
            action,
            depth,
        });
        input = output;
    }
    Ok(Trace {
        frames,
        history,
        output: input,
        tile_versions: store.versions()?,
    })
}

#[test]
fn stale_frozen_frame_rejects_and_mid_op_failure_poison_requires_recovery() -> Result<()> {
    let source = Source::new();
    let mut state = State::with_cursor(Geometry::fixture(), 43, source.cursor())?;
    let mut store = Store::new(43)?;
    let prefix = state.shared.prefix(b"aba")?;
    let mut trace = repeated_trace(&mut store, &prefix, state.geometry)?;
    trace.frames[2].version = 1;
    assert!(credit::gather(&state, &mut store, &prefix, &trace, b'a').is_err());
    assert_eq!(store.updates, [0; TILES]);
    execution::step(&mut state, &mut store, &source, false)?;
    let saved = fixture::restore(&state)?;
    let cursor = state.cursor.clone();
    store.fail_update = true;
    assert!(execution::step(&mut state, &mut store, &source, false).is_err());
    assert_eq!(state.boundary, Boundary::Poisoned);
    assert_eq!(state.cursor, cursor);
    assert!(state.validate_boundary().is_err());
    assert!(execution::step(&mut state, &mut store, &source, true).is_err());
    saved.validate_boundary()?;
    Ok(())
}

#[test]
fn split_continuation_preserves_routes_history_predictor_and_exposure() -> Result<()> {
    let source = Source::new();
    let mut full = State::with_cursor(Geometry::fixture(), 59, source.cursor())?;
    let mut first_store = Store::new(59)?;
    for _ in 0..5 {
        execution::step(&mut full, &mut first_store, &source, false)?;
    }
    let mut resumed = fixture::restore(&full)?;
    let mut second_store = first_store.copy()?;
    let resumed_source = Source::new();
    for _ in 0..7 {
        execution::step(&mut full, &mut first_store, &source, false)?;
        execution::step(&mut resumed, &mut second_store, &resumed_source, false)?;
    }
    fixture::equivalent(&full, &resumed, &first_store, &second_store);
    assert_eq!(full.counters.anchors, 3);
    assert_eq!(full.cursor.epoch, 1);
    assert_eq!(full.cursor.completion_offset, 0);
    assert_eq!(full.counters.local_opportunities, 9);
    Ok(())
}

#[test]
fn replay_selects_newest_four_retained_records_without_pending_target() -> Result<()> {
    let source = Source::new();
    let mut initial = State::with_cursor(Geometry::fixture(), 29, source.cursor())?;
    let mut original_store = Store::new(29)?;
    execution::step(&mut initial, &mut original_store, &source, false)?;
    let mut base = initial.history.records[0].clone();
    base.tile_versions = initial.tile_versions;
    base.shared_version = initial.shared.version;
    for retained in [0, 1, 4, 8] {
        let mut state = fixture::restore(&initial)?;
        let mut store = original_store.copy()?;
        state.history.records.clear();
        for index in 0..retained {
            let mut record = base.clone();
            record.features[0] = index as f32 / 8.0;
            state.history.push(record, state.geometry)?;
        }
        let mut expected = fixture::restore(&state)?.predictor;
        let selected = if retained == 8 {
            &state.history.records[4..8]
        } else {
            &state.history.records[..]
        };
        assert_eq!(state.history.replay(), selected);
        let pairs: Vec<_> = selected
            .iter()
            .map(|r| (r.features.as_slice(), r.correction.as_slice()))
            .collect();
        let old_mse = state.counters.predictor_mse_sum;
        let mut expected_mse = old_mse;
        if !pairs.is_empty() {
            for mse in expected.train_joint(&pairs)?.pair_mse {
                expected_mse += f64::from(mse);
            }
        }
        let reads = source.target_reads.get();
        execution::step(&mut state, &mut store, &source, false)?;
        assert_eq!(source.target_reads.get(), reads);
        assert_eq!(state.predictor.packed(), expected.packed());
        assert_eq!(state.predictor.version, expected.version);
        assert_eq!(
            state.counters.predictor_replay_updates,
            u64::from(retained > 0)
        );
        assert_eq!(
            state.counters.predictor_replay_pairs,
            retained.min(4) as u64
        );
        assert_eq!(
            state.counters.predictor_mse_sum.to_bits(),
            expected_mse.to_bits()
        );
    }
    let mut history = super::history::History::default();
    for index in 0..8 {
        let mut record = base.clone();
        record.features[0] = index as f32;
        record.opportunity = 4;
        record.shared_version = 1;
        record.tile_versions = [1; TILES];
        history.push(record, initial.geometry)?;
    }
    history.records[0].opportunity = 0;
    history.records[3].shared_version = 0;
    history.records[6].tile_versions[2] = 0;
    history.records[7].tile_versions[1] = 3;
    assert_eq!(history.prune(9, &[2; TILES], 2), 4);
    assert_eq!(
        history
            .replay()
            .iter()
            .map(|r| r.features[0])
            .collect::<Vec<_>>(),
        [1.0, 2.0, 4.0, 5.0]
    );
    Ok(())
}
