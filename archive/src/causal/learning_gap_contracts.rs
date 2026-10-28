use super::execution::{self, Step};
use super::geometry::{Geometry, TILES};
use super::immutable;
use super::state::State;
use super::store_api::TileStore;
use super::tile::Tile;
use crate::topology::Mesh;
use anyhow::{ensure, Result};
use serde_json::json;

#[path = "execution_fixture.rs"]
#[allow(clippy::duplicate_mod, dead_code)]
mod fixture;
#[path = "learning_gap_fixture.rs"]
mod measurement;
use fixture::{Source, Store};
use measurement::Probe;

const PREFIX: &[u8] = b"Evidence: leaves are green.\nAnswer: ";
const SEEDS: [u64; 3] = [7, 17, 29];
const LOCALS: usize = 3;
const LABEL: &str = "LEARNING_GAP_DIAGNOSTIC ";

struct RecordedStore {
    inner: Store,
    visits: Vec<usize>,
}

impl TileStore for RecordedStore {
    fn versions(&self) -> Result<[u64; TILES]> {
        self.inner.versions()
    }
    fn read<T>(&mut self, id: usize, visit: impl FnOnce(&Tile) -> Result<T>) -> Result<T> {
        self.inner.read(id, visit)
    }
    fn update<T>(&mut self, id: usize, visit: impl FnOnce(&mut Tile) -> Result<T>) -> Result<T> {
        ensure!(self.visits.len() < 4, "fixture update-visit bound");
        self.visits.push(id);
        self.inner.update(id, visit)
    }
}

#[test]
#[ignore = "historical v2 live-execution diagnostic; ineligible under v3, reproduce only from its frozen v2 snapshot"]
fn fixed_candidate_learning_gap_diagnostic() -> Result<()> {
    for seed in SEEDS {
        scenario(seed)?;
    }
    Ok(())
}

fn scenario(seed: u64) -> Result<()> {
    let source = Source::with_bytes(PREFIX, b"{\"}");
    let geometry = Geometry::candidate();
    let mut anchor = State::with_cursor(geometry, seed, source.cursor())?;
    let mut store = Store::with_geometry(geometry, seed)?;
    let probe = Probe::freeze(&anchor, &mut store, PREFIX)?;
    let before = probe.observe(&anchor, &mut store)?;
    let step = execution::step(&mut anchor, &mut store, &source, false)?;
    match step {
        Step::Committed {
            opportunity: 0,
            loss: Some(loss),
            visits,
            prefix_bytes,
        } => {
            assert_eq!(loss.to_bits(), before.anchor_loss.to_bits());
            assert_eq!(visits, probe.route.len());
            assert_eq!(prefix_bytes, PREFIX.len());
        }
        other => panic!("expected one anchor, got {other:?}"),
    }
    for id in 0..TILES {
        assert_eq!(
            store.updates[id],
            usize::from(probe.route.iter().any(|frame| frame.0 == id))
        );
    }
    let after_anchor = probe.observe(&anchor, &mut store)?;
    let mut local = fixture::restore(&anchor)?;
    let mut recorded = RecordedStore {
        inner: store.copy()?,
        visits: Vec::with_capacity(4),
    };
    fixture::equivalent(&anchor, &local, &store, &recorded.inner);
    assert_eq!(
        after_anchor.state_sha256,
        immutable::identity(&local, &mut recorded.inner)?
    );
    let mut local_rows = Vec::with_capacity(LOCALS);
    for index in 0..LOCALS {
        let route = local_step(&mut local, &mut recorded, &source, index)?;
        assert_eq!(local.shared.packed(), anchor.shared.packed());
        let observed = probe.observe(&local, &mut recorded.inner)?;
        local_rows.push(json!({
            "locals_completed": index + 1,
            "actual_seeded_update_route": route,
            "anchor_loss_delta_from_anchor_only": observed.anchor_loss - after_anchor.anchor_loss,
            "pending_loss_delta_from_anchor_only": observed.pending_loss - after_anchor.pending_loss,
            "observation": observed,
        }));
    }
    assert_eq!(
        after_anchor.state_sha256,
        immutable::identity(&anchor, &mut store)?
    );
    assert_eq!(local.global_opportunity, 4);
    assert_eq!(local.counters.anchors, 1);
    assert_eq!(local.counters.local_opportunities, 3);
    assert_eq!(source.target_reads.get(), 1);
    assert_eq!(source.advances.get(), 1);
    let report = json!({
        "schema": 1, "kind": "conditional_learning_gap_fixture", "seed": seed,
        "geometry": geometry, "kernel": crate::ternary::kernel_name()?,
        "prefix_utf8": std::str::from_utf8(PREFIX)?,
        "anchor_target_byte": 123, "pending_target_byte": 34,
        "fixed_measurement_route": probe.route,
        "measurement_history": "zero",
        "actual_anchor_route": probe.route,
        "actual_anchor_route_origin": "same initial production forward path and seed; loss/visits checked",
        "generated_prefix_appended_byte_frozen_at_initial_state": probe.generated_byte,
        "predictor_comparison": "same frozen initial anchor features and clipped targets in all phases",
        "anchor_loss_delta": after_anchor.anchor_loss - before.anchor_loss,
        "initial": before, "anchor_only": after_anchor, "local_branch": local_rows,
        "executed_anchor_updates": 1, "executed_local_opportunities": LOCALS,
        "scope": "fresh test-only conditional fixture, not E3 replay, learning gain or quality evidence",
    });
    let encoded = serde_json::to_string(&report)?;
    assert!(encoded.len() + LABEL.len() < 32 * 1024);
    println!("{LABEL}{encoded}");
    Ok(())
}

fn local_step(
    state: &mut State,
    store: &mut RecordedStore,
    source: &Source,
    index: usize,
) -> Result<serde_json::Value> {
    store.visits.clear();
    let halts = state.counters.halts;
    let forced = state.counters.forced_exits;
    let cursor = state.cursor.clone();
    let step = execution::step(state, store, source, false)?;
    match step {
        Step::Committed {
            opportunity,
            loss: None,
            visits,
            prefix_bytes,
        } => {
            assert_eq!(opportunity, index as u64 + 1);
            assert_eq!(visits, store.visits.len());
            assert_eq!(prefix_bytes, PREFIX.len() + 1);
        }
        other => panic!("expected target-hidden local, got {other:?}"),
    }
    assert_eq!(state.cursor, cursor);
    assert_eq!(source.target_reads.get(), 1);
    assert_eq!(source.advances.get(), 1);
    assert_eq!(store.visits.first(), Some(&3));
    let mesh = Mesh::new(1)?;
    let mut actions = Vec::with_capacity(store.visits.len());
    for pair in store.visits.windows(2) {
        let neighbors = mesh.neighbors(pair[0])?;
        actions.push(Some(
            neighbors
                .iter()
                .position(|&id| id == pair[1])
                .expect("legal visit"),
        ));
    }
    let halted = state.counters.halts - halts == 1;
    let forced_exit = state.counters.forced_exits - forced == 1;
    assert_ne!(halted, forced_exit);
    if halted {
        assert!(store.visits.len() < 4);
        actions.push(Some(mesh.neighbors(*store.visits.last().unwrap())?.len()));
    } else {
        assert_eq!(store.visits.len(), 4);
        actions.push(None);
    }
    Ok(json!({"tile_ids":store.visits,"actions":actions,"forced_exit":forced_exit}))
}
