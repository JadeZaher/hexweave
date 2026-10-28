use super::super::cursor::{AnchorTarget, Cursor};
use super::super::data::DataIdentity;
use super::super::data_contracts::{Split, TargetKind};
use super::super::execution::TrainingSource;
use super::super::geometry::{Geometry, EVIDENCE_RULE, TILES};
use super::super::predictor::Predictor;
use super::super::shared::Shared;
use super::super::state::{Boundary, State};
use super::super::store_api::TileStore;
use super::super::tile::Tile;
use anyhow::{ensure, Result};
use std::cell::Cell;

pub(super) struct Source {
    pub target_reads: Cell<usize>,
    pub advances: Cell<usize>,
    prefix: &'static [u8],
    targets: &'static [u8],
}

impl Source {
    pub fn new() -> Self {
        Self::with_bytes(b"?", b"aba")
    }
    pub fn with_bytes(prefix: &'static [u8], targets: &'static [u8]) -> Self {
        assert!(!prefix.is_empty() && !targets.is_empty());
        Self {
            target_reads: Cell::new(0),
            advances: Cell::new(0),
            prefix,
            targets,
        }
    }
    pub fn cursor(&self) -> Cursor {
        Cursor {
            schema: 1,
            data: DataIdentity {
                schema: "fixture".into(),
                manifest_sha256: "0".repeat(64),
                artifact_sha256: "1".repeat(64),
                source_pin_manifest_sha256: "2".repeat(64),
                dataset_version: "fixture".into(),
                source_as_of: "fixture".into(),
                split: Split::Train,
                rows: 1,
                completion_targets_per_epoch: self.targets.len() as u64,
            },
            row_index: 0,
            example_id: "fixture".into(),
            completion_offset: 0,
            epoch: 0,
            supervised_targets: 0,
            answer_targets: 0,
            citation_targets: 0,
            format_targets: 0,
        }
    }
}

impl TrainingSource for Source {
    fn validate(&self, cursor: &Cursor) -> Result<()> {
        ensure!(
            cursor.completion_offset < self.targets.len()
                && cursor.supervised_targets
                    == cursor.epoch * self.targets.len() as u64 + cursor.completion_offset as u64
                && cursor.answer_targets == cursor.supervised_targets,
            "fixture cursor mismatch"
        );
        Ok(())
    }
    fn pending_prefix(&self, cursor: &Cursor) -> Result<Vec<u8>> {
        self.validate(cursor)?;
        Ok(self
            .prefix
            .iter()
            .chain(&self.targets[..cursor.completion_offset])
            .copied()
            .collect())
    }
    fn anchor_target(&self, cursor: &Cursor) -> Result<AnchorTarget> {
        self.target_reads.set(self.target_reads.get() + 1);
        Ok(AnchorTarget {
            byte: self.targets[cursor.completion_offset],
            kind: TargetKind::Answer,
        })
    }
    fn advance_anchor(&self, cursor: &mut Cursor) -> Result<()> {
        self.advances.set(self.advances.get() + 1);
        cursor.supervised_targets += 1;
        cursor.answer_targets += 1;
        cursor.completion_offset += 1;
        if cursor.completion_offset == self.targets.len() {
            cursor.completion_offset = 0;
            cursor.epoch += 1;
        }
        self.validate(cursor)
    }
}

pub(super) struct Store {
    pub tiles: Vec<Tile>,
    pub updates: [usize; TILES],
    pub fail_update: bool,
}

impl Store {
    pub fn new(seed: u64) -> Result<Self> {
        Self::with_geometry(Geometry::fixture(), seed)
    }
    pub fn with_geometry(geometry: Geometry, seed: u64) -> Result<Self> {
        let tiles = (0..TILES)
            .map(|id| Tile::new(geometry, id, seed))
            .collect::<Result<_>>()?;
        Ok(Self {
            tiles,
            updates: [0; TILES],
            fail_update: false,
        })
    }
    pub fn copy(&self) -> Result<Self> {
        let tiles = self
            .tiles
            .iter()
            .map(|tile| {
                Tile::from_packed(
                    tile.geometry,
                    tile.id,
                    tile.version,
                    &tile
                        .packed()
                        .iter()
                        .map(|bytes| bytes.to_vec())
                        .collect::<Vec<_>>(),
                    EVIDENCE_RULE,
                )
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            tiles,
            updates: [0; TILES],
            fail_update: false,
        })
    }
}

impl TileStore for Store {
    fn versions(&self) -> Result<[u64; TILES]> {
        Ok(std::array::from_fn(|id| self.tiles[id].version))
    }
    fn read<T>(&mut self, id: usize, visit: impl FnOnce(&Tile) -> Result<T>) -> Result<T> {
        visit(&self.tiles[id])
    }
    fn update<T>(&mut self, id: usize, visit: impl FnOnce(&mut Tile) -> Result<T>) -> Result<T> {
        ensure!(!self.fail_update, "injected store failure");
        self.updates[id] += 1;
        visit(&mut self.tiles[id])
    }
}

pub(super) fn restore(state: &State) -> Result<State> {
    state.validate_boundary()?;
    Ok(State {
        geometry: state.geometry,
        seed: state.seed,
        global_opportunity: state.global_opportunity,
        baseline: state.baseline,
        shared: Shared::from_packed(
            state.geometry,
            state.shared.version,
            state.shared.packed(),
            EVIDENCE_RULE,
        )?,
        predictor: Predictor::from_packed(
            state.geometry,
            state.predictor.version,
            state.predictor.packed(),
            EVIDENCE_RULE,
        )?,
        tile_versions: state.tile_versions,
        history: state.history.clone(),
        cursor: state.cursor.clone(),
        counters: state.counters.clone(),
        boundary: Boundary::Ready,
        assay: state.assay.clone(),
        anchor_only: state.anchor_only,
    })
}

pub(super) fn equivalent(left: &State, right: &State, a: &Store, b: &Store) {
    assert_eq!(left.global_opportunity, right.global_opportunity);
    assert_eq!(left.baseline.to_bits(), right.baseline.to_bits());
    assert_eq!(left.cursor, right.cursor);
    assert_eq!(left.counters, right.counters);
    assert_eq!(left.history, right.history);
    assert_eq!(left.shared.packed(), right.shared.packed());
    assert_eq!(left.predictor.packed(), right.predictor.packed());
    assert_eq!(left.shared.version, right.shared.version);
    assert_eq!(left.predictor.version, right.predictor.version);
    assert_eq!(left.tile_versions, right.tile_versions);
    for (first, second) in a.tiles.iter().zip(&b.tiles) {
        assert_eq!(first.version, second.version);
        assert_eq!(first.packed(), second.packed());
    }
}
