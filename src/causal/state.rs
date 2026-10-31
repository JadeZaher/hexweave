use super::cursor::Cursor;
use super::data::Dataset;
use super::geometry::{Geometry, TILES, VOCAB};
use super::history::History;
use super::predictor::Predictor;
use super::shared::Shared;
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum Boundary {
    Ready,
    InFlight,
    Poisoned,
}

pub(super) use super::state_counters::Counters;

pub(super) struct State {
    pub geometry: Geometry,
    pub seed: u64,
    pub global_opportunity: u64,
    pub baseline: f32,
    pub shared: Shared,
    pub predictor: Predictor,
    pub tile_versions: [u64; TILES],
    pub history: History,
    pub cursor: Cursor,
    pub counters: Counters,
    pub boundary: Boundary,
    pub assay: Option<super::assay_state::Progress>,
    /// Core-path anchor-only schedule without the assay ledger.
    pub anchor_only: bool,
}

pub(super) fn count(value: &mut u64, amount: usize) -> Result<()> {
    *value = value
        .checked_add(u64::try_from(amount)?)
        .context("work counter overflow")?;
    Ok(())
}

impl State {
    pub fn new(geometry: Geometry, seed: u64, dataset: &Dataset) -> Result<Self> {
        Self::with_cursor(geometry, seed, Cursor::start(dataset)?)
    }

    pub fn with_cursor(geometry: Geometry, seed: u64, cursor: Cursor) -> Result<Self> {
        geometry.validate()?;
        ensure!(
            cursor.supervised_targets == 0
                && cursor.epoch == 0
                && cursor.row_index == 0
                && cursor.completion_offset == 0,
            "initial cursor is not at origin"
        );
        Ok(Self {
            geometry,
            seed,
            global_opportunity: 0,
            baseline: (VOCAB as f32).ln(),
            shared: Shared::new(geometry, seed)?,
            predictor: Predictor::new(geometry, seed)?,
            tile_versions: [0; TILES],
            history: History::default(),
            cursor,
            counters: Counters::default(),
            boundary: Boundary::Ready,
            assay: None,
            anchor_only: false,
        })
    }

    /// Every opportunity is an anchor (assay ledger or core path); the clock then advances by 4.
    pub fn anchors_only(&self) -> bool {
        self.assay.is_some() || self.anchor_only
    }

    pub fn numerical_clock(&self) -> Result<u64> {
        if self.anchors_only() {
            self.global_opportunity
                .checked_mul(4)
                .context("assay numerical clock overflow")
        } else {
            Ok(self.global_opportunity)
        }
    }

    #[allow(dead_code)]
    pub fn validate(&self, dataset: &Dataset) -> Result<()> {
        self.validate_boundary()?;
        self.cursor.validate(dataset)
    }

    /// Validate complete continuation state; dataset identity is checked separately.
    pub fn validate_boundary(&self) -> Result<()> {
        self.geometry.validate()?;
        ensure!(
            self.boundary == Boundary::Ready,
            "only complete Ready state can continue or checkpoint"
        );
        ensure!(
            self.shared.geometry == self.geometry && self.predictor.geometry == self.geometry,
            "shared geometry mismatch"
        );
        ensure!(
            self.baseline.is_finite() && self.baseline >= 0.0,
            "invalid anchor baseline"
        );
        self.counters.validate_schedule(
            self.global_opportunity,
            self.geometry,
            self.anchors_only(),
            self.history.records.is_empty(),
        )?;
        if let Some(assay) = &self.assay {
            assay.validate(self)?;
        }
        ensure!(
            self.cursor.supervised_targets == self.counters.anchors,
            "cursor/anchor exposure mismatch"
        );
        let masks = self
            .cursor
            .answer_targets
            .checked_add(self.cursor.citation_targets)
            .and_then(|total| total.checked_add(self.cursor.format_targets));
        ensure!(
            masks == Some(self.cursor.supervised_targets),
            "target mask partition mismatch"
        );
        let predictor_updates = self
            .counters
            .predictor_anchor_updates
            .checked_add(self.counters.predictor_replay_updates)
            .context("predictor update overflow")?;
        let tile_updates = self
            .counters
            .anchor_tile_commits
            .checked_add(self.counters.local_tile_updates)
            .context("tile update overflow")?;
        ensure!(
            self.shared.version <= self.counters.anchors
                && self.predictor.version <= predictor_updates
                && self
                    .tile_versions
                    .iter()
                    .all(|&version| version <= tile_updates),
            "version exceeds possible commits"
        );
        self.history.validate(
            self.geometry,
            self.numerical_clock()?,
            &self.tile_versions,
            self.shared.version,
        )
    }
}
