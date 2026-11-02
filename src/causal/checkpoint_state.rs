//! Bit-exact serialized continuation State (history, counters, cursor) for core checkpoints.

use super::cursor::Cursor;
use super::geometry::{Geometry, EVIDENCE_RULE, TILES};
use super::history::{History, Record};
use super::state::{Boundary, Counters, State};
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct HistoryRecord {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    feature_bits: Vec<u32>,
    correction_bits: Vec<u32>,
    tile_versions: [u64; TILES],
    shared_version: u64,
    opportunity: u64,
    tile_id: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct StateRecord {
    schema: u32,
    evidence_rule: String,
    boundary: Boundary,
    pub geometry: Geometry,
    pub seed: u64,
    pub global_opportunity: u64,
    baseline_bits: u32,
    pub shared_version: u64,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub predictor_version: u64,
    pub tile_versions: [u64; TILES],
    history: Vec<HistoryRecord>,
    pub cursor: Cursor,
    counters: Counters,
    #[serde(default, skip_serializing_if = "is_zero")]
    predictor_mse_bits: u64,
    anchor_loss_bits: u64,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "super::assay_state::present"
    )]
    pub assay: Option<super::assay_state::Progress>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub anchor_only: bool,
}

/// Record schema: 1 interleaved v3, 2 assay ledger, 3 core anchor-only; the two modes are exclusive.
fn schema(assay: bool, anchor_only: bool) -> Result<u32> {
    match (assay, anchor_only) {
        (true, true) => anyhow::bail!("assay ledger and core anchor-only state are exclusive"),
        (true, false) => Ok(2),
        (false, true) => Ok(3),
        (false, false) => Ok(1),
    }
}

fn is_zero(value: &u64) -> bool {
    *value == 0
}

impl StateRecord {
    pub fn capture(state: &State) -> Result<Self> {
        state.validate_boundary()?;
        let mut counters = state.counters.clone();
        counters.predictor_mse_sum = 0.0;
        counters.anchor_loss_sum = 0.0;
        Ok(Self {
            schema: schema(state.assay.is_some(), state.anchor_only)?,
            evidence_rule: EVIDENCE_RULE.into(),
            boundary: state.boundary,
            geometry: state.geometry,
            seed: state.seed,
            global_opportunity: state.global_opportunity,
            baseline_bits: state.baseline.to_bits(),
            shared_version: state.shared.version,
            predictor_version: state.predictor.version,
            tile_versions: state.tile_versions,
            history: state
                .history
                .records
                .iter()
                .map(|record| HistoryRecord {
                    feature_bits: record.features.iter().map(|v| v.to_bits()).collect(),
                    correction_bits: record.correction.iter().map(|v| v.to_bits()).collect(),
                    tile_versions: record.tile_versions,
                    shared_version: record.shared_version,
                    opportunity: record.opportunity,
                    tile_id: record.tile_id,
                })
                .collect(),
            cursor: state.cursor.clone(),
            counters,
            predictor_mse_bits: state.counters.predictor_mse_sum.to_bits(),
            anchor_loss_bits: state.counters.anchor_loss_sum.to_bits(),
            assay: state.assay.clone(),
            anchor_only: state.anchor_only,
        })
    }

    pub fn restore(
        self,
        shared: super::shared::Shared,
        predictor: super::predictor::Predictor,
    ) -> Result<State> {
        ensure!(
            self.schema == schema(self.assay.is_some(), self.anchor_only)?
                && self.evidence_rule == EVIDENCE_RULE
                && self.boundary == Boundary::Ready,
            "checkpoint does not encode complete canonical Ready state"
        );
        ensure!(
            self.history.len() <= 8
                && self.counters.predictor_mse_sum.to_bits() == 0
                && self.counters.anchor_loss_sum.to_bits() == 0,
            "noncanonical state metric encoding"
        );
        ensure!(
            shared.version == self.shared_version && predictor.version == self.predictor_version,
            "shared payload versions mismatch"
        );
        let mut records = Vec::with_capacity(self.history.len());
        for record in self.history {
            ensure!(
                record.feature_bits.len()
                    == if super::tuning::current().fit_predictor {
                        self.geometry.features()
                    } else {
                        0
                    }
                    && record.correction_bits.len() == self.geometry.width,
                "checkpoint history shape mismatch"
            );
            records.push(Record {
                features: record
                    .feature_bits
                    .into_iter()
                    .map(f32::from_bits)
                    .collect(),
                correction: record
                    .correction_bits
                    .into_iter()
                    .map(f32::from_bits)
                    .collect(),
                tile_versions: record.tile_versions,
                shared_version: record.shared_version,
                opportunity: record.opportunity,
                tile_id: record.tile_id,
            });
        }
        let mut counters = self.counters;
        counters.predictor_mse_sum = f64::from_bits(self.predictor_mse_bits);
        counters.anchor_loss_sum = f64::from_bits(self.anchor_loss_bits);
        let state = State {
            geometry: self.geometry,
            seed: self.seed,
            global_opportunity: self.global_opportunity,
            baseline: f32::from_bits(self.baseline_bits),
            shared,
            predictor,
            tile_versions: self.tile_versions,
            history: History { records },
            cursor: self.cursor,
            counters,
            boundary: Boundary::Ready,
            assay: self.assay,
            anchor_only: self.anchor_only,
        };
        state.validate_boundary()?;
        Ok(state)
    }
}
