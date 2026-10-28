//! Retrieval drill: the assay schedule in 1,600-anchor cycles, re-dealt exactly as the harness's
//! validated `Shuffled` stream (salt per cycle); see AGENTS.md §core.

use super::assay_source::{self, Source};
use super::assay_state::{ANCHORS, MANIFEST_SHA, ROWS_SHA, SCHEDULE_SHA};
use super::cursor::{AnchorTarget, Cursor};
use super::data::DataIdentity;
use super::data_contracts::{Split, TargetKind};
use super::execution::TrainingSource;
use super::tuning;
use super::tuning_sources::{is_template, redeal};
use anyhow::{ensure, Context, Result};

/// Anchors per cycle: the assay's 20 passes of 16 rows x 5 target bytes.
pub(super) const CYCLE: u64 = ANCHORS;

/// Foundation rows 0..16 then retention-stress rows 16..32, re-dealt per (cycle, row, pass).
pub(super) struct Drill {
    templates: Vec<Vec<u8>>,
    seed: u64,
}

fn identity() -> DataIdentity {
    DataIdentity {
        schema: "retrieval-drill-v2".into(),
        manifest_sha256: MANIFEST_SHA.into(),
        artifact_sha256: ROWS_SHA.into(),
        source_pin_manifest_sha256: SCHEDULE_SHA.into(),
        dataset_version: "retrieval-drill-assay-cycles-v2".into(),
        source_as_of: "synthetic".into(),
        split: Split::Train,
        rows: 32,
        completion_targets_per_epoch: 80,
    }
}

impl Drill {
    pub fn new(source: &Source, seed: u64) -> Result<Self> {
        let mut templates = Vec::with_capacity(32);
        for group in ["foundation", "retention-stress"] {
            for index in 0..16 {
                let prompt = source.row(&format!("{group}-{index:02}"))?.prompt.clone();
                ensure!(is_template(&prompt), "unexpected drill template");
                templates.push(prompt);
            }
        }
        Ok(Self { templates, seed })
    }

    /// Cursor at target n: assay cursor n % CYCLE, with passes (epochs) counted across cycles.
    pub fn cursor(&self, n: u64) -> Result<Cursor> {
        let local = assay_source::cursor(n % CYCLE)?;
        Ok(Cursor {
            schema: 2,
            data: identity(),
            row_index: local.row_index,
            example_id: format!("drill-{}-{}", n / CYCLE, local.example_id),
            completion_offset: local.completion_offset,
            epoch: n / 80,
            supervised_targets: n,
            answer_targets: n / 5 * 4 + (n % 5).min(4),
            citation_targets: 0,
            format_targets: n / 5,
        })
    }

    /// Prompt and completion at target n; pure in (seed, n).
    fn example(&self, n: u64) -> Result<(Vec<u8>, Vec<u8>)> {
        let local = assay_source::cursor(n % CYCLE)?;
        let group = if local.epoch < 16 { 0 } else { 16 };
        let template = &self.templates[group + local.row_index];
        let salt = tuning::hash(self.seed, &[n / CYCLE]);
        Ok(redeal(
            template,
            tuning::hash(salt, &[local.row_index as u64, local.epoch]),
        ))
    }
}

impl TrainingSource for Drill {
    fn validate(&self, cursor: &Cursor) -> Result<()> {
        ensure!(
            *cursor == self.cursor(cursor.supervised_targets)?,
            "drill cursor/order/mask mismatch"
        );
        Ok(())
    }

    fn pending_prefix(&self, cursor: &Cursor) -> Result<Vec<u8>> {
        self.validate(cursor)?;
        let (mut prompt, completion) = self.example(cursor.supervised_targets)?;
        prompt.extend_from_slice(&completion[..cursor.completion_offset]);
        Ok(prompt)
    }

    fn anchor_target(&self, cursor: &Cursor) -> Result<AnchorTarget> {
        self.validate(cursor)?;
        let (_, completion) = self.example(cursor.supervised_targets)?;
        Ok(AnchorTarget {
            byte: *completion
                .get(cursor.completion_offset)
                .context("drill target offset")?,
            kind: if cursor.completion_offset == 4 {
                TargetKind::Format
            } else {
                TargetKind::Answer
            },
        })
    }

    fn advance_anchor(&self, cursor: &mut Cursor) -> Result<()> {
        self.validate(cursor)?;
        *cursor = self.cursor(cursor.supervised_targets + 1)?;
        Ok(())
    }
}
