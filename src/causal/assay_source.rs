use super::assay_state::{ANCHORS, MANIFEST_SHA, ROWS_SHA, SCHEDULE_SHA};
use super::cursor::{AnchorTarget, Cursor};
use super::data::DataIdentity;
use super::data_contracts::{Split, TargetKind};
use super::execution::TrainingSource;
use anyhow::{ensure, Context, Result};
use std::path::Path;

pub(super) struct Row {
    pub id: String,
    pub group: String,
    pub prompt: Vec<u8>,
    pub completion: Vec<u8>,
}

pub(super) struct Source {
    pub rows: Vec<Row>,
}

pub(super) fn identity() -> DataIdentity {
    DataIdentity {
        schema: "retained-text-source-v1".into(),
        manifest_sha256: MANIFEST_SHA.into(),
        artifact_sha256: ROWS_SHA.into(),
        source_pin_manifest_sha256: SCHEDULE_SHA.into(),
        dataset_version: "retained-text-fixed64-v1".into(),
        source_as_of: "synthetic".into(),
        split: Split::Train,
        rows: 32,
        completion_targets_per_epoch: 80,
    }
}

pub(super) fn cursor(n: u64) -> Result<Cursor> {
    ensure!(n <= ANCHORS, "assay exposure outside fixed course");
    let pass = n / 80;
    let position = (n % 80 / 5) as usize;
    let row = if pass.is_multiple_of(2) {
        position
    } else {
        15 - position
    };
    let row = (row + pass as usize % 16) % 16;
    let group = if pass < 16 {
        "foundation"
    } else {
        "retention-stress"
    };
    Ok(Cursor {
        schema: 2,
        data: identity(),
        row_index: if n == ANCHORS { 0 } else { row },
        example_id: if n == ANCHORS {
            "complete".into()
        } else {
            format!("{group}-{row:02}")
        },
        completion_offset: (n % 5) as usize,
        epoch: pass,
        supervised_targets: n,
        answer_targets: n / 5 * 4 + (n % 5).min(4),
        citation_targets: 0,
        format_targets: n / 5,
    })
}

pub(super) fn validate_cursor(value: &Cursor) -> Result<()> {
    ensure!(
        *value == cursor(value.supervised_targets)?,
        "assay cursor/order/mask mismatch"
    );
    Ok(())
}

impl Source {
    pub fn open(root: &Path, expected: &str) -> Result<Self> {
        ensure!(expected == MANIFEST_SHA, "assay manifest pin");
        let root = root.canonicalize()?;
        for (name, sha, cap) in [
            ("manifest.json", MANIFEST_SHA, 16_384),
            ("rows.jsonl", ROWS_SHA, 65_536),
            ("schedule.json", SCHEDULE_SHA, 32_768),
        ] {
            let path = super::data_io::contained(&root, name)?;
            ensure!(
                super::child_io::hash_file(&path, cap)? == sha,
                "assay file pin: {name}"
            );
        }
        let bytes = super::child_io::read(&root.join("rows.jsonl"), 65_536)?;
        ensure!(
            super::storage_paths::digest(&bytes) == ROWS_SHA,
            "assay rows changed during open"
        );
        let mut rows = Vec::with_capacity(64);
        for line in bytes
            .split(|byte| *byte == b'\n')
            .filter(|line| !line.is_empty())
        {
            let value: serde_json::Value = serde_json::from_slice(line)?;
            let string = |key: &str| -> Result<String> {
                Ok(value[key].as_str().context("assay row field")?.into())
            };
            rows.push(Row {
                id: string("id")?,
                group: string("group")?,
                prompt: string("prompt")?.into_bytes(),
                completion: string("completion")?.into_bytes(),
            });
        }
        ensure!(rows.len() == 64, "assay row count");
        for row in &rows {
            ensure!(
                row.prompt.len() <= 85 && row.completion.len() == 5 && row.completion[4] == b'\n',
                "assay row dimensions"
            );
        }
        Ok(Self { rows })
    }

    pub fn row(&self, id: &str) -> Result<&Row> {
        self.rows
            .iter()
            .find(|row| row.id == id)
            .context("unknown assay row")
    }

    pub fn observations(&self) -> Result<Vec<&Row>> {
        let mut rows = Vec::with_capacity(48);
        for group in ["foundation", "paraphrase", "counterfactual"] {
            for index in 0..16 {
                rows.push(self.row(&format!("{group}-{index:02}"))?);
            }
        }
        Ok(rows)
    }
}

impl TrainingSource for Source {
    fn validate(&self, value: &Cursor) -> Result<()> {
        validate_cursor(value)
    }

    fn pending_prefix(&self, value: &Cursor) -> Result<Vec<u8>> {
        self.validate(value)?;
        ensure!(
            value.supervised_targets < ANCHORS,
            "no target after assay course"
        );
        let row = self.row(&value.example_id)?;
        ensure!(
            matches!(row.group.as_str(), "foundation" | "retention-stress"),
            "probe training forbidden"
        );
        let mut prefix = row.prompt.clone();
        prefix.extend_from_slice(&row.completion[..value.completion_offset]);
        Ok(prefix)
    }

    fn anchor_target(&self, value: &Cursor) -> Result<AnchorTarget> {
        self.validate(value)?;
        ensure!(
            value.supervised_targets < ANCHORS,
            "terminal assay target access"
        );
        Ok(AnchorTarget {
            byte: self.row(&value.example_id)?.completion[value.completion_offset],
            kind: if value.completion_offset == 4 {
                TargetKind::Format
            } else {
                TargetKind::Answer
            },
        })
    }

    fn advance_anchor(&self, value: &mut Cursor) -> Result<()> {
        self.validate(value)?;
        *value = cursor(value.supervised_targets + 1)?;
        Ok(())
    }
}
