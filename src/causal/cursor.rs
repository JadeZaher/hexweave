use super::data::{DataIdentity, Dataset};
use super::data_contracts::TargetKind;
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Cursor {
    pub schema: u32,
    pub data: DataIdentity,
    pub row_index: usize,
    pub example_id: String,
    pub completion_offset: usize,
    pub epoch: u64,
    pub supervised_targets: u64,
    pub answer_targets: u64,
    pub citation_targets: u64,
    pub format_targets: u64,
}

#[cfg(test)]
mod tests {
    use super::super::data_contracts::Split;
    use super::super::data_tests::Fixture;
    use super::*;
    use serde_json::json;

    fn at(dataset: &Dataset, index: usize, offset: usize) -> Cursor {
        let mut cursor = Cursor::start(dataset).unwrap();
        let row = dataset.meta(index).unwrap();
        cursor.row_index = index;
        cursor.example_id = row.id.clone();
        cursor.completion_offset = offset;
        let partial = row.masks.counts(offset);
        let counts: Vec<_> = (0..3).map(|i| row.preceding[i] + partial[i]).collect();
        cursor.answer_targets = counts[0];
        cursor.citation_targets = counts[1];
        cursor.format_targets = counts[2];
        cursor.supervised_targets = counts.iter().sum();
        cursor.validate(dataset).unwrap();
        cursor
    }

    #[test]
    fn local_prefix_and_serialized_continuation_are_exact() {
        let fixture = Fixture::new();
        let dataset = fixture.open().unwrap();
        let row = dataset.record(0).unwrap();
        let mut cursor = Cursor::start(&dataset).unwrap();
        for offset in 0..16 {
            let before = cursor.clone();
            let mut expected = row.prompt.clone();
            expected.extend_from_slice(&row.completion[..offset]);
            for _ in 0..3 {
                assert_eq!(cursor.pending_prefix(&dataset).unwrap(), expected);
            }
            assert_eq!(cursor, before);
            assert_eq!(
                cursor.anchor_target(&dataset).unwrap().byte,
                row.completion[offset]
            );
            let mut resumed: Cursor =
                serde_json::from_slice(&serde_json::to_vec(&cursor).unwrap()).unwrap();
            resumed.validate(&dataset).unwrap();
            cursor.advance_anchor(&dataset).unwrap();
            resumed.advance_anchor(&dataset).unwrap();
            assert_eq!(cursor, resumed);
            assert_eq!(cursor.supervised_targets, offset as u64 + 1);
        }
    }

    #[test]
    fn byte_prefix_can_end_inside_utf8_and_epoch_wrap_scores_newline() {
        let mut fixture = Fixture::new();
        fixture.row(Split::Train, 0, |row| {
            row["answer"] = json!(format!("{}é", row["answer"].as_str().unwrap()));
            let completion = format!(
                "{}\n",
                json!({"answer":row["answer"],"citation":row["citation"]})
            );
            row["utf8_bytes"]["completion"] = json!(completion.len());
            row["utf8_bytes"]["sequence"] =
                json!(row["prompt"].as_str().unwrap().len() + completion.len());
            row["completion"] = json!(completion);
        });
        let dataset = fixture.open().unwrap();
        let row = dataset.record(0).unwrap();
        let offset = row
            .completion
            .windows(2)
            .position(|bytes| bytes == "é".as_bytes())
            .unwrap()
            + 1;
        let cursor = at(&dataset, 0, offset);
        let prefix = cursor.pending_prefix(&dataset).unwrap();
        assert!(std::str::from_utf8(&prefix).is_err());
        assert_eq!(
            cursor.anchor_target(&dataset).unwrap(),
            AnchorTarget {
                byte: 0xa9,
                kind: TargetKind::Answer
            }
        );
        let last = dataset.len() - 1;
        let mut cursor = at(
            &dataset,
            last,
            dataset.meta(last).unwrap().completion_bytes - 1,
        );
        assert_eq!(
            cursor.anchor_target(&dataset).unwrap(),
            AnchorTarget {
                byte: b'\n',
                kind: TargetKind::Format
            }
        );
        cursor.advance_anchor(&dataset).unwrap();
        assert_eq!(
            (cursor.epoch, cursor.row_index, cursor.completion_offset),
            (1, 0, 0)
        );
        assert_eq!(
            cursor.supervised_targets,
            dataset.identity().completion_targets_per_epoch
        );
    }

    #[test]
    fn resume_rejects_inconsistent_identity_position_and_counts() {
        let fixture = Fixture::new();
        let dataset = fixture.open().unwrap();
        let good = Cursor::start(&dataset).unwrap();
        let changes: [fn(&mut Cursor); 8] = [
            |c| c.data.split = Split::Development,
            |c| c.data.artifact_sha256 = "0".repeat(64),
            |c| c.row_index = usize::MAX,
            |c| c.example_id = "wrong".into(),
            |c| c.completion_offset = 192,
            |c| c.supervised_targets = 1,
            |c| c.answer_targets = 1,
            |c| c.epoch = u64::MAX,
        ];
        for change in changes {
            let mut cursor = good.clone();
            change(&mut cursor);
            assert!(cursor.validate(&dataset).is_err());
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct AnchorTarget {
    pub byte: u8,
    pub kind: TargetKind,
}

impl Cursor {
    pub fn start(dataset: &Dataset) -> Result<Self> {
        let first = dataset.meta(0)?;
        let cursor = Self {
            schema: 1,
            data: dataset.identity().clone(),
            row_index: 0,
            example_id: first.id.clone(),
            completion_offset: 0,
            epoch: 0,
            supervised_targets: 0,
            answer_targets: 0,
            citation_targets: 0,
            format_targets: 0,
        };
        cursor.validate(dataset)?;
        Ok(cursor)
    }

    pub fn validate(&self, dataset: &Dataset) -> Result<()> {
        ensure!(
            self.schema == 1 && &self.data == dataset.identity(),
            "cursor data/schema mismatch"
        );
        let row = dataset.meta(self.row_index)?;
        ensure!(
            self.example_id == row.id && self.completion_offset < row.completion_bytes,
            "cursor example/offset mismatch"
        );
        let partial = row.masks.counts(self.completion_offset);
        let mut expected = [0u64; 3];
        for i in 0..3 {
            expected[i] = dataset.epoch_counts()[i]
                .checked_mul(self.epoch)
                .and_then(|n| n.checked_add(row.preceding[i]))
                .and_then(|n| n.checked_add(partial[i]))
                .context("cursor exposure overflow")?;
        }
        let total = expected.iter().try_fold(0u64, |n, v| {
            n.checked_add(*v).context("cursor total overflow")
        })?;
        ensure!(
            expected
                == [
                    self.answer_targets,
                    self.citation_targets,
                    self.format_targets
                ]
                && self.supervised_targets == total,
            "cursor cumulative target/mask mismatch"
        );
        Ok(())
    }

    // Local opportunities receive only the pending causal prefix; see data-API.md.
    pub fn pending_prefix(&self, dataset: &Dataset) -> Result<Vec<u8>> {
        self.validate(dataset)?;
        let row = dataset.record(self.row_index)?;
        let length = row
            .prompt
            .len()
            .checked_add(self.completion_offset)
            .context("prefix length overflow")?;
        ensure!(
            (1..1024).contains(&length),
            "QA prefix does not fit context1024"
        );
        let mut prefix = Vec::with_capacity(length);
        prefix.extend_from_slice(&row.prompt);
        prefix.extend_from_slice(&row.completion[..self.completion_offset]);
        Ok(prefix)
    }

    pub fn anchor_target(&self, dataset: &Dataset) -> Result<AnchorTarget> {
        self.validate(dataset)?;
        let row = dataset.record(self.row_index)?;
        Ok(AnchorTarget {
            byte: row.completion[self.completion_offset],
            kind: row.masks.kind(self.completion_offset),
        })
    }

    // Call only after the complete anchor's model/history/counter transaction.
    pub fn advance_anchor(&mut self, dataset: &Dataset) -> Result<()> {
        self.validate(dataset)?;
        let row = dataset.meta(self.row_index)?;
        let mut next = self.clone();
        next.supervised_targets = next
            .supervised_targets
            .checked_add(1)
            .context("target overflow")?;
        let counter = match row.masks.kind(self.completion_offset) {
            TargetKind::Answer => &mut next.answer_targets,
            TargetKind::Citation => &mut next.citation_targets,
            TargetKind::Format => &mut next.format_targets,
        };
        *counter = counter.checked_add(1).context("mask count overflow")?;
        next.completion_offset = next
            .completion_offset
            .checked_add(1)
            .context("completion offset overflow")?;
        if next.completion_offset == row.completion_bytes {
            next.completion_offset = 0;
            next.row_index = next
                .row_index
                .checked_add(1)
                .context("row index overflow")?;
            if next.row_index == dataset.len() {
                next.row_index = 0;
                next.epoch = next.epoch.checked_add(1).context("epoch overflow")?;
            }
            next.example_id = dataset.meta(next.row_index)?.id.clone();
        }
        next.validate(dataset)?;
        *self = next;
        Ok(())
    }
}
