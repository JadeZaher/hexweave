use crate::causal::cursor::{AnchorTarget, Cursor};
use crate::causal::data::DataIdentity;
use crate::causal::data_contracts::{Split, TargetKind};
use crate::causal::execution::TrainingSource;
use anyhow::{ensure, Result};
use sha2::{Digest, Sha256};
use std::cell::{Cell, RefCell};

pub(super) const TRAIN: [&[u8]; 4] = [
    b"Evidence: plot A has pH 6.\nAnswer: ",
    b"Evidence: plot B has pH 8.\nAnswer: ",
    b"Evidence: plot C has pH 5.\nAnswer: ",
    b"Evidence: plot D has pH 7.\nAnswer: ",
];
pub(super) const HELD_OUT: [&[u8]; 4] = [
    b"Evidence: the measured pH of plot A is 6.\nAnswer: ",
    b"Evidence: the measured pH of plot B is 8.\nAnswer: ",
    b"Evidence: the measured pH of plot C is 5.\nAnswer: ",
    b"Evidence: the measured pH of plot D is 7.\nAnswer: ",
];
pub(super) const ANSWERS: [&[u8]; 4] = [b"6\n", b"8\n", b"5\n", b"7\n"];

pub(super) fn corpus_identity() -> String {
    let mut hash = Sha256::new();
    hash.update(b"anchor-comparison-v3-four-source-copy-cases-v1");
    for bytes in TRAIN.iter().chain(&HELD_OUT).chain(&ANSWERS) {
        hash.update((bytes.len() as u64).to_le_bytes());
        hash.update(bytes);
    }
    hash.update(b"A0,B0,C0,D0,A1,B1,C1,D1");
    format!("{:x}", hash.finalize())
}

pub(super) struct Source {
    pub target_reads: Cell<usize>,
    pub advances: Cell<usize>,
    pub sequence: RefCell<Vec<(usize, usize, u8)>>,
}

impl Source {
    pub fn new() -> Self {
        Self {
            target_reads: Cell::new(0),
            advances: Cell::new(0),
            sequence: RefCell::new(Vec::with_capacity(8)),
        }
    }

    pub fn cursor(&self, completed: u64) -> Cursor {
        let within = completed % 8;
        let row = within as usize % 4;
        let answer_targets = completed / 8 * 4 + within.min(4);
        Cursor {
            schema: 1,
            data: DataIdentity {
                schema: "fixture".into(),
                manifest_sha256: corpus_identity(),
                artifact_sha256: corpus_identity(),
                source_pin_manifest_sha256: corpus_identity(),
                dataset_version: "anchor-comparison-v3-fixture-v1".into(),
                source_as_of: "synthetic-fixed-fixture".into(),
                split: Split::Train,
                rows: 4,
                completion_targets_per_epoch: 8,
            },
            row_index: row,
            example_id: format!("fixture-{}", (b'A' + row as u8) as char),
            completion_offset: within as usize / 4,
            epoch: completed / 8,
            supervised_targets: completed,
            answer_targets,
            citation_targets: 0,
            format_targets: completed - answer_targets,
        }
    }
}

impl TrainingSource for Source {
    fn validate(&self, cursor: &Cursor) -> Result<()> {
        ensure!(
            cursor.supervised_targets <= 8 && *cursor == self.cursor(cursor.supervised_targets),
            "comparison cursor mismatch"
        );
        Ok(())
    }

    fn pending_prefix(&self, cursor: &Cursor) -> Result<Vec<u8>> {
        self.validate(cursor)?;
        let mut bytes = TRAIN[cursor.row_index].to_vec();
        bytes.extend_from_slice(&ANSWERS[cursor.row_index][..cursor.completion_offset]);
        Ok(bytes)
    }

    fn anchor_target(&self, cursor: &Cursor) -> Result<AnchorTarget> {
        self.validate(cursor)?;
        ensure!(self.target_reads.get() < 8, "comparison target-read cap");
        let byte = ANSWERS[cursor.row_index][cursor.completion_offset];
        self.target_reads.set(self.target_reads.get() + 1);
        self.sequence
            .borrow_mut()
            .push((cursor.row_index, cursor.completion_offset, byte));
        Ok(AnchorTarget {
            byte,
            kind: if cursor.completion_offset == 0 {
                TargetKind::Answer
            } else {
                TargetKind::Format
            },
        })
    }

    fn advance_anchor(&self, cursor: &mut Cursor) -> Result<()> {
        self.validate(cursor)?;
        ensure!(self.advances.get() < 8, "comparison cursor-advance cap");
        self.advances.set(self.advances.get() + 1);
        *cursor = self.cursor(cursor.supervised_targets + 1);
        self.validate(cursor)
    }
}
