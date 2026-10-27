use super::cursor::{AnchorTarget, Cursor};
use super::data_contracts::*;
use super::data_guard::Verified;
use super::data_io::*;
use super::execution::TrainingSource;
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct DataIdentity {
    pub schema: String,
    pub manifest_sha256: String,
    pub artifact_sha256: String,
    pub source_pin_manifest_sha256: String,
    pub dataset_version: String,
    pub source_as_of: String,
    pub split: Split,
    pub rows: usize,
    pub completion_targets_per_epoch: u64,
}

#[derive(Clone, Debug)]
pub(super) struct RowMeta {
    pub id: String,
    pub offset: u64,
    pub length: usize,
    pub sha256: String,
    pub completion_bytes: usize,
    pub masks: Masks,
    pub preceding: [u64; 3],
}

#[derive(Debug)]
pub(super) struct Record {
    pub id: String,
    pub prompt: Vec<u8>,
    pub completion: Vec<u8>,
    pub masks: Masks,
    pub evidence_sha256: String,
}

pub(super) struct Dataset {
    root: PathBuf,
    sources: BTreeMap<String, Source>,
    identity: DataIdentity,
    rows: Vec<RowMeta>,
    epoch_counts: [u64; 3],
    verified: Verified,
}

impl Dataset {
    pub fn open(root: &Path, expected_manifest_sha256: &str, split: Split) -> Result<Self> {
        ensure!(
            valid_hash(expected_manifest_sha256),
            "external manifest SHA256 required"
        );
        let root = root.canonicalize()?;
        ensure!(root.is_dir(), "dataset root is not a directory");
        let bytes = bounded(&contained(&root, "manifest.json")?, MANIFEST_CAP)?;
        ensure!(
            sha256(&bytes) == expected_manifest_sha256,
            "manifest pin mismatch"
        );
        let manifest: Manifest = serde_json::from_slice(&bytes)?;
        drop(bytes);
        validate_manifest(&root, &manifest)?;
        let sources = manifest
            .sources
            .iter()
            .map(|s| (s.source_id.clone(), s.clone()))
            .collect();
        let selected = &manifest.artifacts[split.file()];
        let identity = DataIdentity {
            schema: super::data_provenance::identity_schema(&manifest)?.into(),
            manifest_sha256: expected_manifest_sha256.into(),
            artifact_sha256: selected.sha256.clone(),
            source_pin_manifest_sha256: manifest.source_pin_manifest_sha256.clone(),
            dataset_version: manifest.dataset_version.clone(),
            source_as_of: manifest.source_as_of.clone(),
            split,
            rows: selected.rows,
            completion_targets_per_epoch: selected.total_supervised_completion_bytes,
        };
        let mut dataset = Self {
            root,
            sources,
            identity,
            rows: Vec::new(),
            epoch_counts: [0; 3],
            verified: Verified::default(),
        };
        let mut families = BTreeSet::new();
        for partition in [Split::Train, Split::Development] {
            let (rows, counts) = dataset.scan(
                partition,
                &manifest.artifacts[partition.file()],
                &mut families,
            )?;
            if partition == split {
                dataset.rows = rows;
                dataset.epoch_counts = counts;
            }
        }
        dataset.verify_sources()?;
        Ok(dataset)
    }

    pub fn identity(&self) -> &DataIdentity {
        &self.identity
    }
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn meta(&self, index: usize) -> Result<&RowMeta> {
        self.rows
            .get(index)
            .context("row index outside pinned dataset")
    }

    pub fn epoch_counts(&self) -> [u64; 3] {
        self.epoch_counts
    }

    /// Full re-hash of every source verified so far.
    pub fn verify_sources(&self) -> Result<()> {
        self.verified.verify_all(&self.root, &self.sources)
    }

    pub fn record(&self, index: usize) -> Result<Record> {
        let meta = self
            .rows
            .get(index)
            .context("row index outside pinned dataset")?;
        let path = contained(&self.root, self.identity.split.file())?;
        let row = read_record(&path, meta.offset, meta.length, &meta.sha256)?;
        let source = self
            .sources
            .get(&row.source_id)
            .context("unknown source ID")?;
        let masks = validate_row(&row, source, self.identity.split, index)?;
        ensure!(
            row.id == meta.id
                && row.completion.len() == meta.completion_bytes
                && masks == meta.masks,
            "row ledger changed"
        );
        let span = self.verified.span(
            &self.root,
            source,
            row.evidence_start_byte,
            row.evidence_end_byte,
        )?;
        ensure!(
            span == row.evidence.as_bytes(),
            "evidence differs from pinned source span"
        );
        Ok(Record {
            id: row.id,
            prompt: row.prompt.into_bytes(),
            completion: row.completion.into_bytes(),
            masks,
            evidence_sha256: sha256(&span),
        })
    }

    fn scan(
        &self,
        split: Split,
        artifact: &Artifact,
        families: &mut BTreeSet<String>,
    ) -> Result<(Vec<RowMeta>, [u64; 3])> {
        ensure!(
            (1..=ROW_CAP).contains(&artifact.rows),
            "invalid dataset row bound"
        );
        let path = contained(&self.root, split.file())?;
        hash_file(&path, artifact.bytes, &artifact.sha256, FILE_CAP)?;
        let mut reader = BufReader::with_capacity(STREAM_CAP, File::open(&path)?);
        let mut line = Vec::with_capacity(RECORD_CAP + 1);
        let mut rows = Vec::with_capacity(artifact.rows);
        let (mut offset, mut counts, mut answerable) = (0u64, [0u64; 3], 0usize);
        let mut spans = BTreeSet::new();
        let mut maximum = Lengths::default();
        loop {
            line.clear();
            let n = (&mut reader)
                .take(RECORD_CAP as u64 + 1)
                .read_until(b'\n', &mut line)?;
            if n == 0 {
                break;
            }
            ensure!(
                n <= RECORD_CAP && line.last() == Some(&b'\n'),
                "oversized or unterminated JSONL record"
            );
            ensure!(
                rows.len() < artifact.rows && families.len() < ROW_CAP,
                "dataset ledger exceeds row cap"
            );
            let row: Row = serde_json::from_slice(&line)?;
            let source = self
                .sources
                .get(&row.source_id)
                .context("unknown source ID")?;
            let masks = validate_row(&row, source, split, rows.len())?;
            ensure!(
                families.insert(row.fact_family_id.clone()),
                "repeated fact family within/across splits"
            );
            let span = self.verified.span(
                &self.root,
                source,
                row.evidence_start_byte,
                row.evidence_end_byte,
            )?;
            ensure!(
                span == row.evidence.as_bytes(),
                "evidence differs from pinned source span"
            );
            spans.insert((
                row.source_id.clone(),
                row.evidence_start_byte,
                row.evidence_end_byte,
            ));
            rows.push(RowMeta {
                id: row.id,
                offset,
                length: n,
                sha256: sha256(&line),
                completion_bytes: row.completion.len(),
                masks,
                preceding: counts,
            });
            for (count, increment) in counts.iter_mut().zip(masks.counts(row.completion.len())) {
                *count = count
                    .checked_add(increment)
                    .context("completion count overflow")?;
            }
            offset = offset
                .checked_add(n as u64)
                .context("JSONL offset overflow")?;
            ensure!(offset <= artifact.bytes, "JSONL length changed during scan");
            answerable += usize::from(row.answerable);
            maximum.include(row.utf8_bytes);
        }
        let targets = counts
            .iter()
            .try_fold(0u64, |a, b| a.checked_add(*b).context("target overflow"))?;
        ensure!(
            offset == artifact.bytes
                && rows.len() == artifact.rows
                && answerable == artifact.answerable
                && rows.len() - answerable == artifact.abstention
                && artifact.fact_families == rows.len()
                && spans.len() == artifact.unique_evidence_spans
                && maximum == artifact.max_utf8_bytes
                && targets == artifact.total_supervised_completion_bytes,
            "artifact summary mismatch"
        );
        hash_file(&path, artifact.bytes, &artifact.sha256, FILE_CAP)?;
        Ok((rows, counts))
    }
}

impl TrainingSource for Dataset {
    fn validate(&self, cursor: &Cursor) -> Result<()> {
        cursor.validate(self)
    }
    fn pending_prefix(&self, cursor: &Cursor) -> Result<Vec<u8>> {
        cursor.pending_prefix(self)
    }
    fn anchor_target(&self, cursor: &Cursor) -> Result<AnchorTarget> {
        cursor.anchor_target(self)
    }
    fn advance_anchor(&self, cursor: &mut Cursor) -> Result<()> {
        cursor.advance_anchor(self)
    }
    fn verify_boundary(&self) -> Result<()> {
        self.verify_sources()
    }
}
