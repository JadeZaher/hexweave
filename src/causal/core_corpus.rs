//! Byte corpus for next-byte pretraining: the pinned train file is read by window from disk, so memory
//! stays flat however large it grows; the small held-out file is kept for bits per byte. See AGENTS.md §core.

use super::cursor::{AnchorTarget, Cursor};
use super::data::DataIdentity;
use super::data_contracts::{Split, TargetKind};
use super::execution::TrainingSource;
use super::storage_paths::digest;
use anyhow::{ensure, Context, Result};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

/// Virtual first byte of every stream, so each prefix is non-empty.
pub(super) const START: u8 = 0x02;
/// Story terminator written by `scripts/prepare_corpus.py`.
pub(super) const END: u8 = 0x03;
const HELDOUT_CAP: u64 = 64 << 20;
/// Bytes scored per frozen-weight span, and warm-up positions before it; see AGENTS.md §core.
pub(super) const SPAN_BYTES: usize = 128;
pub(super) const SPAN_WARMUP: usize = 2;
/// Scored anchors in the train-ahead probes; see AGENTS.md §core.
pub(super) const AHEAD_ANCHORS: u64 = 512;

#[derive(Deserialize)]
struct Entry {
    bytes: u64,
    sha256: String,
}

#[derive(Deserialize)]
struct Manifest {
    schema: u32,
    kind: String,
    files: BTreeMap<String, Entry>,
}

/// Pinned train stream (read by window), and in-memory held-out bytes.
pub(super) struct Corpus {
    train: File,
    length: u64,
    pub heldout: Vec<u8>,
    pub window: usize,
    identity: DataIdentity,
}

/// SHA-256 of a whole file, streamed in 1 MiB chunks.
fn file_digest(file: &mut File) -> Result<String> {
    let mut hasher = Sha256::new();
    let mut buffer = vec![0; 1 << 20];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            return Ok(format!("{:x}", hasher.finalize()));
        }
        hasher.update(&buffer[..read]);
    }
}

/// The model's view for predicting `text[t]`: the last `window` bytes of `[START] + text[..t]`.
pub(super) fn window_before(text: &[u8], t: usize, window: usize) -> Vec<u8> {
    match (t + 1).saturating_sub(window) {
        0 => [&[START][..], &text[..t]].concat(),
        first => text[first - 1..t].to_vec(),
    }
}

/// Train bytes `[start, end)`, read directly from an open file handle.
fn read_at(file: &File, start: u64, end: u64) -> Result<Vec<u8>> {
    let mut bytes = vec![0; usize::try_from(end - start)?];
    let mut handle = file;
    handle.seek(SeekFrom::Start(start))?;
    handle.read_exact(&mut bytes)?;
    Ok(bytes)
}

impl Corpus {
    pub fn open(root: &Path, manifest_sha256: &str, window: usize) -> Result<Self> {
        let raw = std::fs::read(root.join("manifest.json"))?;
        ensure!(
            digest(&raw) == manifest_sha256,
            "corpus manifest hash mismatch"
        );
        let manifest: Manifest = serde_json::from_slice(&raw)?;
        ensure!(
            manifest.schema == 1 && manifest.kind == "byte-corpus",
            "not a byte-corpus manifest"
        );
        let entry = |name: &str| {
            manifest
                .files
                .get(name)
                .with_context(|| format!("manifest lacks {name}"))
        };
        let (train_entry, heldout_entry) = (entry("train.txt")?, entry("heldout.txt")?);
        let mut train = File::open(root.join("train.txt"))?;
        ensure!(
            train.metadata()?.len() == train_entry.bytes
                && file_digest(&mut train)? == train_entry.sha256,
            "corpus train file does not match its manifest"
        );
        ensure!(
            heldout_entry.bytes <= HELDOUT_CAP,
            "held-out file too large to hold in memory"
        );
        let heldout = std::fs::read(root.join("heldout.txt"))?;
        ensure!(
            heldout.len() as u64 == heldout_entry.bytes && digest(&heldout) == heldout_entry.sha256,
            "corpus held-out file does not match its manifest"
        );
        ensure!(
            train_entry.bytes > 0 && !heldout.is_empty(),
            "empty corpus file"
        );
        ensure!(
            (32..=1023).contains(&window),
            "corpus window must be in 32..=1023"
        );
        let identity = DataIdentity {
            schema: "byte-corpus-v1".into(),
            manifest_sha256: manifest_sha256.into(),
            artifact_sha256: train_entry.sha256.clone(),
            source_pin_manifest_sha256: heldout_entry.sha256.clone(),
            dataset_version: format!("window-{window}"),
            source_as_of: "unknown".into(),
            split: Split::Train,
            rows: 1,
            completion_targets_per_epoch: train_entry.bytes,
        };
        Ok(Self {
            train,
            length: train_entry.bytes,
            heldout,
            window,
            identity,
        })
    }

    /// Cursor at target n: train byte n mod length, epochs counted by wraps.
    pub fn cursor(&self, n: u64) -> Result<Cursor> {
        Ok(Cursor {
            schema: 2,
            data: self.identity.clone(),
            row_index: 0,
            example_id: "corpus".into(),
            completion_offset: usize::try_from(n % self.length)?,
            epoch: n / self.length,
            supervised_targets: n,
            answer_targets: n,
            citation_targets: 0,
            format_targets: 0,
        })
    }

    /// Train bytes `[start, end)`.
    fn read(&self, start: u64, end: u64) -> Result<Vec<u8>> {
        read_at(&self.train, start, end)
    }

    /// `len` (<= length) train bytes from `start`, wrapping once at the file end like
    /// consecutive train targets do; see AGENTS.md §core.
    fn wrapped(&self, start: u64, len: u64) -> Result<Vec<u8>> {
        ensure!(
            start < self.length && len <= self.length,
            "ahead span exceeds corpus length"
        );
        let end = start + len;
        if end <= self.length {
            self.read(start, end)
        } else {
            Ok([
                self.read(start, self.length)?,
                self.read(0, end - self.length)?,
            ]
            .concat())
        }
    }

    /// The train-ahead probe span and its scored count, starting at the cursor's current
    /// position; see AGENTS.md §core.
    pub fn ahead_span(&self, start: u64, window: usize) -> Result<(Vec<u8>, usize)> {
        let scored = usize::try_from(AHEAD_ANCHORS.min(self.length))?;
        let left = start.saturating_sub(SPAN_WARMUP as u64 + window as u64);
        let before = self.read(left, start)?;
        let scored_bytes = self.wrapped(start, scored as u64)?;
        Ok(([before, scored_bytes].concat(), scored))
    }
}

impl TrainingSource for Corpus {
    fn validate(&self, cursor: &Cursor) -> Result<()> {
        ensure!(
            *cursor == self.cursor(cursor.supervised_targets)?,
            "corpus cursor/order mismatch"
        );
        Ok(())
    }

    fn pending_prefix(&self, cursor: &Cursor) -> Result<Vec<u8>> {
        self.validate(cursor)?;
        let t = cursor.completion_offset as u64;
        match (t + 1).saturating_sub(self.window as u64) {
            0 => Ok([vec![START], self.read(0, t)?].concat()),
            first => self.read(first - 1, t),
        }
    }

    fn anchor_target(&self, cursor: &Cursor) -> Result<AnchorTarget> {
        self.validate(cursor)?;
        let t = cursor.completion_offset as u64;
        Ok(AnchorTarget {
            byte: self.read(t, t + 1)?[0],
            kind: TargetKind::Answer,
        })
    }

    fn advance_anchor(&self, cursor: &mut Cursor) -> Result<()> {
        self.validate(cursor)?;
        *cursor = self.cursor(cursor.supervised_targets + 1)?;
        Ok(())
    }
}
