//! Amortized source verification: full hash once, then a metadata guard per span read; see AGENTS.md §amortized.

use super::data_contracts::Source;
use super::data_io::{contained, source_span};
use anyhow::{ensure, Context, Result};
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::fs::{File, Metadata};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

/// Canonical path plus file metadata observed around a successful full verification.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Stamp {
    path: PathBuf,
    bytes: u64,
    modified: SystemTime,
    created: Option<SystemTime>,
}

impl Stamp {
    fn new(path: PathBuf, metadata: &Metadata) -> Result<Self> {
        Ok(Self {
            path,
            bytes: metadata.len(),
            modified: metadata.modified()?,
            created: metadata.created().ok(),
        })
    }

    fn matches(&self, metadata: &Metadata) -> bool {
        metadata.len() == self.bytes
            && metadata.modified().ok() == Some(self.modified)
            && metadata.created().ok() == self.created
    }
}

/// Per-source stamps of fully hash-verified snapshots, keyed by source ID.
#[derive(Default)]
pub(super) struct Verified {
    stamps: RefCell<BTreeMap<String, Stamp>>,
}

impl Verified {
    /// Pinned span: direct read while the stamp guard holds, else a full re-verification.
    pub fn span(&self, root: &Path, source: &Source, start: u64, end: u64) -> Result<Vec<u8>> {
        ensure!(
            start < end && end <= source.bytes && end - start <= 608,
            "invalid source span"
        );
        let path = contained(root, &source.snapshot_path)?;
        let stamp = self.stamps.borrow().get(&source.source_id).cloned();
        if let Some(stamp) = stamp.filter(|stamp| stamp.path == path) {
            let read = guarded(&stamp, start, end)?;
            if let Some(span) = read {
                return Ok(span);
            }
        }
        self.verify(root, source, &path, start, end)
    }

    /// Re-hash every previously verified source in full (`Dataset::open`, after its scan).
    pub fn verify_all(&self, root: &Path, sources: &BTreeMap<String, Source>) -> Result<()> {
        let ids: Vec<String> = self.stamps.borrow().keys().cloned().collect();
        for id in ids {
            let source = sources.get(&id).context("unknown verified source")?;
            let path = contained(root, &source.snapshot_path)?;
            self.verify(root, source, &path, 0, 1)?;
        }
        Ok(())
    }

    fn verify(
        &self,
        root: &Path,
        source: &Source,
        path: &Path,
        start: u64,
        end: u64,
    ) -> Result<Vec<u8>> {
        self.stamps.borrow_mut().remove(&source.source_id);
        let before = Stamp::new(path.to_path_buf(), &std::fs::metadata(path)?)?;
        let span = source_span(root, source, start, end)?;
        let after = Stamp::new(path.to_path_buf(), &std::fs::metadata(path)?)?;
        ensure!(
            before == after && after.bytes == source.bytes,
            "source changed during verification"
        );
        self.stamps
            .borrow_mut()
            .insert(source.source_id.clone(), after);
        Ok(span)
    }
}

/// Bounded span read if the stamp still matches before and after; `None` trips re-verification.
fn guarded(stamp: &Stamp, start: u64, end: u64) -> Result<Option<Vec<u8>>> {
    let mut file = File::open(&stamp.path)?;
    if !stamp.matches(&file.metadata()?) {
        return Ok(None);
    }
    file.seek(SeekFrom::Start(start))?;
    let mut span = vec![0; usize::try_from(end - start)?];
    file.read_exact(&mut span)?;
    Ok(stamp.matches(&file.metadata()?).then_some(span))
}
