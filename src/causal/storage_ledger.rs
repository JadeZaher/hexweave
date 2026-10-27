//! Running persistence byte/file ledger with boundary full-scan cross-check; see AGENTS.md §ledger.

use super::{add, DiskStore};
use crate::causal::storage_paths as paths;
use anyhow::{bail, ensure, Context, Result};

/// Exact owned-directory bytes and entry count, as a full inventory would report them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::causal) struct Ledger {
    pub bytes: u64,
    pub files: usize,
}

impl Ledger {
    /// Ledger from one full directory inventory.
    pub fn scan(root: &std::path::Path) -> Result<Self> {
        let entries = paths::inventory(root)?;
        let bytes = entries.iter().try_fold(0u64, |total, (_, bytes)| {
            total
                .checked_add(*bytes)
                .context("disk accounting overflow")
        })?;
        Ok(Self {
            bytes,
            files: entries.len(),
        })
    }

    fn published(self, bytes: u64) -> Result<Self> {
        Ok(Self {
            bytes: self.bytes.checked_add(bytes).context("ledger overflow")?,
            files: self.files.checked_add(1).context("ledger overflow")?,
        })
    }
}

impl DiskStore {
    /// Current ledger, rebuilt by a full scan only when absent or invalidated.
    fn ledger(&mut self) -> Result<Ledger> {
        if let Some(ledger) = self.ledger {
            return Ok(ledger);
        }
        let ledger = Ledger::scan(&self.directory)?;
        self.ledger = Some(ledger);
        Ok(ledger)
    }

    /// Full rescan cross-check; any ledger/filesystem mismatch poisons the store.
    pub(in crate::causal) fn reconcile(&mut self) -> Result<()> {
        let actual = match Ledger::scan(&self.directory) {
            Ok(actual) => actual,
            Err(error) => {
                self.poisoned = true;
                self.ledger = None;
                return Err(error);
            }
        };
        if let Some(expected) = self.ledger {
            if expected != actual {
                self.poisoned = true;
                self.ledger = None;
                bail!("persistence ledger/filesystem mismatch");
            }
        }
        self.ledger = Some(actual);
        self.record_usage(actual.bytes)
    }

    /// Drop the running ledger so the next write rebuilds it from a full scan.
    pub(in crate::causal) fn invalidate_ledger(&mut self) {
        self.ledger = None;
    }

    /// Debit one hash-verified removed file of `bytes` from the running ledger.
    pub(in crate::causal) fn removed(&mut self, bytes: u64) -> Result<()> {
        let ledger = self.ledger()?;
        let next = ledger
            .bytes
            .checked_sub(bytes)
            .zip(ledger.files.checked_sub(1))
            .map(|(bytes, files)| Ledger { bytes, files });
        let Some(next) = next else {
            self.poisoned = true;
            self.ledger = None;
            bail!("persistence ledger underflow");
        };
        self.ledger = Some(next);
        Ok(())
    }

    fn record_usage(&mut self, used: u64) -> Result<()> {
        ensure!(used <= self.disk_budget_bytes, "owned disk budget exceeded");
        self.io.current_disk_bytes = used;
        self.io.peak_disk_bytes = self.io.peak_disk_bytes.max(used);
        Ok(())
    }

    /// Ledgered immutable write: same budget/count checks and counters as a rescan.
    pub(super) fn write_ledgered(&mut self, name: &str, bytes: &[u8]) -> Result<()> {
        let before = self.ledger()?;
        let length = bytes.len() as u64;
        let possible_peak = before
            .bytes
            .checked_add(2 * length)
            .context("disk peak overflow")?;
        let published = paths::publish_with(
            &self.directory,
            name,
            bytes,
            self.disk_budget_bytes,
            before.bytes,
            before.files,
        )?;
        let after = if published {
            add(&mut self.io.saves, 1)?;
            add(&mut self.io.bytes_written, length)?;
            self.io.peak_disk_bytes = self.io.peak_disk_bytes.max(possible_peak);
            before.published(length)?
        } else {
            before
        };
        self.ledger = Some(after);
        self.record_usage(after.bytes)
    }
}
