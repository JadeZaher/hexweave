//! Test-only RAII scratch directories under `work/validation`; see AGENTS.md §Authored numerical
//! fixtures.

use anyhow::Result;
use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// Per-process sequence, so two guards created within one clock tick never share a directory.
static NEXT: AtomicU64 = AtomicU64::new(0);

/// A unique `work/validation/<prefix>-<pid>-<nanos>-<seq>` directory, removed when dropped.
pub(super) struct ScratchDir {
    path: PathBuf,
}

impl ScratchDir {
    pub fn new(prefix: &str) -> Result<Self> {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos();
        let sequence = NEXT.fetch_add(1, Ordering::Relaxed);
        let name = format!("{prefix}-{}-{nanos}-{sequence}", std::process::id());
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("work/validation")
            .join(name);
        std::fs::create_dir_all(&path)?;
        Ok(Self { path })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Deref for ScratchDir {
    type Target = Path;

    fn deref(&self) -> &Path {
        &self.path
    }
}

impl AsRef<Path> for ScratchDir {
    fn as_ref(&self) -> &Path {
        &self.path
    }
}

impl Drop for ScratchDir {
    /// Best effort, also while unwinding: a failed removal must not turn into a second panic.
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}
