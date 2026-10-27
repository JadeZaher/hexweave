use super::profile::{self, Bucket};
use anyhow::{ensure, Context, Result};
use sha2::{Digest, Sha256};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

pub(super) const FILE_CAP: u64 = 131_072;
pub(super) const FILE_COUNT_CAP: usize = 4096;
pub(super) const DISK_CAP: u64 = 2 * 1024 * 1024 * 1024;
static TEMP_ID: AtomicU64 = AtomicU64::new(0);

pub(super) fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

pub(super) fn hash_valid(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

pub(super) fn identifier(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 80
        && value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
}

fn regular(path: &Path, directory: bool) -> Result<()> {
    let metadata = fs::symlink_metadata(path)?;
    ensure!(
        !metadata.file_type().is_symlink(),
        "symlink in persistence path"
    );
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        ensure!(
            metadata.file_attributes() & 0x400 == 0,
            "reparse point in persistence path"
        );
    }
    ensure!(
        if directory {
            metadata.is_dir()
        } else {
            metadata.is_file()
        },
        "unexpected persistence entry type"
    );
    Ok(())
}

pub(super) fn attempt(workspace: &Path, directory: &Path) -> Result<PathBuf> {
    let workspace = workspace.canonicalize()?;
    regular(&workspace, true)?;
    let work = workspace.join("work");
    regular(&work, true)?;
    ensure!(
        directory.is_absolute(),
        "attempt directory must be absolute"
    );
    ensure!(
        !directory
            .components()
            .any(|c| matches!(c, Component::CurDir | Component::ParentDir)),
        "noncanonical attempt path"
    );
    for ancestor in directory.ancestors() {
        regular(ancestor, true)?;
    }
    let canonical = directory.canonicalize()?;
    ensure!(
        canonical != work && canonical.starts_with(&work),
        "attempt escaped owned workspace"
    );
    Ok(canonical)
}

pub(super) fn file(root: &Path, name: &str) -> Result<PathBuf> {
    ensure!(
        !name.is_empty()
            && name.len() <= 160
            && name
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"-_.".contains(&b))
            && name != "."
            && name != "..",
        "noncanonical persistence filename"
    );
    regular(root, true)?;
    let path = root.join(name);
    if path.try_exists()? {
        regular(&path, false)?;
    }
    Ok(path)
}

pub(super) fn read(root: &Path, name: &str, cap: u64) -> Result<Vec<u8>> {
    ensure!(cap <= FILE_CAP, "invalid persistence read bound");
    let path = file(root, name)?;
    let mut input = File::open(&path)?;
    let length = input.metadata()?.len();
    ensure!(length <= cap, "persistence file exceeds bound");
    let mut bytes = Vec::with_capacity(usize::try_from(length)?);
    (&mut input).take(cap + 1).read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() as u64 == length && bytes.len() as u64 <= cap,
        "persistence file changed or grew"
    );
    Ok(bytes)
}

pub(super) fn inventory(root: &Path) -> Result<Vec<(String, u64)>> {
    profile::time(Bucket::Scan, || scan(root))
}

fn scan(root: &Path) -> Result<Vec<(String, u64)>> {
    regular(root, true)?;
    let mut entries = Vec::new();
    for entry in fs::read_dir(root)? {
        ensure!(
            entries.len() < FILE_COUNT_CAP,
            "persistence file count exceeds bound"
        );
        let entry = entry?;
        let name = entry
            .file_name()
            .into_string()
            .map_err(|_| anyhow::anyhow!("non-UTF8 persistence name"))?;
        let path = file(root, &name)?;
        let length = fs::metadata(path)?.len();
        ensure!(length <= FILE_CAP, "oversized persistence entry");
        entries.push((name, length));
    }
    entries.sort_unstable();
    Ok(entries)
}

#[cfg(test)]
pub(super) fn usage(root: &Path) -> Result<u64> {
    inventory(root)?.iter().try_fold(0u64, |total, (_, bytes)| {
        total
            .checked_add(*bytes)
            .context("disk accounting overflow")
    })
}

/// Publish after one full directory scan for file-count and byte accounting.
pub(super) fn publish(root: &Path, name: &str, bytes: &[u8], budget: u64) -> Result<bool> {
    let entries = inventory(root)?;
    let used = entries.iter().try_fold(0u64, |total, (_, bytes)| {
        total
            .checked_add(*bytes)
            .context("disk accounting overflow")
    })?;
    publish_with(root, name, bytes, budget, used, entries.len())
}

/// Publish by same-directory hard link, which never replaces an existing name.
/// `used`/`files` are the caller's exact current directory bytes and entry count.
pub(super) fn publish_with(
    root: &Path,
    name: &str,
    bytes: &[u8],
    budget: u64,
    used: u64,
    files: usize,
) -> Result<bool> {
    ensure!(
        !bytes.is_empty() && bytes.len() as u64 <= FILE_CAP && budget <= DISK_CAP,
        "invalid write bounds"
    );
    let destination = file(root, name)?;
    if destination.try_exists()? {
        ensure!(
            read(root, name, FILE_CAP)? == bytes,
            "immutable persistence collision"
        );
        return Ok(false);
    }
    ensure!(
        files + 2 <= FILE_COUNT_CAP,
        "persistence file count budget exhausted"
    );
    let peak = used
        .checked_add(2 * bytes.len() as u64)
        .context("disk budget overflow")?;
    ensure!(peak <= budget, "owned persistence disk budget exhausted");
    let temporary = format!(
        ".pending-{}-{}",
        std::process::id(),
        TEMP_ID.fetch_add(1, Ordering::Relaxed)
    );
    let path = file(root, &temporary)?;
    profile::time(Bucket::Publish, || -> Result<bool> {
        let result = (|| -> Result<()> {
            let mut output = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)?;
            output.write_all(bytes)?;
            profile::time(Bucket::Sync, || output.sync_all())?;
            drop(output);
            fs::hard_link(&path, &destination)
                .context("atomic no-replace publication requires same-volume hard links")?;
            Ok(())
        })();
        if path.try_exists()? {
            fs::remove_file(file(root, &temporary)?)?;
        }
        result?;
        Ok(true)
    })
}

pub(super) fn remove(root: &Path, name: &str, expected: &str) -> Result<u64> {
    let bytes = read(root, name, FILE_CAP)?;
    ensure!(
        digest(&bytes) == expected,
        "refuse deletion of changed persistence payload"
    );
    fs::remove_file(file(root, name)?)?;
    Ok(bytes.len() as u64)
}
