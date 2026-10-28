//! SHA-256 digests and pin syntax for checkpoints and sources; test-only safe file paths.

#[cfg(test)]
use anyhow::{ensure, Result};
use sha2::{Digest, Sha256};
#[cfg(test)]
use std::fs;
#[cfg(test)]
use std::path::{Path, PathBuf};

pub(super) const FILE_CAP: u64 = 131_072;

pub(super) fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

pub(super) fn hash_valid(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

#[cfg(test)]
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

#[cfg(test)]
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
