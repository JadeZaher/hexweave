//! Bounded regular-file reads and streamed SHA-256 pins; JSON publication is test-only.

#[cfg(test)]
use super::storage_paths;
use anyhow::{ensure, Context, Result};
use sha2::{Digest, Sha256};
#[cfg(test)]
use std::fs::OpenOptions;
use std::fs::{self, File};
use std::io::Read;
#[cfg(test)]
use std::io::{self, Write};
use std::path::Path;

pub(super) const RESULT_CAP: u64 = 131_072;

#[cfg(test)]
pub(super) fn json(value: &impl serde::Serialize) -> Result<Vec<u8>> {
    struct Limited(Vec<u8>);
    impl Write for Limited {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            if self
                .0
                .len()
                .checked_add(bytes.len())
                .is_none_or(|length| length > RESULT_CAP as usize)
            {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "child JSON exceeds bound",
                ));
            }
            self.0.extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    let mut output = Limited(Vec::with_capacity(RESULT_CAP as usize));
    serde_json::to_writer(&mut output, value)?;
    Ok(output.0)
}

pub(super) fn read(path: &Path, cap: u64) -> Result<Vec<u8>> {
    ensure!(cap <= RESULT_CAP, "invalid child file cap");
    regular(path)?;
    let mut file = File::open(path)?;
    let length = file.metadata()?.len();
    ensure!(length <= cap, "child file exceeds bound");
    let mut bytes = Vec::with_capacity(usize::try_from(length)?);
    Read::by_ref(&mut file)
        .take(cap + 1)
        .read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() as u64 == length && length <= cap,
        "child file changed while reading"
    );
    Ok(bytes)
}

pub(super) fn regular(path: &Path) -> Result<()> {
    let metadata = fs::symlink_metadata(path)?;
    ensure!(
        metadata.is_file() && !metadata.file_type().is_symlink(),
        "child file is not regular"
    );
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        ensure!(
            metadata.file_attributes() & 0x400 == 0,
            "child file is a reparse point"
        );
    }
    Ok(())
}

/// Streamed SHA-256 of a regular file bounded by `maximum`; fails if it changes while read.
pub(super) fn hash_file(path: &Path, maximum: u64) -> Result<String> {
    regular(path)?;
    let mut file = File::open(path)?;
    let expected = file.metadata()?.len();
    ensure!(expected <= maximum, "pinned file exceeds bound");
    let (mut buffer, mut length, mut hasher) = ([0u8; 8192], 0u64, Sha256::new());
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        length = length
            .checked_add(count as u64)
            .context("pinned file length overflow")?;
        ensure!(length <= maximum, "pinned file grew beyond bound");
        hasher.update(&buffer[..count]);
    }
    ensure!(length == expected, "pinned file length changed");
    Ok(format!("{:x}", hasher.finalize()))
}

#[cfg(test)]
pub(super) fn publish(attempt: &Path, name: &str, bytes: &[u8]) -> Result<String> {
    ensure!(
        !bytes.is_empty() && bytes.len() as u64 <= RESULT_CAP,
        "child result exceeds bound"
    );
    let destination = storage_paths::file(attempt, name)?;
    ensure!(!destination.try_exists()?, "child output must be fresh");
    let temporary = storage_paths::file(attempt, &format!(".child-{}-{name}", std::process::id()))?;
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)?;
    output.write_all(bytes)?;
    output.sync_all()?;
    drop(output);
    let publication =
        fs::hard_link(&temporary, &destination).context("child atomic publication failed");
    fs::remove_file(&temporary)?;
    publication?;
    Ok(storage_paths::digest(bytes))
}
