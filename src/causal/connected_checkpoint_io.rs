//! Bounded filesystem operations for connected checkpoints; see AGENTS.md.

use anyhow::{ensure, Context, Result};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;
use std::time::Duration;

pub(super) const COMMON_CAP: u64 = 16 * 1024 * 1024;
pub(super) const TOTAL_CAP: usize = 64 * 1024 * 1024;

pub(super) fn retry<T>(mut action: impl FnMut() -> std::io::Result<T>) -> std::io::Result<T> {
    let mut delay = Duration::from_millis(25);
    for _ in 0..7 {
        match action() {
            Err(error) if matches!(error.raw_os_error(), Some(5 | 32)) => {
                std::thread::sleep(delay);
                delay *= 2;
            }
            result => return result,
        }
    }
    action()
}

pub(super) fn name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 64
        && !name.starts_with('.')
        && name != "manifest.json"
        && name != "tiles"
        && name
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'-' | b'_' | b'.'))
}

pub(super) fn read(path: &Path, cap: u64) -> Result<Vec<u8>> {
    let file = retry(|| File::open(path))?;
    ensure!(
        file.metadata()?.len() <= cap,
        "connected payload exceeds size bound"
    );
    let mut bytes = Vec::new();
    file.take(cap + 1).read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() as u64 <= cap,
        "connected payload grew beyond bound"
    );
    Ok(bytes)
}

pub(super) fn write(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = retry(|| File::create(path))?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

fn transfer(source: &Path, destination: Option<&Path>, cap: u64) -> Result<String> {
    let input = retry(|| File::open(source))?;
    let length = input.metadata()?.len();
    ensure!(length <= cap, "connected file exceeds size bound");
    let mut input = input.take(cap + 1);
    let mut output = destination
        .map(|path| retry(|| File::create(path)))
        .transpose()?;
    let mut digest = Sha256::new();
    let mut buffer = [0; 64 * 1024];
    let mut copied = 0;
    loop {
        let count = input.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        copied += count as u64;
        ensure!(copied <= cap, "connected file grew beyond bound");
        digest.update(&buffer[..count]);
        if let Some(output) = &mut output {
            output.write_all(&buffer[..count])?;
        }
    }
    ensure!(copied == length, "connected file changed during copy");
    if let Some(output) = output {
        output.sync_all()?;
    }
    Ok(format!("{:x}", digest.finalize()))
}

pub(super) fn hash(path: &Path, cap: u64) -> Result<String> {
    transfer(path, None, cap)
}

pub(super) fn copy(source: &Path, destination: &Path, cap: u64) -> Result<String> {
    transfer(source, Some(destination), cap)
}

pub(super) fn remove_checkpoint(out: &Path, path: &Path) -> Result<()> {
    let root = out.join("checkpoints").canonicalize()?;
    let resolved = path.canonicalize()?;
    ensure!(
        resolved.parent() == Some(root.as_path()),
        "checkpoint prune escaped its root"
    );
    let name = resolved
        .file_name()
        .and_then(|name| name.to_str())
        .context("checkpoint name encoding")?;
    ensure!(
        name.len() == 13 && name.starts_with('a') && name[1..].bytes().all(|c| c.is_ascii_digit()),
        "checkpoint prune name mismatch"
    );
    retry(|| std::fs::remove_dir_all(&resolved))?;
    Ok(())
}
