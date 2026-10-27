use super::data_contracts::*;
use anyhow::{ensure, Context, Result};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Component, Path, PathBuf};

pub(super) fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
pub(super) fn valid_hash(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
pub(super) fn valid_id(value: &str, maximum: usize) -> bool {
    !value.is_empty()
        && value.len() <= maximum
        && value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
}

pub(super) fn contained(root: &Path, relative: &str) -> Result<PathBuf> {
    ensure!(
        !relative.is_empty() && relative.len() <= 160 && !relative.contains(['\\', ':']),
        "invalid dataset path"
    );
    ensure!(
        relative
            .split('/')
            .all(|c| !c.is_empty() && c != "." && c != ".."),
        "noncanonical dataset path"
    );
    ensure!(
        Path::new(relative)
            .components()
            .all(|c| matches!(c, Component::Normal(_))),
        "non-relative dataset path"
    );
    let path = root
        .join(relative)
        .canonicalize()
        .context("resolve dataset path")?;
    ensure!(
        path.starts_with(root) && path.is_file(),
        "dataset path escapes root or is not a file"
    );
    Ok(path)
}

pub(super) fn bounded(path: &Path, cap: usize) -> Result<Vec<u8>> {
    let file = File::open(path)?;
    ensure!(
        file.metadata()?.len() <= cap as u64,
        "input exceeds byte cap"
    );
    let mut bytes = Vec::with_capacity(cap + 1);
    file.take(cap as u64 + 1).read_to_end(&mut bytes)?;
    ensure!(bytes.len() <= cap, "input grew beyond byte cap");
    Ok(bytes)
}

pub(super) fn hash_file(
    path: &Path,
    expected_bytes: u64,
    expected_hash: &str,
    cap: u64,
) -> Result<()> {
    ensure!(
        expected_bytes > 0 && expected_bytes <= cap && valid_hash(expected_hash),
        "invalid file pin"
    );
    let mut file = File::open(path)?;
    ensure!(
        file.metadata()?.len() == expected_bytes,
        "pinned file length changed"
    );
    let mut hash = Sha256::new();
    let mut count = 0u64;
    let mut buffer = [0u8; STREAM_CAP];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        count = count
            .checked_add(n as u64)
            .context("file length overflow")?;
        ensure!(count <= expected_bytes, "pinned file grew");
        hash.update(&buffer[..n]);
    }
    ensure!(
        count == expected_bytes && format!("{:x}", hash.finalize()) == expected_hash,
        "pinned file hash changed"
    );
    Ok(())
}

pub(super) fn source_span(root: &Path, source: &Source, start: u64, end: u64) -> Result<Vec<u8>> {
    ensure!(
        start < end && end <= source.bytes && end - start <= 608,
        "invalid source span"
    );
    let path = contained(root, &source.snapshot_path)?;
    let mut file = File::open(path)?;
    ensure!(
        source.bytes <= SOURCE_CAP && file.metadata()?.len() == source.bytes,
        "source length changed"
    );
    let mut hash = Sha256::new();
    let mut offset = 0u64;
    let mut span = Vec::with_capacity((end - start) as usize);
    let mut buffer = [0u8; STREAM_CAP + 3];
    let mut carry = 0usize;
    loop {
        let n = file.read(&mut buffer[carry..carry + STREAM_CAP])?;
        if n == 0 {
            break;
        }
        let next = offset
            .checked_add(n as u64)
            .context("source offset overflow")?;
        ensure!(next <= source.bytes, "source grew");
        hash.update(&buffer[carry..carry + n]);
        let (left, right) = (offset.max(start), next.min(end));
        if left < right {
            span.extend_from_slice(
                &buffer[carry + (left - offset) as usize..carry + (right - offset) as usize],
            );
        }
        let end_of_text = carry + n;
        let valid = match std::str::from_utf8(&buffer[..end_of_text]) {
            Ok(_) => end_of_text,
            Err(error) => {
                ensure!(error.error_len().is_none(), "source is not UTF-8");
                error.valid_up_to()
            }
        };
        carry = end_of_text - valid;
        ensure!(carry <= 3, "invalid UTF-8 carry");
        buffer.copy_within(valid..end_of_text, 0);
        offset = next;
    }
    ensure!(
        carry == 0 && offset == source.bytes && format!("{:x}", hash.finalize()) == source.sha256,
        "source UTF-8/hash changed"
    );
    ensure!(
        span.len() == (end - start) as usize,
        "source span incomplete"
    );
    Ok(span)
}

pub(super) fn read_record(path: &Path, offset: u64, length: usize, expected: &str) -> Result<Row> {
    ensure!(
        (1..=RECORD_CAP).contains(&length),
        "record exceeds byte cap"
    );
    let mut file = File::open(path)?;
    file.seek(SeekFrom::Start(offset))?;
    let mut bytes = vec![0; length];
    file.read_exact(&mut bytes)?;
    ensure!(
        sha256(&bytes) == expected && bytes.last() == Some(&b'\n'),
        "record identity changed"
    );
    Ok(serde_json::from_slice(&bytes)?)
}

pub(super) fn validate_manifest(root: &Path, manifest: &Manifest) -> Result<()> {
    super::data_provenance::identity_schema(manifest)?;
    ensure!(
        manifest.prompt_template == TEMPLATE
            && manifest.limits
                == serde_json::json!({
                    "envelope_bytes":80,"question_bytes":144,"evidence_bytes":608,"completion_bytes":192,"sequence_positions":1024
                }),
        "model rendering/limits changed"
    );
    ensure!(
        manifest.artifacts.len() == 2
            && manifest.artifacts.contains_key("train.jsonl")
            && manifest.artifacts.contains_key("development.jsonl"),
        "only train/development artifact paths are authorized"
    );
    ensure!(
        (1..=16).contains(&manifest.sources.len())
            && valid_hash(&manifest.source_pin_manifest_sha256),
        "invalid source ledger"
    );
    let pins_bytes = bounded(&contained(root, "source-pins.json")?, MANIFEST_CAP)?;
    ensure!(
        sha256(&pins_bytes) == manifest.source_pin_manifest_sha256,
        "source-pin manifest mismatch"
    );
    let pins = super::data_provenance::read_pins(&pins_bytes, manifest)?;
    drop(pins_bytes);
    let pinned = &pins;
    ensure!(
        pinned.len() == manifest.sources.len(),
        "source-pin count mismatch"
    );
    let (mut ids, mut paths, mut hashes) = (BTreeSet::new(), BTreeSet::new(), BTreeSet::new());
    for source in &manifest.sources {
        ensure!(
            valid_id(&source.source_id, 16)
                && valid_hash(&source.sha256)
                && (1..=SOURCE_CAP).contains(&source.bytes),
            "invalid source identity"
        );
        ensure!(
            ids.insert(&source.source_id)
                && paths.insert(&source.original_path)
                && hashes.insert(&source.sha256),
            "duplicate source identity/path/content"
        );
        ensure!(
            source.original_path.len() <= 152
                && !source.original_path.contains(['\\', ':'])
                && source
                    .original_path
                    .split('/')
                    .all(|c| !c.is_empty() && c != "." && c != "..")
                && Path::new(&source.original_path)
                    .components()
                    .all(|c| matches!(c, std::path::Component::Normal(_)))
                && source.snapshot_path == format!("sources/{}", source.original_path),
            "invalid source snapshot path"
        );
        let family = super::data_provenance::family(manifest, &source.partition)?;
        ensure!(
            pinned
                .iter()
                .filter(|p| p.path == source.original_path
                    && p.sha256 == source.sha256
                    && p.bytes == source.bytes
                    && p.family == family)
                .count()
                == 1,
            "source pin/provenance mismatch"
        );
    }
    Ok(())
}
