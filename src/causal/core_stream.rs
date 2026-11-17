//! Bounded checkpoint payload writers; see AGENTS.md §Core v2.

use super::resident::ResidentStore;
use super::state_codec::encode_tile;
use super::tuning::{self, Latent};
use anyhow::{ensure, Result};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

struct HashWriter<W> {
    inner: W,
    hash: Sha256,
}

impl<W: Write> Write for HashWriter<W> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let written = self.inner.write(bytes)?;
        self.hash.update(&bytes[..written]);
        Ok(written)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

fn tiles<W: Write>(out: &mut W, store: &ResidentStore) -> Result<()> {
    for tile in &store.tiles {
        let bytes = encode_tile(tile)?;
        out.write_all(&(bytes.len() as u64).to_le_bytes())?;
        out.write_all(&bytes)?;
    }
    Ok(())
}

fn latents<W: Write>(out: &mut W, values: &BTreeMap<(usize, usize), Latent>) -> Result<()> {
    out.write_all(&(values.len() as u64).to_le_bytes())?;
    for (&(owner, matrix), latent) in values {
        ensure!(
            latent.moments.len() == 2 * latent.weights.len(),
            "latent moment shape"
        );
        for value in [
            owner as u64,
            matrix as u64,
            latent.step,
            latent.weights.len() as u64,
        ] {
            out.write_all(&value.to_le_bytes())?;
        }
        for value in latent.weights.iter().chain(&latent.moments) {
            out.write_all(&value.to_le_bytes())?;
        }
    }
    Ok(())
}

fn file(
    path: &Path,
    write: impl FnOnce(&mut BufWriter<HashWriter<File>>) -> Result<()>,
) -> Result<String> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(HashWriter {
        inner: file,
        hash: Sha256::new(),
    });
    write(&mut writer)?;
    writer.flush()?;
    writer.get_ref().inner.sync_all()?;
    Ok(format!("{:x}", writer.get_ref().hash.clone().finalize()))
}

/// Stream one tile at a time and return the synced payload's SHA-256.
pub(super) fn write_tiles(path: &Path, store: &ResidentStore) -> Result<String> {
    file(path, |writer| tiles(writer, store))
}

/// Stream borrowed optimizer state and return its SHA-256 and draw counter.
pub(super) fn write_latents(path: &Path) -> Result<(String, u64)> {
    tuning::with_latents(|values, calls| Ok((file(path, |writer| latents(writer, values))?, calls)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::causal::core_codec::{encode_latents, encode_tiles};
    use crate::causal::geometry::Geometry;

    #[test]
    fn streamed_payloads_match_the_original_codec() -> Result<()> {
        let store = ResidentStore::new(Geometry::fixture(), 7)?;
        let mut hashed = BufWriter::with_capacity(
            7,
            HashWriter {
                inner: Vec::new(),
                hash: Sha256::new(),
            },
        );
        tiles(&mut hashed, &store)?;
        hashed.flush()?;
        let expected = encode_tiles(&store)?;
        assert_eq!(hashed.get_ref().inner, expected);
        assert_eq!(
            format!("{:x}", hashed.get_ref().hash.clone().finalize()),
            super::super::storage_paths::digest(&expected)
        );
        let values = BTreeMap::from([
            (
                (0, 2),
                Latent {
                    weights: vec![0.25, -0.5],
                    moments: vec![0.0, 1.0, -1.0, 2.0],
                    step: 4,
                },
            ),
            (
                (7, 0),
                Latent {
                    weights: vec![1.0],
                    moments: vec![0.5, -0.5],
                    step: 9,
                },
            ),
        ]);
        let mut output = Vec::new();
        latents(&mut output, &values)?;
        let cloned: Vec<_> = values.into_iter().collect();
        assert_eq!(output, encode_latents(&cloned)?);
        Ok(())
    }
}
