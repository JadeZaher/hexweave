//! Streaming publication for the test-only paged store.

use super::super::tuning::Latent;
use anyhow::{ensure, Result};
use sha2::{Digest, Sha256};
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

pub(super) fn validate(values: &[((usize, usize), Latent)]) -> Result<()> {
    for (_, latent) in values {
        ensure!(
            latent.moments.len() == 2 * latent.weights.len(),
            "latent moment shape"
        );
    }
    Ok(())
}

fn payload(out: &mut impl Write, tile: &[u8], values: &[((usize, usize), Latent)]) -> Result<()> {
    out.write_all(&(tile.len() as u64).to_le_bytes())?;
    out.write_all(tile)?;
    out.write_all(&(values.len() as u64).to_le_bytes())?;
    for ((owner, matrix), latent) in values {
        for number in [
            *owner as u64,
            *matrix as u64,
            latent.step,
            latent.weights.len() as u64,
        ] {
            out.write_all(&number.to_le_bytes())?;
        }
        for number in latent.weights.iter().chain(&latent.moments) {
            out.write_all(&number.to_le_bytes())?;
        }
    }
    Ok(())
}

pub(super) fn write(
    path: &Path,
    tile: &[u8],
    values: &[((usize, usize), Latent)],
) -> Result<String> {
    let mut out = BufWriter::new(HashWriter {
        inner: File::create(path)?,
        hash: Sha256::new(),
    });
    payload(&mut out, tile, values)?;
    out.flush()?;
    out.get_ref().inner.sync_all()?;
    Ok(format!("{:x}", out.get_ref().hash.clone().finalize()))
}

#[test]
fn streamed_publication_matches_original_codec_and_hash() -> Result<()> {
    use super::super::{core_codec::encode_latents, storage_paths::digest, tuning::Latent};
    let tile = b"packed tile";
    let values = vec![(
        (2, 1),
        Latent {
            weights: vec![0.25, -0.5],
            moments: vec![1.0, 2.0, -1.0, 0.0],
            step: 9,
        },
    )];
    validate(&values)?;
    let mut expected = (tile.len() as u64).to_le_bytes().to_vec();
    expected.extend_from_slice(tile);
    expected.extend_from_slice(&encode_latents(&values)?);
    let mut out = BufWriter::with_capacity(
        7,
        HashWriter {
            inner: Vec::new(),
            hash: Sha256::new(),
        },
    );
    payload(&mut out, tile, &values)?;
    out.flush()?;
    assert_eq!(out.get_ref().inner, expected);
    assert_eq!(
        format!("{:x}", out.get_ref().hash.clone().finalize()),
        digest(&expected)
    );
    Ok(())
}
