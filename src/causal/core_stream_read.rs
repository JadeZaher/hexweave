//! Hash-checked incremental checkpoint readers; see AGENTS.md §Core v2.

use super::geometry::{Geometry, MATRIX_COUNT, TILES, VOCAB};
use super::resident::ResidentStore;
use super::state_codec::decode_tile;
use super::tuning::{Latent, Latents};
use anyhow::{bail, ensure, Context, Result};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{self, BufReader, ErrorKind, Read};
use std::path::Path;
use std::time::Duration;

struct HashReader {
    input: File,
    hash: Sha256,
}

impl Read for HashReader {
    fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
        let read = self.input.read(bytes)?;
        self.hash.update(&bytes[..read]);
        Ok(read)
    }
}

struct Reader {
    input: BufReader<HashReader>,
    remaining: u64,
}

impl Reader {
    fn new(file: File) -> Result<Self> {
        let remaining = file.metadata()?.len();
        Ok(Self {
            input: BufReader::new(HashReader {
                input: file,
                hash: Sha256::new(),
            }),
            remaining,
        })
    }

    fn take(&mut self, bytes: &mut [u8]) -> Result<()> {
        ensure!(
            u64::try_from(bytes.len())? <= self.remaining,
            "truncated checkpoint payload"
        );
        if let Err(error) = self.input.read_exact(bytes) {
            if error.kind() == ErrorKind::UnexpectedEof {
                bail!("truncated checkpoint payload");
            }
            return Err(error.into());
        }
        self.remaining -= bytes.len() as u64;
        Ok(())
    }

    fn u64(&mut self) -> Result<u64> {
        let mut bytes = [0; 8];
        self.take(&mut bytes)?;
        Ok(u64::from_le_bytes(bytes))
    }

    fn floats(&mut self, count: usize) -> Result<Vec<f32>> {
        let mut values = Vec::with_capacity(count);
        for _ in 0..count {
            let mut bytes = [0; 4];
            self.take(&mut bytes)?;
            values.push(f32::from_le_bytes(bytes));
        }
        Ok(values)
    }

    fn finish(mut self, expected: &str) -> Result<()> {
        ensure!(self.remaining == 0, "trailing checkpoint bytes");
        let mut extra = [0; 1];
        ensure!(
            self.input.read(&mut extra)? == 0,
            "checkpoint grew during read"
        );
        ensure!(
            format!("{:x}", self.input.get_ref().hash.clone().finalize()) == expected,
            "checkpoint hash mismatch"
        );
        Ok(())
    }
}

/// Decode and hash one packed tile at a time.
pub(super) fn tiles(
    file: File,
    expected: &str,
    geometry: Geometry,
    versions: [u64; TILES],
) -> Result<ResidentStore> {
    let mut input = Reader::new(file)?;
    let mut tiles = Vec::with_capacity(TILES);
    for (id, version) in versions.into_iter().enumerate() {
        let length = usize::try_from(input.u64()?)?;
        let expected_length = 48
            + geometry
                .shapes(id)?
                .iter()
                .map(|(r, c)| 12 + (r * c).div_ceil(4))
                .sum::<usize>();
        ensure!(length == expected_length, "packed tile length mismatch");
        let mut bytes = vec![0; length];
        input.take(&mut bytes)?;
        tiles.push(decode_tile(&bytes, geometry, id, version)?);
    }
    input.finish(expected)?;
    ResidentStore::from_tiles(tiles, geometry)
}

/// Decode and hash latent values without a second complete payload buffer.
pub(super) fn latents(file: File, expected: &str, geometry: Geometry) -> Result<Latents> {
    geometry.validate()?;
    let mut input = Reader::new(file)?;
    let count = input.u64()?;
    ensure!(
        count <= (TILES * MATRIX_COUNT + 1) as u64 && count <= input.remaining / 32,
        "latent count exceeds payload"
    );
    let mut result = Vec::new();
    let mut seen = BTreeSet::new();
    for _ in 0..count {
        let owner = usize::try_from(input.u64()?)?;
        let matrix = usize::try_from(input.u64()?)?;
        let step = input.u64()?;
        let length = usize::try_from(input.u64()?)?;
        let expected_length = if owner < TILES {
            let shapes = geometry.shapes(owner)?;
            let (rows, cols) = shapes.get(matrix).context("invalid latent matrix")?;
            rows * cols
        } else {
            ensure!(owner == TILES && matrix == 0, "invalid latent owner");
            geometry.width * VOCAB
        };
        ensure!(seen.insert((owner, matrix)), "duplicate latent key");
        ensure!(length == expected_length, "latent shape mismatch");
        let moments = length.checked_mul(2).context("latent length overflow")?;
        let bytes = length
            .checked_mul(3)
            .and_then(|n| n.checked_mul(4))
            .context("latent length overflow")?;
        ensure!(
            u64::try_from(bytes)? <= input.remaining,
            "latent payload shorter than its declared length"
        );
        let weights = input.floats(length)?;
        let moments = input.floats(moments)?;
        ensure!(
            weights.iter().chain(&moments).all(|v| v.is_finite()),
            "nonfinite latent value"
        );
        result.push((
            (owner, matrix),
            Latent {
                weights,
                moments,
                step,
            },
        ));
    }
    input.finish(expected)?;
    Ok(result)
}

/// Load both verified payloads before installing any optimizer state.
pub(super) fn load(
    directory: &Path,
    files: &BTreeMap<String, String>,
    geometry: Geometry,
    versions: [u64; TILES],
) -> Result<(ResidentStore, Latents)> {
    let open = |name: &str| -> Result<(File, String)> {
        let expected = files
            .get(name)
            .context("checkpoint manifest lacks payload hash")?
            .clone();
        let file = File::open(directory.join(name))?;
        Ok((file, expected))
    };
    let store = retry_payload(|| {
        let (file, hash) = open("tiles.bin")?;
        tiles(file, &hash, geometry, versions)
    })?;
    let latents = retry_payload(|| {
        let (file, hash) = open("latent.bin")?;
        latents(file, &hash, geometry)
    })?;
    Ok((store, latents))
}

fn retry_payload<T>(mut operation: impl FnMut() -> Result<T>) -> Result<T> {
    let mut delay = Duration::from_millis(50);
    for _ in 0..8 {
        match operation() {
            Err(error)
                if error.chain().any(|cause| {
                    cause
                        .downcast_ref::<io::Error>()
                        .is_some_and(|error| matches!(error.raw_os_error(), Some(5 | 32)))
                }) =>
            {
                std::thread::sleep(delay);
                delay *= 2;
            }
            result => return result,
        }
    }
    operation()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::causal::core_codec::{encode_latents, encode_tiles};
    use crate::causal::storage_paths::digest;

    #[test]
    fn reads_existing_codec_and_rejects_corrupt_or_truncated_payloads() -> Result<()> {
        let directory = super::super::core_contracts::fresh_temp("stream-read")?;
        std::fs::create_dir_all(&directory)?;
        let path = directory.join("payload.bin");
        let geometry = Geometry::fixture();
        let store = ResidentStore::new(geometry, 7)?;
        let tile_bytes = encode_tiles(&store)?;
        std::fs::write(&path, &tile_bytes)?;
        let loaded = tiles(
            File::open(&path)?,
            &digest(&tile_bytes),
            geometry,
            [0; TILES],
        )?;
        assert_eq!(encode_tiles(&loaded)?, tile_bytes);
        assert!(tiles(
            File::open(&path)?,
            &digest(b"different"),
            geometry,
            [0; TILES]
        )
        .is_err());

        let values = vec![(
            (TILES, 0),
            Latent {
                weights: vec![0.25; 4096],
                moments: vec![-0.5; 8192],
                step: 4,
            },
        )];
        let bytes = encode_latents(&values)?;
        std::fs::write(&path, &bytes)?;
        assert_eq!(
            latents(File::open(&path)?, &digest(&bytes), geometry)?,
            values
        );
        let mut trailing = bytes.clone();
        trailing.push(0);
        std::fs::write(&path, &trailing)?;
        let error = latents(File::open(&path)?, &digest(&bytes), geometry).unwrap_err();
        assert!(super::super::core_checkpoint::integrity_failure(&error));
        let mut malformed = bytes.clone();
        malformed[32..40].copy_from_slice(&u64::MAX.to_le_bytes());
        std::fs::write(&path, &malformed)?;
        let error = latents(File::open(&path)?, &digest(&malformed), geometry).unwrap_err();
        assert!(super::super::core_checkpoint::integrity_failure(&error));
        std::fs::write(&path, &bytes[..bytes.len() - 1])?;
        let error = latents(File::open(&path)?, &digest(&bytes), geometry).unwrap_err();
        assert!(super::super::core_checkpoint::integrity_failure(&error));
        let mut attempts = 0;
        retry_payload(|| {
            attempts += 1;
            if attempts == 1 {
                Err(io::Error::from_raw_os_error(32).into())
            } else {
                Ok(())
            }
        })?;
        assert_eq!(attempts, 2);
        let mut attempts = 0;
        let failure: Result<()> = retry_payload(|| {
            attempts += 1;
            Err(io::Error::from(ErrorKind::PermissionDenied).into())
        });
        assert!(failure.is_err());
        assert_eq!(attempts, 1);
        std::fs::remove_file(&path)?;
        std::fs::remove_dir(&directory)?;
        Ok(())
    }
}
