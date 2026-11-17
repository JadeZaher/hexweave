//! Streaming tile-plus-Adam payloads for connected training; see AGENTS.md.

use super::geometry::{Geometry, MATRIX_COUNT};
use super::state_codec::encode_tile;
use super::tile::Tile;
use super::tuning::Latents;
use anyhow::{ensure, Result};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

pub(super) const MAGIC: &[u8; 8] = b"HWCONN02";

struct Hashed<W> {
    inner: W,
    hash: Sha256,
    bytes: u64,
}

impl<W: Write> Write for Hashed<W> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let n = self.inner.write(bytes)?;
        self.hash.update(&bytes[..n]);
        self.bytes += n as u64;
        Ok(n)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

pub(super) fn packed_length(geometry: Geometry, id: usize) -> Result<usize> {
    Ok(48
        + geometry
            .shapes(id)?
            .iter()
            .map(|(r, c)| 12 + (r * c).div_ceil(4))
            .sum::<usize>())
}

pub(super) fn cap(geometry: Geometry, id: usize) -> Result<u64> {
    let shapes = geometry.shapes(id)?;
    Ok((32
        + packed_length(geometry, id)?
        + (shapes.len() + 1) * 32
        + shapes.iter().map(|(r, c)| r * c * 12).sum::<usize>()
        + geometry.routes(id)? * geometry.width * 12
        + shapes.iter().map(|(r, c)| r * c).sum::<usize>()) as u64)
}

pub(super) fn latent_length(geometry: Geometry, id: usize, matrix: usize) -> Result<usize> {
    if matrix == MATRIX_COUNT {
        return Ok(geometry.routes(id)? * geometry.width);
    }
    ensure!(matrix < MATRIX_COUNT, "invalid connected latent matrix");
    let (rows, cols) = geometry.shapes(id)?[matrix];
    Ok(rows * cols)
}

pub(super) fn validate(tile: &Tile, values: &Latents) -> Result<()> {
    tile.geometry.validate()?;
    let mut previous = None;
    for ((owner, matrix), latent) in values {
        ensure!(*owner == tile.id, "foreign tile latent");
        ensure!(
            previous.is_none_or(|old| old < *matrix),
            "noncanonical tile latent order"
        );
        previous = Some(*matrix);
        let length = latent_length(tile.geometry, tile.id, *matrix)?;
        super::connected_adam::validate(latent, length)?;
    }
    Ok(())
}

fn encode(
    out: &mut impl Write,
    tile: &Tile,
    values: &Latents,
    last_direction: &[i8],
) -> Result<()> {
    validate(tile, values)?;
    ensure!(
        last_direction.len() == tile.geometry.tile_coefficients(tile.id)?
            && last_direction.iter().all(|value| (-1..=1).contains(value)),
        "invalid trit direction history"
    );
    let packed = encode_tile(tile)?;
    out.write_all(MAGIC)?;
    out.write_all(&(packed.len() as u64).to_le_bytes())?;
    out.write_all(&packed)?;
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
        floats(out, latent.weights.iter().chain(&latent.moments))?;
    }
    out.write_all(&(last_direction.len() as u64).to_le_bytes())?;
    let mut buffer = [0; 16 * 1024];
    for chunk in last_direction.chunks(buffer.len()) {
        for (target, value) in buffer.iter_mut().zip(chunk) {
            *target = *value as u8;
        }
        out.write_all(&buffer[..chunk.len()])?;
    }
    Ok(())
}

fn floats<'a>(out: &mut impl Write, values: impl Iterator<Item = &'a f32>) -> Result<()> {
    let mut buffer = [0; 16 * 1024];
    let mut used = 0;
    for value in values {
        buffer[used..used + 4].copy_from_slice(&value.to_le_bytes());
        used += 4;
        if used == buffer.len() {
            out.write_all(&buffer)?;
            used = 0;
        }
    }
    out.write_all(&buffer[..used])?;
    Ok(())
}

pub(super) fn write(
    path: &Path,
    tile: &Tile,
    values: &Latents,
    last_direction: &[i8],
) -> Result<(String, u64)> {
    let mut writer = BufWriter::new(Hashed {
        inner: File::create(path)?,
        hash: Sha256::new(),
        bytes: 0,
    });
    encode(&mut writer, tile, values, last_direction)?;
    writer.flush()?;
    writer.get_ref().inner.sync_all()?;
    Ok((
        format!("{:x}", writer.get_ref().hash.clone().finalize()),
        writer.get_ref().bytes,
    ))
}

/// The exact bytes and hash `write` would publish, kept in RAM for write-behind.
pub(super) fn encode_bytes(
    tile: &Tile,
    values: &Latents,
    last_direction: &[i8],
) -> Result<(String, Vec<u8>)> {
    let mut writer = Hashed {
        inner: Vec::with_capacity(usize::try_from(cap(tile.geometry, tile.id)?)?),
        hash: Sha256::new(),
        bytes: 0,
    };
    encode(&mut writer, tile, values, last_direction)?;
    // The cap over-reserves; release the slack so pending byte accounting is real RAM.
    writer.inner.shrink_to_fit();
    Ok((format!("{:x}", writer.hash.finalize()), writer.inner))
}

/// Durably writes already-encoded bundle bytes to `path`.
pub(super) fn write_bytes(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = File::create(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

pub(super) fn digest(tile: &Tile, values: &Latents, last_direction: &[i8]) -> Result<String> {
    let mut writer = Hashed {
        inner: io::sink(),
        hash: Sha256::new(),
        bytes: 0,
    };
    encode(&mut writer, tile, values, last_direction)?;
    Ok(format!("{:x}", writer.hash.finalize()))
}
