//! Bounded verification of one connected tile bundle; see AGENTS.md.

use super::connected_store::TileMeta;
use super::connected_store_codec::{cap, latent_length, packed_length, MAGIC};
use super::geometry::{Geometry, MATRIX_COUNT};
use super::state_codec::decode_tile;
use super::tile::Tile;
use super::tuning::{Latent, Latents};
use anyhow::{ensure, Result};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

struct Reader<R> {
    file: R,
    hash: Sha256,
    remaining: u64,
}

pub(super) struct Bundle {
    pub tile: Tile,
    pub latents: Latents,
    pub last_direction: Vec<i8>,
    pub bytes: u64,
}

impl<R: Read> Reader<R> {
    fn bytes(&mut self, bytes: &mut [u8]) -> Result<()> {
        ensure!(
            bytes.len() as u64 <= self.remaining,
            "truncated connected tile"
        );
        self.file.read_exact(bytes)?;
        self.hash.update(&*bytes);
        self.remaining -= bytes.len() as u64;
        Ok(())
    }

    fn number(&mut self) -> Result<u64> {
        let mut bytes = [0; 8];
        self.bytes(&mut bytes)?;
        Ok(u64::from_le_bytes(bytes))
    }

    fn floats(&mut self, length: usize) -> Result<Vec<f32>> {
        ensure!(
            length as u64 * 4 <= self.remaining,
            "truncated connected latent"
        );
        let mut values = Vec::with_capacity(length);
        let mut buffer = [0; 16 * 1024];
        while values.len() < length {
            let count = (length - values.len()).min(buffer.len() / 4) * 4;
            self.bytes(&mut buffer[..count])?;
            for bytes in buffer[..count].chunks_exact(4) {
                let value = f32::from_le_bytes(bytes.try_into()?);
                ensure!(value.is_finite(), "nonfinite connected latent");
                values.push(value);
            }
        }
        Ok(values)
    }
}

pub(super) fn load(path: &Path, geometry: Geometry, id: usize, meta: &TileMeta) -> Result<Bundle> {
    geometry.validate()?;
    let file = File::open(path)?;
    let length = file.metadata()?.len();
    let limit = cap(geometry, id)?;
    ensure!(length <= limit, "connected tile exceeds payload bound");
    decode(
        BufReader::new(file.take(limit + 1)),
        length,
        geometry,
        id,
        meta,
    )
}

/// Verifies and decodes write-behind bytes held in RAM exactly as a disk bundle.
pub(super) fn load_bytes(
    bytes: &[u8],
    geometry: Geometry,
    id: usize,
    meta: &TileMeta,
) -> Result<Bundle> {
    geometry.validate()?;
    ensure!(
        bytes.len() as u64 <= cap(geometry, id)?,
        "connected tile exceeds payload bound"
    );
    decode(bytes, bytes.len() as u64, geometry, id, meta)
}

fn decode(
    file: impl Read,
    length: u64,
    geometry: Geometry,
    id: usize,
    meta: &TileMeta,
) -> Result<Bundle> {
    let mut reader = Reader {
        file,
        hash: Sha256::new(),
        remaining: length,
    };
    let mut magic = [0; 8];
    reader.bytes(&mut magic)?;
    ensure!(&magic == MAGIC, "connected tile magic mismatch");
    let packed_length = packed_length(geometry, id)?;
    ensure!(
        reader.number()? == packed_length as u64,
        "connected packed length mismatch"
    );
    let mut packed = vec![0; packed_length];
    reader.bytes(&mut packed)?;
    let tile = decode_tile(&packed, geometry, id, meta.version)?;
    drop(packed);
    let values = read_latents(&mut reader, geometry, id)?;
    let last_direction = read_directions(&mut reader, geometry.tile_coefficients(id)?)?;
    ensure!(reader.remaining == 0, "trailing connected tile bytes");
    let mut trailing = [0];
    ensure!(
        reader.file.read(&mut trailing)? == 0,
        "connected tile grew during read"
    );
    ensure!(
        format!("{:x}", reader.hash.finalize()) == meta.hash,
        "connected tile hash mismatch"
    );
    ensure!(
        values
            .iter()
            .map(|(_, value)| value.step)
            .max()
            .unwrap_or(0)
            == meta.steps,
        "connected optimizer-step mismatch"
    );
    Ok(Bundle {
        tile,
        latents: values,
        last_direction,
        bytes: length,
    })
}

fn read_directions(reader: &mut Reader<impl Read>, length: usize) -> Result<Vec<i8>> {
    ensure!(
        reader.number()? == length as u64,
        "trit direction history shape mismatch"
    );
    let mut values = Vec::with_capacity(length);
    let mut buffer = [0; 16 * 1024];
    while values.len() < length {
        let count = (length - values.len()).min(buffer.len());
        reader.bytes(&mut buffer[..count])?;
        for value in &buffer[..count] {
            let value = *value as i8;
            ensure!((-1..=1).contains(&value), "invalid trit direction history");
            values.push(value);
        }
    }
    Ok(values)
}

fn read_latents(reader: &mut Reader<impl Read>, geometry: Geometry, id: usize) -> Result<Latents> {
    let count = reader.number()?;
    ensure!(
        count <= (MATRIX_COUNT + 1) as u64,
        "too many connected tile latents"
    );
    let mut values = Vec::with_capacity(count as usize);
    let mut previous = None;
    for _ in 0..count {
        let owner = reader.number()?;
        let matrix = usize::try_from(reader.number()?)?;
        let step = reader.number()?;
        let length = reader.number()?;
        ensure!(
            owner == id as u64 && matrix <= MATRIX_COUNT,
            "foreign connected latent"
        );
        ensure!(
            previous.is_none_or(|old| old < matrix),
            "duplicate or unordered connected latent"
        );
        previous = Some(matrix);
        let expected = latent_length(geometry, id, matrix)?;
        ensure!(
            length == expected as u64 && length * 12 <= reader.remaining,
            "connected latent shape mismatch"
        );
        let weights = reader.floats(expected)?;
        let moments = reader.floats(2 * expected)?;
        let latent = Latent {
            weights,
            moments,
            step,
        };
        super::connected_adam::validate(&latent, expected)?;
        values.push(((id, matrix), latent));
    }
    Ok(values)
}
