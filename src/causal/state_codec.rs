use super::geometry::{Geometry, EVIDENCE_RULE, MATRIX_COUNT, VOCAB};
use super::predictor::Predictor;
use super::shared::Shared;
use super::tile::Tile;
use anyhow::{ensure, Context, Result};

const MAGIC: &[u8; 8] = b"HWCPACK1";

fn encode(
    geometry: Geometry,
    kind: u32,
    id: u32,
    version: u64,
    shapes: &[(usize, usize)],
    packed: &[&[u8]],
) -> Result<Vec<u8>> {
    geometry.validate()?;
    ensure!(
        shapes.len() == packed.len() && shapes.len() <= MATRIX_COUNT,
        "codec matrix count"
    );
    let capacity = 48 + shapes.len() * 12 + packed.iter().map(|b| b.len()).sum::<usize>();
    ensure!(
        capacity <= super::storage_paths::FILE_CAP as usize,
        "packed state exceeds file bound"
    );
    let mut bytes = Vec::with_capacity(capacity);
    bytes.extend_from_slice(MAGIC);
    for value in [
        geometry.width,
        geometry.ffn,
        geometry.heads,
        geometry.context,
    ] {
        bytes.extend_from_slice(&u32::try_from(value)?.to_le_bytes());
    }
    for value in [kind, id] {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes.extend_from_slice(&version.to_le_bytes());
    bytes.extend_from_slice(&1u32.to_le_bytes());
    bytes.extend_from_slice(&u32::try_from(shapes.len())?.to_le_bytes());
    for ((rows, cols), values) in shapes.iter().zip(packed) {
        ensure!(
            values.len() == (rows * cols).div_ceil(4),
            "packed matrix length mismatch"
        );
        for value in [*rows, *cols, values.len()] {
            bytes.extend_from_slice(&u32::try_from(value)?.to_le_bytes());
        }
        bytes.extend_from_slice(values);
    }
    Ok(bytes)
}

struct Reader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, count: usize) -> Result<&'a [u8]> {
        let end = self
            .offset
            .checked_add(count)
            .context("codec offset overflow")?;
        let result = self
            .bytes
            .get(self.offset..end)
            .context("truncated packed state")?;
        self.offset = end;
        Ok(result)
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into()?))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into()?))
    }
}

fn decode(
    bytes: &[u8],
    geometry: Geometry,
    kind: u32,
    id: u32,
    version: u64,
    shapes: &[(usize, usize)],
) -> Result<Vec<Vec<u8>>> {
    geometry.validate()?;
    let expected = 48
        + shapes.len() * 12
        + shapes
            .iter()
            .map(|(r, c)| (r * c).div_ceil(4))
            .sum::<usize>();
    ensure!(
        bytes.len() == expected,
        "packed state exact byte count mismatch"
    );
    let mut reader = Reader { bytes, offset: 0 };
    ensure!(reader.take(8)? == MAGIC, "packed state magic mismatch");
    for value in [
        geometry.width,
        geometry.ffn,
        geometry.heads,
        geometry.context,
    ] {
        ensure!(
            reader.u32()? as usize == value,
            "packed state geometry mismatch"
        );
    }
    ensure!(
        reader.u32()? == kind && reader.u32()? == id && reader.u64()? == version,
        "packed state kind/id/version mismatch"
    );
    ensure!(
        reader.u32()? == 1 && reader.u32()? as usize == shapes.len(),
        "packed evidence/schema mismatch"
    );
    let mut result = Vec::with_capacity(shapes.len());
    for (rows, cols) in shapes {
        let length = (rows * cols).div_ceil(4);
        ensure!(
            reader.u32()? as usize == *rows
                && reader.u32()? as usize == *cols
                && reader.u32()? as usize == length,
            "packed matrix shape mismatch"
        );
        result.push(reader.take(length)?.to_vec());
    }
    ensure!(reader.offset == bytes.len(), "trailing packed state bytes");
    Ok(result)
}

pub(super) fn encode_tile(tile: &Tile) -> Result<Vec<u8>> {
    encode(
        tile.geometry,
        0,
        tile.id as u32,
        tile.version,
        &tile.geometry.shapes(tile.id)?,
        &tile.packed(),
    )
}

pub(super) fn decode_tile(
    bytes: &[u8],
    geometry: Geometry,
    id: usize,
    version: u64,
) -> Result<Tile> {
    let packed = decode(
        bytes,
        geometry,
        0,
        u32::try_from(id)?,
        version,
        &geometry.shapes(id)?,
    )?;
    Tile::from_packed(geometry, id, version, &packed, EVIDENCE_RULE)
}

pub(super) fn encode_shared(shared: &Shared) -> Result<Vec<u8>> {
    encode(
        shared.geometry,
        1,
        0,
        shared.version,
        &[(shared.geometry.width, VOCAB)],
        &[shared.packed()],
    )
}

pub(super) fn decode_shared(bytes: &[u8], geometry: Geometry, version: u64) -> Result<Shared> {
    let packed = decode(bytes, geometry, 1, 0, version, &[(geometry.width, VOCAB)])?;
    Shared::from_packed(geometry, version, &packed[0], EVIDENCE_RULE)
}

pub(super) fn encode_predictor(predictor: &Predictor) -> Result<Vec<u8>> {
    let geometry = predictor.geometry;
    encode(
        geometry,
        2,
        0,
        predictor.version,
        &[(geometry.width, geometry.features())],
        &[predictor.packed()],
    )
}

pub(super) fn decode_predictor(
    bytes: &[u8],
    geometry: Geometry,
    version: u64,
) -> Result<Predictor> {
    let packed = decode(
        bytes,
        geometry,
        2,
        0,
        version,
        &[(geometry.width, geometry.features())],
    )?;
    Predictor::from_packed(geometry, version, &packed[0], EVIDENCE_RULE)
}
