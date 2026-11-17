//! Binary payloads of core checkpoints: length-prefixed tiles and little-endian latent state.

use super::geometry::{Geometry, TILES};
use super::resident::ResidentStore;
use super::state_codec::decode_tile;
#[cfg(test)]
use super::state_codec::encode_tile;
#[cfg(test)]
use super::tuning::Latent;
use super::tuning::Latents;
use anyhow::{ensure, Context, Result};

#[cfg(test)]
pub(super) fn encode_tiles(store: &ResidentStore) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    for tile in &store.tiles {
        let encoded = encode_tile(tile)?;
        bytes.extend_from_slice(&(encoded.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&encoded);
    }
    Ok(bytes)
}

pub(super) fn decode_tiles(
    bytes: &[u8],
    geometry: Geometry,
    versions: [u64; TILES],
) -> Result<ResidentStore> {
    let mut reader = Reader { bytes, offset: 0 };
    let mut tiles = Vec::with_capacity(TILES);
    for (id, version) in versions.into_iter().enumerate() {
        let length = usize::try_from(reader.u64()?)?;
        tiles.push(decode_tile(reader.take(length)?, geometry, id, version)?);
    }
    ensure!(reader.offset == bytes.len(), "trailing tile bytes");
    ResidentStore::from_tiles(tiles, geometry)
}

/// Count, then per entry: owner, matrix, step, length, weights, interleaved moments.
#[cfg(test)]
pub(super) fn encode_latents(latents: &[((usize, usize), Latent)]) -> Result<Vec<u8>> {
    let mut bytes = (latents.len() as u64).to_le_bytes().to_vec();
    for ((owner, matrix), latent) in latents {
        ensure!(
            latent.moments.len() == 2 * latent.weights.len(),
            "latent moment shape"
        );
        for value in [
            *owner as u64,
            *matrix as u64,
            latent.step,
            latent.weights.len() as u64,
        ] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        for value in latent.weights.iter().chain(&latent.moments) {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    Ok(bytes)
}

pub(super) fn decode_latents(bytes: &[u8]) -> Result<Latents> {
    let mut reader = Reader { bytes, offset: 0 };
    let count = reader.u64()?;
    let mut latents = Vec::new();
    for _ in 0..count {
        let (owner, matrix) = (
            usize::try_from(reader.u64()?)?,
            usize::try_from(reader.u64()?)?,
        );
        let (step, length) = (reader.u64()?, usize::try_from(reader.u64()?)?);
        let total = length.checked_mul(3).context("latent length overflow")?;
        ensure!(
            total
                .checked_mul(4)
                .is_some_and(|needed| needed <= reader.remaining()),
            "latent payload shorter than its declared length"
        );
        let mut weights = Vec::with_capacity(length);
        for _ in 0..length {
            weights.push(f32::from_le_bytes(reader.take(4)?.try_into()?));
        }
        let mut moments = Vec::with_capacity(total - length);
        for _ in length..total {
            moments.push(f32::from_le_bytes(reader.take(4)?.try_into()?));
        }
        latents.push((
            (owner, matrix),
            Latent {
                weights,
                moments,
                step,
            },
        ));
    }
    ensure!(reader.offset == bytes.len(), "trailing latent bytes");
    Ok(latents)
}

struct Reader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl Reader<'_> {
    fn take(&mut self, length: usize) -> Result<&[u8]> {
        let end = self
            .offset
            .checked_add(length)
            .context("checkpoint length overflow")?;
        let slice = self
            .bytes
            .get(self.offset..end)
            .context("truncated checkpoint payload")?;
        self.offset = end;
        Ok(slice)
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into()?))
    }

    fn remaining(&self) -> usize {
        self.bytes.len() - self.offset
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn latent_roundtrip_preserves_bits_and_rejects_truncation() -> Result<()> {
        let values = vec![
            (
                (3, 0),
                Latent {
                    weights: vec![-0.0, f32::from_bits(0x7fc0_1234)],
                    moments: vec![1.0, -2.0, 0.0, 4.0],
                    step: 9,
                },
            ),
            (
                (3, 1),
                Latent {
                    weights: vec![],
                    moments: vec![],
                    step: 0,
                },
            ),
        ];
        let encoded = encode_latents(&values)?;
        assert_eq!(encode_latents(&decode_latents(&encoded)?)?, encoded);
        assert!(decode_latents(&encoded[..encoded.len() - 1]).is_err());
        Ok(())
    }
}
