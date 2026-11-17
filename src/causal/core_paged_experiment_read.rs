//! Bounded streaming reads for the test-only paged store.

use super::super::{
    geometry::{Geometry, MATRIX_COUNT},
    state_codec::decode_tile,
    tile::Tile,
    tuning::Latent,
};
use super::{Latents, Slot};
use anyhow::{ensure, Result};
use sha2::{Digest, Sha256};
use std::{
    fs::File,
    io::{self, BufReader, Read},
};

struct HashReader<R> {
    inner: R,
    hash: Sha256,
}

impl<R: Read> Read for HashReader<R> {
    fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
        let n = self.inner.read(bytes)?;
        self.hash.update(&bytes[..n]);
        Ok(n)
    }
}

struct Reader<R> {
    input: BufReader<HashReader<R>>,
    consumed: u64,
    length: u64,
}

impl<R: Read> Reader<R> {
    fn bytes(&mut self, bytes: &mut [u8]) -> Result<()> {
        self.input.read_exact(bytes)?;
        self.consumed += bytes.len() as u64;
        Ok(())
    }
    fn u64(&mut self) -> Result<u64> {
        let mut bytes = [0; 8];
        self.bytes(&mut bytes)?;
        Ok(u64::from_le_bytes(bytes))
    }
    fn floats(&mut self, length: usize) -> Result<Vec<f32>> {
        let mut values = Vec::with_capacity(length);
        for _ in 0..length {
            let mut bytes = [0; 4];
            self.bytes(&mut bytes)?;
            let value = f32::from_le_bytes(bytes);
            ensure!(value.is_finite(), "nonfinite tile latent");
            values.push(value);
        }
        Ok(values)
    }
}

pub(super) fn load(geometry: Geometry, id: usize, slot: &Slot) -> Result<(Tile, Latents)> {
    let shapes = geometry.shapes(id)?;
    let coefficients: usize = shapes.iter().map(|(r, c)| r * c).sum();
    let tile_length = 48
        + shapes.len() * 12
        + shapes
            .iter()
            .map(|(r, c)| (r * c).div_ceil(4))
            .sum::<usize>();
    let cap = 8 + tile_length + 8 + shapes.len() * 32 + coefficients * 12;
    let file = File::open(&slot.path)?;
    let length = file.metadata()?.len();
    ensure!(length <= cap as u64, "tile payload exceeds bound");
    let mut reader = Reader {
        input: BufReader::new(HashReader {
            inner: file.take(cap as u64 + 1),
            hash: Sha256::new(),
        }),
        consumed: 0,
        length,
    };
    ensure!(
        reader.u64()? == tile_length as u64,
        "invalid packed tile length"
    );
    ensure!(
        reader.length.saturating_sub(reader.consumed) >= tile_length as u64 + 8,
        "truncated tile payload"
    );
    let mut bytes = vec![0; tile_length];
    reader.bytes(&mut bytes)?;
    let tile = decode_tile(&bytes, geometry, id, slot.version)?;
    drop(bytes);
    let count = reader.u64()?;
    ensure!(count <= shapes.len() as u64, "too many tile latents");
    let mut values = Vec::with_capacity(count as usize);
    let mut seen = [false; MATRIX_COUNT];
    for _ in 0..count {
        let owner = reader.u64()?;
        let matrix = usize::try_from(reader.u64()?)?;
        let step = reader.u64()?;
        let length = reader.u64()?;
        ensure!(
            owner == id as u64 && matrix < shapes.len() && !seen[matrix],
            "invalid tile latent key"
        );
        seen[matrix] = true;
        let expected = shapes[matrix].0 * shapes[matrix].1;
        ensure!(length == expected as u64, "invalid tile latent shape");
        ensure!(
            reader.length.saturating_sub(reader.consumed) >= (expected as u64) * 12,
            "truncated tile latent"
        );
        let weights = reader.floats(expected)?;
        let moments = reader.floats(expected * 2)?;
        values.push((
            (id, matrix),
            Latent {
                weights,
                moments,
                step,
            },
        ));
    }
    let mut trailing = [0];
    ensure!(
        reader.input.read(&mut trailing)? == 0,
        "trailing tile bytes"
    );
    ensure!(
        format!("{:x}", reader.input.get_ref().hash.clone().finalize()) == slot.hash,
        "tile payload hash mismatch"
    );
    ensure!(
        values
            .iter()
            .map(|(_, latent)| latent.step)
            .max()
            .unwrap_or(0)
            == slot.steps,
        "tile step metadata mismatch"
    );
    Ok((tile, values))
}

#[test]
fn rejects_malformed_streams_before_installing_latents() -> Result<()> {
    use super::super::{core_codec::encode_latents, scratch_dir::ScratchDir, tuning::Knobs};
    use super::{digest, fs, tuning, Paged};
    let geometry = Geometry::fixture();
    let _guard = tuning::scoped(Knobs::core(0.003, 7));
    let scratch = ScratchDir::new("core-paged-stream-corruption")?;
    let mut store = Paged::new(&scratch.path().join("store"), geometry, 7)?;
    let id = 3;
    let original = fs::read(&store.slots[id].path)?;
    let tile_len = u64::from_le_bytes(original[..8].try_into()?) as usize;
    let prefix = &original[..8 + tile_len];
    let shapes = geometry.shapes(id)?;
    let matrix = shapes
        .iter()
        .enumerate()
        .min_by_key(|(_, (r, c))| r * c)
        .unwrap()
        .0;
    let length = shapes[matrix].0 * shapes[matrix].1;
    let latent = Latent {
        weights: vec![0.0; length],
        moments: vec![0.0; length * 2],
        step: 1,
    };
    let one = vec![((id, matrix), latent.clone())];
    let mut valid = prefix.to_vec();
    valid.extend_from_slice(&encode_latents(&one)?);
    store.slots[id].steps = 1;
    fs::write(&store.slots[id].path, &valid)?;
    store.slots[id].hash = digest(&valid);
    assert!(store.load(id).is_ok());

    let mut bad_tile = valid.clone();
    bad_tile[..8].copy_from_slice(&u64::MAX.to_le_bytes());
    let mut bad_length = valid.clone();
    let offset = prefix.len() + 8 + 24;
    bad_length[offset..offset + 8].copy_from_slice(&u64::MAX.to_le_bytes());
    let mut duplicate = prefix.to_vec();
    duplicate.extend_from_slice(&encode_latents(&[one[0].clone(), one[0].clone()])?);
    let mut trailing = valid.clone();
    trailing.push(0);
    let mut nonfinite = valid.clone();
    nonfinite[offset + 8..offset + 12].copy_from_slice(&f32::NAN.to_le_bytes());
    let mut truncated = valid.clone();
    truncated.pop();
    for (case, bytes, hash) in [
        ("tile length", bad_tile, true),
        ("latent length", bad_length, true),
        ("duplicate", duplicate, true),
        ("trailing", trailing, true),
        ("nonfinite", nonfinite, true),
        ("truncated", truncated, true),
        ("hash", valid, false),
    ] {
        fs::write(&store.slots[id].path, &bytes)?;
        store.slots[id].hash = if hash { digest(&bytes) } else { "wrong".into() };
        assert!(store.enter(id, false).is_err(), "accepted {case}");
        assert!(!store.live, "attached {case}");
        assert!(tuning::steps()
            .iter()
            .all(|((owner, _), _)| *owner >= super::TILES));
    }
    Ok(())
}
