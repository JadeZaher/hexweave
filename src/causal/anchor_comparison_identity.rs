use crate::causal::execution::checked_tile;
use crate::causal::state::State;
use crate::causal::store_api::TileStore;
use anyhow::{ensure, Result};
use sha2::{Digest, Sha256};
use std::io::{self, Write};

struct HashWriter(Sha256);

impl Write for HashWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0.update(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// Hash fixture state exactly without asserting production admission.
pub(super) fn identity<S: TileStore>(state: &State, store: &mut S) -> Result<String> {
    ensure!(store.versions()? == state.tile_versions, "fixture versions");
    let mut writer = HashWriter(Sha256::new());
    writer.write_all(b"anchor-comparison-v3-test-only-identity-v1")?;
    serde_json::to_writer(
        &mut writer,
        &(
            state.geometry,
            state.seed,
            state.global_opportunity,
            &state.cursor,
            &state.counters,
            state.boundary,
            state.tile_versions,
        ),
    )?;
    writer.write_all(&state.baseline.to_bits().to_le_bytes())?;
    writer.write_all(&state.shared.version.to_le_bytes())?;
    writer.write_all(state.shared.packed())?;
    writer.write_all(&state.predictor.version.to_le_bytes())?;
    writer.write_all(state.predictor.packed())?;
    writer.write_all(&(state.history.records.len() as u64).to_le_bytes())?;
    for record in &state.history.records {
        serde_json::to_writer(
            &mut writer,
            &(
                record.tile_versions,
                record.shared_version,
                record.opportunity,
                record.tile_id,
            ),
        )?;
        writer.write_all(&(record.features.len() as u64).to_le_bytes())?;
        writer.write_all(&(record.correction.len() as u64).to_le_bytes())?;
        for value in record.features.iter().chain(&record.correction) {
            writer.write_all(&value.to_bits().to_le_bytes())?;
        }
    }
    for id in 0..7 {
        store.read(id, |tile| {
            checked_tile(tile, state.geometry, id, state.tile_versions[id])?;
            for packed in tile.packed() {
                writer.write_all(packed)?;
            }
            Ok(())
        })?;
    }
    Ok(format!("{:x}", writer.0.finalize()))
}
