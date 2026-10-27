use super::state::State;
use super::store_api::TileStore;
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

/// Hash exact model and continuation values without a second model allocation.
pub(super) fn identity<S: TileStore>(state: &State, store: &mut S) -> Result<String> {
    state.validate_boundary()?;
    ensure!(
        store.versions()? == state.tile_versions,
        "read-only state/store version mismatch"
    );
    let mut writer = HashWriter(Sha256::new());
    if let Some(assay) = &state.assay {
        writer.write_all(b"causal-retained-immutable-v1")?;
        serde_json::to_writer(&mut writer, assay)?;
    } else {
        writer.write_all(b"causal-immutable-state-v1")?;
    }
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
        for value in record.features.iter().chain(&record.correction) {
            writer.write_all(&value.to_bits().to_le_bytes())?;
        }
    }
    for id in 0..7 {
        store.read(id, |tile| {
            super::execution::checked_tile(tile, state.geometry, id, state.tile_versions[id])?;
            for packed in tile.packed() {
                writer.write_all(packed)?;
            }
            Ok(())
        })?;
    }
    Ok(format!("{:x}", writer.0.finalize()))
}
