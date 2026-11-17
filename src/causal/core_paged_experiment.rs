//! Test-only one-tile disk store experiment; see AGENTS.md §Core v2.

use super::geometry::{Geometry, TILES};
use super::state_codec::encode_tile;
use super::storage_paths::digest;
use super::store_api::TileStore;
use super::tile::Tile;
use super::tuning::{self, Latents};
use anyhow::{ensure, Result};
use std::fs;
use std::path::{Path, PathBuf};

#[path = "core_paged_experiment_publish.rs"]
mod publish;
#[path = "core_paged_experiment_read.rs"]
mod read;
#[path = "core_paged_experiment_snapshot.rs"]
mod snapshot;

struct Slot {
    path: PathBuf,
    hash: String,
    version: u64,
    steps: u64,
}

struct Paged {
    directory: PathBuf,
    geometry: Geometry,
    slots: Vec<Slot>,
    generation: u64,
    live: bool,
    peak_live: usize,
    poisoned: bool,
}

impl Paged {
    fn new(directory: &Path, geometry: Geometry, seed: u64) -> Result<Self> {
        geometry.validate()?;
        ensure!(
            tuning::steps()
                .iter()
                .all(|((owner, _), _)| *owner >= TILES),
            "tile latents already resident"
        );
        fs::create_dir(directory)?;
        let mut store = Self {
            directory: directory.into(),
            geometry,
            slots: Vec::new(),
            generation: 0,
            live: false,
            peak_live: 0,
            poisoned: false,
        };
        for id in 0..TILES {
            let tile = Tile::new(geometry, id, seed)?;
            let slot = store.publish(&tile, &[])?;
            store.slots.push(slot);
        }
        Ok(store)
    }

    fn publish(
        &mut self,
        tile: &Tile,
        values: &[((usize, usize), tuning::Latent)],
    ) -> Result<Slot> {
        let tile_bytes = encode_tile(tile)?;
        publish::validate(values)?;
        self.generation = self
            .generation
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("generation overflow"))?;
        let path = self
            .directory
            .join(format!("tile-{}-{}.bin", tile.id, self.generation));
        let partial = path.with_extension("partial");
        let hash = publish::write(&partial, &tile_bytes, values)?;
        fs::rename(&partial, &path)?;
        Ok(Slot {
            path,
            hash,
            version: tile.version,
            steps: values
                .iter()
                .map(|(_, latent)| latent.step)
                .max()
                .unwrap_or(0),
        })
    }

    fn load(&self, id: usize) -> Result<(Tile, Latents)> {
        ensure!(
            !self.poisoned && !self.live && id < self.slots.len(),
            "paged store unavailable"
        );
        read::load(self.geometry, id, &self.slots[id])
    }

    fn enter(&mut self, id: usize, fingerprint: bool) -> Result<(Tile, Option<[u8; 32]>)> {
        let (tile, values) = self.load(id)?;
        let original = fingerprint
            .then(|| snapshot::latent_digest(&values))
            .transpose()?;
        tuning::put_tile(id, values)?;
        self.live = true;
        self.peak_live = self.peak_live.max(1);
        Ok((tile, original))
    }

    fn leave(&mut self, id: usize) -> Latents {
        let values = tuning::take_tile(id);
        self.live = false;
        values
    }
}

impl TileStore for Paged {
    fn versions(&self) -> Result<[u64; TILES]> {
        ensure!(
            !self.poisoned && !self.live && self.slots.len() == TILES,
            "paged store unavailable"
        );
        Ok(std::array::from_fn(|id| self.slots[id].version))
    }

    fn optimizer_steps(&self) -> Result<[u64; TILES]> {
        self.versions()?;
        Ok(std::array::from_fn(|id| self.slots[id].steps))
    }

    fn read<T>(&mut self, id: usize, visit: impl FnOnce(&Tile) -> Result<T>) -> Result<T> {
        let (tile, original) = self.enter(id, true)?;
        let (result, values) = self.finish_visit(id, || visit(&tile));
        let result = result.and_then(|value| {
            ensure!(
                Some(snapshot::latent_digest(&values)?) == original,
                "read changed tile optimizer state"
            );
            Ok(value)
        });
        if result.is_err() {
            self.poisoned = true;
        }
        result
    }

    fn update<T>(&mut self, id: usize, visit: impl FnOnce(&mut Tile) -> Result<T>) -> Result<T> {
        let (mut tile, _) = self.enter(id, false)?;
        let (result, values) = self.finish_visit(id, || visit(&mut tile));
        let committed = (|| {
            let result = result?;
            ensure!(
                tile.id == id && tile.geometry == self.geometry,
                "tile identity changed"
            );
            let slot = self.publish(&tile, &values)?;
            let old = std::mem::replace(&mut self.slots[id], slot);
            fs::remove_file(old.path)?;
            Ok(result)
        })();
        if committed.is_err() {
            self.poisoned = true;
        }
        committed
    }
}

#[path = "execution_fixture.rs"]
#[allow(clippy::duplicate_mod, dead_code)]
mod fixture;

#[test]
fn one_tile_disk_experiment_matches_resident_updates() -> Result<()> {
    use super::{resident::ResidentStore, state::State, tuning::Knobs};
    let geometry = Geometry::fixture();
    for (batch, float_forward, threaded) in [(4, false, false), (8, false, true), (1, true, false)]
    {
        let knobs = Knobs {
            batch,
            float_forward,
            history_ticks: 8,
            fit_predictor: false,
            explore: 0.7,
            ..Knobs::core(0.003, 7)
        };
        let _guard = tuning::scoped(knobs);
        let source = fixture::Source::new();
        let mut state = State::with_cursor(geometry, 7, source.cursor())?;
        state.anchor_only = true;
        let mut resident = ResidentStore::new(geometry, 7)?;
        for _ in 0..3 {
            super::core_batch::step(&mut state, &mut resident, &source, threaded)?;
        }
        let expected = snapshot::fingerprint_hashes(&state, &resident)?;
        let expected_state =
            serde_json::to_vec(&super::checkpoint_state::StateRecord::capture(&state)?)?;
        drop(resident);
        drop(state);
        tuning::install(knobs);
        let source = fixture::Source::new();
        let mut state = State::with_cursor(geometry, 7, source.cursor())?;
        state.anchor_only = true;
        let scratch =
            super::scratch_dir::ScratchDir::new(&format!("core-paged-experiment-{batch}"))?;
        let mut paged = Paged::new(&scratch.path().join("store"), geometry, 7)?;
        for _ in 0..3 {
            super::core_batch::step(&mut state, &mut paged, &source, threaded)?;
        }
        assert_eq!(paged.peak_live, 1);
        assert!(tuning::steps()
            .iter()
            .all(|((owner, _), _)| *owner >= TILES));
        let (resident, mut all) = paged.snapshot()?;
        let (mut shared, calls) = tuning::export();
        all.append(&mut shared);
        all.sort_by_key(|(key, _)| *key);
        tuning::import(all, calls);
        assert_eq!(snapshot::fingerprint_hashes(&state, &resident)?, expected);
        assert_eq!(
            serde_json::to_vec(&super::checkpoint_state::StateRecord::capture(&state)?)?,
            expected_state
        );
        fs::write(&paged.slots[super::execution::START_TILE].path, b"corrupt")?;
        assert!(super::core_batch::step(&mut state, &mut paged, &source, false).is_err());
        assert_eq!(state.boundary, super::state::Boundary::Poisoned);
    }
    Ok(())
}
