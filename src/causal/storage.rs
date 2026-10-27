use super::geometry::{Geometry, TILES};
use super::profile::{self, Bucket};
use super::state_codec::{decode_tile, encode_tile};
use super::storage_paths as paths;
use super::store_api::TileStore;
use super::tile::Tile;
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[path = "storage_ledger.rs"]
mod ledger;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Blob {
    pub file: String,
    pub sha256: String,
    pub bytes: u64,
}

impl Blob {
    pub fn validate(&self, prefix: &str) -> Result<()> {
        ensure!(
            paths::hash_valid(&self.sha256) && self.bytes > 0 && self.bytes <= paths::FILE_CAP,
            "invalid immutable blob identity"
        );
        ensure!(
            self.file == format!("{prefix}-{}.bin", self.sha256),
            "noncanonical immutable blob name"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct TileRef {
    pub id: usize,
    pub version: u64,
    pub blob: Blob,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct IoCounters {
    pub loads: u64,
    pub saves: u64,
    pub bytes_read: u64,
    pub bytes_written: u64,
    pub peak_live_tiles: usize,
    pub current_disk_bytes: u64,
    pub peak_disk_bytes: u64,
}

pub(super) struct DiskStore {
    pub directory: PathBuf,
    pub geometry: Geometry,
    pub seed: u64,
    pub disk_budget_bytes: u64,
    pub io: IoCounters,
    pub(super) tiles: Vec<TileRef>,
    live_tiles: usize,
    poisoned: bool,
    ledger: Option<ledger::Ledger>,
}

fn add(value: &mut u64, amount: u64) -> Result<()> {
    *value = value
        .checked_add(amount)
        .context("storage counter overflow")?;
    Ok(())
}

impl DiskStore {
    pub fn create(
        workspace: &Path,
        attempt: &Path,
        geometry: Geometry,
        seed: u64,
        budget: u64,
    ) -> Result<Self> {
        let mut store = Self::empty(workspace, attempt, geometry, seed, budget)?;
        for id in 0..TILES {
            let tile = Tile::new(geometry, id, seed)?;
            store.live_tiles = 1;
            store.io.peak_live_tiles = 1;
            let blob = store.write_blob("tile", &encode_tile(&tile)?)?;
            store.tiles.push(TileRef {
                id,
                version: 0,
                blob,
            });
            drop(tile);
            store.live_tiles = 0;
        }
        Ok(store)
    }

    pub(super) fn empty(
        workspace: &Path,
        attempt: &Path,
        geometry: Geometry,
        seed: u64,
        budget: u64,
    ) -> Result<Self> {
        geometry.validate()?;
        ensure!(
            (1_048_576..=paths::DISK_CAP).contains(&budget),
            "invalid explicit persistence disk budget"
        );
        let attempt = paths::attempt(workspace, attempt)?;
        let directory = attempt.join("persistence");
        std::fs::create_dir(&directory).context("persistence directory must be fresh")?;
        let marker = format!("causal-persistence-v1\n{}\n", attempt.display());
        paths::publish(&directory, "OWNED", marker.as_bytes(), budget)?;
        let mut store = Self {
            directory,
            geometry,
            seed,
            disk_budget_bytes: budget,
            io: IoCounters::default(),
            tiles: Vec::with_capacity(TILES),
            live_tiles: 0,
            poisoned: false,
            ledger: None,
        };
        store.reconcile()?;
        Ok(store)
    }

    pub fn references(&self) -> Result<Vec<TileRef>> {
        self.versions()?;
        Ok(self.tiles.clone())
    }

    pub(super) fn write_blob(&mut self, prefix: &str, bytes: &[u8]) -> Result<Blob> {
        let sha256 = paths::digest(bytes);
        let blob = Blob {
            file: format!("{prefix}-{sha256}.bin"),
            sha256,
            bytes: bytes.len() as u64,
        };
        blob.validate(prefix)?;
        self.write_named(&blob.file, bytes)?;
        Ok(blob)
    }

    pub(super) fn write_named(&mut self, name: &str, bytes: &[u8]) -> Result<()> {
        profile::time(Bucket::StoreWrite, || {
            let result = self.write_ledgered(name, bytes);
            if result.is_err() {
                self.invalidate_ledger();
            }
            result
        })
    }

    pub(super) fn read_blob(&mut self, blob: &Blob, prefix: &str) -> Result<Vec<u8>> {
        let bytes = profile::time(Bucket::StoreRead, || {
            read_blob(&self.directory, blob, prefix)
        })?;
        add(&mut self.io.loads, 1)?;
        add(&mut self.io.bytes_read, blob.bytes)?;
        Ok(bytes)
    }

    fn load(&mut self, id: usize) -> Result<Tile> {
        ensure!(
            !self.poisoned && self.live_tiles == 0 && id < self.tiles.len(),
            "store unavailable or tile out of range"
        );
        let reference = self.tiles[id].clone();
        ensure!(reference.id == id, "tile reference order mismatch");
        let bytes = self.read_blob(&reference.blob, "tile")?;
        let tile = decode_tile(&bytes, self.geometry, id, reference.version)?;
        self.live_tiles = 1;
        self.io.peak_live_tiles = self.io.peak_live_tiles.max(1);
        Ok(tile)
    }
}

pub(super) fn read_blob(directory: &Path, blob: &Blob, prefix: &str) -> Result<Vec<u8>> {
    blob.validate(prefix)?;
    let bytes = paths::read(directory, &blob.file, blob.bytes)?;
    ensure!(
        bytes.len() as u64 == blob.bytes && paths::digest(&bytes) == blob.sha256,
        "immutable blob length/hash mismatch"
    );
    Ok(bytes)
}

impl TileStore for DiskStore {
    fn versions(&self) -> Result<[u64; TILES]> {
        ensure!(
            !self.poisoned && self.live_tiles == 0 && self.tiles.len() == TILES,
            "incomplete or poisoned tile store"
        );
        for (id, reference) in self.tiles.iter().enumerate() {
            ensure!(reference.id == id, "tile reference order mismatch");
            reference.blob.validate("tile")?;
        }
        Ok(std::array::from_fn(|id| self.tiles[id].version))
    }

    fn read<T>(&mut self, id: usize, visit: impl FnOnce(&Tile) -> Result<T>) -> Result<T> {
        let tile = self.load(id)?;
        let result = visit(&tile);
        drop(tile);
        self.live_tiles = 0;
        result
    }

    fn update<T>(&mut self, id: usize, visit: impl FnOnce(&mut Tile) -> Result<T>) -> Result<T> {
        let mut tile = self.load(id)?;
        let previous = self.tiles[id].clone();
        let original: Vec<_> = tile.packed().iter().map(|b| paths::digest(b)).collect();
        let result = (|| -> Result<T> {
            let result = visit(&mut tile)?;
            ensure!(
                tile.id == id && tile.geometry == self.geometry,
                "tile update changed identity"
            );
            let bytes = encode_tile(&tile)?;
            let changed = tile
                .packed()
                .iter()
                .map(|b| paths::digest(b))
                .collect::<Vec<_>>()
                != original;
            ensure!(
                (!changed && tile.version == previous.version)
                    || (changed && previous.version.checked_add(1) == Some(tile.version)),
                "tile version does not match mutation"
            );
            let blob = self.write_blob("tile", &bytes)?;
            self.tiles[id] = TileRef {
                id,
                version: tile.version,
                blob,
            };
            Ok(result)
        })();
        drop(tile);
        self.live_tiles = 0;
        if result.is_err() {
            self.poisoned = true;
        }
        result
    }
}
