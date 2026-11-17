//! Bounded tile ownership and write-back caching; see CONNECTED.md.

use super::connected_store_codec as codec;
use super::geometry::{Geometry, TILES};
use super::tile::Tile;
use super::tuning::Latents;
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[path = "connected_store_access.rs"]
mod access;
#[path = "connected_store_cache.rs"]
mod cache;
#[path = "connected_store_fingerprint.rs"]
mod fingerprint;
#[path = "connected_store_paging.rs"]
mod paging;
#[path = "connected_store_tracking.rs"]
mod tracking;

pub(super) use paging::{Stats, MAX_WRITE_BEHIND};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct TileMeta {
    pub hash: String,
    pub version: u64,
    pub steps: u64,
    pub revision: u64,
}

#[derive(Clone)]
pub(super) struct Slot {
    pub path: PathBuf,
    pub meta: TileMeta,
}

pub(super) struct Lease {
    tile: Tile,
    latents: Latents,
    id: usize,
    source: TileMeta,
    before_trits: Option<Vec<i8>>,
    last_direction: Vec<i8>,
}

impl Lease {
    pub fn tile(&self) -> &Tile {
        &self.tile
    }

    pub fn latents(&self) -> &Latents {
        &self.latents
    }

    pub fn parts_mut(&mut self) -> Result<(&mut Tile, &mut Latents)> {
        if self.before_trits.is_none() {
            self.before_trits = Some(tracking::trits(&self.tile)?);
        }
        Ok((&mut self.tile, &mut self.latents))
    }
}

pub(super) struct Store {
    pub geometry: Geometry,
    pub precision: super::connected_precision::Precision,
    pub view: super::connected_precision::View,
    pub slots: Vec<Slot>,
    root: PathBuf,
    workers: usize,
    /// Decoded-bundle cap (leased plus idle cached), `workers` unless `set_paging` raises it.
    resident: usize,
    write_behind: usize,
    pending: Vec<paging::Pending>,
    heat: [f64; TILES],
    leased: [bool; TILES],
    cached: Vec<Lease>,
    eager: bool,
    poisoned: bool,
    stats: Stats,
}

impl Store {
    #[cfg(test)]
    pub fn new(root: &Path, geometry: Geometry, seed: u64, workers: usize) -> Result<Self> {
        Self::new_precision(
            root,
            geometry,
            seed,
            workers,
            super::connected_precision::Precision::Ternary,
        )
    }

    pub fn new_precision(
        root: &Path,
        geometry: Geometry,
        seed: u64,
        workers: usize,
        precision: super::connected_precision::Precision,
    ) -> Result<Self> {
        ensure!(!root.exists(), "connected tile store already exists");
        let mut store = Self::empty(root, geometry, workers)?;
        store.precision = precision;
        for id in 0..TILES {
            let mut tile = Tile::new(geometry, id, seed)?;
            let values = if precision.is_ternary() {
                Vec::new()
            } else {
                super::connected_precision::fresh(&tile)?
            };
            if !precision.is_ternary() {
                super::connected_precision::attach_tile(&mut tile, &values, store.view)?;
            }
            let last_direction = vec![0; geometry.tile_coefficients(id)?];
            store.stats.peak_view_bytes = store
                .stats
                .peak_view_bytes
                .max(tile.weights.as_ref().map_or(0, |w| w.bytes()));
            let slot = store.write(&tile, &values, &last_direction, 0)?;
            store.slots.push(slot);
            store.stats.peak_live = 1;
            store.stats.peak_resident = 1;
        }
        Ok(store)
    }

    fn empty(root: &Path, geometry: Geometry, workers: usize) -> Result<Self> {
        geometry.validate()?;
        ensure!(
            (1..=4).contains(&workers),
            "connected workers must be 1..=4"
        );
        std::fs::create_dir_all(root)?;
        Ok(Self {
            geometry,
            precision: super::connected_precision::Precision::Ternary,
            view: super::connected_precision::View::Fp32,
            slots: Vec::new(),
            root: root.into(),
            workers,
            resident: workers,
            write_behind: 0,
            pending: Vec::new(),
            heat: [0.0; TILES],
            leased: [false; TILES],
            cached: Vec::new(),
            eager: false,
            poisoned: false,
            stats: Stats {
                resident_cap: workers,
                ..Stats::default()
            },
        })
    }

    pub fn stats(&self) -> Stats {
        self.stats.clone()
    }

    pub fn quiescent(&self) -> Result<()> {
        ensure!(
            !self.poisoned && self.stats.current_live == 0 && self.slots.len() == TILES,
            "connected store has active leases or is poisoned"
        );
        Ok(())
    }

    pub fn versions(&self) -> Result<[u64; TILES]> {
        ensure!(
            !self.poisoned && self.slots.len() == TILES,
            "connected store unavailable"
        );
        Ok(std::array::from_fn(|id| self.slots[id].meta.version))
    }

    pub fn optimizer_steps(&self) -> Result<[u64; TILES]> {
        self.versions()?;
        Ok(std::array::from_fn(|id| self.slots[id].meta.steps))
    }

    fn valid_lease(&self, lease: &Lease) -> Result<usize> {
        let id = lease.id;
        ensure!(
            !self.poisoned && id < TILES && self.leased[id],
            "inactive connected lease"
        );
        ensure!(
            lease.tile.id == id
                && lease.tile.geometry == self.geometry
                && self.slots[id].meta == lease.source,
            "stale connected tile lease"
        );
        Ok(id)
    }

    fn release(&mut self, id: usize) {
        self.leased[id] = false;
        self.stats.current_live -= 1;
        self.residency();
    }

    pub fn abort(&mut self, id: usize) {
        self.poisoned = true;
        if id < TILES && self.leased[id] {
            self.release(id);
        }
    }

    pub fn return_read(&mut self, lease: Lease) -> Result<()> {
        let result = (|| {
            let id = self.valid_lease(&lease)?;
            ensure!(
                lease.before_trits.is_none(),
                "read changed connected tile state"
            );
            Ok(id)
        })();
        match result {
            Ok(id) => self.retain(id, lease),
            Err(error) => {
                self.abort(lease.id);
                Err(error)
            }
        }
    }

    pub fn commit(&mut self, mut lease: Lease) -> Result<()> {
        let result = (|| {
            let id = self.valid_lease(&lease)?;
            ensure!(
                lease.tile.version >= lease.source.version,
                "connected tile version regressed"
            );
            let revision = lease
                .source
                .revision
                .checked_add(1)
                .context("connected revision overflow")?;
            let (transitions, reversals) = tracking::record(&mut lease)?;
            codec::validate(&lease.tile, &lease.latents)?;
            if !self.precision.is_ternary() {
                super::connected_precision::validate_masters(&lease.tile, &lease.latents)?;
                ensure!(
                    self.view == super::connected_precision::View::Fp32,
                    "cannot train a quantized diagnostic view"
                );
                let weights = lease
                    .tile
                    .weights
                    .as_ref()
                    .context("missing FP32 committed view")?;
                ensure!(
                    weights.mode == self.view
                        && weights.version == lease.tile.version
                        && weights.matrices.len() == super::geometry::MATRIX_COUNT
                        && weights
                            .matrices
                            .iter()
                            .zip(&lease.latents)
                            .all(|(view, (_, master))| *view == master.weights),
                    "stale FP32 committed view"
                );
            }
            let steps = lease
                .latents
                .iter()
                .map(|(_, state)| state.step)
                .max()
                .unwrap_or(0);
            ensure!(steps >= lease.source.steps, "connected optimizer regressed");
            self.slots[id].meta = TileMeta {
                hash: String::new(),
                version: lease.tile.version,
                steps,
                revision,
            };
            self.stats.trit_transitions += transitions;
            self.stats.trit_reversals += reversals;
            Ok(id)
        })();
        match result {
            Ok(id) => self.retain(id, lease),
            Err(error) => {
                self.abort(lease.id);
                Err(error)
            }
        }
    }
}
