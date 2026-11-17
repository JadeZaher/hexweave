//! Serial visits and owned runtime cleanup for connected storage.

use super::super::{
    connected_checkpoint_io as io, geometry::TILES, store_api::TileStore, tile::Tile, tuning,
};
use super::{codec, Geometry, Latents, Path, Slot, Store, TileMeta};
use anyhow::{bail, ensure, Result};
use std::path::PathBuf;
use std::time::Instant;

impl Store {
    pub(super) fn remove_obsolete(&self) -> Result<()> {
        let root = self.root.canonicalize()?;
        for entry in std::fs::read_dir(&root)? {
            let entry = entry?;
            if !entry.file_type()?.is_file() {
                continue;
            }
            let name = entry.file_name();
            let name = name.to_string_lossy();
            let blob = name.len() == 68
                && name.ends_with(".bin")
                && name[..64].bytes().all(|byte| byte.is_ascii_hexdigit());
            let partial = name.ends_with(".partial")
                && (name.starts_with("tile-") || name.starts_with("restore-"))
                && name
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.'));
            if !(blob || partial) {
                continue;
            }
            let path = entry.path().canonicalize()?;
            ensure!(
                path.parent() == Some(root.as_path()),
                "connected cleanup escaped runtime root"
            );
            if !self
                .slots
                .iter()
                .any(|slot| slot.path.file_name() == entry.path().file_name())
            {
                io::retry(|| std::fs::remove_file(&path))?;
            }
        }
        Ok(())
    }
}

impl TileStore for Store {
    fn versions(&self) -> Result<[u64; TILES]> {
        Store::versions(self)
    }

    fn optimizer_steps(&self) -> Result<[u64; TILES]> {
        Store::optimizer_steps(self)
    }

    fn read<T>(&mut self, id: usize, visit: impl FnOnce(&Tile) -> Result<T>) -> Result<T> {
        ensure!(
            !tuning::current().float_forward,
            "connected reads require ternary forward"
        );
        let lease = self.checkout(id)?;
        if !tuning::take_tile(id).is_empty() {
            self.abort(id);
            bail!("unexpected optimizer state before connected read");
        }
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| visit(lease.tile())));
        let changed = !tuning::take_tile(id).is_empty();
        match result {
            Ok(Ok(value)) if !changed => {
                self.return_read(lease)?;
                Ok(value)
            }
            Ok(Ok(_)) => {
                self.abort(id);
                bail!("read created connected optimizer state")
            }
            Ok(Err(error)) => {
                self.abort(id);
                Err(error)
            }
            Err(error) => {
                self.abort(id);
                std::panic::resume_unwind(error)
            }
        }
    }

    fn update<T>(&mut self, id: usize, visit: impl FnOnce(&mut Tile) -> Result<T>) -> Result<T> {
        self.visit(id, visit)
    }
}

impl Store {
    fn visit<T>(&mut self, id: usize, visit: impl FnOnce(&mut Tile) -> Result<T>) -> Result<T> {
        ensure!(
            self.precision.is_ternary(),
            "FP32 updates require an explicit mutable lease"
        );
        let mut lease = self.checkout(id)?;
        let (tile, latents) = match lease.parts_mut() {
            Ok(parts) => parts,
            Err(error) => {
                self.abort(id);
                return Err(error);
            }
        };
        if let Err(error) = tuning::put_tile(id, std::mem::take(latents)) {
            tuning::take_tile(id);
            self.abort(id);
            return Err(error);
        }
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| visit(tile)));
        *latents = tuning::take_tile(id);
        match result {
            Ok(Ok(value)) => {
                self.commit(lease)?;
                Ok(value)
            }
            Ok(Err(error)) => {
                self.abort(id);
                Err(error)
            }
            Err(error) => {
                self.abort(id);
                std::panic::resume_unwind(error)
            }
        }
    }
}

impl Store {
    pub fn restore(
        root: &Path,
        geometry: Geometry,
        workers: usize,
        slots: Vec<Slot>,
        precision: super::super::connected_precision::Precision,
    ) -> Result<Self> {
        ensure!(slots.len() == TILES, "connected tile count mismatch");
        let mut store = Self::empty(root, geometry, workers)?;
        store.precision = precision;
        store.slots = slots;
        for id in 0..TILES {
            let lease = store.checkout(id)?;
            store.return_read(lease)?;
        }
        for id in 0..TILES {
            let slot = &store.slots[id];
            let partial = store.root.join(format!("restore-{id}.partial"));
            let hash = io::copy(&slot.path, &partial, codec::cap(geometry, id)?)?;
            ensure!(hash == slot.meta.hash, "connected restore copy changed");
            let path = store.root.join(format!("{hash}.bin"));
            io::retry(|| std::fs::rename(&partial, &path))?;
            store.slots[id].path = path;
        }
        store.remove_obsolete()?;
        Ok(store)
    }

    pub(super) fn write(
        &mut self,
        tile: &Tile,
        latents: &Latents,
        last_direction: &[i8],
        revision: u64,
    ) -> Result<Slot> {
        let start = Instant::now();
        let partial = self
            .root
            .join(format!("tile-{}-{revision}.partial", tile.id));
        let (hash, bytes) = codec::write(&partial, tile, latents, last_direction)?;
        let path = self.place(&partial, tile.id, &hash)?;
        self.stats.writes += 1;
        self.stats.written_bytes += bytes;
        self.stats.write_seconds += start.elapsed().as_secs_f64();
        Ok(Slot {
            path,
            meta: TileMeta {
                hash,
                version: tile.version,
                steps: latents
                    .iter()
                    .map(|(_, state)| state.step)
                    .max()
                    .unwrap_or(0),
                revision,
            },
        })
    }

    /// Renames a synced partial to its content-addressed blob, reusing a verified duplicate.
    pub(super) fn place(&self, partial: &Path, id: usize, hash: &str) -> Result<PathBuf> {
        let path = self.root.join(format!("{hash}.bin"));
        if path.exists() {
            ensure!(
                io::hash(&path, codec::cap(self.geometry, id)?)? == hash,
                "existing connected blob is corrupt"
            );
            std::fs::remove_file(partial)?;
        } else {
            std::fs::rename(partial, &path)?;
        }
        Ok(path)
    }
}
