//! Visit-frequency eviction and durable dirty-bundle publication; see CONNECTED.md.

use super::super::{connected_checkpoint_io as io, connected_store_read};
use super::{Lease, Store};
use anyhow::{ensure, Result};
use std::time::Instant;

impl Store {
    pub(super) fn residency(&mut self) {
        self.stats.current_resident = self.cached.len() + self.stats.current_live;
        self.stats.current_view_bytes = self
            .cached
            .iter()
            .map(|lease| lease.tile.weights.as_ref().map_or(0, |w| w.bytes()))
            .sum::<usize>()
            + self
                .leased
                .iter()
                .enumerate()
                .filter(|(_, live)| **live && !self.precision.is_ternary())
                .map(|(id, _)| self.geometry.tile_coefficients(id).unwrap_or(0) * 4)
                .sum::<usize>();
        self.stats.peak_view_bytes = self
            .stats
            .peak_view_bytes
            .max(self.stats.current_view_bytes);
        self.stats.peak_resident = self.stats.peak_resident.max(self.stats.current_resident);
        debug_assert!(self.stats.current_resident <= self.resident);
    }

    pub fn checkout(&mut self, id: usize) -> Result<Lease> {
        ensure!(
            !self.poisoned && id < self.slots.len(),
            "connected store unavailable"
        );
        ensure!(
            !self.leased[id] && self.stats.current_live < self.workers,
            "connected tile lease collision or residency limit"
        );
        let lease = if let Some(index) = self.cached.iter().position(|lease| lease.tile.id == id) {
            self.stats.cache_hits += 1;
            self.cached.remove(index)
        } else {
            if self.cached.len() + self.stats.current_live == self.resident {
                self.evict()?;
            }
            self.stats.cache_misses += 1;
            self.load(id)?
        };
        self.warm(id);
        self.leased[id] = true;
        self.stats.current_live += 1;
        self.stats.peak_live = self.stats.peak_live.max(self.stats.current_live);
        self.residency();
        Ok(lease)
    }

    fn load(&mut self, id: usize) -> Result<Lease> {
        let mut bundle = match self.take_pending(id)? {
            Some(bundle) => bundle,
            None => {
                let start = Instant::now();
                let slot = &self.slots[id];
                let result = connected_store_read::load(&slot.path, self.geometry, id, &slot.meta);
                self.stats.read_seconds += start.elapsed().as_secs_f64();
                match result {
                    Ok(value) => {
                        self.stats.reads += 1;
                        self.stats.read_bytes += value.bytes;
                        value
                    }
                    Err(error) => {
                        self.poisoned = true;
                        return Err(error);
                    }
                }
            }
        };
        if !self.precision.is_ternary() {
            if let Err(error) = super::super::connected_precision::attach_tile(
                &mut bundle.tile,
                &bundle.latents,
                self.view,
            ) {
                self.poisoned = true;
                return Err(error);
            }
        }
        Ok(Lease {
            tile: bundle.tile,
            latents: bundle.latents,
            id,
            source: self.slots[id].meta.clone(),
            before_trits: None,
            last_direction: bundle.last_direction,
        })
    }

    pub(super) fn retain(&mut self, id: usize, mut lease: Lease) -> Result<()> {
        lease.before_trits = None;
        lease.source = self.slots[id].meta.clone();
        if self.eager {
            if let Err(error) = self.persist(&mut lease) {
                self.abort(id);
                return Err(error);
            }
            self.release(id);
        } else {
            self.release(id);
            self.cached.push(lease);
            self.residency();
        }
        Ok(())
    }

    pub(super) fn evict(&mut self) -> Result<()> {
        ensure!(!self.cached.is_empty(), "no idle connected cache slot");
        let index = self.coldest();
        let mut lease = self.cached.remove(index);
        let result = if self.write_behind > 0 {
            self.defer(&lease)
        } else {
            self.persist(&mut lease)
        };
        if let Err(error) = result {
            self.cached.insert(index, lease);
            self.poisoned = true;
            return Err(error);
        }
        self.stats.evictions += 1;
        self.residency();
        Ok(())
    }

    fn persist(&mut self, lease: &mut Lease) -> Result<()> {
        let id = lease.tile.id;
        if !self.slots[id].meta.hash.is_empty() {
            return Ok(());
        }
        let slot = self.write(
            &lease.tile,
            &lease.latents,
            &lease.last_direction,
            self.slots[id].meta.revision,
        )?;
        self.install(id, slot)?;
        lease.source = self.slots[id].meta.clone();
        Ok(())
    }

    /// Replaces a dirty slot with its durable publication and retires the superseded blob.
    pub(super) fn install(&mut self, id: usize, slot: super::Slot) -> Result<()> {
        let meta = &self.slots[id].meta;
        ensure!(
            meta.hash.is_empty()
                && slot.meta.version == meta.version
                && slot.meta.steps == meta.steps
                && slot.meta.revision == meta.revision,
            "dirty connected cache metadata mismatch"
        );
        let old = std::mem::replace(&mut self.slots[id], slot);
        self.stats.dirty_flushes += 1;
        if old.path.parent() == Some(self.root.as_path())
            && !self.slots.iter().any(|slot| slot.path == old.path)
        {
            if let Err(error) = io::retry(|| std::fs::remove_file(&old.path)) {
                eprintln!(
                    "connected obsolete blob retained {}: {error}",
                    old.path.display()
                );
            }
        }
        Ok(())
    }

    pub fn set_view(&mut self, view: super::super::connected_precision::View) -> Result<()> {
        self.quiescent()?;
        ensure!(
            !self.precision.is_ternary(),
            "quantized views require FP32 masters"
        );
        ensure!(
            self.pending.is_empty() && self.slots.iter().all(|slot| !slot.meta.hash.is_empty()),
            "diagnostic view requires durable frozen bundles"
        );
        self.cached.clear();
        self.view = view;
        self.residency();
        Ok(())
    }

    pub fn flush(&mut self) -> Result<()> {
        self.quiescent()?;
        self.drain()?;
        for index in 0..self.cached.len() {
            let mut lease = self.cached.remove(index);
            let result = self.persist(&mut lease);
            self.cached.insert(index, lease);
            if let Err(error) = result {
                self.poisoned = true;
                return Err(error);
            }
        }
        ensure!(
            self.slots.iter().all(|slot| !slot.meta.hash.is_empty()),
            "connected dirty tile missing from cache"
        );
        Ok(())
    }

    #[cfg(test)]
    pub fn set_eager(&mut self, eager: bool) -> Result<()> {
        self.flush()?;
        if eager {
            self.cached.clear();
            self.residency();
        }
        self.eager = eager;
        Ok(())
    }
}
