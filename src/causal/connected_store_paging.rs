//! Write-behind pending bundles and visit-frequency eviction; see CONNECTED.md §Write-behind paging.

use super::super::{connected_store_read, geometry::TILES};
use super::{codec, Lease, Slot, Store, TileMeta};
use anyhow::{ensure, Result};
use serde::Serialize;
use std::time::Instant;

/// Per-checkout decay of each tile's visit heat: a half-life of about 16 checkouts.
const HEAT_DECAY: f64 = 0.957_603_280_698_573_7;

/// Most encoded bundles write-behind may hold in RAM.
pub(in super::super) const MAX_WRITE_BEHIND: usize = 64;

#[derive(Clone, Default, Debug, Serialize)]
/// Session counters reset on resume; persisted per-coefficient directions do not.
pub(in super::super) struct Stats {
    pub current_view_bytes: usize,
    pub peak_view_bytes: usize,
    pub peak_live: usize,
    pub current_live: usize,
    pub peak_resident: usize,
    pub current_resident: usize,
    pub resident_cap: usize,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub evictions: u64,
    pub dirty_flushes: u64,
    pub reads: u64,
    pub writes: u64,
    pub read_bytes: u64,
    pub written_bytes: u64,
    pub read_seconds: f64,
    pub write_seconds: f64,
    pub trit_transitions: u64,
    pub trit_reversals: u64,
    pub writes_deferred: u64,
    pub pending_hits: u64,
    pub pending_peak: usize,
    pub pending_bytes_peak: usize,
}

/// One evicted dirty bundle encoded in RAM, authoritative until published or checked out.
pub(super) struct Pending {
    id: usize,
    meta: TileMeta,
    bytes: Vec<u8>,
}

impl Store {
    /// Sets the decoded-bundle cap (`workers..=TILES`) and the write-behind depth (0..=64).
    pub fn set_paging(&mut self, resident: usize, write_behind: usize) -> Result<()> {
        self.quiescent()?;
        ensure!(
            (self.workers..=TILES).contains(&resident) && write_behind <= MAX_WRITE_BEHIND,
            "connected resident tiles or write-behind depth out of range"
        );
        self.drain()?;
        while self.cached.len() > resident {
            self.evict()?;
        }
        self.drain()?;
        self.resident = resident;
        self.write_behind = write_behind;
        self.stats.resident_cap = resident;
        Ok(())
    }

    /// Counts one checkout of `id` in the exponentially decayed visit heat.
    pub(super) fn warm(&mut self, id: usize) {
        self.heat.iter_mut().for_each(|heat| *heat *= HEAT_DECAY);
        self.heat[id] += 1.0;
    }

    /// The idle cached entry with the least heat; ties keep oldest-first order.
    pub(super) fn coldest(&self) -> usize {
        let mut best = 0;
        for (index, lease) in self.cached.iter().enumerate() {
            if self.heat[lease.tile.id] < self.heat[self.cached[best].tile.id] {
                best = index;
            }
        }
        best
    }

    /// Encodes a dirty evicted bundle into the pending buffer, publishing the oldest when full.
    pub(super) fn defer(&mut self, lease: &Lease) -> Result<()> {
        let id = lease.tile.id;
        if !self.slots[id].meta.hash.is_empty() {
            return Ok(());
        }
        let (hash, bytes) =
            codec::encode_bytes(&lease.tile, &lease.latents, &lease.last_direction)?;
        let meta = TileMeta {
            hash,
            ..self.slots[id].meta.clone()
        };
        ensure!(
            meta.version == lease.tile.version,
            "dirty connected cache metadata mismatch"
        );
        if self.pending.len() == self.write_behind {
            self.publish_oldest()?;
        }
        self.pending.push(Pending { id, meta, bytes });
        self.stats.writes_deferred += 1;
        self.stats.pending_peak = self.stats.pending_peak.max(self.pending.len());
        let held = self
            .pending
            .iter()
            .map(|entry| entry.bytes.capacity())
            .sum();
        self.stats.pending_bytes_peak = self.stats.pending_bytes_peak.max(held);
        Ok(())
    }

    /// Removes and decodes `id`'s pending bytes, if any; a failure poisons the store.
    pub(super) fn take_pending(
        &mut self,
        id: usize,
    ) -> Result<Option<connected_store_read::Bundle>> {
        let Some(index) = self.pending.iter().position(|entry| entry.id == id) else {
            return Ok(None);
        };
        let entry = self.pending.remove(index);
        let result = connected_store_read::load_bytes(&entry.bytes, self.geometry, id, &entry.meta)
            .and_then(|bundle| {
                ensure!(
                    self.slots[id].meta.hash.is_empty()
                        && self.slots[id].meta.version == entry.meta.version
                        && self.slots[id].meta.steps == entry.meta.steps
                        && self.slots[id].meta.revision == entry.meta.revision,
                    "pending connected bundle metadata mismatch"
                );
                Ok(bundle)
            });
        match result {
            Ok(bundle) => {
                self.stats.pending_hits += 1;
                Ok(Some(bundle))
            }
            Err(error) => {
                self.poisoned = true;
                Err(error)
            }
        }
    }

    /// Publishes every pending bundle, oldest first; required before on-disk state is read.
    pub(super) fn drain(&mut self) -> Result<()> {
        while !self.pending.is_empty() {
            self.publish_oldest()?;
        }
        Ok(())
    }

    fn publish_oldest(&mut self) -> Result<()> {
        let entry = self.pending.remove(0);
        let result = self.publish(&entry);
        if let Err(error) = result {
            self.pending.insert(0, entry);
            self.poisoned = true;
            return Err(error);
        }
        Ok(())
    }

    fn publish(&mut self, entry: &Pending) -> Result<()> {
        let start = Instant::now();
        let partial = self
            .root
            .join(format!("tile-{}-{}.partial", entry.id, entry.meta.revision));
        codec::write_bytes(&partial, &entry.bytes)?;
        let path = self.place(&partial, entry.id, &entry.meta.hash)?;
        self.stats.writes += 1;
        self.stats.written_bytes += entry.bytes.len() as u64;
        self.stats.write_seconds += start.elapsed().as_secs_f64();
        let slot = Slot {
            path,
            meta: entry.meta.clone(),
        };
        self.install(entry.id, slot)
    }
}
