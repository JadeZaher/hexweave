//! All-tiles-resident store for the core path; checkpoints persist it whole. See AGENTS.md §core.

use super::geometry::{Geometry, TILES};
use super::store_api::TileStore;
use super::tile::Tile;
use anyhow::{ensure, Result};

pub(super) struct ResidentStore {
    // Sparse route visits select computation; this Vec still owns every packed tile.
    pub tiles: Vec<Tile>,
}

impl ResidentStore {
    /// Fresh seeded tiles, identical to every other store's initialization.
    pub fn new(geometry: Geometry, seed: u64) -> Result<Self> {
        let tiles = (0..TILES)
            .map(|id| Tile::new(geometry, id, seed))
            .collect::<Result<_>>()?;
        Ok(Self { tiles })
    }

    /// Adopt restored tiles, checking order and geometry.
    pub fn from_tiles(tiles: Vec<Tile>, geometry: Geometry) -> Result<Self> {
        ensure!(
            tiles.len() == TILES
                && tiles
                    .iter()
                    .enumerate()
                    .all(|(id, tile)| tile.id == id && tile.geometry == geometry),
            "resident tile set mismatch"
        );
        Ok(Self { tiles })
    }
}

impl TileStore for ResidentStore {
    fn versions(&self) -> Result<[u64; TILES]> {
        Ok(std::array::from_fn(|id| self.tiles[id].version))
    }

    fn optimizer_steps(&self) -> Result<[u64; TILES]> {
        Ok(super::core_schedule::tile_steps())
    }

    fn read<T>(&mut self, id: usize, visit: impl FnOnce(&Tile) -> Result<T>) -> Result<T> {
        visit(&self.tiles[id])
    }

    fn update<T>(&mut self, id: usize, visit: impl FnOnce(&mut Tile) -> Result<T>) -> Result<T> {
        // Publication happens at checkpoints; returning from this callback does not evict.
        visit(&mut self.tiles[id])
    }
}
