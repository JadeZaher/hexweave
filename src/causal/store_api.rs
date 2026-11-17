use super::geometry::TILES;
use super::tile::Tile;
use anyhow::Result;

/// Keep decoded tile ownership inside each storage operation.
pub(super) trait TileStore {
    fn versions(&self) -> Result<[u64; TILES]>;
    fn optimizer_steps(&self) -> Result<[u64; TILES]>;
    // The callback permits eviction after a visit; each implementation chooses residency.
    fn read<T>(&mut self, id: usize, visit: impl FnOnce(&Tile) -> Result<T>) -> Result<T>;
    fn update<T>(&mut self, id: usize, visit: impl FnOnce(&mut Tile) -> Result<T>) -> Result<T>;
}
