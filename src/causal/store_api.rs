use super::geometry::TILES;
use super::tile::Tile;
use anyhow::Result;

/// Keep decoded tile ownership inside each storage operation.
pub(super) trait TileStore {
    fn versions(&self) -> Result<[u64; TILES]>;
    fn read<T>(&mut self, id: usize, visit: impl FnOnce(&Tile) -> Result<T>) -> Result<T>;
    fn update<T>(&mut self, id: usize, visit: impl FnOnce(&mut Tile) -> Result<T>) -> Result<T>;
}
