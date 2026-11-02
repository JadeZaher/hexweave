use super::cursor::{AnchorTarget, Cursor};
use super::geometry::{Geometry, TILES, VISITS};
use super::shared::Prefix;
use super::store_api::TileStore;
use super::tile::Tile;
use crate::topology::Mesh;
use anyhow::{ensure, Result};

/// Every route enters the mesh here: training, evaluation and generation alike.
pub(super) const START_TILE: usize = 3;

pub(super) trait TrainingSource {
    fn validate(&self, cursor: &Cursor) -> Result<()>;
    fn pending_prefix(&self, cursor: &Cursor) -> Result<Vec<u8>>;
    fn anchor_target(&self, cursor: &Cursor) -> Result<AnchorTarget>;
    fn advance_anchor(&self, cursor: &mut Cursor) -> Result<()>;
}

pub(super) struct Frame {
    pub input: Vec<f32>,
    pub tile_id: usize,
    pub version: u64,
    pub action: Option<usize>,
    pub depth: usize,
}

pub(super) struct Trace {
    pub frames: Vec<Frame>,
    pub history: Vec<f32>,
    pub output: Vec<f32>,
    pub tile_versions: [u64; TILES],
}

#[derive(Debug)]
pub(super) enum Step {
    #[cfg(test)]
    Stopped,
    #[allow(dead_code)]
    Committed {
        opportunity: u64,
        loss: Option<f32>,
        visits: usize,
        prefix_bytes: usize,
    },
}

pub(super) fn checked_tile(tile: &Tile, geometry: Geometry, id: usize, version: u64) -> Result<()> {
    ensure!(
        tile.id == id && tile.geometry == geometry && tile.version == version,
        "tile identity, geometry or frozen version mismatch"
    );
    Ok(())
}

pub(super) fn destination(mesh: &Mesh, id: usize, action: Option<usize>) -> Result<Option<usize>> {
    let Some(action) = action else {
        return Ok(None);
    };
    let neighbors = mesh.neighbors(id)?;
    ensure!(action <= neighbors.len(), "route action outside legal set");
    Ok(neighbors.get(action).copied())
}

/// Pure realized route trace; weights and forward caches never enter frames.
pub(super) fn forward_path<S: TileStore>(
    store: &mut S,
    geometry: Geometry,
    prefix: &Prefix,
    history: Vec<f32>,
    random: &mut Option<u64>,
    versions: [u64; TILES],
) -> Result<Trace> {
    let mesh = Mesh::new(1)?;
    let (mut id, mut input) = (START_TILE, prefix.initial(geometry).to_vec());
    let mut frames = Vec::with_capacity(VISITS);
    for depth in 0..VISITS {
        let (forward, action) = store.read(id, |tile| {
            checked_tile(tile, geometry, id, versions[id])?;
            let forward = tile.forward(&input, &prefix.memory, &history)?;
            let action = if depth + 1 == VISITS {
                None
            } else {
                Some(tile.route(&forward, &history, random)?)
            };
            Ok((forward, action))
        })?;
        let old_input = std::mem::replace(&mut input, forward.output);
        frames.push(Frame {
            input: old_input,
            tile_id: id,
            version: versions[id],
            action,
            depth,
        });
        match destination(&mesh, id, action)? {
            Some(next) => id = next,
            None => break,
        }
    }
    Ok(Trace {
        frames,
        history,
        output: input,
        tile_versions: versions,
    })
}

#[cfg(test)]
pub(super) use super::execution_legacy::step;
