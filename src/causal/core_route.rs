//! Frozen v2 route with training-only exploration; see AGENTS.md §Core v2.

use super::execution::{checked_tile, destination, Frame, Trace, START_TILE};
use super::geometry::{TILES, VISITS};
use super::resident::ResidentStore;
use super::shared::Prefix;
use super::state::State;
use super::{core_schedule, tuning};
use crate::topology::Mesh;
use anyhow::Result;

pub(super) fn forward(
    state: &State,
    store: &ResidentStore,
    prefix: &Prefix,
    history: Vec<f32>,
    anchor: u64,
    counts: &[u64; TILES],
) -> Result<Trace> {
    let mesh = Mesh::new(1)?;
    let knobs = tuning::current();
    let (mut id, mut input) = (START_TILE, prefix.initial(state.geometry).to_vec());
    let mut frames = Vec::with_capacity(VISITS);
    for depth in 0..VISITS {
        let tile = &store.tiles[id];
        checked_tile(tile, state.geometry, id, state.tile_versions[id])?;
        let forward = tile.forward(&input, &prefix.memory, &history)?;
        let action = if depth + 1 == VISITS {
            None
        } else {
            let greedy = tile.route(&forward, &history, &mut None)?;
            Some(
                core_schedule::explore(knobs, anchor, depth, &mesh.neighbors(id)?, counts)
                    .unwrap_or(greedy),
            )
        };
        frames.push(Frame {
            input: std::mem::replace(&mut input, forward.output),
            tile_id: id,
            version: tile.version,
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
        tile_versions: state.tile_versions,
    })
}
