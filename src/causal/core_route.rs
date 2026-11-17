//! Frozen v2 route with training-only exploration; see AGENTS.md §Core v2.

use super::execution::{checked_tile, destination, Frame, Trace, START_TILE};
use super::geometry::{TILES, VISITS};
use super::shared::Prefix;
use super::state::State;
use super::store_api::TileStore;
use super::{core_schedule, tuning};
use crate::topology::Mesh;
use anyhow::Result;

pub(super) fn forward<S: TileStore>(
    state: &State,
    store: &mut S,
    prefix: &Prefix,
    history: Vec<f32>,
    anchor: u64,
    counts: &[u64; TILES],
) -> Result<Trace> {
    let mesh = Mesh::new(1)?;
    let knobs = tuning::current();
    // Begin at the center tile with the final visible byte's embedding/position row.
    let (mut id, mut input) = (START_TILE, prefix.initial(state.geometry).to_vec());
    let mut frames = Vec::with_capacity(VISITS);
    // Visits are sequential: each selected tile transforms the previous tile's output.
    for depth in 0..VISITS {
        let (forward, action, version) = store.read(id, |tile| {
            checked_tile(tile, state.geometry, id, state.tile_versions[id])?;
            let forward = tile.forward(&input, &prefix.memory, &history)?;
            let action = if depth + 1 == VISITS {
                // The fourth visit ends the route regardless of learned halt scores.
                None
            } else {
                let greedy = tile.route(&forward, &history, &mut None)?;
                Some(
                    core_schedule::explore(knobs, anchor, depth, &mesh.neighbors(id)?, counts)
                        .unwrap_or(greedy),
                )
            };
            Ok((forward, action, tile.version))
        })?;
        // Retain replay inputs and route metadata, not per-visit caches or packed weights.
        frames.push(Frame {
            input: std::mem::replace(&mut input, forward.output),
            tile_id: id,
            version,
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
