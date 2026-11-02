//! Ordered position gradients with at most four scoped workers; see AGENTS.md §Core v2.

use super::core_batch::Position;
use super::execution::checked_tile;
use super::geometry::Geometry;
use super::tile::Tile;
use super::tile_gradients::TileGradients;
use super::tuning::{self, Knobs};
use anyhow::{ensure, Result};

fn position(tile: &Tile, position: &Position, baseline: f32) -> Result<TileGradients> {
    let mut gradient = TileGradients::zeros(tile.geometry, tile.id)?;
    let advantage = position.credit.loss - baseline;
    ensure!(advantage.is_finite(), "invalid route advantage");
    for (frame, correction) in position
        .trace
        .frames
        .iter()
        .zip(&position.credit.corrections)
    {
        if frame.tile_id != tile.id {
            continue;
        }
        checked_tile(tile, tile.geometry, frame.tile_id, frame.version)?;
        tile.backward(
            &frame.input,
            &position.prefix.memory,
            &position.trace.history,
            correction,
            None,
            Some(&mut gradient),
            frame.action.map(|action| (action, advantage)),
        )?;
    }
    Ok(gradient)
}

pub(super) fn mean(
    tile: &Tile,
    positions: &[Position],
    baseline: f32,
    knobs: Knobs,
    threaded: bool,
) -> Result<TileGradients> {
    let gradients = if threaded && positions.len() > 1 {
        ensure!(!knobs.float_forward, "workers cannot access float latents");
        std::thread::scope(|scope| -> Result<Vec<TileGradients>> {
            let workers = positions.len().min(4);
            let handles: Vec<_> = (0..workers)
                .map(|worker| {
                    scope.spawn(move || -> Result<Vec<(usize, TileGradients)>> {
                        tuning::install(knobs);
                        (worker..positions.len())
                            .step_by(workers)
                            .map(|index| Ok((index, position(tile, &positions[index], baseline)?)))
                            .collect()
                    })
                })
                .collect();
            let mut indexed = Vec::with_capacity(positions.len());
            let mut error = None;
            for handle in handles {
                match handle
                    .join()
                    .map_err(|_| anyhow::anyhow!("gradient worker panicked"))
                    .and_then(|result| result)
                {
                    Ok(values) => indexed.extend(values),
                    Err(found) => error = Some(found),
                }
            }
            if let Some(error) = error {
                return Err(error);
            }
            indexed.sort_by_key(|(index, _)| *index);
            Ok(indexed.into_iter().map(|(_, gradient)| gradient).collect())
        })?
    } else {
        positions
            .iter()
            .map(|p| position(tile, p, baseline))
            .collect::<Result<Vec<_>>>()?
    };
    reduce(gradients, tile.geometry, tile.id)
}

fn reduce(gradients: Vec<TileGradients>, geometry: Geometry, id: usize) -> Result<TileGradients> {
    let count = gradients.len();
    let mut iter = gradients.into_iter();
    let mut sum = iter.next().ok_or_else(|| anyhow::anyhow!("empty batch"))?;
    for gradient in iter {
        for (output, values) in sum.values.iter_mut().zip(gradient.values) {
            for (value, add) in output.iter_mut().zip(values) {
                *value += add;
            }
        }
    }
    if count > 1 {
        for values in &mut sum.values {
            for value in values {
                *value /= count as f32;
            }
        }
    }
    sum.validate(geometry, id)?;
    Ok(sum)
}
