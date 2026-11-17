//! Ordered position gradients with at most four scoped workers; see AGENTS.md §Core v2.

use super::core_batch::Position;
use super::execution::checked_tile;
use super::geometry::Geometry;
use super::tile::Tile;
use super::tile_gradients::TileGradients;
use super::tuning::{self, Knobs};
use anyhow::{ensure, Result};

/// Sum this position's repeated visits; an unvisited tile contributes a dense zero gradient.
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

/// Reduce in position order, dividing by the whole batch rather than the tile's visit count.
pub(super) fn mean(
    tile: &Tile,
    positions: &[Position],
    baseline: f32,
    knobs: Knobs,
    threaded: bool,
) -> Result<TileGradients> {
    ensure!(!positions.is_empty(), "empty batch");
    let mut sum = None;
    // Four results plus one accumulator bound gradient retention, not the retained batch tapes.
    for chunk in positions.chunks(4) {
        let gradients = gradients(
            tile,
            chunk,
            baseline,
            knobs,
            threaded && positions.len() > 1,
        )?;
        for gradient in gradients {
            if let Some(sum) = &mut sum {
                add(sum, gradient);
            } else {
                sum = Some(gradient);
            }
        }
    }
    finish(sum.unwrap(), positions.len(), tile.geometry, tile.id)
}

fn gradients(
    tile: &Tile,
    positions: &[Position],
    baseline: f32,
    knobs: Knobs,
    threaded: bool,
) -> Result<Vec<TileGradients>> {
    if threaded {
        ensure!(!knobs.float_forward, "workers cannot access float latents");
        // Each group creates fresh workers; scope lifetime is also a thread-creation cost.
        std::thread::scope(|scope| -> Result<Vec<TileGradients>> {
            let handles: Vec<_> = positions
                .iter()
                .map(|position| {
                    scope.spawn(move || -> Result<TileGradients> {
                        tuning::install(knobs);
                        self::position(tile, position, baseline)
                    })
                })
                .collect();
            let mut ordered = Vec::with_capacity(positions.len());
            let mut error = None;
            // Join in position order so scheduling cannot reorder floating-point additions.
            for handle in handles {
                match handle
                    .join()
                    .map_err(|_| anyhow::anyhow!("gradient worker panicked"))
                    .and_then(|result| result)
                {
                    Ok(gradient) => ordered.push(gradient),
                    Err(found) => error = Some(found),
                }
            }
            if let Some(error) = error {
                return Err(error);
            }
            Ok(ordered)
        })
    } else {
        positions
            .iter()
            .map(|p| position(tile, p, baseline))
            .collect::<Result<Vec<_>>>()
    }
}

fn add(sum: &mut TileGradients, gradient: TileGradients) {
    for (output, values) in sum.values.iter_mut().zip(gradient.values) {
        for (value, add) in output.iter_mut().zip(values) {
            *value += add;
        }
    }
}

fn finish(
    mut sum: TileGradients,
    count: usize,
    geometry: Geometry,
    id: usize,
) -> Result<TileGradients> {
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

#[cfg(test)]
pub(super) fn reference(
    tile: &Tile,
    positions: &[Position],
    baseline: f32,
) -> Result<TileGradients> {
    let gradients = positions
        .iter()
        .map(|p| position(tile, p, baseline))
        .collect::<Result<Vec<_>>>()?;
    let mut iter = gradients.into_iter();
    let mut sum = iter.next().ok_or_else(|| anyhow::anyhow!("empty batch"))?;
    for gradient in iter {
        for (output, values) in sum.values.iter_mut().zip(gradient.values) {
            for (value, add) in output.iter_mut().zip(values) {
                *value += add;
            }
        }
    }
    if positions.len() > 1 {
        for values in &mut sum.values {
            for value in values {
                *value /= positions.len() as f32;
            }
        }
    }
    sum.validate(tile.geometry, tile.id)?;
    Ok(sum)
}
