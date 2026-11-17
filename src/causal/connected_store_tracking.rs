//! Per-coefficient trit direction history; see AGENTS.md for resumed telemetry scope.

use super::super::tile::Tile;
use super::Lease;
use anyhow::{ensure, Result};

pub(super) fn trits(tile: &Tile) -> Result<Vec<i8>> {
    let mut values = Vec::with_capacity(tile.geometry.tile_coefficients(tile.id)?);
    for matrix in &tile.matrices {
        for index in 0..matrix.len() {
            values.push(matrix.value(index)?);
        }
    }
    Ok(values)
}

pub(super) fn record(lease: &mut Lease) -> Result<(u64, u64)> {
    let Some(before_trits) = lease.before_trits.as_ref() else {
        return Ok((0, 0));
    };
    let count = lease.tile.geometry.tile_coefficients(lease.tile.id)?;
    ensure!(
        before_trits.len() == count && lease.last_direction.len() == count,
        "trit direction tracking shape mismatch"
    );
    let (mut transitions, mut reversals, mut offset) = (0, 0, 0);
    for matrix in &lease.tile.matrices {
        for index in 0..matrix.len() {
            let direction = (matrix.value(index)? - before_trits[offset]).signum();
            if direction != 0 {
                transitions += 1;
                if lease.last_direction[offset] == -direction {
                    reversals += 1;
                }
                lease.last_direction[offset] = direction;
            }
            offset += 1;
        }
    }
    Ok((transitions, reversals))
}
