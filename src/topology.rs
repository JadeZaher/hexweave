use crate::config::tile_count;
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Axial {
    pub q: i32,
    pub r: i32,
}

#[derive(Clone, Debug)]
pub struct Mesh {
    coordinates: Vec<Axial>,
}

impl Mesh {
    /// Build axial coordinates in stable q-then-r order.
    pub fn new(radius: usize) -> Result<Self> {
        let count = tile_count(radius)?;
        let radius = radius as i32;
        let mut coordinates = Vec::with_capacity(count);
        for q in -radius..=radius {
            let minimum = (-radius).max(-q - radius);
            let maximum = radius.min(-q + radius);
            for r in minimum..=maximum {
                coordinates.push(Axial { q, r });
            }
        }
        Ok(Self { coordinates })
    }

    pub fn len(&self) -> usize {
        self.coordinates.len()
    }

    pub fn is_empty(&self) -> bool {
        self.coordinates.is_empty()
    }

    pub fn coordinates(&self) -> &[Axial] {
        &self.coordinates
    }

    pub fn neighbors(&self, tile: usize) -> Result<Vec<usize>> {
        ensure!(tile < self.len(), "tile index {tile} is outside the mesh");
        let origin = self.coordinates[tile];
        Ok(self
            .coordinates
            .iter()
            .enumerate()
            .filter_map(|(id, point)| {
                let dq = (point.q - origin.q).abs();
                let dr = (point.r - origin.r).abs();
                let ds = (point.q + point.r - origin.q - origin.r).abs();
                (dq.max(dr).max(ds) == 1).then_some(id)
            })
            .collect())
    }
}
