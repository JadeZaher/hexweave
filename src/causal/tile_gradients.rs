use super::geometry::{Geometry, MATRIX_COUNT};
use super::norm;
use anyhow::{ensure, Result};

pub(super) const Q: usize = 0;
pub(super) const K: usize = 1;
pub(super) const V: usize = 2;
pub(super) const O: usize = 3;
pub(super) const G: usize = 4;
pub(super) const A: usize = 5;
pub(super) const DOWN: usize = 6;
pub(super) const CURVE: usize = 7;
pub(super) const ROUTER: usize = 8;
pub(super) const U: usize = 9;
pub(super) const P: usize = 10;

pub(super) struct TileGradients {
    pub values: [Vec<f32>; MATRIX_COUNT],
}

impl TileGradients {
    pub fn zeros(geometry: Geometry, id: usize) -> Result<Self> {
        geometry.validate()?;
        let shapes = geometry.shapes(id)?;
        Ok(Self {
            values: std::array::from_fn(|i| vec![0.0; shapes[i].0 * shapes[i].1]),
        })
    }

    pub fn validate(&self, geometry: Geometry, id: usize) -> Result<()> {
        for (values, (r, c)) in self.values.iter().zip(geometry.shapes(id)?) {
            norm::vector(values, r * c)?;
        }
        Ok(())
    }
}

pub(super) fn outer(
    gradient: Option<&mut TileGradients>,
    index: usize,
    left: &[f32],
    right: &[f32],
    scale: f32,
) -> Result<()> {
    if let Some(gradient) = gradient {
        add_outer(&mut gradient.values[index], left, right, scale)?;
    }
    Ok(())
}

pub(super) fn add_outer(output: &mut [f32], left: &[f32], right: &[f32], scale: f32) -> Result<()> {
    ensure!(
        !left.is_empty() && !right.is_empty(),
        "empty outer product shape"
    );
    ensure!(
        output.len() == left.len() * right.len(),
        "outer product shape mismatch"
    );
    norm::finite(left)?;
    norm::finite(right)?;
    ensure!(scale.is_finite(), "nonfinite outer scale");
    for (row, &l) in output.chunks_exact_mut(right.len()).zip(left) {
        for (value, &r) in row.iter_mut().zip(right) {
            *value += (l * scale) * r;
        }
    }
    norm::finite(output)
}
