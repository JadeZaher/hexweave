//! Weight products for tiles and the tied embedding: ternary kernels, or each matrix's f32 latent
//! weights in float-forward mode (`tuning::Knobs::float_forward`); see AGENTS.md §core.

use super::geometry::TILES;
use super::norm;
use super::shared::Shared;
use super::tile::Tile;
use super::tuning;
use crate::ternary::TernaryMatrix;
use anyhow::{ensure, Context, Result};

/// Rows `[start, start + count)` of row-major `weights` (with `cols` columns) times `input`.
fn rows_product(
    weights: &[f32],
    cols: usize,
    start: usize,
    count: usize,
    input: &[f32],
) -> Result<Vec<f32>> {
    ensure!(
        input.len() == cols && (start + count) * cols <= weights.len(),
        "float product shape"
    );
    let output: Vec<f32> = (start..start + count)
        .map(|row| {
            weights[row * cols..(row + 1) * cols]
                .iter()
                .zip(input)
                .map(|(weight, value)| weight * value)
                .sum()
        })
        .collect();
    norm::finite(&output)?;
    Ok(output)
}

/// Transpose of `rows_product`: the row block's columns weighted by `input`.
fn rows_transpose(
    weights: &[f32],
    cols: usize,
    start: usize,
    count: usize,
    input: &[f32],
) -> Result<Vec<f32>> {
    ensure!(
        input.len() == count && (start + count) * cols <= weights.len(),
        "float transpose shape"
    );
    let mut output = vec![0.0; cols];
    for (offset, &value) in input.iter().enumerate() {
        let row = &weights[(start + offset) * cols..(start + offset + 1) * cols];
        for (sum, weight) in output.iter_mut().zip(row) {
            *sum += weight * value;
        }
    }
    norm::finite(&output)?;
    Ok(output)
}

/// `scale * W x` for the matrix keyed `key`.
fn project(
    key: (usize, usize),
    matrix: &TernaryMatrix,
    input: &[f32],
    scale: f32,
) -> Result<Vec<f32>> {
    match tuning::with_float(key, |w| {
        rows_product(w, matrix.cols(), 0, matrix.rows(), input)
    }) {
        Some(output) => {
            let mut output = output?;
            norm::scale_in_place(&mut output, scale)?;
            Ok(output)
        }
        None => norm::project(matrix, input, scale),
    }
}

/// `scale * W^T x` for the matrix keyed `key`.
fn transpose(
    key: (usize, usize),
    matrix: &TernaryMatrix,
    input: &[f32],
    scale: f32,
) -> Result<Vec<f32>> {
    match tuning::with_float(key, |w| {
        rows_transpose(w, matrix.cols(), 0, matrix.rows(), input)
    }) {
        Some(output) => {
            let mut output = output?;
            norm::scale_in_place(&mut output, scale)?;
            Ok(output)
        }
        None => norm::transpose(matrix, input, scale),
    }
}

impl Tile {
    /// `scale * W x` for matrix `index`.
    pub fn project(&self, index: usize, input: &[f32], scale: f32) -> Result<Vec<f32>> {
        if let Some(weights) =
            super::connected_precision::matrix(&self.weights, index, self.version)?
        {
            let mut out = rows_product(
                weights,
                self.matrices[index].cols(),
                0,
                self.matrices[index].rows(),
                input,
            )?;
            norm::scale_in_place(&mut out, scale)?;
            return Ok(out);
        }
        project((self.id, index), &self.matrices[index], input, scale)
    }

    /// `scale * W^T x` for matrix `index`.
    pub fn transpose(&self, index: usize, input: &[f32], scale: f32) -> Result<Vec<f32>> {
        if let Some(weights) =
            super::connected_precision::matrix(&self.weights, index, self.version)?
        {
            let mut out = rows_transpose(
                weights,
                self.matrices[index].cols(),
                0,
                self.matrices[index].rows(),
                input,
            )?;
            norm::scale_in_place(&mut out, scale)?;
            return Ok(out);
        }
        transpose((self.id, index), &self.matrices[index], input, scale)
    }

    /// Rows `[start, start + count)` of matrix `index` times `input` (one attention head).
    pub fn project_rows(
        &self,
        index: usize,
        start: usize,
        count: usize,
        input: &[f32],
    ) -> Result<Vec<f32>> {
        let matrix = &self.matrices[index];
        if let Some(weights) =
            super::connected_precision::matrix(&self.weights, index, self.version)?
        {
            return rows_product(weights, matrix.cols(), start, count, input);
        }
        tuning::with_float((self.id, index), |w| {
            rows_product(w, matrix.cols(), start, count, input)
        })
        .unwrap_or_else(|| matrix.forward_rows(start, count, input))
    }

    /// Transpose of `project_rows`.
    pub fn transpose_rows(
        &self,
        index: usize,
        start: usize,
        count: usize,
        input: &[f32],
    ) -> Result<Vec<f32>> {
        let matrix = &self.matrices[index];
        if let Some(weights) =
            super::connected_precision::matrix(&self.weights, index, self.version)?
        {
            return rows_transpose(weights, matrix.cols(), start, count, input);
        }
        tuning::with_float((self.id, index), |w| {
            rows_transpose(w, matrix.cols(), start, count, input)
        })
        .unwrap_or_else(|| matrix.transpose_rows(start, count, input))
    }

    /// Element `element` of matrix `index` (the activation-curve offsets).
    pub fn weight(&self, index: usize, element: usize) -> Result<f32> {
        if let Some(weights) =
            super::connected_precision::matrix(&self.weights, index, self.version)?
        {
            return weights
                .get(element)
                .copied()
                .context("explicit weight index out of bounds");
        }
        match tuning::with_float((self.id, index), |w| w.get(element).copied()) {
            Some(value) => value.context("float weight index out of bounds"),
            None => Ok(f32::from(self.matrices[index].value(element)?)),
        }
    }
}

impl Shared {
    /// Column `index` of the tied embedding E (one byte's vector).
    pub fn embedding_column(&self, index: usize) -> Result<Vec<f32>> {
        let (rows, cols) = (self.embedding.rows(), self.embedding.cols());
        if let Some(weights) = super::connected_precision::matrix(&self.weights, 0, self.version)? {
            ensure!(
                index < cols && weights.len() == rows * cols,
                "explicit embedding shape"
            );
            return Ok((0..rows).map(|row| weights[row * cols + index]).collect());
        }
        match tuning::with_float((TILES, 0), |w| -> Result<Vec<f32>> {
            ensure!(
                index < cols && w.len() == rows * cols,
                "embedding latent shape"
            );
            Ok((0..rows).map(|row| w[row * cols + index]).collect())
        }) {
            Some(column) => column,
            None => self.embedding.column(index),
        }
    }

    /// `scale * E x`.
    pub fn embedding_project(&self, input: &[f32], scale: f32) -> Result<Vec<f32>> {
        if let Some(weights) = super::connected_precision::matrix(&self.weights, 0, self.version)? {
            let mut out = rows_product(
                weights,
                self.embedding.cols(),
                0,
                self.embedding.rows(),
                input,
            )?;
            norm::scale_in_place(&mut out, scale)?;
            return Ok(out);
        }
        project((TILES, 0), &self.embedding, input, scale)
    }

    /// `scale * E^T x` (the readout logits).
    pub fn embedding_transpose(&self, input: &[f32], scale: f32) -> Result<Vec<f32>> {
        if let Some(weights) = super::connected_precision::matrix(&self.weights, 0, self.version)? {
            let mut out = rows_transpose(
                weights,
                self.embedding.cols(),
                0,
                self.embedding.rows(),
                input,
            )?;
            norm::scale_in_place(&mut out, scale)?;
            return Ok(out);
        }
        transpose((TILES, 0), &self.embedding, input, scale)
    }
}
