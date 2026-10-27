use super::{accumulate, validate_vector, TernaryMatrix};
use anyhow::{ensure, Result};

impl TernaryMatrix {
    /// Project a contiguous output-row block; see ternary/AGENTS.md.
    pub(crate) fn forward_rows(
        &self,
        start: usize,
        count: usize,
        input: &[f32],
    ) -> Result<Vec<f32>> {
        self.check_rows(start, count)?;
        validate_vector(input, self.cols)?;
        let mut output = vec![0.0; count];
        for (offset, sum) in output.iter_mut().enumerate() {
            for (column, &value) in input.iter().enumerate() {
                accumulate(sum, self.read((start + offset) * self.cols + column), value);
            }
        }
        ensure!(
            output.iter().all(|v| v.is_finite()),
            "nonfinite row projection"
        );
        Ok(output)
    }

    /// Transpose a contiguous output-row block without padded inputs.
    pub(crate) fn transpose_rows(
        &self,
        start: usize,
        count: usize,
        input: &[f32],
    ) -> Result<Vec<f32>> {
        self.check_rows(start, count)?;
        validate_vector(input, count)?;
        let mut output = vec![0.0; self.cols];
        for (offset, &value) in input.iter().enumerate() {
            for (column, sum) in output.iter_mut().enumerate() {
                accumulate(sum, self.read((start + offset) * self.cols + column), value);
            }
        }
        ensure!(
            output.iter().all(|v| v.is_finite()),
            "nonfinite row transpose"
        );
        Ok(output)
    }

    fn check_rows(&self, start: usize, count: usize) -> Result<()> {
        ensure!(
            count > 0 && start.checked_add(count).is_some_and(|end| end <= self.rows),
            "invalid matrix row block"
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_row_blocks_match_independent_integer_products_and_reject_shapes() -> Result<()> {
        let values = [-1, 0, 1, 1, -1, 0, 0, 1, -1, -1, 1, 1];
        let matrix = TernaryMatrix::from_values(4, 3, &values)?;
        assert_eq!(matrix.forward_rows(1, 2, &[2.0, 3.0, 5.0])?, [-1.0, -2.0]);
        assert_eq!(matrix.transpose_rows(1, 2, &[2.0, 3.0])?, [2.0, 1.0, -3.0]);
        for (start, count) in [(0, 0), (4, 1), (usize::MAX, 2), (3, 2)] {
            assert!(matrix.forward_rows(start, count, &[1.0; 3]).is_err());
            assert!(matrix.transpose_rows(start, count, &[1.0; 2]).is_err());
        }
        assert!(matrix.forward_rows(0, 1, &[f32::NAN; 3]).is_err());
        assert!(matrix.transpose_rows(0, 1, &[f32::INFINITY]).is_err());
        assert!(matrix.transpose_rows(0, 1, &[1.0; 2]).is_err());
        Ok(())
    }
}
