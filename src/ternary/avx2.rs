//! Independent-output-lane AVX2; see AGENTS.md for the safety contract.

use std::arch::x86_64::*;

use crate::ternary::{accumulate, TernaryMatrix};

/// Accumulate eight independent rows in scalar column order.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn forward(matrix: &TernaryMatrix, input: &[f32]) -> Vec<f32> {
    let mut output = vec![0.0; matrix.rows];
    let full_rows = matrix.rows / 8 * 8;
    for row in (0..full_rows).step_by(8) {
        let mut sum = _mm256_setzero_ps();
        for (col, &value) in input.iter().enumerate() {
            let codes: [i32; 8] = std::array::from_fn(|lane| {
                i32::from(matrix.read((row + lane) * matrix.cols + col)) + 1
            });
            // The stack array contains exactly eight readable i32 lanes.
            let codes = unsafe { _mm256_loadu_si256(codes.as_ptr().cast()) };
            sum = accumulate_lanes(sum, codes, _mm256_set1_ps(value));
        }
        // row + 8 <= full_rows <= output.len(); unaligned stores are allowed.
        unsafe { _mm256_storeu_ps(output.as_mut_ptr().add(row), sum) };
    }
    for (row, sum) in output.iter_mut().enumerate().skip(full_rows) {
        for (col, &value) in input.iter().enumerate() {
            accumulate(sum, matrix.read(row * matrix.cols + col), value);
        }
    }
    output
}

/// Accumulate eight contiguous columns in scalar row order.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn transpose(matrix: &TernaryMatrix, input: &[f32]) -> Vec<f32> {
    let mut output = vec![0.0; matrix.cols];
    let full_cols = matrix.cols / 8 * 8;
    let shifts = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14);
    let code_mask = _mm256_set1_epi32(3);
    for col in (0..full_cols).step_by(8) {
        let mut sum = _mm256_setzero_ps();
        for (row, &value) in input.iter().enumerate() {
            let index = row * matrix.cols + col;
            let word = packed_window(matrix, index);
            let codes = _mm256_and_si256(
                _mm256_srlv_epi32(_mm256_set1_epi32(word as i32), shifts),
                code_mask,
            );
            sum = accumulate_lanes(sum, codes, _mm256_set1_ps(value));
        }
        // col + 8 <= full_cols <= output.len(); unaligned stores are allowed.
        unsafe { _mm256_storeu_ps(output.as_mut_ptr().add(col), sum) };
    }
    for (row, &value) in input.iter().enumerate() {
        for (col, sum) in output.iter_mut().enumerate().skip(full_cols) {
            accumulate(sum, matrix.read(row * matrix.cols + col), value);
        }
    }
    output
}

#[target_feature(enable = "avx2")]
fn accumulate_lanes(sum: __m256, codes: __m256i, value: __m256) -> __m256 {
    let negative = _mm256_castsi256_ps(_mm256_cmpeq_epi32(codes, _mm256_setzero_si256()));
    let positive = _mm256_castsi256_ps(_mm256_cmpeq_epi32(codes, _mm256_set1_epi32(2)));
    let minus = _mm256_sub_ps(sum, value);
    let plus = _mm256_add_ps(sum, value);
    _mm256_blendv_ps(_mm256_blendv_ps(sum, minus, negative), plus, positive)
}

fn packed_window(matrix: &TernaryMatrix, index: usize) -> u32 {
    let byte = index / 4;
    let word = u32::from(matrix.packed[byte])
        | (u32::from(matrix.packed.get(byte + 1).copied().unwrap_or(0)) << 8)
        | (u32::from(matrix.packed.get(byte + 2).copied().unwrap_or(0)) << 16);
    word >> ((index % 4) * 2)
}
