//! SIMD broadcast scaling with scalar parity and no fused multiply-add; see AGENTS.md.

use super::norm;
use anyhow::{ensure, Result};

pub(super) fn scale(values: &mut [f32], factor: f32) -> Result<()> {
    ensure!(factor.is_finite() && factor >= 0.0, "invalid update scale");
    norm::finite(values)?;
    #[cfg(target_arch = "x86_64")]
    if crate::ternary::kernel_name()? == "avx2" {
        // Runtime dispatch establishes AVX2; chunks and tail cover only this mutable slice.
        unsafe {
            avx2(values, factor);
        }
        return norm::finite(values);
    }
    scalar(values, factor);
    norm::finite(values)
}

pub(super) fn scalar(values: &mut [f32], factor: f32) {
    for value in values {
        *value *= factor;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(super) unsafe fn avx2(values: &mut [f32], factor: f32) {
    use std::arch::x86_64::{_mm256_loadu_ps, _mm256_mul_ps, _mm256_set1_ps, _mm256_storeu_ps};
    let factor = _mm256_set1_ps(factor);
    let mut chunks = values.chunks_exact_mut(8);
    for chunk in &mut chunks {
        let input = _mm256_loadu_ps(chunk.as_ptr());
        _mm256_storeu_ps(chunk.as_mut_ptr(), _mm256_mul_ps(input, factor));
    }
    let remainder = chunks.into_remainder();
    let mut factors = [0.0; 8];
    _mm256_storeu_ps(factors.as_mut_ptr(), factor);
    scalar(remainder, factors[0]);
}
