//! Checked packed-matrix dispatch; see AGENTS.md.

use std::sync::OnceLock;

use anyhow::{anyhow, ensure, Result};

use super::{accumulate, validate_vector, TernaryMatrix};

#[cfg(target_arch = "x86_64")]
#[path = "avx2.rs"]
mod avx2;

#[derive(Clone, Copy)]
enum Kernel {
    Scalar,
    Avx2,
}

static SELECTED: OnceLock<std::result::Result<Kernel, String>> = OnceLock::new();

/// Report CPU support without requiring AVX2 for the whole executable.
pub fn avx2_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Resolve HEXWEAVE_KERNEL once per process: auto, scalar, or avx2.
pub fn kernel_name() -> Result<&'static str> {
    Ok(match selected()? {
        Kernel::Scalar => "scalar",
        Kernel::Avx2 => "avx2",
    })
}

fn selected() -> Result<Kernel> {
    SELECTED
        .get_or_init(|| {
            let request = match std::env::var("HEXWEAVE_KERNEL") {
                Ok(value) => value,
                Err(std::env::VarError::NotPresent) => "auto".to_owned(),
                Err(error) => return Err(error.to_string()),
            };
            match request.as_str() {
                "scalar" => Ok(Kernel::Scalar),
                "auto" if !avx2_available() => Ok(Kernel::Scalar),
                "auto" | "avx2" if avx2_available() => Ok(Kernel::Avx2),
                "avx2" => Err("AVX2 is unavailable on this CPU".to_owned()),
                _ => Err("HEXWEAVE_KERNEL must be auto, scalar, or avx2".to_owned()),
            }
        })
        .as_ref()
        .copied()
        .map_err(|message| anyhow!(message.clone()))
}

impl TernaryMatrix {
    /// Use the selected kernel without changing reduction order.
    pub fn forward(&self, input: &[f32]) -> Result<Vec<f32>> {
        match selected()? {
            Kernel::Avx2 if self.rows >= 8 => self.forward_avx2(input),
            _ => self.forward_scalar(input),
        }
    }

    /// Use the selected transposed kernel without materializing weights.
    pub fn transpose(&self, input: &[f32]) -> Result<Vec<f32>> {
        match selected()? {
            Kernel::Avx2 if self.cols >= 8 => self.transpose_avx2(input),
            _ => self.transpose_scalar(input),
        }
    }

    /// Preserve the original scalar add/subtract/skip reference.
    pub fn forward_scalar(&self, input: &[f32]) -> Result<Vec<f32>> {
        validate_vector(input, self.cols)?;
        let mut output = vec![0.0; self.rows];
        for (row, sum) in output.iter_mut().enumerate() {
            for (col, &value) in input.iter().enumerate() {
                accumulate(sum, self.read(row * self.cols + col), value);
            }
        }
        finite_output(output)
    }

    /// Preserve the original scalar transpose reference.
    pub fn transpose_scalar(&self, input: &[f32]) -> Result<Vec<f32>> {
        validate_vector(input, self.rows)?;
        let mut output = vec![0.0; self.cols];
        for (row, &value) in input.iter().enumerate() {
            for (col, sum) in output.iter_mut().enumerate() {
                accumulate(sum, self.read(row * self.cols + col), value);
            }
        }
        finite_output(output)
    }

    /// Run checked AVX2 explicitly; unsupported CPUs return an error.
    pub fn forward_avx2(&self, input: &[f32]) -> Result<Vec<f32>> {
        validate_vector(input, self.cols)?;
        ensure!(avx2_available(), "AVX2 is unavailable on this CPU");
        #[cfg(target_arch = "x86_64")]
        {
            // Runtime detection above satisfies the target-feature contract.
            finite_output(unsafe { avx2::forward(self, input) })
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            unreachable!("unsupported CPUs return before dispatch")
        }
    }

    /// Run checked transposed AVX2 explicitly; see AGENTS.md for lane order.
    pub fn transpose_avx2(&self, input: &[f32]) -> Result<Vec<f32>> {
        validate_vector(input, self.rows)?;
        ensure!(avx2_available(), "AVX2 is unavailable on this CPU");
        #[cfg(target_arch = "x86_64")]
        {
            // Runtime detection above satisfies the target-feature contract.
            finite_output(unsafe { avx2::transpose(self, input) })
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            unreachable!("unsupported CPUs return before dispatch")
        }
    }
}

fn finite_output(output: Vec<f32>) -> Result<Vec<f32>> {
    ensure!(
        output.iter().all(|value| value.is_finite()),
        "nonfinite output"
    );
    Ok(output)
}
