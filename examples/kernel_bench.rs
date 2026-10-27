//! Fixed-shape microbenchmark; see docs/PERFORMANCE-LOOPS.md.

use anyhow::{ensure, Result};
use hexweave::ternary::TernaryMatrix;
use serde_json::json;
use std::hint::black_box;
use std::time::Instant;

fn main() -> Result<()> {
    let mut cases = Vec::new();
    for (rows, cols) in [(8, 8), (8, 26), (7, 16), (16, 256), (256, 16), (64, 64)] {
        let matrix = TernaryMatrix::new(rows, cols, 7)?;
        let input: Vec<_> = (0..cols).map(|i| (i as f32 - 3.0) / 17.0).collect();
        let returned: Vec<_> = (0..rows).map(|i| (i as f32 - 2.0) / 11.0).collect();
        for transpose in [false, true] {
            let (reference, selected) = if transpose {
                (
                    matrix.transpose_scalar(&returned)?,
                    matrix.transpose(&returned)?,
                )
            } else {
                (matrix.forward_scalar(&input)?, matrix.forward(&input)?)
            };
            ensure!(
                reference
                    .iter()
                    .zip(&selected)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "selected kernel differs from scalar reference"
            );
            let mut samples = Vec::new();
            let mut checksum = 0.0_f64;
            for repetition in 0..8 {
                let started = Instant::now();
                for _ in 0..1000 {
                    let output = if transpose {
                        matrix.transpose(black_box(&returned))?
                    } else {
                        matrix.forward(black_box(&input))?
                    };
                    checksum += black_box(output[0]) as f64;
                    black_box(output);
                }
                if repetition > 0 {
                    samples.push(started.elapsed().as_secs_f64());
                }
            }
            samples.sort_by(f64::total_cmp);
            cases.push(json!({"rows": rows, "cols": cols, "transpose": transpose,
                "iterations_per_sample": 1000, "median_seconds": samples[3],
                "samples_seconds": samples, "checksum": checksum,
                "all_output_bits_match_scalar": true}));
        }
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "kernel": hexweave::ternary::kernel_name()?, "cases": cases,
            "scope": "matvec plus validation/allocation; not end-to-end training"
        }))?
    );
    Ok(())
}
