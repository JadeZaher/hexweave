//! Contextual rows: E's gradient through a row normalization holding a constant state; see
//! CONNECTED.md §Contextual rows.

use super::connected_context::{self, Context};
use super::geometry::VOCAB;
use super::tuning::{self, Knobs};
use anyhow::Result;

#[test]
fn embedding_gradient_stays_exact_with_a_constant_state_inside_the_row_norm() -> Result<()> {
    let _guard = tuning::scoped(Knobs {
        connected: true,
        connected_fp32: true,
        ..Knobs::ORACLE
    });
    let (shared, _, _) = super::connected_precision_contracts::explicit(43)?;
    let (bytes, d) = (b"abab", shared.geometry.width);
    // Positions 1 and 2 hold states; row 0 has none and row 3 is the newest (always raw).
    let mut context = Context::new(8);
    for key in 1..3u64 {
        let state = (0..d).map(|j| 0.6 * ((key as usize * d + j) as f32 * 0.7).cos());
        context.push(key, state.collect());
    }
    let (prefix, tally) = connected_context::rows(&shared, bytes, &context, 3, 0)?;
    assert_eq!((tally.rows, tally.filled), (3, 2));
    // f64 oracle on v_i = E[:, b_i] + p_i + s_i: dv_i / dE[:, b_i] is the identity, s_i constant.
    let mut rows = Vec::new();
    for (index, &byte) in bytes.iter().enumerate() {
        let mut column = shared.embedding_column(usize::from(byte))?;
        super::positions::add(index, &mut column)?;
        let state = context.state(index as u64).filter(|_| index < 3);
        let mut row: Vec<f64> = column.iter().map(|v| f64::from(*v)).collect();
        for (value, added) in row.iter_mut().zip(state.unwrap_or(&[])) {
            *value += f64::from(*added);
        }
        rows.push(row);
    }
    let normalize = |row: &[f64]| {
        let inverse = (row.iter().map(|v| v * v).sum::<f64>() / d as f64 + 1e-5)
            .sqrt()
            .recip();
        row.iter().map(|v| v * inverse).collect::<Vec<f64>>()
    };
    let weights: Vec<f64> = (0..4 * d).map(|i| (i as f64 * 0.37).sin()).collect();
    let loss = |rows: &[Vec<f64>]| -> f64 {
        let flat = rows.iter().flat_map(|row| normalize(row));
        flat.zip(&weights).map(|(z, w)| z * w).sum()
    };
    let expected: Vec<f64> = rows.iter().flat_map(|row| normalize(row)).collect();
    for (built, oracle) in prefix.memory.iter().zip(&expected) {
        assert!((f64::from(*built) - oracle).abs() < 1e-5, "row value");
    }
    let adjoint: Vec<f32> = weights.iter().map(|w| *w as f32).collect();
    let mut gradient = vec![0.0; shared.embedding.len()];
    shared.add_prefix_gradient(&prefix, &adjoint, &mut gradient)?;
    for byte in [b'a', b'b'] {
        for j in 0..d {
            let shifted = |step: f64| {
                let mut moved = rows.clone();
                for (row, _) in moved.iter_mut().zip(bytes).filter(|(_, b)| **b == byte) {
                    row[j] += step;
                }
                loss(&moved)
            };
            let numeric = (shifted(1e-5) - shifted(-1e-5)) / 2e-5;
            let analytic = f64::from(gradient[j * VOCAB + usize::from(byte)]);
            assert!(
                (analytic - numeric).abs() <= 1e-3 * (1.0 + numeric.abs()),
                "byte {byte} row {j}: {analytic} vs {numeric}"
            );
        }
    }
    Ok(())
}
