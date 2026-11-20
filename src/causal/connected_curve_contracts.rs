//! Memory-conditioned knot, pulse and normalized-visit adjoints on a bounded float oracle.

use super::connected_config::{CreditMode, Options, SliceMode};
use super::connected_precision::Precision;
use super::connected_state::State;
use super::connected_store::Store;
use super::geometry::Geometry;
use super::scratch_dir::ScratchDir;
use super::tile::Tile;
use super::tile_gradients::{TileGradients, A, CURVE, DOWN, G, P, U};
use super::tuning::{self, Knobs};
use super::{connected_step, ffn, pulse};
use anyhow::Result;

#[path = "execution_fixture.rs"]
#[allow(clippy::duplicate_mod, dead_code)]
mod fixture;

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

fn close(actual: f32, expected: f32) {
    assert!(
        (actual - expected).abs() < 2e-4 + 0.02 * expected.abs(),
        "connected curve derivative {actual} versus {expected}"
    );
}

#[test]
fn shared_memory_changes_knots_and_knot_adjoints_match_finite_differences() -> Result<()> {
    let _guard = tuning::scoped(Knobs {
        connected: true,
        ..Knobs::PINNED
    });
    let tile = Tile::new(Geometry::fixture(), 3, 37)?;
    let input: Vec<_> = (0..16).map(|j| (j as f32 * 0.31).sin()).collect();
    let history: Vec<_> = (0..16).map(|j| 0.12 * (j as f32 * 0.27).cos()).collect();
    let controls = pulse::forward(&tile, &input, &history)?.curve_controls();
    let neutral = pulse::forward(&tile, &input, &[0.0; 16])?.curve_controls();
    assert!(controls
        .iter()
        .zip(neutral)
        .any(|(a, b)| (a - b).abs() > 1e-5));
    let upstream: Vec<_> = (0..16).map(|j| 0.1 * (j as f32 * 0.43).cos()).collect();
    let cache = ffn::forward(&tile, &input, controls)?;
    let (_, derivatives) = ffn::backward(&tile, &input, &cache, &upstream, None)?;
    assert!(derivatives.iter().any(|value| value.abs() > 1e-5));
    for index in 0..3 {
        let (mut plus, mut minus) = (controls, controls);
        plus[index] += 0.001;
        minus[index] -= 0.001;
        let expected = (dot(&ffn::forward(&tile, &input, plus)?.output, &upstream)
            - dot(&ffn::forward(&tile, &input, minus)?.output, &upstream))
            / 0.002;
        close(derivatives[index], expected);
    }
    Ok(())
}

#[test]
fn connected_curve_saturates_outside_the_knots_and_other_paths_extrapolate() -> Result<()> {
    let tile = Tile::new(Geometry::fixture(), 3, 7)?;
    let knots = [-1.0f32, 0.25, 0.5];
    // Controls place channel 0's knot heights (at z = -2, 0, 2) exactly on `knots`.
    let mut controls = [0.0; 3];
    for (k, control) in controls.iter_mut().enumerate() {
        *control =
            knots[k] - (k as f32 - 1.0) - 0.25 * tile.weight(CURVE, k * tile.geometry.ffn)?;
    }
    // (z, saturated value, saturated slope, extrapolated value, extrapolated slope)
    let table = [
        (-5.0, -1.0, 0.0, -2.875, 0.625),
        (-2.0, -1.0, 0.625, -1.0, 0.625),
        (-1.0, -0.375, 0.625, -0.375, 0.625),
        (0.0, 0.25, 0.125, 0.25, 0.125),
        (1.0, 0.375, 0.125, 0.375, 0.125),
        (2.0, 0.5, 0.125, 0.5, 0.125),
        (5.0, 0.5, 0.0, 0.875, 0.125),
    ];
    for connected in [true, false] {
        let _guard = tuning::scoped(Knobs {
            connected,
            ..Knobs::PINNED
        });
        for (z, value, slope, extrapolated, extrapolated_slope) in table {
            let (actual, actual_slope) = ffn::curve(&tile, 0, z, &controls)?;
            let expected = if connected {
                (value, slope)
            } else {
                (extrapolated, extrapolated_slope)
            };
            assert!(
                (actual - expected.0).abs() < 1e-6 && (actual_slope - expected.1).abs() < 1e-6,
                "z {z}, connected {connected}: ({actual}, {actual_slope}) versus {expected:?}"
            );
        }
    }
    Ok(())
}

#[test]
fn normalized_visit_input_pulse_curve_and_down_credit_match_central_differences() -> Result<()> {
    // The last row turns post-visit normalization off (`connected.visit_norm: false`).
    for (slice_mode, visit_norm) in [
        (SliceMode::Full, true),
        (SliceMode::Windowed, true),
        (SliceMode::Hybrid, true),
        (SliceMode::Full, false),
    ] {
        check_visit_adjoints(slice_mode, visit_norm)?;
    }
    Ok(())
}

/// Verifies post-visit RMS normalization (or its absence), input adjoints and FFN/pulse gradients.
fn check_visit_adjoints(slice_mode: SliceMode, visit_norm: bool) -> Result<()> {
    let _guard = tuning::scoped(Knobs {
        connected: true,
        connected_fp32: true,
        slice_mode,
        visit_norm,
        ..Knobs::ORACLE
    });
    let (_, mut tile, baseline) = super::connected_precision_contracts::explicit(71)?;
    let input: Vec<_> = (0..16)
        .map(|j| 3.0 * (0.19 + (j as f32 * 0.73).sin()))
        .collect();
    let memory: Vec<_> = (0..48).map(|j| (j as f32 * 0.39 + 0.17).cos()).collect();
    let history: Vec<_> = (0..16).map(|j| 0.02 * (j as f32 * 0.31).cos()).collect();
    let visible = tile.forward(&input, &memory, &history)?.output;
    let rms = (dot(&visible, &visible) / 16.0).sqrt();
    // The input has RMS about 2, so an unnormalized output is far from RMS 1.
    assert_eq!((rms - 1.0).abs() < 1e-3, visit_norm, "post-visit RMS {rms}");
    if slice_mode != SliceMode::Full {
        let mut flipped = input.clone();
        flipped[0] = -flipped[0];
        assert_ne!(visible, tile.forward(&flipped, &memory, &history)?.output);
    }
    let upstream: Vec<_> = (0..16)
        .map(|j| 0.03 * (j as f32 + 1.0).sin() + 0.02 * visible[j])
        .collect();
    let mut gradient = TileGradients::zeros(tile.geometry, tile.id)?;
    let correction = tile.backward(
        &input,
        &memory,
        &history,
        &upstream,
        None,
        Some(&mut gradient),
        None,
    )?;
    if slice_mode == SliceMode::Windowed {
        assert!(gradient.values[CURVE][..48].contains(&0.0));
    }
    for j in [0, 7, 15] {
        let mut values = [0.0; 2];
        for (slot, offset) in [0.001, -0.001].into_iter().enumerate() {
            let mut moved = input.clone();
            moved[j] += offset;
            values[slot] = dot(&tile.forward(&moved, &memory, &history)?.output, &upstream);
        }
        close(correction[j], (values[0] - values[1]) / 0.002);
    }
    let view = super::connected_precision::View::Fp32;
    for matrix in [U, P, CURVE, DOWN, G, A] {
        let end = if matrix == P {
            3 * 16
        } else {
            gradient.values[matrix].len()
        };
        let index = (0..end)
            .max_by(|&a, &b| {
                gradient.values[matrix][a]
                    .abs()
                    .total_cmp(&gradient.values[matrix][b].abs())
            })
            .unwrap();
        assert!(gradient.values[matrix][index].abs() > 1e-6);
        let mut values = [0.0; 2];
        for (slot, offset) in [0.001, -0.001].into_iter().enumerate() {
            let mut perturbed = baseline.clone();
            perturbed
                .iter_mut()
                .find(|(key, _)| *key == (3, matrix))
                .unwrap()
                .1
                .weights[index] += offset;
            super::connected_precision::attach_tile(&mut tile, &perturbed, view)?;
            values[slot] = dot(&tile.forward(&input, &memory, &history)?.output, &upstream);
        }
        super::connected_precision::attach_tile(&mut tile, &baseline, view)?;
        close(
            gradient.values[matrix][index],
            (values[0] - values[1]) / 0.002,
        );
    }
    Ok(())
}

#[test]
fn credit_modes_step_anchor_and_hybrid_paths_with_windowed_slicing() -> Result<()> {
    let scratch = ScratchDir::new("connected-credit-modes")?;
    let source = fixture::Source::new();
    let geometry = Geometry::fixture();
    for (slot, (credit_mode, slice_mode)) in [
        (CreditMode::AnchorExact, SliceMode::Windowed),
        (CreditMode::AnchorPredicted, SliceMode::Windowed),
        (CreditMode::Hybrid, SliceMode::Hybrid),
    ]
    .into_iter()
    .enumerate()
    {
        let options = Options {
            precision: Precision::Fp32,
            lanes: 2,
            workers: 2,
            visits: 2,
            credit_mode,
            slice_mode,
            ..Options::default()
        };
        let _guard = tuning::scoped(Knobs {
            connected: true,
            connected_fp32: true,
            slice_mode,
            ..Knobs::core(0.003, 7)
        });
        let dir = scratch.join(format!("mode-{slot}"));
        let mut state = State::new(geometry, 7, source.cursor(), &options)?;
        let mut store =
            Store::new_precision(&dir, geometry, 7, options.workers, options.precision)?;
        for _ in 0..2 {
            connected_step::step(&mut state, &mut store, &source, options, 0.003, false)?;
        }
        state.validate()?;
        if credit_mode == CreditMode::AnchorExact {
            assert_eq!(state.counters.predicted_records, 0);
            assert!(state.counters.local_updates > 0);
        } else if credit_mode == CreditMode::AnchorPredicted {
            assert_eq!(state.counters.predicted_records, 0);
        } else {
            assert!(state.counters.predicted_records > 0);
            assert!(state.counters.local_updates > 0);
        }
    }
    Ok(())
}
