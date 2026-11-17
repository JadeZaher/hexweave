//! Off-lattice connected products, adjoints, readout bias and continuous-version regressions.

use super::connected_config::Options;
use super::connected_precision::{self as precision, Precision, View};
use super::connected_update::{apply, apply_bias, apply_embedding, bound_scale, Mode};
use super::geometry::{Geometry, VOCAB};
use super::shared::Shared;
use super::tile::Tile;
use super::tile_gradients::{TileGradients, P, Q, ROUTER};
use super::tuning::{self, Knobs, Latents};
use anyhow::{Context, Result};

pub(super) fn explicit(seed: u64) -> Result<(Shared, Tile, Latents)> {
    let geometry = Geometry::fixture();
    let mut tile = Tile::new(geometry, 3, seed)?;
    let mut latents = precision::fresh(&tile)?;
    for (_, latent) in &mut latents {
        for (i, weight) in latent.weights.iter_mut().enumerate() {
            *weight = *weight * 0.4 + 0.17 * (i as f32 * 0.13).sin();
        }
    }
    latents[11].1.weights = latents[P].1.weights[2 * geometry.width * geometry.width..].to_vec();
    precision::attach_tile(&mut tile, &latents, View::Fp32)?;
    let mut shared = Shared::new(geometry, seed)?;
    let mut latent = super::connected_adam::fresh(&shared.embedding, 0, shared.embedding.len())?;
    for (i, weight) in latent.weights.iter_mut().enumerate() {
        *weight = *weight * 0.4 + 0.17 * (i as f32 * 0.13).sin();
    }
    precision::attach_shared(&mut shared, &latent, View::Fp32)?;
    Ok((shared, tile, latents))
}

#[test]
fn explicit_fp32_products_and_adjoints_use_off_lattice_weights_without_tls_fallback() -> Result<()>
{
    let _guard = tuning::scoped(Knobs {
        connected: true,
        connected_fp32: true,
        ..Knobs::ORACLE
    });
    let (shared, mut tile, latents) = explicit(41)?;
    assert!(Shared::new(Geometry::fixture(), 41)?
        .prefix(b"abc")
        .is_err());
    let prefix = shared.prefix(b"aba")?;
    let x = prefix.initial(shared.geometry);
    assert!(Tile::new(shared.geometry, 3, 41)?
        .forward(x, &prefix.memory, &[0.0; 16])
        .is_err());
    assert_ne!(shared.embedding_column(97)?, shared.embedding.column(97)?);
    // Labels read that active view: RMS-normalized columns, one distinct feature per byte.
    for byte in [b'a', b'b'] {
        let label = shared.target_feature(byte)?;
        let column = shared.embedding_column(usize::from(byte))?;
        assert_eq!(label, super::norm::rms(&column)?.values);
    }
    assert_ne!(shared.target_feature(b'a')?, shared.target_feature(b'b')?);
    assert_ne!(tile.weight(Q, 1)?, f32::from(tile.matrices[Q].value(1)?));
    super::numerical_tests::check_conditional_model(&shared, &tile)?;
    super::numerical_tests::check_tied_model(&shared, &tile)?;
    let expected = tile.project(Q, x, 1.0)?;
    for (row, value) in expected.iter().enumerate() {
        let direct: f32 = latents[Q].1.weights[row * 16..(row + 1) * 16]
            .iter()
            .zip(x)
            .map(|(w, x)| w * x)
            .sum();
        assert_eq!(*value, direct);
    }
    let mut missing = latents.clone();
    missing.remove(0);
    assert!(precision::attach_tile(&mut tile, &missing, View::Fp32).is_err());
    Ok(())
}

#[test]
fn float_master_updates_advance_versions_without_trit_changes_and_keep_masked_rows() -> Result<()> {
    let _guard = tuning::scoped(Knobs {
        connected: true,
        connected_fp32: true,
        ..Knobs::core(0.003, 7)
    });
    let mut tile = Tile::new(Geometry::fixture(), 3, 7)?;
    let mut latents = precision::fresh(&tile)?;
    precision::attach_tile(&mut tile, &latents, View::Fp32)?;
    let packed = tile.packed().map(<[u8]>::to_vec);
    let mut gradient = TileGradients::zeros(tile.geometry, tile.id)?;
    gradient.values[Q][0] = 1.0;
    let options = Options::default();
    let update = apply(
        &mut tile,
        &mut latents,
        &gradient,
        0.0001,
        1.0,
        &options,
        Mode::Local,
    )?;
    assert_eq!(update.transitions, 0);
    assert_eq!(tile.version, 1);
    assert_eq!(tile.weight(Q, 0)?, latents[Q].1.weights[0]);
    assert_ne!(tile.weight(Q, 0)?, f32::from(tile.matrices[Q].value(0)?));
    let q = latents[Q].1.clone();
    let start = 2 * 16 * 16;
    gradient.values[ROUTER][0] = 1.0;
    gradient.values[P][start] = 1.0;
    apply(
        &mut tile,
        &mut latents,
        &gradient,
        0.0001,
        1.0,
        &options,
        Mode::RouteOnly,
    )?;
    assert_eq!(latents[Q].1, q);
    assert_eq!(tile.version, 2);
    assert_eq!(tile.packed().map(<[u8]>::to_vec), packed);
    assert_eq!(tile.weight(P, start)?, latents[11].1.weights[0]);
    let mut shared = Shared::new(tile.geometry, 7)?;
    let mut master = Some(super::connected_adam::fresh(
        &shared.embedding,
        0,
        shared.embedding.len(),
    )?);
    precision::attach_shared(&mut shared, master.as_ref().unwrap(), View::Fp32)?;
    let old_prefix = shared.prefix(b"a")?;
    let mut gradient = vec![0.0; shared.embedding.len()];
    gradient[97] = 1.0;
    let update = apply_embedding(&mut shared, &mut master, &gradient, 0.0001, &options)?;
    assert_eq!(update.transitions, 0);
    assert_eq!(shared.version, 1);
    assert!(shared
        .add_prefix_gradient(&old_prefix, &[0.0; 16], &mut gradient)
        .is_err());
    Ok(())
}

#[test]
fn coordinates_without_adam_delta_decay_at_the_realized_step_multiplier() -> Result<()> {
    let (geometry, mut options, rate) = (Geometry::fixture(), Options::default(), 0.1);
    let initial = Tile::new(geometry, 3, 7)?;
    let route_start = 2 * geometry.width * geometry.width;
    // Decay follows the realized multiplier, gain after cap: trusted; barely trusted; capped.
    for (gain, cap) in [(1.0f32, 1.0), (0.01, 1.0), (1.0, 0.001)] {
        options.max_update_norm = cap;
        let (mut tile, mut latents) = (Tile::new(geometry, 3, 7)?, Vec::new());
        let mut gradient = TileGradients::zeros(geometry, 3)?;
        gradient.values[Q][0] = 1.0;
        let update = apply(
            &mut tile,
            &mut latents,
            &gradient,
            rate,
            gain,
            &options,
            Mode::Local,
        )?;
        let factor = bound_scale(gain, update.base_norm, cap);
        let keep = 1.0 - options.weight_decay * (rate * f64::from(factor));
        for ((_, matrix), latent) in &latents {
            for (index, weight) in latent.weights.iter().enumerate() {
                let start = f64::from(initial.matrices[*matrix].value(index)?);
                // Local mode leaves P's route rows alone; Q[0] carries the only Adam delta.
                let route = *matrix == P && index >= route_start;
                let kept = if route { 1.0 } else { keep };
                let expected = (kept * start) as f32;
                if (*matrix, index) != (Q, 0) {
                    assert_eq!(*weight, expected, "case {gain}/{cap} at {matrix}/{index}");
                }
            }
        }
    }
    Ok(())
}

#[test]
fn only_v5_protocols_parse_and_each_names_its_precision() -> Result<()> {
    for (precision, name) in [
        (Precision::Ternary, "causal-connected-v5"),
        (Precision::Fp32, "causal-connected-fp32-v5"),
    ] {
        assert_eq!(precision.protocol(), name);
        assert_eq!(Precision::from_protocol(name)?, precision);
    }
    // Every earlier generation of either precision needs its pinned binary.
    for version in 1..=4 {
        for prefix in ["causal-connected", "causal-connected-fp32"] {
            assert!(Precision::from_protocol(&format!("{prefix}-v{version}")).is_err());
        }
    }
    Ok(())
}

fn close(actual: f32, expected: f32) {
    let tolerance = 2e-4 + 1e-2 * expected.abs();
    assert!(
        (actual - expected).abs() <= tolerance,
        "{actual} versus {expected}"
    );
}

/// Readout cross-entropy at byte `a`, accumulated in f64 for central differences.
fn cross_entropy(shared: &Shared, output: &[f32]) -> Result<f64> {
    let logits = shared.readout(output)?.logits;
    let logits: Vec<f64> = logits.iter().map(|value| f64::from(*value)).collect();
    let maximum = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let sum: f64 = logits.iter().map(|value| (value - maximum).exp()).sum();
    Ok(maximum + sum.ln() - logits[usize::from(b'a')])
}

/// Central difference of `loss` for a ±0.01 move of `values[index]`.
fn central(
    values: &[f32],
    index: usize,
    mut loss: impl FnMut(Vec<f32>) -> Result<f64>,
) -> Result<f32> {
    let mut sides = [0.0; 2];
    for (side, step) in [0.01, -0.01].into_iter().enumerate() {
        let mut moved = values.to_vec();
        moved[index] += step;
        sides[side] = loss(moved)?;
    }
    Ok(((sides[0] - sides[1]) / 0.02) as f32)
}

#[test]
fn readout_bias_and_output_credit_match_finite_differences_and_bias_steps_descend() -> Result<()> {
    let mut shared = Shared::new(Geometry::fixture(), 7)?;
    let mut bias: Vec<f32> = (0..VOCAB).map(|k| 0.3 * (k as f32 * 0.11).sin()).collect();
    bias[usize::from(b' ')] = 3.0;
    bias[usize::from(b'e')] = 2.0;
    shared.bias = Some(bias.clone());
    let output: Vec<f32> = (0..16)
        .map(|j| 3.0 * (j as f32 * 0.53 + 0.2).sin())
        .collect();
    let task = shared.task(&output, b'a')?;
    let gradient = task
        .bias_gradient
        .clone()
        .context("missing readout-bias credit")?;
    for byte in [b'a', b' ', b'e', b'z'].map(usize::from) {
        let expected = central(&bias, byte, |moved| {
            shared.bias = Some(moved);
            cross_entropy(&shared, &output)
        })?;
        close(gradient[byte], expected);
    }
    shared.bias = Some(bias);
    for j in [0, 7, 15] {
        let expected = central(&output, j, |moved| cross_entropy(&shared, &moved))?;
        close(task.input_correction[j], expected);
    }
    let rms = (output.iter().map(|v| v * v).sum::<f32>() / 16.0).sqrt();
    assert!((task.output_rms - rms).abs() < 1e-4 * rms);
    apply_bias(
        &mut shared,
        &mut None,
        &gradient,
        0.003,
        &Options::default(),
    )?;
    assert!(cross_entropy(&shared, &output)? < f64::from(task.loss));
    Ok(())
}

#[test]
fn coherent_embedding_drift_engages_the_master_rms_cap() -> Result<()> {
    let _guard = tuning::scoped(Knobs {
        connected: true,
        connected_fp32: true,
        ..Knobs::core(0.003, 7)
    });
    let options = Options::default();
    let mut shared = Shared::new(Geometry::fixture(), 7)?;
    let fresh = super::connected_adam::fresh(&shared.embedding, 0, shared.embedding.len())?;
    precision::attach_shared(&mut shared, &fresh, View::Fp32)?;
    let mut latent = Some(fresh);
    // One fixed gradient, repeated: the unigram-aligned drift that took the v2 E master to RMS 3.
    let gradient: Vec<f32> = (0..shared.embedding.len())
        .map(|i| if i % 5 == 0 { -0.25 } else { 0.25 })
        .collect();
    let mut last = super::connected_update::UpdateMetrics::default();
    for _ in 0..2000 {
        last = apply_embedding(&mut shared, &mut latent, &gradient, 0.003, &options)?;
    }
    let cap = options.master_rms_cap;
    assert!(
        last.max_rms > 0.999 * cap && last.max_rms <= cap,
        "cap not engaged: {}",
        last.max_rms
    );
    let master = latent.context("missing embedding master")?;
    assert!(super::connected_update_scale::rms(&master.weights) <= cap);
    // The FP32 view the readout uses is the capped master, not a stale copy.
    let view = shared.weights.as_ref().context("missing FP32 view")?;
    assert_eq!(view.matrices[0], master.weights);
    Ok(())
}
