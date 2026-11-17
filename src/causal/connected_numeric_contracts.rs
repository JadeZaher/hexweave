//! Contracts for observed-only credit, contextual sharing, direction-preserving updates and
//! master scale control.

use super::connected_config::Options;
use super::connected_gain::{lambert_w, Gain};
use super::connected_precision::{self as precision, View};
use super::connected_predictor::FitMetrics;
use super::connected_signal::{Kind, Memory, Record};
use super::connected_update::{self as update, Mode, UpdateMetrics, ROUTE_P_SLOT};
use super::connected_update_scale::rms;
use super::geometry::{Geometry, MATRIX_COUNT, TILES};
use super::norm;
use super::tile::Tile;
use super::tile_gradients::{TileGradients, P, ROUTER};
use super::tuning::{self, Knobs, Latents};
use anyhow::Result;

fn record(geometry: Geometry, kind: Kind) -> Record {
    let d = geometry.width;
    let mut correction = vec![0.0; d];
    correction[0] = 0.2;
    Record {
        input: vec![0.25; d],
        output: vec![0.5; d],
        history: vec![0.0; d],
        correction,
        tile_id: 3,
        revision: 0,
        shared_revision: 0,
        epoch: 0,
        kind,
        lane: if kind == Kind::Predicted {
            Some(0)
        } else {
            None
        },
    }
}

#[test]
fn memory_shares_contextual_credit_with_explicit_freshness() -> Result<()> {
    let geometry = Geometry::fixture();
    let mut memory = Memory::new(geometry, 8)?;
    let mut first = record(geometry, Kind::Predicted);
    first.lane = Some(0);
    memory.publish(first.clone())?;
    let revisions = [0; TILES];
    let result = memory.query_report(&first.input, 1, 0, &revisions, 0, Some(1))?;
    assert_eq!(result.records, 1);
    assert_eq!(result.cross_lane_records, 1);
    assert!(result.values[0] > 0.19);
    let mut opposite = first.clone();
    opposite.input.fill(-0.25);
    opposite.output.fill(-0.5);
    opposite.correction[0] = -0.2;
    opposite.lane = Some(2);
    memory.publish(opposite.clone())?;
    let positive = memory.query(&first.input, 1, 0, &revisions, 0)?;
    let negative = memory.query(&opposite.input, 1, 0, &revisions, 0)?;
    assert!(positive[0] > 0.19 && negative[0] < -0.19);
    for (epoch, version, shared) in [(65, 0, 0), (0, 9, 0), (0, 0, 2)] {
        let mut revisions = revisions;
        revisions[3] = version;
        assert_eq!(
            memory.query(&first.input, 1, epoch, &revisions, shared)?,
            vec![0.0; geometry.width]
        );
    }
    let mut future = first;
    future.epoch = 10;
    let mut future_memory = Memory::new(geometry, 8)?;
    future_memory.publish(future)?;
    assert_eq!(
        future_memory.query(&opposite.input, 1, 9, &revisions, 0)?,
        vec![0.0; geometry.width]
    );
    memory.validate()?;
    let restored: Memory = serde_json::from_slice(&serde_json::to_vec(&memory)?)?;
    assert_eq!(memory, restored);
    Ok(())
}

#[test]
fn lambert_gain_damps_amplifies_and_expires() -> Result<()> {
    for input in [0.0, 1e-12, 0.1, 1.0, std::f64::consts::E, 6.0, 1e6] {
        let w = lambert_w(input)?;
        assert!((w * w.exp() - input).abs() <= 1e-10 * input.max(1.0));
    }
    assert!(lambert_w(-0.1).is_err());
    let options = Options::default();
    let good = FitMetrics {
        mse: 0.0,
        zero_mse: 0.01,
        cosine: 1.0,
        prediction_norm: 0.4,
        target_norm: 0.4,
    };
    let bad = FitMetrics {
        mse: 0.04,
        cosine: -1.0,
        ..good
    };
    let (mut boost, mut damp) = (Gain::new(), Gain::new());
    for epoch in 0..16 {
        boost.observe(&good, epoch, epoch, &options)?;
        damp.observe(&bad, epoch, epoch, &options)?;
    }
    assert!(boost.value_at(15, 15, &options) > 2.0);
    assert!(damp.value_at(15, 15, &options) < 0.5);
    assert!(boost.value_at(31, 15, &options) < boost.value_at(15, 15, &options));
    assert_eq!(boost.value_at(80, 15, &options), 1.0);
    assert_eq!(boost.value_at(15, 24, &options), 1.0);
    let quiet = FitMetrics {
        mse: 0.0,
        zero_mse: 0.0,
        cosine: 0.0,
        prediction_norm: 0.0,
        target_norm: 0.0,
    };
    let mut neutral = Gain::new();
    neutral.observe(&quiet, 1, 1, &options)?;
    assert_eq!(neutral, Gain::new());
    let previous = boost.clone();
    boost.observe(&quiet, 16, 16, &options)?;
    assert_eq!(boost, previous);
    Ok(())
}

fn gradients(geometry: Geometry) -> Result<TileGradients> {
    let mut gradient = TileGradients::zeros(geometry, 3)?;
    for values in &mut gradient.values {
        for (index, value) in values.iter_mut().enumerate() {
            *value = ((index % 7) as f32 - 3.0) * 0.01;
        }
    }
    Ok(gradient)
}

#[test]
fn tile_gain_preserves_whole_update_direction_and_uniform_norm_cap() -> Result<()> {
    let geometry = Geometry::fixture();
    let (mut base, mut scaled) = (Tile::new(geometry, 3, 7)?, Tile::new(geometry, 3, 7)?);
    let (mut a, mut b) = (Vec::new(), Vec::new());
    let gradient = gradients(geometry)?;
    let run = |tile: &mut Tile, latents: &mut Latents, gain, max_update_norm| {
        let options = Options {
            max_update_norm,
            ..Options::default()
        };
        update::apply(tile, latents, &gradient, 0.01, gain, &options, Mode::Local)
    };
    let base_metrics = run(&mut base, &mut a, 1.0, 4.0)?;
    let scaled_metrics = run(&mut scaled, &mut b, 4.0, 0.05)?;
    assert_eq!(base_metrics.base_norm, scaled_metrics.base_norm);
    assert!(scaled_metrics.applied_norm <= 0.05);
    let ratio = scaled_metrics.applied_norm / base_metrics.applied_norm;
    let original = Tile::new(geometry, 3, 7)?;
    // Each run decays at its realized multiplier, applied / base norm; P's route rows stay put.
    let keep = |metrics: &UpdateMetrics| {
        1.0 - Options::default().weight_decay * 0.01 * metrics.applied_norm / metrics.base_norm
    };
    let route_start = 2 * geometry.width * geometry.width;
    for (((owner, matrix), left), ((_, right_matrix), right)) in a.iter().zip(&b) {
        assert_eq!(*owner, 3);
        assert_eq!(matrix, right_matrix);
        for (index, (x, y)) in left.weights.iter().zip(&right.weights).enumerate() {
            let route = *matrix == P && index >= route_start;
            let kept = |metrics| if route { 1.0 } else { keep(metrics) };
            let initial = f64::from(original.matrices[*matrix].value(index)?);
            let moved = |w: &f32, metrics| f64::from(*w) - initial * kept(metrics);
            assert!((moved(x, &base_metrics) * ratio - moved(y, &scaled_metrics)).abs() < 2e-7);
        }
    }
    Ok(())
}

#[test]
fn policy_and_local_updates_keep_disjoint_moments_and_clocks() -> Result<()> {
    let geometry = Geometry::fixture();
    let mut tile = Tile::new(geometry, 3, 7)?;
    let mut latents = Vec::new();
    let (gradient, options) = (gradients(geometry)?, Options::default());
    let step = |tile: &mut Tile, latents: &mut Latents, mode| {
        update::apply(tile, latents, &gradient, 0.01, 1.0, &options, mode)
    };
    step(&mut tile, &mut latents, Mode::Local)?;
    let local_before = latents.clone();
    step(&mut tile, &mut latents, Mode::RouteOnly)?;
    assert!(latents.windows(2).all(|pair| pair[0].0 < pair[1].0));
    let start = 2 * geometry.width * geometry.width;
    for (key, before) in local_before {
        let after = &latents.iter().find(|(k, _)| *k == key).unwrap().1;
        assert_eq!(before.step, after.step);
        assert_eq!(before.moments, after.moments);
        if key.1 == P {
            assert_eq!(before.weights[..start], after.weights[..start]);
        } else {
            assert_eq!(before.weights, after.weights);
        }
    }
    let routes: Vec<_> = latents
        .iter()
        .filter(|(key, _)| key.1 == ROUTER || key.1 == ROUTE_P_SLOT)
        .cloned()
        .collect();
    step(&mut tile, &mut latents, Mode::Local)?;
    for (key, before) in routes {
        assert_eq!(before, latents.iter().find(|(k, _)| *k == key).unwrap().1);
    }
    assert!(latents.len() <= MATRIX_COUNT + 1);
    assert!(latents.windows(2).all(|pair| pair[0].0 < pair[1].0));
    let mut route_first = Tile::new(geometry, 3, 7)?;
    let mut route_latents = Vec::new();
    for mode in [Mode::RouteOnly, Mode::Local] {
        step(&mut route_first, &mut route_latents, mode)?;
        assert!(route_latents.windows(2).all(|pair| pair[0].0 < pair[1].0));
    }
    Ok(())
}

#[test]
fn broadcast_gain_simd_matches_scalar_including_tail() -> Result<()> {
    let input: Vec<_> = (0..35).map(|i| (i as f32 - 17.0) * 0.17).collect();
    for factor in [0.25, 1.0, 2.75] {
        let mut scalar = input.clone();
        super::connected_simd::scalar(&mut scalar, factor);
        let mut dispatched = input.clone();
        super::connected_simd::scale(&mut dispatched, factor)?;
        assert_eq!(scalar, dispatched);
        #[cfg(target_arch = "x86_64")]
        if crate::ternary::avx2_available() {
            let mut vectorized = input.clone();
            unsafe {
                super::connected_simd::avx2(&mut vectorized, factor);
            }
            assert_eq!(scalar, vectorized);
        }
    }
    Ok(())
}

#[test]
fn coherent_local_drift_stays_under_the_master_rms_cap_with_bounded_output() -> Result<()> {
    let _guard = tuning::scoped(Knobs {
        connected: true,
        connected_fp32: true,
        ..Knobs::core(0.003, 7)
    });
    let options = Options::default();
    let mut tile = Tile::new(Geometry::fixture(), 3, 7)?;
    let mut latents = precision::fresh(&tile)?;
    precision::attach_tile(&mut tile, &latents, View::Fp32)?;
    let rows: Vec<f32> = (0..48).map(|j| (j as f32 * 0.39 + 0.17).cos()).collect();
    let mut memory = Vec::new();
    for row in rows.chunks(16) {
        memory.extend(norm::rms(row)?.values);
    }
    let (input, history) = (memory[32..].to_vec(), vec![0.0; 16]);
    // One fixed unit-norm correction, repeated: the coherent drift behind the v2 collapse.
    let correction: Vec<f32> = (0..16)
        .map(|j| if j % 3 == 0 { -0.25 } else { 0.25 })
        .collect();
    let mut gradient = TileGradients::zeros(tile.geometry, 3)?;
    tile.backward(
        &input,
        &memory,
        &history,
        &correction,
        None,
        Some(&mut gradient),
        None,
    )?;
    let mut last = UpdateMetrics::default();
    for _ in 0..2000 {
        last = update::apply(
            &mut tile,
            &mut latents,
            &gradient,
            0.003,
            1.0,
            &options,
            Mode::Local,
        )?;
    }
    let cap = options.master_rms_cap;
    // The projection must be engaged, or this contract would also pass without a cap.
    assert!(last.max_rms > 0.999 * cap && last.max_rms <= cap);
    assert!(latents
        .iter()
        .all(|(_, latent)| rms(&latent.weights) <= cap));
    // The step guard's raw visit RMS: bounded after drift, about 1000 for a 1000x input.
    let raw = tile.forward(&input, &memory, &history)?.raw_rms;
    let scaled: Vec<f32> = input.iter().map(|x| 1000.0 * x).collect();
    let loud = tile.forward(&scaled, &memory, &history)?.raw_rms;
    assert!(raw < 10.0, "drifted raw RMS {raw}");
    assert!((loud / 1000.0 - 1.0).abs() < 0.1, "scaled raw RMS {loud}");
    Ok(())
}
