use super::geometry::{Geometry, EVIDENCE_RULE};
use super::norm;
use super::shared::Shared;
use super::tile::Tile;
use super::tile_gradients::{TileGradients, P, Q, ROUTER, U};
use super::tuning::Knobs;
use anyhow::Result;

#[path = "numerical_models.rs"]
mod models;
pub(super) use models::{check_conditional_model, check_tied_model};

#[test]
fn joint_batch_rejects_invalid_inputs_before_commit_and_preserves_stalls() -> Result<()> {
    let geometry = super::geometry::Geometry::fixture();
    let mut predictor = super::predictor::Predictor::new(geometry, 7)?;
    let packed = predictor.packed().to_vec();
    let features = vec![0.0; geometry.features()];
    let target = vec![0.0; geometry.width];
    let bad = vec![f32::NAN; geometry.width];
    assert!(predictor.train_joint(&[]).is_err());
    assert!(predictor
        .train_joint(&[(features.as_slice(), target.as_slice()); 5])
        .is_err());
    assert!(predictor.train_joint(&[(&features[..1], &target)]).is_err());
    assert!(predictor
        .train_joint(&[(&features, &target), (&features, &bad)])
        .is_err());
    assert_eq!(predictor.packed(), packed);
    assert_eq!(predictor.version, 0);
    let update = predictor.train_joint(&[(&features, &target)])?;
    assert_eq!(update.transitions, 0);
    assert_eq!(update.pair_mse, [0.0]);
    assert_eq!(predictor.packed(), packed);
    assert_eq!(predictor.version, 0);
    Ok(())
}

fn close(actual: f32, expected: f32) {
    assert!(
        (actual - expected).abs() <= 0.004 + 0.025 * expected.abs(),
        "{actual} versus {expected}"
    );
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

#[test]
fn rms_adjoint_includes_epsilon_and_rejects_nonfinite() -> Result<()> {
    for input in [
        vec![0.2, -0.7, 0.4, 1.3],
        vec![0.0001, -0.0002, 0.0003, 0.0004],
    ] {
        let upstream = [0.7, -0.1, 0.3, 0.2];
        let cache = norm::rms(&input)?;
        let derivative = norm::backward(&cache.values, cache.inverse, &upstream)?;
        let epsilon = if input[0] < 0.001 { 1e-6 } else { 1e-3 };
        for j in 0..input.len() {
            let mut plus = input.clone();
            let mut minus = input.clone();
            plus[j] += epsilon;
            minus[j] -= epsilon;
            let expected = (dot(&norm::rms(&plus)?.values, &upstream)
                - dot(&norm::rms(&minus)?.values, &upstream))
                / (2.0 * epsilon);
            close(derivative[j], expected);
        }
    }
    assert!(norm::rms(&[f32::NAN]).is_err());
    assert!(norm::rms(&[f32::MAX]).is_err());
    Ok(())
}

#[test]
fn tile_conditional_adjoint_covers_current_state_and_every_prefix_position() -> Result<()> {
    check_conditional_adjoint()
}

pub(super) fn check_conditional_adjoint() -> Result<()> {
    let geometry = Geometry::fixture();
    let shared = Shared::new(geometry, 37)?;
    let tile = Tile::new(geometry, 3, 37)?;
    check_conditional_model(&shared, &tile)
}

#[test]
fn anchor_phase_sinks_and_detached_policy_preserve_task_adjoints() -> Result<()> {
    let geometry = Geometry::fixture();
    let shared = Shared::new(geometry, 19)?;
    let tile = Tile::new(geometry, 3, 19)?;
    let prefix = shared.prefix(b"abc")?;
    let input = prefix.initial(geometry);
    let history = vec![0.0; geometry.width];
    let upstream = vec![0.07; geometry.width];
    let mut both_memory = vec![0.0; prefix.memory.len()];
    let mut gather_memory = vec![0.0; prefix.memory.len()];
    let mut both = TileGradients::zeros(geometry, 3)?;
    let mut parameters = TileGradients::zeros(geometry, 3)?;
    let mut task = TileGradients::zeros(geometry, 3)?;
    let a = tile.backward(
        input,
        &prefix.memory,
        &history,
        &upstream,
        Some(&mut both_memory),
        Some(&mut both),
        Some((0, 0.7)),
    )?;
    let b = tile.backward(
        input,
        &prefix.memory,
        &history,
        &upstream,
        Some(&mut gather_memory),
        None,
        None,
    )?;
    let c = tile.backward(
        input,
        &prefix.memory,
        &history,
        &upstream,
        None,
        Some(&mut parameters),
        Some((0, 0.7)),
    )?;
    tile.backward(
        input,
        &prefix.memory,
        &history,
        &upstream,
        None,
        Some(&mut task),
        None,
    )?;
    assert_eq!(a, b);
    assert_eq!(b, c);
    assert_eq!(both_memory, gather_memory);
    assert_eq!(both.values, parameters.values);
    assert_eq!(task.values[U], parameters.values[U]);
    assert_eq!(
        task.values[P][..2 * geometry.width * geometry.width],
        parameters.values[P][..2 * geometry.width * geometry.width]
    );
    assert!(task.values[ROUTER].iter().all(|v| *v == 0.0));
    assert!(parameters.values[ROUTER].iter().any(|v| *v != 0.0));
    Ok(())
}

#[test]
fn tied_embedding_gradient_sums_readout_and_repeated_prefix_uses_once() -> Result<()> {
    for knobs in [Knobs::PINNED, Knobs::ORACLE] {
        let _pinned_on_exit = super::tuning::scoped(knobs);
        check_tied_embedding()?;
    }
    Ok(())
}

pub(super) fn check_tied_embedding() -> Result<()> {
    let geometry = Geometry::fixture();
    let shared = Shared::new(geometry, 41)?;
    let tile = Tile::new(geometry, 3, 41)?;
    check_tied_model(&shared, &tile)
}

#[test]
fn tile_update_is_versioned_and_rejects_nonfinite_before_mutation() -> Result<()> {
    let geometry = Geometry::fixture();
    let mut tile = Tile::new(geometry, 3, 7)?;
    let mut gradient = TileGradients::zeros(geometry, 3)?;
    gradient.values[Q][0] = if tile.matrices[Q].value(0)? == -1 {
        -1.0
    } else {
        1.0
    };
    let changes = tile.apply(&gradient)?;
    assert_eq!(changes.iter().sum::<usize>(), 1);
    assert_eq!(tile.version, 1);
    let packed: Vec<_> = tile.packed().iter().map(|bytes| bytes.to_vec()).collect();
    gradient.values[P][0] = f32::NAN;
    assert!(tile.apply(&gradient).is_err());
    assert_eq!(tile.version, 1);
    assert_eq!(
        tile.packed()
            .iter()
            .map(|bytes| bytes.to_vec())
            .collect::<Vec<_>>(),
        packed
    );
    let restored = Tile::from_packed(geometry, 3, tile.version, &packed, EVIDENCE_RULE)?;
    assert_eq!(tile.packed(), restored.packed());
    assert!(Tile::from_packed(geometry, 3, 1, &packed, "threshold-four").is_err());
    Ok(())
}
