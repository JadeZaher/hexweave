use super::geometry::{Geometry, EVIDENCE_RULE, VOCAB};
use super::shared::Shared;
use super::tile::Tile;
use super::tile_gradients::{TileGradients, P, Q, ROUTER, U};
use super::tuning::Knobs;
use super::{norm, positions};
use anyhow::Result;

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
    let prefix = shared.prefix(b"aba")?;
    let input = prefix.initial(geometry).to_vec();
    let history = vec![0.0; geometry.width];
    let upstream: Vec<_> = (0..geometry.width)
        .map(|j| (j as f32 + 1.0) * 0.013)
        .collect();
    let mut memory_adjoint = vec![0.0; prefix.memory.len()];
    let input_adjoint = tile.backward(
        &input,
        &prefix.memory,
        &history,
        &upstream,
        Some(&mut memory_adjoint),
        None,
        None,
    )?;
    for (memory, indices) in [(false, vec![0, 5, 15]), (true, vec![0, 20, 47])] {
        for index in indices {
            let source = if memory { &prefix.memory } else { &input };
            let mut plus = source.to_vec();
            let mut minus = source.to_vec();
            plus[index] += 0.001;
            minus[index] -= 0.001;
            let positive = if memory {
                tile.forward(&input, &plus, &history)?
            } else {
                tile.forward(&plus, &prefix.memory, &history)?
            };
            let negative = if memory {
                tile.forward(&input, &minus, &history)?
            } else {
                tile.forward(&minus, &prefix.memory, &history)?
            };
            let finite_difference =
                (dot(&positive.output, &upstream) - dot(&negative.output, &upstream)) / 0.002;
            close(
                if memory {
                    memory_adjoint[index]
                } else {
                    input_adjoint[index]
                },
                finite_difference,
            );
        }
    }
    assert!(shared.prefix(&[0; 9]).is_err());
    assert!(tile.forward(&input, &[], &history).is_err());
    Ok(())
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
    let prefix = shared.prefix(b"aba")?;
    let history = vec![0.0; geometry.width];
    let forward = tile.forward(prefix.initial(geometry), &prefix.memory, &history)?;
    let mut task = shared.task(&forward.output, b'a')?;
    let mut memory = vec![0.0; prefix.memory.len()];
    let input = tile.backward(
        prefix.initial(geometry),
        &prefix.memory,
        &history,
        &task.input_correction,
        Some(&mut memory),
        None,
        None,
    )?;
    let last = memory.len() - geometry.width;
    norm::add(&mut memory[last..], &input)?;
    shared.add_prefix_gradient(&prefix, &memory, &mut task.embedding_gradient)?;
    for index in [
        usize::from(b'a'),
        3 * VOCAB + usize::from(b'b'),
        7 * VOCAB + usize::from(b'z'),
    ] {
        let positive = perturbed_loss(&shared, &tile, b"aba", index, 0.001)?;
        let negative = perturbed_loss(&shared, &tile, b"aba", index, -0.001)?;
        close(
            task.embedding_gradient[index],
            (positive - negative) / 0.002,
        );
    }
    Ok(())
}

fn perturbed_loss(
    shared: &Shared,
    tile: &Tile,
    bytes: &[u8],
    index: usize,
    delta: f32,
) -> Result<f32> {
    let d = shared.geometry.width;
    let mut memory = Vec::new();
    for (position, &byte) in bytes.iter().enumerate() {
        let mut embedding = shared.embedding_column(usize::from(byte))?;
        if index % VOCAB == usize::from(byte) {
            embedding[index / VOCAB] += delta;
        }
        positions::add(position, &mut embedding)?;
        memory.extend(norm::rms(&embedding)?.values);
    }
    let output = tile
        .forward(&memory[memory.len() - d..], &memory, &vec![0.0; d])?
        .output;
    let readout = shared.readout(&output)?;
    let mut logits = readout.logits;
    let scale = (d as f32).sqrt() * super::tuning::current().temperature;
    logits[index % VOCAB] += delta * readout.normalized.values[index / VOCAB] / scale;
    let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    Ok(
        maximum + logits.iter().map(|v| (v - maximum).exp()).sum::<f32>().ln()
            - logits[usize::from(b'a')],
    )
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
