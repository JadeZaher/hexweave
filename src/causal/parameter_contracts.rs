use super::geometry::Geometry;
use super::tile::Tile;
use super::tile_gradients::{TileGradients, A, CURVE, DOWN, G, K, O, P, Q, ROUTER, U, V};
use super::tuning::Knobs;
use anyhow::Result;

#[path = "continuous_fixture.rs"]
mod continuous;

fn close(actual: f32, expected: f64) {
    assert!(
        (f64::from(actual) - expected).abs() < 0.00002 + 0.004 * expected.abs(),
        "parameter adjoint {actual} versus continuous finite difference {expected}"
    );
}

fn largest(values: &[f32], start: usize, end: usize) -> usize {
    (start..end)
        .max_by(|&left, &right| values[left].abs().total_cmp(&values[right].abs()))
        .unwrap()
}

#[test]
fn continuous_reference_checks_every_task_matrix_and_detached_policy_heads() -> Result<()> {
    let float = Knobs {
        float_forward: true,
        ..Knobs::ORACLE
    };
    for knobs in [Knobs::PINNED, Knobs::ORACLE, float] {
        let _pinned_on_exit = super::tuning::scoped(knobs);
        if knobs.float_forward {
            let tile = Tile::new(Geometry::fixture(), 3, 71)?;
            let keys: Vec<_> = (0..tile.matrices.len())
                .map(|i| ((3, i), &tile.matrices[i]))
                .collect();
            super::tuning::ripple_latents(&keys)?;
        }
        check_reference()?;
    }
    Ok(())
}

fn check_reference() -> Result<()> {
    let geometry = Geometry::fixture();
    let tile = Tile::new(geometry, 3, 71)?;
    let input: Vec<_> = (0..geometry.width)
        .map(|j| 0.19 + (j as f32 * 0.73).sin())
        .collect();
    let memory: Vec<_> = (0..3 * geometry.width)
        .map(|j| (j as f32 * 0.39 + 0.17).cos())
        .collect();
    let history: Vec<_> = (0..geometry.width)
        .map(|j| 0.02 * (j as f32 * 0.31).cos())
        .collect();
    let upstream: Vec<_> = (0..geometry.width)
        .map(|j| 0.03 * (j as f32 + 1.0).sin())
        .collect();
    let mut task = TileGradients::zeros(geometry, 3)?;
    tile.backward(
        &input,
        &memory,
        &history,
        &upstream,
        None,
        Some(&mut task),
        None,
    )?;
    let mut reference = continuous::Reference::new(&tile)?;
    let x: Vec<_> = input.iter().map(|&v| f64::from(v)).collect();
    let m: Vec<_> = memory.iter().map(|&v| f64::from(v)).collect();
    let h: Vec<_> = history.iter().map(|&v| f64::from(v)).collect();
    let frozen = reference.forward(&x, &m, &h);
    for (&actual, &expected) in tile
        .forward(&input, &memory, &history)?
        .output
        .iter()
        .zip(&frozen.values)
    {
        close(actual, expected);
    }
    let objective = |model: &continuous::Reference| {
        model
            .forward(&x, &m, &h)
            .values
            .iter()
            .zip(&upstream)
            .map(|(value, &weight)| value * f64::from(weight))
            .sum::<f64>()
    };
    for matrix in [Q, K, V, O, G, A, DOWN, CURVE, U] {
        let index = largest(&task.values[matrix], 0, task.values[matrix].len());
        assert!(
            task.values[matrix][index].abs() > 1e-8,
            "degenerate matrix fixture {matrix}"
        );
        close(
            task.values[matrix][index],
            difference(&mut reference, matrix, index, &objective),
        );
    }
    for row_start in [0, geometry.width] {
        let start = row_start * geometry.width;
        let index = largest(
            &task.values[P],
            start,
            start + geometry.width * geometry.width,
        );
        close(
            task.values[P][index],
            difference(&mut reference, P, index, &objective),
        );
    }
    assert!(task.values[ROUTER].iter().all(|&value| value == 0.0));
    assert!(task.values[P][2 * geometry.width * geometry.width..]
        .iter()
        .all(|&value| value == 0.0));
    check_policy(
        &tile,
        &input,
        &memory,
        &history,
        &upstream,
        &task,
        &mut reference,
        &frozen,
    )
}

#[allow(clippy::too_many_arguments)]
fn check_policy(
    tile: &Tile,
    input: &[f32],
    memory: &[f32],
    history: &[f32],
    upstream: &[f32],
    task: &TileGradients,
    reference: &mut continuous::Reference,
    frozen: &continuous::Output,
) -> Result<()> {
    let geometry = tile.geometry;
    let mut combined = TileGradients::zeros(geometry, 3)?;
    tile.backward(
        input,
        memory,
        history,
        upstream,
        None,
        Some(&mut combined),
        Some((2, 0.7)),
    )?;
    for matrix in [Q, K, V, O, G, A, DOWN, CURVE, U] {
        assert_eq!(task.values[matrix], combined.values[matrix]);
    }
    assert_eq!(
        task.values[P][..2 * geometry.width * geometry.width],
        combined.values[P][..2 * geometry.width * geometry.width]
    );
    let policy =
        |model: &continuous::Reference| model.detached_policy(frozen, 2, f64::from(0.7f32));
    for (matrix, start) in [(ROUTER, 0), (P, 2 * geometry.width * geometry.width)] {
        let index = largest(
            &combined.values[matrix],
            start,
            combined.values[matrix].len(),
        );
        close(
            combined.values[matrix][index],
            difference(reference, matrix, index, &policy),
        );
    }
    Ok(())
}

fn difference(
    model: &mut continuous::Reference,
    matrix: usize,
    index: usize,
    objective: &impl Fn(&continuous::Reference) -> f64,
) -> f64 {
    let original = model.weights[matrix][index];
    model.weights[matrix][index] = original + 1e-5;
    let plus = objective(model);
    model.weights[matrix][index] = original - 1e-5;
    let minus = objective(model);
    model.weights[matrix][index] = original;
    (plus - minus) / 2e-5
}
