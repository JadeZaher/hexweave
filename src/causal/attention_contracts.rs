use super::attention;
use super::geometry::Geometry;
use super::tile::Tile;
use super::tile_gradients::{TileGradients, K, O, Q, V};
use anyhow::Result;

#[path = "attention_reference.rs"]
mod reference;

fn close(actual: f32, expected: f64) {
    assert!(
        (f64::from(actual) - expected).abs() <= 0.00002 + 0.0004 * expected.abs(),
        "factored attention {actual} versus independent legacy {expected}"
    );
}

fn case(geometry: Geometry, length: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let input = (0..geometry.width)
        .map(|i| 0.2 + (i as f32 * 0.51).sin())
        .collect();
    let memory = (0..length * geometry.width)
        .map(|i| (i as f32 * 0.71 + 0.19).cos())
        .collect();
    let upstream = (0..geometry.width)
        .map(|i| 0.03 * (i as f32 + 0.3).cos())
        .collect();
    (input, memory, upstream)
}

#[test]
fn independent_legacy_reference_checks_every_attention_parameter_and_input_adjoint() -> Result<()> {
    let geometry = Geometry::fixture();
    let tile = Tile::new(geometry, 3, 119)?;
    let (input, memory, upstream) = case(geometry, 3);
    let cache = attention::forward(&tile, &input, &memory)?;
    let mut gradient = TileGradients::zeros(geometry, 3)?;
    let mut prefix = vec![0.0; memory.len()];
    let returned = attention::backward(
        &tile,
        &memory,
        &cache,
        &upstream,
        Some(&mut prefix),
        Some(&mut gradient),
    )?;
    let mut model = reference::Reference::new(&tile)?;
    let x: Vec<_> = input.iter().map(|v| f64::from(*v)).collect();
    let m: Vec<_> = memory.iter().map(|v| f64::from(*v)).collect();
    for ((&base, &o), expected) in input.iter().zip(&cache.output).zip(model.forward(&x, &m)) {
        close(base + 0.25 * o, expected);
    }
    for matrix in [Q, K, V, O] {
        for index in 0..model.weights[matrix].len() {
            let original = model.weights[matrix][index];
            model.weights[matrix][index] = original + 1e-5;
            let plus = model.objective(&x, &m, &upstream);
            model.weights[matrix][index] = original - 1e-5;
            let minus = model.objective(&x, &m, &upstream);
            model.weights[matrix][index] = original;
            close(gradient.values[matrix][index], (plus - minus) / 2e-5);
        }
    }
    for (is_memory, expected) in [(false, returned), (true, prefix)] {
        for (index, &actual) in expected.iter().enumerate() {
            let mut plus = if is_memory { m.clone() } else { x.clone() };
            let mut minus = plus.clone();
            plus[index] += 1e-5;
            minus[index] -= 1e-5;
            let (positive, negative) = if is_memory {
                (
                    model.objective(&x, &plus, &upstream),
                    model.objective(&x, &minus, &upstream),
                )
            } else {
                (
                    model.objective(&plus, &m, &upstream),
                    model.objective(&minus, &m, &upstream),
                )
            };
            close(actual, (positive - negative) / 2e-5);
        }
    }
    Ok(())
}

#[test]
fn factored_optional_sinks_add_once_and_handle_zero_score_gradients() -> Result<()> {
    let geometry = Geometry::fixture();
    let tile = Tile::new(geometry, 3, 71)?;
    let (input, normal, upstream) = case(geometry, 4);
    for memory in [normal, vec![0.0; 4 * geometry.width]] {
        let cache = attention::forward(&tile, &input, &memory)?;
        let mut both = TileGradients::zeros(geometry, 3)?;
        let mut parameters = TileGradients::zeros(geometry, 3)?;
        let mut prefix = vec![0.0; memory.len()];
        let mut gather = prefix.clone();
        let a = attention::backward(
            &tile,
            &memory,
            &cache,
            &upstream,
            Some(&mut prefix),
            Some(&mut both),
        )?;
        let b = attention::backward(&tile, &memory, &cache, &upstream, Some(&mut gather), None)?;
        let c = attention::backward(
            &tile,
            &memory,
            &cache,
            &upstream,
            None,
            Some(&mut parameters),
        )?;
        let d = attention::backward(&tile, &memory, &cache, &upstream, None, None)?;
        assert_eq!(a, b);
        assert_eq!(b, c);
        assert_eq!(c, d);
        assert_eq!(prefix, gather);
        assert_eq!(both.values, parameters.values);
        let before = prefix.clone();
        attention::backward(&tile, &memory, &cache, &upstream, Some(&mut prefix), None)?;
        for (actual, once) in prefix.iter().zip(before) {
            close(*actual, 2.0 * f64::from(once));
        }
        if memory.iter().all(|v| *v == 0.0) {
            for matrix in [Q, K, V, O] {
                assert!(both.values[matrix].iter().all(|v| *v == 0.0));
            }
        }
    }
    Ok(())
}

#[test]
fn context_cap_forward_has_all_heads_and_bounded_factored_cache() -> Result<()> {
    for geometry in [Geometry::fixture(), Geometry::candidate()] {
        let tile = Tile::new(geometry, 3, 43)?;
        let (input, memory, _) = case(geometry, geometry.context);
        let cache = attention::forward(&tile, &input, &memory)?;
        let model = reference::Reference::new(&tile)?;
        let x: Vec<_> = input.iter().map(|v| f64::from(*v)).collect();
        let m: Vec<_> = memory.iter().map(|v| f64::from(*v)).collect();
        for ((&base, &o), expected) in input.iter().zip(&cache.output).zip(model.forward(&x, &m)) {
            close(base + 0.25 * o, expected);
        }
        assert_eq!(
            cache.score_vectors.capacity(),
            geometry.heads * geometry.width
        );
        assert_eq!(cache.summaries.capacity(), geometry.heads * geometry.width);
        assert_eq!(
            cache.probabilities.capacity(),
            geometry.heads * geometry.context
        );
        let mut bad = memory;
        bad[geometry.width] = f32::NAN;
        assert!(attention::forward(&tile, &input, &bad).is_err());
    }
    Ok(())
}
