//! Reusable conditional and tied-embedding numerical probes.

use super::super::{geometry::VOCAB, norm, positions};
use super::{close, dot, Shared, Tile};
use anyhow::Result;

pub(in crate::causal) fn check_conditional_model(shared: &Shared, tile: &Tile) -> Result<()> {
    let geometry = shared.geometry;
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

pub(in crate::causal) fn check_tied_model(shared: &Shared, tile: &Tile) -> Result<()> {
    let geometry = shared.geometry;
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
        let positive = perturbed_loss(shared, tile, b"aba", index, 0.001)?;
        let negative = perturbed_loss(shared, tile, b"aba", index, -0.001)?;
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
    let scale = (d as f32).sqrt() * super::super::tuning::current().temperature;
    logits[index % VOCAB] += delta * readout.normalized.values[index / VOCAB] / scale;
    let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    Ok(
        maximum + logits.iter().map(|v| (v - maximum).exp()).sum::<f32>().ln()
            - logits[usize::from(b'a')],
    )
}
