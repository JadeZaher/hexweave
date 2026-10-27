use super::history::{History, Record};
use super::math::{add_outer, clipped};
use super::node::{Frame, Gradients};
use super::residency::Residents;
use super::state::Model;
use super::{THRESHOLD, TILES, VOCAB, WIDTH};
use anyhow::{ensure, Result};

pub(super) struct Anchor {
    pub loss: f32,
    pub node_updates: usize,
    pub router_updates: usize,
    pub predictor_updates: usize,
    pub predictor_error_sum: f32,
}

pub(super) struct AnchorRequest<'a> {
    pub frames: &'a [Frame],
    pub context: &'a [f32],
    pub token: usize,
    pub target: usize,
    pub step: usize,
    pub baseline: f32,
    pub train_predictor: bool,
}

/// Return conditional-path adjoints before committing any changed weight.
pub(super) fn anchor(
    model: &mut Model,
    residents: &mut Residents,
    history: &mut History,
    request: AnchorRequest<'_>,
) -> Result<Anchor> {
    let final_hidden = &request.frames.last().unwrap().output;
    let (loss, logits_gradient) = model.task(final_hidden, request.target)?;
    let logits_gradient: Vec<_> = logits_gradient
        .iter()
        .map(|v| v / (WIDTH as f32).sqrt())
        .collect();
    let (corrections, input_gradient) = returned_wave(model, request.frames, &logits_gradient)?;
    let records = request
        .train_predictor
        .then(|| records(model, &request, &corrections));
    let mut result = Anchor {
        loss,
        node_updates: 0,
        router_updates: 0,
        predictor_updates: 0,
        predictor_error_sum: 0.0,
    };
    for id in 0..TILES {
        if !request.frames.iter().any(|frame| frame.node == id) {
            continue;
        }
        let mut gradients = Gradients::zeros(model.nodes[id].router.len());
        for (frame, correction) in request.frames.iter().zip(&corrections) {
            if frame.node != id {
                continue;
            }
            gradients.add_transform(frame, correction);
            if let Some(route) = &frame.route {
                gradients.add_route(route, loss - request.baseline);
                result.router_updates += 1;
            }
        }
        residents.apply(id, &mut model.nodes[id], &gradients)?;
        result.node_updates += 1;
    }
    update_shared(
        model,
        request.token,
        &input_gradient,
        final_hidden,
        &logits_gradient,
    )?;
    if let Some(records) = records {
        for record in records {
            result.predictor_error_sum +=
                model.train_predictor(&record.features, &record.correction)?;
            result.predictor_updates += 1;
            history.push(record);
        }
    }
    Ok(result)
}

fn returned_wave(
    model: &Model,
    frames: &[Frame],
    logits_gradient: &[f32],
) -> Result<(Vec<Vec<f32>>, Vec<f32>)> {
    let mut correction = model.readout.transpose(logits_gradient)?;
    let mut corrections = vec![vec![0.0; WIDTH]; frames.len()];
    for (index, frame) in frames.iter().enumerate().rev() {
        ensure!(
            model.nodes[frame.node].version == frame.version,
            "anchor weight version changed"
        );
        corrections[index].clone_from(&correction);
        correction = model.nodes[frame.node].returned_correction(frame, &correction)?;
    }
    Ok((corrections, correction))
}

fn records(model: &Model, request: &AnchorRequest<'_>, corrections: &[Vec<f32>]) -> Vec<Record> {
    request
        .frames
        .iter()
        .zip(corrections)
        .map(|(frame, correction)| Record {
            node: frame.node,
            node_version: frame.version,
            shared_version: model.shared_version,
            dependency_versions: std::array::from_fn(|id| model.nodes[id].version),
            step: request.step,
            features: model.features(frame, request.context),
            correction: correction.clone(),
        })
        .collect()
}

fn update_shared(
    model: &mut Model,
    token: usize,
    input_gradient: &[f32],
    final_hidden: &[f32],
    logits_gradient: &[f32],
) -> Result<()> {
    let mut embedding_gradient = vec![0.0; WIDTH * VOCAB];
    for (row, value) in input_gradient.iter().enumerate() {
        embedding_gradient[row * VOCAB + token] = 0.5 * value;
    }
    let mut readout_gradient = vec![0.0; VOCAB * WIDTH];
    add_outer(&mut readout_gradient, logits_gradient, final_hidden);
    let embedding = model.embedding.apply_gradient(
        &mut model.embedding_evidence,
        &clipped(&embedding_gradient)?,
        THRESHOLD,
    )?;
    let readout = model.readout.apply_gradient(
        &mut model.readout_evidence,
        &clipped(&readout_gradient)?,
        THRESHOLD,
    )?;
    model.embedding_transitions += embedding;
    model.readout_transitions += readout;
    if embedding + readout > 0 {
        model.shared_version += 1;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_allocations_preserve_dense_revisit_updates() -> Result<()> {
        let mut model = Model::new(7)?;
        let mut reference = Model::new(7)?;
        let center = model.center();
        let neighbor = model.mesh.neighbors(center)?[0];
        let mut hidden = model.input(6)?;
        let mut frames = Vec::new();
        for id in [center, neighbor, center] {
            let frame = model.nodes[id].forward(id, &hidden)?;
            hidden.clone_from(&frame.output);
            frames.push(frame);
        }
        let (_, logits) = reference.task(&hidden, 7)?;
        let logits: Vec<_> = logits.iter().map(|v| v / (WIDTH as f32).sqrt()).collect();
        let (corrections, input) = returned_wave(&reference, &frames, &logits)?;
        let mut reference_residents = Residents::new(1);
        for id in 0..TILES {
            let mut gradients = Gradients::zeros(reference.nodes[id].router.len());
            let mut selected = false;
            for (frame, correction) in frames.iter().zip(&corrections) {
                if frame.node == id {
                    selected = true;
                    gradients.add_transform(frame, correction);
                }
            }
            if selected {
                reference_residents.apply(id, &mut reference.nodes[id], &gradients)?;
            }
        }
        update_shared(&mut reference, 6, &input, &hidden, &logits)?;
        let mut residents = Residents::new(1);
        let mut history = History::new();
        let result = anchor(
            &mut model,
            &mut residents,
            &mut history,
            AnchorRequest {
                frames: &frames,
                context: &[0.0; WIDTH],
                token: 6,
                target: 7,
                step: 0,
                baseline: 0.0,
                train_predictor: false,
            },
        )?;
        assert_eq!(model.all_matrices(), reference.all_matrices());
        for (actual, expected) in model.nodes.iter().zip(&reference.nodes) {
            assert_eq!(actual.version, expected.version);
        }
        assert_eq!(residents.loads, reference_residents.loads);
        assert_eq!(residents.evictions, reference_residents.evictions);
        assert_eq!(result.node_updates, 2);
        assert_eq!(result.predictor_updates, 0);
        assert_eq!(history.peak, 0);
        Ok(())
    }
}
