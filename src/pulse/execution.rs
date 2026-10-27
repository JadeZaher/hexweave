use super::history::History;
use super::math::Random;
use super::node::{Frame, Gradients};
use super::report::Counts;
use super::residency::Residents;
use super::state::Model;
use super::{DEPTH, TILES, VOCAB, WIDTH};
use anyhow::Result;

pub(super) struct Step<'a> {
    pub context: &'a [f32],
    pub token: usize,
    pub local: bool,
    pub step: usize,
    pub seed: u64,
}

pub(super) fn trajectory(
    model: &mut Model,
    residents: &mut Residents,
    history: &mut History,
    request: Step<'_>,
    counts: &mut Counts,
) -> Result<Vec<Frame>> {
    let mut random = Random::new(request.seed);
    let mut hidden = model.input(request.token)?;
    let mut node = model.center();
    let mut frames = Vec::with_capacity(if request.local { 0 } else { DEPTH });
    for depth in 0..DEPTH {
        let context = if request.local {
            history.prune(model, request.step);
            history.summary()
        } else {
            request.context.to_vec()
        };
        let mut frame = model.nodes[node].forward(node, &hidden)?;
        counts.training_tile_visits += 1;
        counts.visits_per_tile[node] += 1;
        let next = if depth + 1 < DEPTH {
            let route = model.nodes[node].route(
                &frame.output,
                &context,
                &model.mesh.neighbors(node)?,
                Some(&mut random),
            )?;
            let selected = route.rows[route.selected];
            frame.route = Some(route);
            counts.training_router_calls += 1;
            counts.learned_halts += usize::from(selected == TILES);
            selected
        } else {
            counts.forced_depth_exits += 1;
            TILES
        };
        if request.local {
            let correction = model.predict(&model.features(&frame, &context))?;
            let mut gradients = Gradients::zeros(model.nodes[node].router.len());
            gradients.add_transform(&frame, &correction);
            residents.apply(node, &mut model.nodes[node], &gradients)?;
            counts.local_node_updates += 1;
            counts.predictor_forward_calls += 1;
        }
        hidden.clone_from(&frame.output);
        if !request.local {
            frames.push(frame);
        }
        if next == TILES {
            break;
        }
        node = next;
    }
    Ok(frames)
}

pub(super) fn evaluate(model: &Model, counts: &mut Counts) -> Result<f32> {
    let mut loss = 0.0;
    for token in 12..VOCAB {
        let mut hidden = model.input(token)?;
        let mut node = model.center();
        for depth in 0..DEPTH {
            let frame = model.nodes[node].forward(node, &hidden)?;
            counts.evaluation_tile_visits += 1;
            hidden = frame.output;
            if depth + 1 == DEPTH {
                break;
            }
            let route = model.nodes[node].route(
                &hidden,
                &[0.0; WIDTH],
                &model.mesh.neighbors(node)?,
                None,
            )?;
            counts.evaluation_router_calls += 1;
            node = route.rows[route.selected];
            if node == TILES {
                break;
            }
        }
        loss += model.task(&hidden, (token + 1) % VOCAB)?.0;
    }
    Ok(loss / 4.0)
}
