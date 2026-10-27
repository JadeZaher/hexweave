use super::gradients::Gradients;
use super::history::History;
use super::node::Frame;
use super::shared::Shared;
use super::storage::Store;
use super::{RouteCredit, DEPTH, TILES, WIDTH};
use anyhow::Result;
use serde::Serialize;

#[derive(Default, Serialize)]
pub(super) struct Counts {
    #[serde(skip)]
    pub credit: Option<super::credit_diagnostic::Diagnostic>,
    pub tile_visits: usize,
    pub local_updates: usize,
    pub anchor_targets: usize,
    pub returned_frames: usize,
    pub learned_halts: usize,
    pub forced_exits: usize,
    pub revisits: usize,
    pub route_calls: usize,
    pub predictor_predictions: usize,
    pub predictor_updates: usize,
    pub predictor_replays: usize,
    pub transitions: [usize; 5],
    pub visits_by_tile: [usize; TILES],
    pub path_lengths: [usize; DEPTH],
    pub predictor_error_sum: f32,
}

pub(super) struct ForwardRequest<'a> {
    pub context: &'a [u8],
    pub history: &'a [f32],
    pub random_seed: Option<u64>,
    pub local: bool,
    pub step: usize,
}

pub(super) fn execute(
    shared: &Shared,
    store: &mut Store,
    history: &mut History,
    counts: &mut Counts,
    request: ForwardRequest<'_>,
) -> Result<Vec<Frame>> {
    let mut hidden = shared.input(request.context)?;
    let mut id = store
        .mesh
        .coordinates()
        .iter()
        .position(|p| p.q == 0 && p.r == 0)
        .unwrap();
    let mut random = request.random_seed;
    let mut frames: Vec<Frame> = Vec::with_capacity(DEPTH);
    let mut visited = [false; TILES];
    let mut path_length = 0;
    for depth in 0..DEPTH {
        history.prune(&store.versions, shared.version, request.step);
        let signal = if request.local {
            history.summary()
        } else {
            request.history.to_vec()
        };
        let neighbors = store.mesh.neighbors(id)?;
        let coordinate = store.mesh.coordinates()[id];
        let frame = store.with_node(id, |node| {
            let mut frame = node.forward(id, &hidden, &signal)?;
            if depth + 1 < DEPTH {
                frame.route = Some(node.route(&frame, &signal, &neighbors, &mut random)?);
            }
            if request.local {
                let features: Vec<_> = frame
                    .input
                    .iter()
                    .chain(&frame.output)
                    .chain(&signal)
                    .copied()
                    .chain([coordinate.q as f32, coordinate.r as f32])
                    .collect();
                let correction = shared.predict(&features)?;
                let mut gradient = Gradients::zeros(node);
                gradient.add(node, &frame, &correction, None, RouteCredit::Coupled)?;
                let transitions = node.apply(&gradient)?;
                for (count, change) in counts.transitions.iter_mut().zip(transitions) {
                    *count += change;
                }
                counts.local_updates += 1;
                counts.predictor_predictions += 1;
            }
            Ok(frame)
        })?;
        counts.revisits += usize::from(visited[id]);
        visited[id] = true;
        path_length += 1;
        counts.tile_visits += 1;
        counts.visits_by_tile[id] += 1;
        let next = frame
            .route
            .as_ref()
            .map(|route| route.destinations[route.selected]);
        counts.route_calls += usize::from(frame.route.is_some());
        hidden.clone_from(&frame.output);
        if !request.local {
            frames.push(frame);
        }
        match next {
            Some(TILES) => {
                counts.learned_halts += 1;
                break;
            }
            Some(next) => id = next,
            None => {
                counts.forced_exits += 1;
                break;
            }
        }
    }
    counts.path_lengths[path_length - 1] += 1;
    Ok(frames)
}

pub(super) fn evaluate(shared: &Shared, store: &mut Store, bytes: &[u8]) -> Result<(f32, usize)> {
    let identity = shared.identity()?;
    let versions = store.versions;
    let examples = (bytes.len() - super::CONTEXT).min(64);
    let mut total = 0.0;
    for index in 0..examples {
        let (context, target) = super::data::example(bytes, index);
        let frames = execute(
            shared,
            store,
            &mut History::default(),
            &mut Counts::default(),
            ForwardRequest {
                context,
                history: &[0.0; WIDTH],
                random_seed: None,
                local: false,
                step: 0,
            },
        )?;
        total += shared.task(&frames.last().unwrap().output, target)?.0;
    }
    anyhow::ensure!(
        identity == shared.identity()? && versions == store.versions,
        "evaluation mutated weights"
    );
    Ok((total / examples as f32, examples))
}
