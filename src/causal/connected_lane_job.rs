//! Loss-free read-only tile jobs and bounded worker execution; see CONNECTED.md §Learning loop
//! (label feature: §v5).

use super::super::connected_config::Options;
use super::super::connected_predictor::Predictor;
use super::super::connected_signal::{Kind, Record};
use super::super::connected_store::Lease;
use super::super::execution::destination;
use super::super::shared::Prefix;
use super::super::tile_gradients::TileGradients;
use super::super::tuning::{self, Knobs};
use anyhow::Result;

pub(super) struct Job {
    pub lane: usize,
    pub lease: Lease,
    pub input: Vec<f32>,
    pub history: Vec<f32>,
    pub depth: usize,
    pub epoch: u64,
    pub revision: u64,
    pub shared_revision: u64,
    pub gain: f32,
    pub draw: super::super::connected_route::Draw,
}

pub(super) struct Outcome {
    pub lane: usize,
    pub lease: Lease,
    pub output: Vec<f32>,
    pub next: Option<usize>,
    pub record: Record,
    pub gradient: Option<TileGradients>,
}

/// A job computes no loss; its upstream credit is the predictor's output given the `label`
/// feature of the phase's already-read-out byte.
fn execute(
    job: Job,
    prefix: &Prefix,
    label: &[f32],
    predictor: &Predictor,
    options: Options,
    train: bool,
) -> Result<Outcome> {
    let tile = job.lease.tile();
    let forward = tile.forward(&job.input, &prefix.memory, &job.history)?;
    let features = predictor.features(&job.input, &forward.output, &job.history, label, tile.id)?;
    let correction = predictor.predict(&features, tile.id)?;
    let action = if job.depth + 1 < options.visits {
        Some({ job.draw }.route(tile, &forward, &job.history)?.0)
    } else {
        None
    };
    let next = destination(super::super::geometry::mesh(), tile.id, action)?;
    let record = Record {
        input: job.input,
        output: forward.output.clone(),
        history: job.history,
        correction,
        tile_id: tile.id,
        revision: job.revision,
        shared_revision: job.shared_revision,
        epoch: job.epoch,
        kind: Kind::Predicted,
        lane: Some(job.lane),
    };
    // Zero trust gives zero gain: no local differentiation, update or Adam moment change.
    let gradient = if train && job.gain > 0.0 {
        let mut gradient = TileGradients::zeros(tile.geometry, tile.id)?;
        tile.backward(
            &record.input,
            &prefix.memory,
            &record.history,
            &record.correction,
            None,
            Some(&mut gradient),
            None,
        )?;
        Some(gradient)
    } else {
        None
    };
    Ok(Outcome {
        lane: job.lane,
        lease: job.lease,
        output: forward.output,
        next,
        record,
        gradient,
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn compute(
    jobs: Vec<Job>,
    prefix: &Prefix,
    label: &[f32],
    predictor: &Predictor,
    options: Options,
    knobs: Knobs,
    threaded: bool,
    train: bool,
) -> Result<Vec<Outcome>> {
    if !threaded || jobs.len() == 1 {
        return jobs
            .into_iter()
            .map(|job| execute(job, prefix, label, predictor, options, train))
            .collect();
    }
    std::thread::scope(|scope| {
        let handles: Vec<_> = jobs
            .into_iter()
            .map(|job| {
                scope.spawn(move || {
                    tuning::install(knobs);
                    execute(job, prefix, label, predictor, options, train)
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| {
                handle
                    .join()
                    .map_err(|_| anyhow::anyhow!("connected tile worker panicked"))?
            })
            .collect()
    })
}
