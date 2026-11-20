//! Frozen anchor routes and observed credit for connected training; see AGENTS.md.

use super::execution::{destination, START_TILE};
use super::geometry::{Geometry, TILES};
use super::shared::{Prefix, Shared};
use super::store_api::TileStore;
use super::tile_gradients::TileGradients;
use super::{norm, routing};
use anyhow::{ensure, Result};

pub(super) struct Frame {
    pub tile: usize,
    pub version: u64,
    pub input: Vec<f32>,
    pub output: Vec<f32>,
    pub history: Vec<f32>,
    pub heads: Vec<f32>,
    pub latent: Vec<f32>,
    pub action: Option<usize>,
    pub knot_overflow: f32,
    /// RMS of the visit's raw output before post-visit normalization.
    pub raw_rms: f32,
    /// The visit-balance bonus's effect on this frame's route choice; None without one.
    pub steer: Option<super::connected_balance::Steer>,
}

pub(super) struct Trace {
    pub frames: Vec<Frame>,
    pub output: Vec<f32>,
    pub versions: [u64; TILES],
    /// The shared-section read at route entry; None without `connected.shared`.
    pub read: Option<super::connected_sections_read::Read>,
}

impl Trace {
    /// Mean fraction of FFN preactivations beyond the outer knots over the frames.
    pub fn knot_overflow(&self) -> f64 {
        let total: f64 = self
            .frames
            .iter()
            .map(|frame| f64::from(frame.knot_overflow))
            .sum();
        total / self.frames.len().max(1) as f64
    }

    /// Mean and maximum raw (pre-normalization) visit RMS over the frames.
    pub fn visit_rms(&self) -> (f64, f64) {
        let values = self.frames.iter().map(|frame| f64::from(frame.raw_rms));
        let maximum = values.clone().fold(0.0, f64::max);
        let mean = values.sum::<f64>() / self.frames.len().max(1) as f64;
        (mean, maximum)
    }
}

pub(super) struct Credit {
    pub loss: f32,
    pub corrections: Vec<Vec<f32>>,
    pub gradients: Vec<TileGradients>,
    pub embedding: Vec<f32>,
    pub bias_gradient: Option<Vec<f32>>,
    pub output_rms: f32,
    /// L2 norm of each frame's raw output adjoint, the seed of its exact gradient.
    pub adjoint_norms: Vec<f64>,
    /// The exact adjoint at the shared read (route entry); None without a read.
    pub read_adjoint: Option<Vec<f32>>,
}

/// Execute a target-free route from the prefix's last row; each frame retains its exact detached
/// memory input.
pub(super) fn forward<S: TileStore>(
    store: &mut S,
    geometry: Geometry,
    prefix: &Prefix,
    visits: usize,
    draw: super::connected_route::Draw,
    memory: impl FnMut(usize, &[f32]) -> Result<Vec<f32>>,
) -> Result<Trace> {
    let entry = (prefix.initial(geometry).to_vec(), None);
    forward_from(store, geometry, prefix, entry, visits, draw, memory)
}

/// `forward` from an explicit route-entry state and its shared read (`connected_sections_read`).
pub(super) fn forward_from<S: TileStore>(
    store: &mut S,
    geometry: Geometry,
    prefix: &Prefix,
    entry: (Vec<f32>, Option<super::connected_sections_read::Read>),
    visits: usize,
    mut draw: super::connected_route::Draw,
    mut memory: impl FnMut(usize, &[f32]) -> Result<Vec<f32>>,
) -> Result<Trace> {
    ensure!((1..=4).contains(&visits), "invalid connected visit bound");
    let mesh = super::geometry::mesh();
    let versions = store.versions()?;
    let (input, read) = entry;
    ensure!(input.len() == geometry.width, "connected route entry width");
    let (mut tile_id, mut input) = (START_TILE, input);
    let mut frames = Vec::with_capacity(visits);
    for depth in 0..visits {
        let history = memory(tile_id, &input)?;
        let (forward, choice) = store.read(tile_id, |tile| {
            ensure!(tile.version == versions[tile_id], "anchor tile changed");
            let forward = tile.forward(&input, &prefix.memory, &history)?;
            let choice = if depth + 1 == visits {
                None
            } else {
                Some(draw.route(tile, &forward, &history)?)
            };
            Ok((forward, choice))
        })?;
        let (action, steer) = choice.map_or((None, None), |(action, steer)| (Some(action), steer));
        let next_input = forward.output.clone();
        frames.push(Frame {
            tile: tile_id,
            version: versions[tile_id],
            input,
            output: forward.output,
            history,
            heads: forward.heads,
            latent: forward.latent,
            action,
            knot_overflow: forward.knot_overflow,
            raw_rms: forward.raw_rms,
            steer,
        });
        input = next_input;
        match destination(mesh, tile_id, action)? {
            Some(next) => tile_id = next,
            None => break,
        }
    }
    Ok(Trace {
        frames,
        output: input,
        versions,
        read,
    })
}

/// Observe the target only after forward, differentiating its frozen realized route.
pub(super) fn gather<S: TileStore>(
    shared: &Shared,
    store: &mut S,
    prefix: &Prefix,
    trace: &Trace,
    target: u8,
) -> Result<Credit> {
    ensure!(
        store.versions()? == trace.versions,
        "anchor versions changed"
    );
    ensure!(!trace.frames.is_empty(), "empty connected anchor");
    let task = shared.task(&trace.output, target)?;
    let mut upstream = task.input_correction;
    let mut corrections = Vec::with_capacity(trace.frames.len());
    let mut gradients = Vec::with_capacity(trace.frames.len());
    let mut adjoint_norms = Vec::with_capacity(trace.frames.len());
    let mut prefix_adjoint = vec![0.0; prefix.memory.len()];
    for frame in trace.frames.iter().rev() {
        corrections.push(norm::unit_clip(&upstream)?);
        adjoint_norms.push(super::connected_signal::length(&upstream));
        let mut gradient = TileGradients::zeros(shared.geometry, frame.tile)?;
        upstream = store.read(frame.tile, |tile| {
            ensure!(
                tile.version == frame.version,
                "stale connected anchor frame"
            );
            tile.backward(
                &frame.input,
                &prefix.memory,
                &frame.history,
                &upstream,
                Some(&mut prefix_adjoint),
                Some(&mut gradient),
                None,
            )
        })?;
        gradients.push(gradient);
    }
    corrections.reverse();
    gradients.reverse();
    adjoint_norms.reverse();
    let final_row = prefix_adjoint.len() - shared.geometry.width;
    let upstream = super::connected_sections_read::entry_adjoint(trace.read.as_ref(), upstream)?;
    norm::add(&mut prefix_adjoint[final_row..], &upstream)?;
    let read_adjoint = trace.read.as_ref().map(|_| upstream);
    let mut embedding = task.embedding_gradient;
    shared.add_prefix_gradient(prefix, &prefix_adjoint, &mut embedding)?;
    Ok(Credit {
        loss: task.loss,
        corrections,
        gradients,
        embedding,
        bias_gradient: task.bias_gradient,
        output_rms: task.output_rms,
        adjoint_norms,
        read_adjoint,
    })
}

/// Actual task outcomes train route matrices; transformation credit stays local and predicted.
pub(super) fn route_gradient<S: TileStore>(
    store: &mut S,
    geometry: Geometry,
    trace: &Trace,
    tile_id: usize,
    advantage: f32,
) -> Result<TileGradients> {
    let mut gradient = TileGradients::zeros(geometry, tile_id)?;
    store.read(tile_id, |tile| {
        for frame in trace.frames.iter().filter(|frame| frame.tile == tile_id) {
            if let Some(action) = frame.action {
                routing::policy(
                    tile,
                    &frame.output,
                    &frame.heads,
                    &frame.latent,
                    &frame.history,
                    action,
                    advantage,
                    &mut gradient,
                )?;
            }
        }
        Ok(())
    })?;
    Ok(gradient)
}
