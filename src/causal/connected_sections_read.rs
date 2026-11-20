//! Shared-section read at route entry, per-anchor write credit and teaching; see CONNECTED.md
//! §Shared sections.

use super::connected_anchor::{Credit, Trace};
use super::connected_plasticity::dial;
use super::connected_restart::TileRestart;
use super::connected_sections::{Sections, SharedCredit};
use super::connected_state::State;
use super::connected_store::Store;
use super::connected_update_scale::rms;
use super::geometry::Geometry;
use super::norm;
use super::shared::Prefix;
use super::store_api::TileStore;
use super::tile_gradients::TileGradients;
use anyhow::{ensure, Context, Result};

/// A route entry's shared read: the pre-read state, the mean section output and, under
/// `visit_norm`, the normalized post-read state with its inverse RMS.
pub(super) struct Read {
    pub base: Vec<f32>,
    pub value: Vec<f32>,
    norm: Option<(Vec<f32>, f32)>,
}

/// One anchor's write credit: pre-read entry state, prefix memory and mixed output adjoint.
pub(super) struct Item {
    pub base: Vec<f32>,
    pub memory: Vec<f32>,
    pub upstream: Vec<f32>,
}

/// The state entering the route: the prefix's last row x, or N(x + read) with sections.
pub(super) fn entry(
    sections: Option<&Sections>,
    geometry: Geometry,
    prefix: &Prefix,
) -> Result<(Vec<f32>, Option<Read>)> {
    let base = prefix.initial(geometry).to_vec();
    let Some(sections) = sections else {
        return Ok((base, None));
    };
    let value = read(sections, &base, &prefix.memory)?;
    let sum: Vec<f32> = base.iter().zip(&value).map(|(x, r)| x + r).collect();
    let knobs = super::tuning::current();
    // The convention a tile visit's output gets: RMS-normalized under connected `visit_norm`.
    let (input, norm) = if knobs.connected && knobs.visit_norm {
        let normalized = norm::rms(&sum)?;
        let input = normalized.values.clone();
        (input, Some((normalized.values, normalized.inverse)))
    } else {
        (sum, None)
    };
    Ok((input, Some(Read { base, value, norm })))
}

/// Mean of every section's output at `base` under zero history: the frozen shared read.
fn read(sections: &Sections, base: &[f32], memory: &[f32]) -> Result<Vec<f32>> {
    let history = vec![0.0; base.len()];
    let mut total = vec![0.0f32; base.len()];
    for section in &sections.sections {
        norm::add(
            &mut total,
            &section.tile.forward(base, memory, &history)?.output,
        )?;
    }
    norm::scale_in_place(&mut total, (sections.sections.len() as f32).recip())?;
    Ok(total)
}

/// The adjoint at the pre-read state: through the post-read normalization, the read detached.
pub(super) fn entry_adjoint(read: Option<&Read>, upstream: Vec<f32>) -> Result<Vec<f32>> {
    match read.and_then(|read| read.norm.as_ref()) {
        Some((values, inverse)) => norm::backward(values, *inverse, &upstream),
        None => Ok(upstream),
    }
}

/// Write weights `W(e x_i) / sum_j W(e x_j)` from per-visit surprise `x_i` (distance / bar),
/// W capped at the plasticity dial's 4; equal weights when unavailable or all zero. The flag
/// tells whether the Lambert weights were used.
pub(super) fn weights(surprise: Option<&[f64]>, visits: usize) -> Result<(Vec<f64>, bool)> {
    ensure!(visits > 0, "connected shared write weights need a visit");
    if let Some(values) = surprise {
        ensure!(
            values.len() == visits && values.iter().all(|x| x.is_finite() && *x >= 0.0),
            "invalid connected shared surprise"
        );
        let dials = values
            .iter()
            .map(|&x| dial(x, 1.0))
            .collect::<Result<Vec<_>>>()?;
        let total: f64 = dials.iter().sum();
        if total > 0.0 {
            return Ok((dials.iter().map(|w| w / total).collect(), true));
        }
    }
    Ok((vec![1.0 / visits as f64; visits], false))
}

/// Each visit's `distance / bar` when every visited tile's signal is warm and its bar positive.
pub(super) fn surprise(restarts: Option<&[TileRestart]>, tiles: &[usize]) -> Option<Vec<f64>> {
    let restarts = restarts?;
    let surprise = |tile: usize| {
        let entry = &restarts[tile];
        (entry.signal.is_warm() && entry.bar > 0.0).then(|| entry.signal.distance() / entry.bar)
    };
    tiles.iter().map(|&tile| surprise(tile)).collect()
}

/// After `gather`: queue the anchor's write credit, fold the monitors and apply `teach`.
pub(super) fn observe(
    state: &mut State,
    store: &mut Store,
    prefix: &Prefix,
    trace: &Trace,
    credit: &mut Credit,
) -> Result<()> {
    let (Some(read), Some(shared)) = (trace.read.as_ref(), state.options.shared) else {
        return Ok(());
    };
    let tiles: Vec<usize> = trace.frames.iter().map(|frame| frame.tile).collect();
    let surprise = surprise(state.restarts.as_deref(), &tiles);
    let (weights, lambert) = weights(surprise.as_deref(), tiles.len())?;
    // `mix`: the weighted mix of the visited tiles' output corrections.
    let mut sum = vec![0.0f64; state.geometry.width];
    for (weight, correction) in weights.iter().zip(&credit.corrections) {
        for (total, value) in sum.iter_mut().zip(correction) {
            *total += weight * f64::from(*value);
        }
    }
    let mixed: Vec<f32> = sum.iter().map(|&value| value as f32).collect();
    let exact = credit
        .read_adjoint
        .as_deref()
        .context("connected shared read adjoint missing")?;
    let alignment = cosine(&mixed, exact);
    let upstream = match shared.credit {
        SharedCredit::Mix => mixed,
        // `entry`: dL/ds_k = g_r / K for each of the K sections averaged into r.
        SharedCredit::Entry => {
            let count = shared.sections as f32;
            norm::unit_clip(&exact.iter().map(|value| value / count).collect::<Vec<_>>())?
        }
    };
    if shared.teach > 0.0 {
        let (tau, norms) = (shared.teach, &credit.adjoint_norms);
        teach(
            store,
            state.geometry,
            prefix,
            trace,
            &read.value,
            tau,
            norms,
            &mut credit.gradients,
        )?;
    }
    let sections = state
        .sections
        .as_mut()
        .context("connected shared sections missing")?;
    let queue = &mut sections.queue;
    for (&tile, weight) in tiles.iter().zip(&weights) {
        queue.tile_weight_sum[tile] += weight;
        queue.tile_weight_visits[tile] += 1;
    }
    let delta: Vec<f32> = read
        .value
        .iter()
        .zip(&read.base)
        .map(|(r, x)| r - x)
        .collect();
    queue.read_delta_sum += rms(&delta) / rms(&read.base).max(f64::MIN_POSITIVE);
    queue.alignment_sum += alignment;
    queue.reads += 1;
    queue.lambert_reads += u64::from(lambert);
    let (base, memory) = (read.base.clone(), prefix.memory.clone());
    sections.pending.push(Item {
        base,
        memory,
        upstream,
    });
    Ok(())
}

/// Cosine of two vectors in f64; 0 when either is zero.
pub(super) fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot = |x: &[f32], y: &[f32]| -> f64 {
        x.iter()
            .zip(y)
            .map(|(p, q)| f64::from(*p) * f64::from(*q))
            .sum()
    };
    let norms = (dot(a, a) * dot(b, b)).sqrt();
    if norms > 0.0 {
        dot(a, b) / norms
    } else {
        0.0
    }
}

/// Blends each visit's local target toward the read s. With g the raw output adjoint its exact
/// gradient came from, target' = (1 - tau)(y - g) + tau t has adjoint (1 - tau) g + tau p, so by
/// linearity G' = (1 - tau) G(g) + tau G(p); p = (y - s) |g| / |y - s| keeps tau a true share.
#[allow(clippy::too_many_arguments)]
fn teach(
    store: &mut Store,
    geometry: Geometry,
    prefix: &Prefix,
    trace: &Trace,
    read: &[f32],
    tau: f64,
    norms: &[f64],
    gradients: &mut [TileGradients],
) -> Result<()> {
    let (keep, pull) = ((1.0 - tau) as f32, tau as f32);
    let visits = trace.frames.iter().zip(norms).zip(gradients.iter_mut());
    for ((frame, &norm), gradient) in visits {
        let gap: Vec<f32> = frame.output.iter().zip(read).map(|(y, s)| y - s).collect();
        let scale = (norm / super::connected_signal::length(&gap).max(1e-12)) as f32;
        let direction: Vec<f32> = gap.iter().map(|value| value * scale).collect();
        let mut pulled = TileGradients::zeros(geometry, frame.tile)?;
        store.read(frame.tile, |tile| {
            ensure!(
                tile.version == frame.version,
                "stale connected teaching frame"
            );
            let (memory, sink) = (&prefix.memory, Some(&mut pulled));
            tile.backward(
                &frame.input,
                memory,
                &frame.history,
                &direction,
                None,
                sink,
                None,
            )
        })?;
        for (values, extra) in gradient.values.iter_mut().zip(&pulled.values) {
            for (value, pull_value) in values.iter_mut().zip(extra) {
                *value = keep * *value + pull * pull_value;
            }
        }
    }
    Ok(())
}
