//! Interval and lifetime connected measurements; see CONNECTED.md.

use super::super::connected_state::{Counters, State};
use super::super::connected_store::Store;
use super::super::connected_update_scale::rms;
use super::super::geometry::{TILES, VOCAB};
use anyhow::Result;
use serde_json::{json, Value};
use std::fs::File;
use std::io::Write;
use std::time::Instant;

#[path = "connected_retention.rs"]
mod retention;

pub(super) struct Log {
    file: File,
    previous: Counters,
    previous_anchors: u64,
    previous_seconds: f64,
    initial_anchors: u64,
    started: Instant,
    references: retention::References,
    /// Readout-bias provenance, written once on the next row (a fresh run's first).
    readout_prior: Option<&'static str>,
}

impl Log {
    pub fn new(
        file: File,
        state: &State,
        started: Instant,
        readout_prior: Option<&'static str>,
    ) -> Self {
        Self {
            file,
            previous: state.counters.clone(),
            previous_anchors: state.anchors,
            previous_seconds: 0.0,
            initial_anchors: state.anchors,
            started,
            references: retention::References::new(state.anchors),
            readout_prior,
        }
    }

    pub fn assess(&mut self, evaluation: &mut Value, anchors: u64) {
        self.references.annotate(evaluation, anchors);
    }

    pub fn record(
        &mut self,
        state: &State,
        store: &Store,
        eval: Option<Value>,
        seconds: f64,
    ) -> Result<()> {
        let current = &state.counters;
        let interval_anchors = state.anchors - self.previous_anchors;
        let interval_seconds = seconds - self.previous_seconds;
        let mut line = json!({
            "protocol": state.options.precision.protocol(), "precision": state.options.precision, "anchors": state.anchors, "epoch": state.cursor.epoch,
            "elapsed_seconds": self.started.elapsed().as_secs_f64(), "training_seconds": seconds,
            "anchors_per_training_second": (state.anchors - self.initial_anchors) as f64 / seconds.max(1e-9),
            "train_loss": average(current.loss_sum, state.anchors),
            "interval": {
                "from_anchor": self.previous_anchors, "to_anchor": state.anchors,
                "anchors": interval_anchors, "training_seconds": interval_seconds,
                "anchors_per_training_second": (interval_anchors > 0).then_some(interval_anchors as f64 / interval_seconds.max(1e-9)),
                "train_loss": average(current.loss_sum - self.previous.loss_sum, interval_anchors),
                "predictor": predictor(current, &self.previous),
                "local_updates": current.local_updates - self.previous.local_updates,
                "local_visits": difference(&current.local_visits, &self.previous.local_visits),
                "anchor_visits": difference(&current.anchor_visits, &self.previous.anchor_visits),
            },
            "predictor_cumulative": predictor(current, &Counters::default()),
            "eval": eval, "mechanism": current, "paging": store.stats(),
            "gains": gains(state, store), "trust": trust(state),
            "memory_records": state.memory.records.len(),
        });
        if state.options.replay.is_some() {
            line["replay"] = state.replay.report();
        }
        if let Some(restart) = super::super::connected_restart::report(state)? {
            line["restart"] = restart;
        }
        add_scale(&mut line, state, &self.previous, interval_anchors);
        if let Some(prior) = self.readout_prior.take() {
            line["readout_prior"] = json!(prior);
        }
        writeln!(self.file, "{line}")?;
        self.file.flush()?;
        self.previous = current.clone();
        self.previous_anchors = state.anchors;
        self.previous_seconds = seconds;
        Ok(())
    }
}

pub(super) fn gains(state: &State, store: &Store) -> Vec<f32> {
    state
        .gains
        .iter()
        .enumerate()
        .map(|(id, gain)| gain.value_at(state.epoch, store.slots[id].meta.revision, &state.options))
        .collect()
}

/// Per-tile predictor trust; local update multipliers are `gains * trust`.
pub(super) fn trust(state: &State) -> Vec<f64> {
    (0..TILES).map(|tile| state.predictor.trust(tile)).collect()
}

/// Adds readout-RMS, raw visit-RMS, knot-overflow and change means since `previous`, plus maxima.
pub(super) fn add_scale(value: &mut Value, state: &State, previous: &Counters, anchors: u64) {
    let current = &state.counters;
    let readout = current.readout_rms_sum - previous.readout_rms_sum;
    let visit = current.visit_rms_sum - previous.visit_rms_sum;
    let overflow = current.knot_overflow_sum - previous.knot_overflow_sum;
    let change = current.change_sum - previous.change_sum;
    let embedding = state
        .shared_latent
        .as_ref()
        .map(|latent| latent.weights.as_slice());
    value["readout_rms"] = json!(average(readout, anchors));
    value["visit_rms"] = json!(average(visit, anchors));
    value["visit_rms_max"] = json!(current.visit_rms_max);
    value["knot_overflow"] = json!(average(overflow, anchors));
    // Mean relative E-plus-local-tile change per anchor: the scale restart thresholds compare.
    value["change_per_anchor"] = json!(average(change, anchors));
    value["master_rms"] = json!(current.master_rms_max);
    value["embedding_rms"] = json!(embedding.map(rms));
    value["embedding_spectral_ratio"] =
        json!(embedding.and_then(|weights| spectral_ratio(weights, state.geometry.width, VOCAB)));
}

/// Row-major sigma1/sigma2: power iteration on A^T A with one deflation, 32 steps in f64;
/// https://en.wikipedia.org/wiki/Power_iteration. `None` only when the second value is exactly
/// zero; a near-rank-1 matrix yields a large finite ratio.
pub(super) fn spectral_ratio(weights: &[f32], rows: usize, cols: usize) -> Option<f64> {
    if rows == 0 || cols == 0 || weights.len() != rows * cols {
        return None;
    }
    let (first, direction) = dominant(weights, cols, None)?;
    let (second, _) = dominant(weights, cols, Some(&direction))?;
    let ratio = first / second;
    ratio.is_finite().then_some(ratio)
}

/// Largest singular value and right vector of A, restricted orthogonal to `deflate`.
fn dominant(weights: &[f32], cols: usize, deflate: Option<&[f64]>) -> Option<(f64, Vec<f64>)> {
    // Golden-ratio Weyl start: deterministic and not orthogonal to sign-structured vectors.
    let start = (0..cols).map(|j| 1.0 + (j as f64 * 0.618_033_988_749_895).fract());
    let mut vector = unit(start.collect(), deflate)?;
    for _ in 0..32 {
        let image = multiply(weights, cols, &vector);
        let mut next = vec![0.0; cols];
        for (row, scale) in weights.chunks_exact(cols).zip(&image) {
            for (value, a) in next.iter_mut().zip(row) {
                *value += f64::from(*a) * scale;
            }
        }
        vector = unit(next, deflate)?;
    }
    let image = multiply(weights, cols, &vector);
    Some((dot(&image, &image).sqrt(), vector))
}

/// Projects out the unit `deflate` direction, then normalizes; `None` for a vanishing vector.
fn unit(mut vector: Vec<f64>, deflate: Option<&[f64]>) -> Option<Vec<f64>> {
    if let Some(direction) = deflate {
        let overlap = dot(&vector, direction);
        for (value, d) in vector.iter_mut().zip(direction) {
            *value -= overlap * d;
        }
    }
    let norm = dot(&vector, &vector).sqrt();
    (norm > 0.0 && norm.is_finite()).then(|| vector.iter().map(|v| v / norm).collect())
}

fn multiply(weights: &[f32], cols: usize, vector: &[f64]) -> Vec<f64> {
    weights
        .chunks_exact(cols)
        .map(|row| row.iter().zip(vector).map(|(a, v)| f64::from(*a) * v).sum())
        .collect()
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

fn average(sum: f64, count: u64) -> Option<f64> {
    (count > 0).then_some(sum / count.max(1) as f64)
}

fn difference<const N: usize>(current: &[u64; N], previous: &[u64; N]) -> Vec<u64> {
    current
        .iter()
        .zip(previous)
        .map(|(now, old)| now - old)
        .collect()
}

fn predictor(current: &Counters, previous: &Counters) -> Value {
    let fits = current.predictor_fits - previous.predictor_fits;
    let mse = current.predictor_mse_sum - previous.predictor_mse_sum;
    let zero = current.predictor_zero_mse_sum - previous.predictor_zero_mse_sum;
    let cosine = current.predictor_cosine_sum - previous.predictor_cosine_sum;
    json!({
        "fits": fits, "prefit_mse": average(mse, fits),
        "zero_mse": average(zero, fits), "prefit_cosine": average(cosine, fits),
        "relative_mse_improvement": (fits > 0 && zero > 1e-12).then_some(1.0 - mse / zero.max(1e-12)),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interval_predictor_metrics_use_only_new_prefit_observations() {
        let previous = Counters {
            predictor_fits: 8,
            predictor_mse_sum: 32.0,
            predictor_zero_mse_sum: 16.0,
            predictor_cosine_sum: 2.0,
            ..Counters::default()
        };
        let current = Counters {
            predictor_fits: 10,
            predictor_mse_sum: 33.0,
            predictor_zero_mse_sum: 20.0,
            predictor_cosine_sum: 3.5,
            ..Counters::default()
        };
        let interval = predictor(&current, &previous);
        assert_eq!(interval["fits"], 2);
        assert_eq!(interval["prefit_mse"], 0.5);
        assert_eq!(interval["zero_mse"], 2.0);
        assert_eq!(interval["prefit_cosine"], 0.75);
        assert_eq!(interval["relative_mse_improvement"], 0.75);
        assert!(predictor(&current, &current)["prefit_mse"].is_null());
        assert!(predictor(&current, &current)["relative_mse_improvement"].is_null());
    }

    #[test]
    fn spectral_ratio_recovers_a_known_rank_two_spectrum() {
        // A = 3 u1 v1^T + u2 v2^T; u1, v1 constant and u2, v2 alternating, all unit: ratio 3.
        let (rows, cols) = (128, 256);
        let sign = |i: usize| if i.is_multiple_of(2) { 1.0 } else { -1.0 };
        let norm = ((rows * cols) as f64).sqrt();
        let matrix: Vec<f32> = (0..rows * cols)
            .map(|index| ((3.0 + sign(index / cols) * sign(index % cols)) / norm) as f32)
            .collect();
        let ratio = spectral_ratio(&matrix, rows, cols).expect("rank-two ratio");
        assert!((ratio - 3.0).abs() <= 0.03, "ratio {ratio}");
    }
}
