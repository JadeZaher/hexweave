use crate::ternary::TernaryMatrix;
use anyhow::{ensure, Result};
use serde::Serialize;
use serde_json::{json, Value};

#[derive(Clone, Default, Serialize)]
pub(super) struct Metrics {
    pub frames: usize,
    pub unique_tile_commits: usize,
    pub repeated_visit_commits: usize,
    pub coordinate_opportunities: usize,
    pub task_nonzero: usize,
    pub route_nonzero: usize,
    pub both_nonzero: usize,
    pub opposite_sign: usize,
    pub task_transitions: usize,
    pub coupled_transitions: usize,
    pub actual_transitions: usize,
    pub applied_transitions: usize,
    pub different_results: usize,
    pub different_commits: usize,
    pub task_opposing: usize,
    pub suppressed_task: usize,
    pub new_where_task_zero: usize,
    pub task_route_dot: f64,
    pub task_norm_squared: f64,
    pub route_norm_squared: f64,
    pub task_descent: f64,
    pub coupled_projection: f64,
    pub deficit: f64,
    pub residual_max: f64,
}

impl Metrics {
    pub fn merge(&mut self, value: &Self) {
        self.frames += value.frames;
        self.unique_tile_commits += value.unique_tile_commits;
        self.repeated_visit_commits += value.repeated_visit_commits;
        self.coordinate_opportunities += value.coordinate_opportunities;
        self.task_nonzero += value.task_nonzero;
        self.route_nonzero += value.route_nonzero;
        self.both_nonzero += value.both_nonzero;
        self.opposite_sign += value.opposite_sign;
        self.task_transitions += value.task_transitions;
        self.coupled_transitions += value.coupled_transitions;
        self.actual_transitions += value.actual_transitions;
        self.applied_transitions += value.applied_transitions;
        self.different_results += value.different_results;
        self.different_commits += value.different_commits;
        self.task_opposing += value.task_opposing;
        self.suppressed_task += value.suppressed_task;
        self.new_where_task_zero += value.new_where_task_zero;
        self.task_route_dot += value.task_route_dot;
        self.task_norm_squared += value.task_norm_squared;
        self.route_norm_squared += value.route_norm_squared;
        self.task_descent += value.task_descent;
        self.coupled_projection += value.coupled_projection;
        self.deficit += value.deficit;
        self.residual_max = self.residual_max.max(value.residual_max);
    }

    pub fn report(&self) -> Value {
        json!({"raw": self, "ratios": {
            "opposite_over_both_nonzero": ratio(self.opposite_sign as f64, self.both_nonzero as f64),
            "different_over_coordinates": ratio(self.different_results as f64, self.coordinate_opportunities as f64),
            "deficit_over_task_descent": ratio(self.deficit, self.task_descent),
            "task_route_cosine": ratio(self.task_route_dot, (self.task_norm_squared * self.route_norm_squared).sqrt())
        }})
    }
}

pub(super) fn measure(
    weights: &TernaryMatrix,
    task: &[f32],
    route: &[f32],
    coupled: &[f32],
    actual: &[f32],
    frames: usize,
) -> Result<Metrics> {
    ensure!(frames > 0 && frames <= 3, "invalid diagnostic frame count");
    for values in [task, route, coupled, actual] {
        ensure!(
            values.len() == weights.len(),
            "credit diagnostic shape mismatch"
        );
        ensure!(
            values.iter().all(|x| x.is_finite()),
            "nonfinite shadow credit"
        );
    }
    let mut metrics = Metrics {
        frames,
        unique_tile_commits: 1,
        repeated_visit_commits: usize::from(frames > 1),
        coordinate_opportunities: weights.len(),
        ..Metrics::default()
    };
    for index in 0..weights.len() {
        let w = weights.value(index)?;
        coordinate(
            &mut metrics,
            w,
            f64::from(task[index]),
            f64::from(route[index]),
            f64::from(coupled[index]),
            f64::from(actual[index]),
        );
    }
    metrics.different_commits = usize::from(metrics.different_results > 0);
    for value in [
        metrics.task_route_dot,
        metrics.task_norm_squared,
        metrics.route_norm_squared,
        metrics.task_descent,
        metrics.coupled_projection,
        metrics.deficit,
        metrics.residual_max,
    ] {
        ensure!(value.is_finite(), "nonfinite credit diagnostic sum");
    }
    Ok(metrics)
}

fn coordinate(metrics: &mut Metrics, w: i8, t: f64, r: f64, c: f64, a: f64) {
    let (dt, dc, da) = (delta(w, t), delta(w, c), delta(w, a));
    metrics.task_nonzero += usize::from(t != 0.0);
    metrics.route_nonzero += usize::from(r != 0.0);
    metrics.both_nonzero += usize::from(t != 0.0 && r != 0.0);
    metrics.opposite_sign += usize::from(t * r < 0.0);
    metrics.task_transitions += usize::from(dt != 0);
    metrics.coupled_transitions += usize::from(dc != 0);
    metrics.actual_transitions += usize::from(da != 0);
    metrics.different_results += usize::from(dt != dc);
    metrics.task_opposing += usize::from(t * f64::from(dc) > 0.0);
    metrics.suppressed_task += usize::from(dt != 0 && dc == 0);
    metrics.new_where_task_zero += usize::from(t == 0.0 && dc != 0);
    metrics.task_route_dot += t * r;
    metrics.task_norm_squared += t * t;
    metrics.route_norm_squared += r * r;
    metrics.task_descent -= t * f64::from(dt);
    metrics.coupled_projection += t * f64::from(dc);
    metrics.deficit += t * f64::from(dc - dt);
    metrics.residual_max = metrics.residual_max.max((c - t - r).abs());
}

fn delta(weight: i8, gradient: f64) -> i8 {
    let sign = if gradient > 0.0 {
        1
    } else if gradient < 0.0 {
        -1
    } else {
        0
    };
    (weight - sign).clamp(-1, 1) - weight
}

pub(super) fn ratio(numerator: f64, denominator: f64) -> Option<f64> {
    (denominator > 0.0).then(|| numerator / denominator)
}
