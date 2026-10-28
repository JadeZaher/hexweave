use super::credit_metrics::{measure, ratio, Metrics};
use super::gradients::Gradients;
use super::node::{Frame, Node};
use super::{RouteCredit, TILES, WIDTH};
use anyhow::{ensure, Result};
use serde_json::{json, Value};

#[derive(Default)]
pub(super) struct Diagnostic {
    pub total: Metrics,
    pub tiles: [Metrics; TILES],
    pub anchors: usize,
    pub anchors_with_different_results: usize,
}

impl Diagnostic {
    pub fn commit(&mut self, id: usize, mut value: Metrics, applied: usize) -> Result<bool> {
        ensure!(
            value.actual_transitions == applied,
            "predicted/actual U transition mismatch"
        );
        value.applied_transitions = applied;
        let different = value.different_results > 0;
        self.total.merge(&value);
        self.tiles[id].merge(&value);
        Ok(different)
    }

    pub fn report(&self) -> Value {
        json!({"schema": 1, "scope": "frozen anchor unique-tile commits only",
            "coordinates_per_commit": 2 * WIDTH * WIDTH,
            "anchors": self.anchors, "anchors_with_different_results": self.anchors_with_different_results,
            "anchor_difference_fraction": ratio(self.anchors_with_different_results as f64, self.anchors as f64),
            "total": self.total.report(),
            "per_tile": self.tiles.iter().enumerate().map(|(id, value)| json!({"tile": id, "metrics": value.report()})).collect::<Vec<_>>()})
    }
}

pub(super) fn observe(
    node: &Node,
    frames: &[Frame],
    corrections: &[Vec<f32>],
    actual: &Gradients,
    request: (usize, f32, RouteCredit),
) -> Result<Metrics> {
    let (id, advantage, mode) = request;
    let mut task = Gradients::zeros(node);
    let mut coupled = Gradients::zeros(node);
    let mut route = Gradients::zeros(node);
    let mut frame_count = 0;
    for (frame, correction) in frames.iter().zip(corrections) {
        if frame.id == id {
            task.add(
                node,
                frame,
                correction,
                Some(advantage),
                RouteCredit::RouteToUDetached,
            )?;
            coupled.add(
                node,
                frame,
                correction,
                Some(advantage),
                RouteCredit::Coupled,
            )?;
            route.add(
                node,
                frame,
                &[0.0; WIDTH],
                Some(advantage),
                RouteCredit::Coupled,
            )?;
            frame_count += 1;
        }
    }
    let expected = match mode {
        RouteCredit::Coupled => &coupled.pulse_context,
        RouteCredit::RouteToUDetached => &task.pulse_context,
    };
    ensure!(
        &actual.pulse_context == expected,
        "shadow changed actual U gradient arithmetic"
    );
    measure(
        &node.pulse.context,
        &task.pulse_context,
        &route.pulse_context,
        &coupled.pulse_context,
        &actual.pulse_context,
        frame_count,
    )
}
