//! Prequential train-ahead accumulator (corpus runs only); see AGENTS.md §core.

use super::core_corpus::AHEAD_ANCHORS;

/// Online-adaptation loss over the first `AHEAD_ANCHORS` anchors trained since the last
/// evaluation; in-memory only, never persisted, checkpointed or part of the run identity.
#[derive(Default)]
pub(super) struct Ahead {
    loss_sum: f64,
    anchors: u64,
}

impl Ahead {
    /// Add one just-committed anchor's loss, while fewer than `AHEAD_ANCHORS` are accumulated.
    pub fn add(&mut self, loss: f32) {
        if self.anchors < AHEAD_ANCHORS {
            self.loss_sum += f64::from(loss);
            self.anchors += 1;
        }
    }

    /// Mean bits per byte over exactly `AHEAD_ANCHORS` anchors, or None short of that; resets.
    pub fn take(&mut self) -> Option<f64> {
        let result = (self.anchors == AHEAD_ANCHORS)
            .then(|| self.loss_sum / self.anchors as f64 / std::f64::consts::LN_2);
        *self = Self::default();
        result
    }
}
