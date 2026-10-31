//! Prequential train-ahead accumulator (corpus runs only); see AGENTS.md §core.

use super::core_corpus::AHEAD_ANCHORS;

/// Online-adaptation accumulator for one window, paired with its start anchor; in-memory only, never persisted or checkpointed.
#[derive(Default)]
pub(super) struct Ahead {
    loss_sum: f64,
    anchors: u64,
    start: u64,
    disarmed: bool,
    short: bool,
}

impl Ahead {
    /// Arm a fresh window at `start` anchor; `disarmed` nulls this one window (a resume's cursor has no logged frozen probe to pair with), `short` nulls every window.
    pub fn arm(start: u64, disarmed: bool, short: bool) -> Self {
        Self {
            start,
            disarmed,
            short,
            ..Self::default()
        }
    }

    /// Whether this accumulator nulls every window (a corpus shorter than `AHEAD_ANCHORS`).
    pub fn short(&self) -> bool {
        self.short
    }

    /// Add one just-committed anchor's loss, while fewer than `AHEAD_ANCHORS` are accumulated.
    pub fn add(&mut self, loss: f32) {
        if !self.short && self.anchors < AHEAD_ANCHORS {
            self.loss_sum += f64::from(loss);
            self.anchors += 1;
        }
    }

    /// Mean bits per byte over exactly `AHEAD_ANCHORS` anchors paired with the window's start anchor; both null short of that, when disarmed, or on a short corpus.
    pub fn take(&self) -> (Option<f64>, Option<u64>) {
        let ready = !self.disarmed && !self.short && self.anchors == AHEAD_ANCHORS;
        (
            ready.then(|| self.loss_sum / self.anchors as f64 / std::f64::consts::LN_2),
            ready.then_some(self.start),
        )
    }
}
