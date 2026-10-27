//! Bounded ternary pulse experiment; see `AGENTS.md` for the learning contract.

mod credit;
mod execution;
mod history;
mod math;
mod node;
mod report;
mod residency;
mod runner;
mod state;

#[cfg(test)]
mod contracts;

use anyhow::{ensure, Result};
use serde::Serialize;
use serde_json::Value;

pub(crate) const WIDTH: usize = 8;
pub(crate) const VOCAB: usize = 16;
pub(crate) const TILES: usize = 7;
pub(crate) const DEPTH: usize = 3;
pub(crate) const FEATURES: usize = 3 * WIDTH + 2;
pub(crate) const THRESHOLD: i16 = 1;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Mode {
    Pulse,
    AnchorOnly,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct Options {
    pub steps: usize,
    pub seed: u64,
    pub mode: Mode,
    pub resident_limit: usize,
    pub anchor_interval: usize,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            steps: 32,
            seed: 7,
            mode: Mode::Pulse,
            resident_limit: 1,
            anchor_interval: 4,
        }
    }
}

impl Options {
    pub fn validate(self) -> Result<()> {
        ensure!((1..=256).contains(&self.steps), "steps must be in 1..=256");
        ensure!(
            (1..=3).contains(&self.resident_limit),
            "resident_limit must be in 1..=3"
        );
        ensure!(
            (1..=16).contains(&self.anchor_interval),
            "anchor_interval must be in 1..=16"
        );
        Ok(())
    }
}

/// Run the small synthetic-ID experiment with the default memory bounds.
pub fn run(steps: usize, seed: u64, mode: Mode) -> Result<Value> {
    run_with_options(Options {
        steps,
        seed,
        mode,
        ..Options::default()
    })
}

/// Run one finite experiment; this does not launch a production trainer.
pub fn run_with_options(options: Options) -> Result<Value> {
    options.validate()?;
    runner::run(options)
}
