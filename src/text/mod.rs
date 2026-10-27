//! Bounded byte prediction with one live tile; see text/AGENTS.md.

mod credit;
mod credit_diagnostic;
mod credit_metrics;
mod data;
mod execution;
mod gradients;
mod history;
mod math;
mod node;
mod report;
mod runner;
mod shared;
mod storage;
mod training_pulse;

#[cfg(test)]
mod credit_diagnostic_tests;
#[cfg(test)]
mod tests;

use anyhow::{ensure, Result};
use serde::Serialize;
use serde_json::Value;
use std::path::Path;

pub(super) const WIDTH: usize = 16;
pub(super) const VOCAB: usize = 256;
pub(super) const CONTEXT: usize = 8;
pub(super) const TILES: usize = 7;
pub(super) const DEPTH: usize = 3;
pub(super) const FEATURES: usize = WIDTH * 3 + 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Mode {
    Pulse,
    AnchorOnly,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum RouteCredit {
    Coupled,
    RouteToUDetached,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Profile {
    Fixture,
    T2Credit,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum CreditDiagnostics {
    Off,
    On,
}

#[derive(Clone, Debug, Serialize)]
pub struct Options {
    pub steps: usize,
    pub seed: u64,
    pub anchor_interval: usize,
    pub mode: Mode,
    pub route_credit: RouteCredit,
    pub profile: Profile,
    pub credit_diagnostics: CreditDiagnostics,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            steps: 256,
            seed: 7,
            anchor_interval: 4,
            mode: Mode::Pulse,
            route_credit: RouteCredit::Coupled,
            profile: Profile::Fixture,
            credit_diagnostics: CreditDiagnostics::Off,
        }
    }
}

impl Options {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.credit_diagnostics == CreditDiagnostics::Off || self.profile == Profile::T2Credit,
            "credit diagnostics require explicit t2-credit profile"
        );
        ensure!(
            self.profile != Profile::T2Credit || self.mode == Mode::Pulse,
            "T2 requires pulse mode"
        );
        ensure!(
            (1..=1024).contains(&self.steps),
            "text steps must be 1..=1024"
        );
        ensure!(
            (1..=16).contains(&self.anchor_interval),
            "anchor interval must be 1..=16"
        );
        Ok(())
    }
}

/// Train a bounded byte fixture, preserving all artifacts in a fresh owned directory.
pub fn run(options: &Options, train: &Path, valid: &Path, work_dir: &Path) -> Result<Value> {
    options.validate()?;
    let data = data::Corpus::load(train, valid)?;
    data.validate_profile(options.profile)?;
    runner::run(options, &data, work_dir)
}
