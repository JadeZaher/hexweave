//! Bounded measurement policy outside the learning identity; see CONNECTED.md.

use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub(super) struct Options {
    pub budget_seconds: u64,
    pub retention_spans: usize,
    pub sample_bytes: usize,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            budget_seconds: 120,
            retention_spans: 4,
            sample_bytes: 64,
        }
    }
}

impl Options {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            (10..=600).contains(&self.budget_seconds),
            "connected evaluation budget must be in 10..=600 seconds"
        );
        ensure!(
            self.retention_spans <= 6,
            "connected retention spans must be in 0..=6"
        );
        ensure!(
            (1..=128).contains(&self.sample_bytes),
            "connected sample bytes must be in 1..=128"
        );
        Ok(())
    }
}
