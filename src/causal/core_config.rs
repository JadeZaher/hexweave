//! Core run configuration: data kinds, validation and the persisted run identity; see AGENTS.md §core.

use super::core_checkpoint::PROTOCOL;
use super::geometry::Geometry;
use super::storage_paths::digest;
use super::tuning::Knobs;
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::path::{Component, PathBuf};

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields, tag = "kind", rename_all = "snake_case")]
pub(super) enum Data {
    /// Source-grounded QA rows (train split) with dev-split assessment.
    Science {
        root: PathBuf,
        manifest_sha256: String,
        /// Re-deal each epoch's SOURCE values and citations (`core_redeal`); absent means off.
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        redeal_citations: bool,
    },
    /// Unbounded retrieval drill over the retained-text assay templates.
    Drill { root: PathBuf },
    /// Next-byte pretraining over a pinned plain-text corpus (`core_corpus`).
    Corpus {
        root: PathBuf,
        manifest_sha256: String,
        /// Bytes of context per prediction, at most the 1,023 the candidate's context allows.
        window: usize,
    },
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Config {
    pub schema: u32,
    pub out: PathBuf,
    pub data: Data,
    pub seed: u64,
    pub rate: f64,
    pub anchors: u64,
    pub wall_seconds: u64,
    pub checkpoint_every: u64,
    pub eval_every: u64,
    pub eval_cases: usize,
    pub resume: bool,
    /// Compute with f32 latent weights instead of trits (`weights.rs`); absent means ternary.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub float_forward: bool,
    /// Zero history input, skipping records/predictor training (`tuning::Knobs::history`); absent means on.
    #[serde(default = "default_true", skip_serializing_if = "is_true")]
    pub history: bool,
    /// Held-out probe span count (`core_eval::heldout_nll`); absent means 16. Not part of the run
    /// identity; stamped in `best/selection.json` so selections at a different sample size are
    /// never compared (`core_best`).
    #[serde(
        default = "default_eval_spans",
        skip_serializing_if = "is_default_eval_spans"
    )]
    pub eval_spans: usize,
    #[serde(default = "default_batch")]
    pub batch: u32,
    #[serde(default)]
    pub decay: super::core_schedule::Decay,
    #[serde(default)]
    pub explore: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub connected: Option<super::connected_config::Options>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub connected_eval: Option<super::connected_eval_config::Options>,
    /// Compiled tile count the run needs; absent means 7. Checked, never run identity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tiles: Option<usize>,
}

fn default_batch() -> u32 {
    1
}

fn default_true() -> bool {
    true
}

fn is_true(value: &bool) -> bool {
    *value
}

fn default_eval_spans() -> usize {
    16
}

fn is_default_eval_spans(value: &usize) -> bool {
    *value == default_eval_spans()
}

impl Config {
    pub fn validate(&self) -> Result<()> {
        if let Some(mismatch) = super::geometry::tile_mismatch(self.tiles.unwrap_or(7)) {
            anyhow::bail!("run config needs {mismatch}");
        }
        super::core_schedule::validate(self)?;
        if let Some(options) = self.connected {
            options.validate_run(self)?;
            ensure!(
                self.eval_spans <= 128 && self.eval_cases <= 4,
                "connected evaluation supports at most 128 spans and four samples"
            );
        }
        if let Some(options) = &self.connected_eval {
            ensure!(
                self.connected.is_some(),
                "connected_eval requires connected training"
            );
            options.validate()?;
        }
        ensure!(self.schema == 1, "core config schema must be 1");
        let parts: Vec<Component<'_>> = self.out.components().collect();
        ensure!(
            matches!(parts.as_slice(), [Component::Normal(work), Component::Normal(runs), Component::Normal(_)]
                if *work == "work" && *runs == "core-runs"),
            "out must be work/core-runs/<name>"
        );
        ensure!(
            self.rate.is_finite() && self.rate > 0.0 && self.rate <= 1.0,
            "rate must be in (0, 1]"
        );
        ensure!(
            self.anchors > 0
                && self.checkpoint_every > 0
                && self.eval_every > 0
                && self.wall_seconds > 0,
            "anchors, checkpoint_every, eval_every and wall_seconds must be positive"
        );
        ensure!(
            (1..=32).contains(&self.eval_cases),
            "eval_cases must be in 1..=32"
        );
        ensure!(self.eval_spans >= 1, "eval_spans must be at least 1");
        if let Data::Corpus { window, .. } = &self.data {
            ensure!(
                (32..=1023).contains(window),
                "corpus window must be in 32..=1023"
            );
        }
        Ok(())
    }

    /// The numerics profile this run installs, checks and resumes under.
    pub fn knobs(&self) -> Knobs {
        Knobs {
            float_forward: self.float_forward,
            history: self.history,
            batch: self.batch,
            decay: self.decay,
            explore: self.explore,
            history_ticks: 8,
            fit_predictor: false,
            connected: self.connected.is_some(),
            connected_fp32: self
                .connected
                .is_some_and(|o| o.precision == super::connected_precision::Precision::Fp32),
            slice_mode: self
                .connected
                .map_or(super::connected_config::SliceMode::Full, |o| o.slice_mode),
            visit_norm: self.connected.is_none_or(|o| o.visit_norm),
            ..Knobs::core(self.rate, self.seed)
        }
    }

    /// Numerics identity: resuming requires the same protocol, geometry, data, seed, rate and knobs.
    pub fn identity(&self) -> Result<String> {
        let protocol = if let Some(options) = self.connected {
            options.precision.protocol()
        } else {
            PROTOCOL
        };
        let mut value = json!({ "protocol": protocol, "geometry": Geometry::candidate(),
                            "data": &self.data, "seed": self.seed, "rate": self.rate,
                            "numerics": self.knobs() });
        if let Some(options) = self.connected {
            value["connected"] = serde_json::to_value(options)?;
        }
        // Seven-tile identities predate this key, so only other tile counts record it.
        if super::geometry::TILES != 7 {
            value["tiles"] = json!(super::geometry::TILES);
        }
        Ok(digest(&serde_json::to_vec(&value)?))
    }
}

/// Run identity of a config document, for the identity contracts.
#[cfg(test)]
pub(super) fn identity_of(config: &[u8]) -> Result<String> {
    serde_json::from_slice::<Config>(config)?.identity()
}
