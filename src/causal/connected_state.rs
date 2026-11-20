//! Complete connected-learning state at quiescent checkpoint boundaries; see AGENTS.md.

use super::connected_config::Options;
use super::connected_gain::Gain;
use super::connected_predictor::Predictor;
use super::connected_sections;
use super::connected_signal::Memory;
use super::connected_wave::Waves;
use super::cursor::Cursor;
use super::geometry::{Geometry, TILES, VOCAB};
use super::shared::Shared;
use super::state_codec::{decode_shared, encode_shared};
use super::tuning::Latent;
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[path = "connected_state_validate.rs"]
mod validation;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Counters {
    pub loss_sum: f64,
    pub predictor_fits: u64,
    pub predictor_mse_sum: f64,
    pub predictor_zero_mse_sum: f64,
    pub predictor_cosine_sum: f64,
    pub observed_records: u64,
    pub predicted_records: u64,
    pub local_updates: u64,
    pub local_visits: [u64; TILES],
    pub anchor_visits: [u64; TILES],
    pub route_updates: u64,
    pub local_groups: u64,
    pub collision_deferrals: u64,
    pub cross_lane_reads: u64,
    pub memory_reads: u64,
    pub retrieved_records: u64,
    pub tile_transitions: u64,
    pub embedding_transitions: u64,
    pub opposed_momentum: u64,
    pub local_base_norm_sum: f64,
    pub local_applied_norm_sum: f64,
    pub local_boosted: u64,
    pub local_damped: u64,
    #[serde(default)]
    pub readout_rms_sum: f64,
    /// Per anchor, the mean raw (pre-normalization) RMS of its observed visits, summed.
    #[serde(default)]
    pub visit_rms_sum: f64,
    /// Running maximum of any observed visit's raw RMS.
    #[serde(default)]
    pub visit_rms_max: f64,
    #[serde(default)]
    pub knot_overflow_sum: f64,
    /// Running per-tile maximum of `UpdateMetrics::max_rms`.
    #[serde(default)]
    pub master_rms_max: [f64; TILES],
    /// Sum over anchors of their relative E-plus-local-tile change (CONNECTED.md §Warm restarts).
    #[serde(default)]
    pub change_sum: f64,
    /// Anchor waves completed and tiles they stepped (CONNECTED.md §Anchor waves); omitted at 0.
    #[serde(default, skip_serializing_if = "Waves::is_unused")]
    pub waves: Waves,
}

pub(super) struct State {
    pub options: Options,
    pub geometry: Geometry,
    pub seed: u64,
    pub cursor: Cursor,
    pub anchors: u64,
    pub epoch: u64,
    pub shared: Shared,
    pub shared_latent: Option<Latent>,
    /// Adam state whose weights are the readout-bias master; None before the first anchor.
    pub bias_latent: Option<Latent>,
    pub predictor: Predictor,
    pub memory: Memory,
    pub gains: Vec<Gain>,
    pub baseline: f32,
    pub counters: Counters,
    pub ready: bool,
    /// Session-only prioritized replay buffer; never checkpointed (CONNECTED.md §Replay).
    pub replay: super::connected_replay::Buffer,
    /// Session-only live memory-read tally for the logs (CONNECTED.md §Memory reach).
    pub reach: super::connected_reach::Reach,
    /// Per-tile warm-restart state, present exactly when `options.restart` is.
    pub restarts: Option<Vec<super::connected_restart::TileRestart>>,
    /// Per-tile visit-balance estimates, present exactly when `options.balance` is.
    pub balance: Option<super::connected_balance::Estimate>,
    /// Shared sections and their queue, present exactly when `options.shared` is.
    pub sections: Option<super::connected_sections::Sections>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct AdamRecord {
    weights: Vec<f32>,
    moments: Vec<f32>,
    step: u64,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Record {
    schema: u32,
    geometry: Geometry,
    options: Options,
    seed: u64,
    cursor: Cursor,
    anchors: u64,
    epoch: u64,
    shared_version: u64,
    shared_latent: Option<AdamRecord>,
    bias: Vec<f32>,
    bias_latent: Option<AdamRecord>,
    predictor: Predictor,
    memory: Memory,
    gains: Vec<Gain>,
    baseline: f32,
    counters: Counters,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    restarts: Option<Vec<super::connected_restart::TileRestart>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    balance: Option<super::connected_balance::Estimate>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    shared: Option<super::connected_sections::Saved>,
}

impl AdamRecord {
    fn of(latent: &Latent) -> Self {
        Self {
            weights: latent.weights.clone(),
            moments: latent.moments.clone(),
            step: latent.step,
        }
    }

    fn latent(self) -> Latent {
        Latent {
            weights: self.weights,
            moments: self.moments,
            step: self.step,
        }
    }
}

impl State {
    pub fn new(geometry: Geometry, seed: u64, cursor: Cursor, options: &Options) -> Result<Self> {
        let mut shared = Shared::new(geometry, seed)?;
        shared.bias = Some(vec![0.0; VOCAB]);
        let mut state = Self {
            options: *options,
            geometry,
            seed,
            cursor,
            anchors: 0,
            epoch: 0,
            shared,
            shared_latent: None,
            bias_latent: None,
            predictor: Predictor::new(geometry, seed)?,
            memory: super::connected_balance::memory(geometry, options)?,
            gains: (0..TILES).map(|_| Gain::new()).collect(),
            baseline: (VOCAB as f32).ln(),
            counters: Counters::default(),
            ready: true,
            replay: Default::default(),
            reach: Default::default(),
            restarts: super::connected_restart::fresh(options),
            balance: super::connected_balance::fresh(options),
            sections: super::connected_sections::fresh(geometry, seed, options)?,
        };
        if options.precision == super::connected_precision::Precision::Fp32 {
            state.shared_latent = Some(super::connected_adam::fresh(
                &state.shared.embedding,
                0,
                state.shared.embedding.len(),
            )?);
            state.attach_view(super::connected_precision::View::Fp32)?;
        }
        Ok(state)
    }

    /// Set a fresh run's readout bias (VOCAB log-probabilities) and its entropy as the baseline.
    pub fn install_readout_prior(&mut self, prior: &[f32]) -> Result<()> {
        ensure!(
            self.anchors == 0 && self.bias_latent.is_none(),
            "connected readout prior after training began"
        );
        super::norm::vector(prior, VOCAB)?;
        // -sum exp(p) p: the expected loss of predicting the prior itself; see CONNECTED.md v4.
        let entropy: f64 = prior
            .iter()
            .map(|&p| -f64::from(p).exp() * f64::from(p))
            .sum();
        let baseline = entropy as f32;
        ensure!(baseline.is_finite(), "nonfinite connected readout prior");
        self.shared.bias = Some(prior.to_vec());
        self.baseline = baseline;
        Ok(())
    }

    pub fn attach_view(&mut self, mode: super::connected_precision::View) -> Result<()> {
        if self.options.precision == super::connected_precision::Precision::Fp32 {
            let latent = self
                .shared_latent
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("missing FP32 shared master"))?;
            super::connected_precision::attach_shared(&mut self.shared, latent, mode)?;
        }
        super::connected_sections::attach_view(&mut self.sections, self.options.precision, mode)
    }

    pub fn payloads(&self) -> Result<BTreeMap<String, Vec<u8>>> {
        self.validate()?;
        let (sections, shared) = super::connected_sections::save(self)?;
        let record = Record {
            schema: 1,
            geometry: self.geometry,
            options: self.options,
            seed: self.seed,
            cursor: self.cursor.clone(),
            anchors: self.anchors,
            epoch: self.epoch,
            shared_latent: self.shared_latent.as_ref().map(AdamRecord::of),
            shared_version: self.shared.version,
            bias: self.shared.bias.clone().unwrap_or_default(),
            bias_latent: self.bias_latent.as_ref().map(AdamRecord::of),
            predictor: self.predictor.clone(),
            memory: self.memory.clone(),
            gains: self.gains.clone(),
            baseline: self.baseline,
            counters: self.counters.clone(),
            restarts: self.restarts.clone(),
            balance: self.balance.clone(),
            shared,
        };
        let mut payloads = BTreeMap::from([
            ("connected.json".into(), serde_json::to_vec(&record)?),
            ("shared.bin".into(), encode_shared(&self.shared)?),
        ]);
        payloads.extend(sections);
        Ok(payloads)
    }

    pub fn restore(
        payloads: &BTreeMap<String, Vec<u8>>,
        geometry: Geometry,
        seed: u64,
    ) -> Result<Self> {
        let raw = payloads
            .get("connected.json")
            .ok_or_else(|| anyhow::anyhow!("missing connected state"))?;
        let record: Record = serde_json::from_slice(raw)?;
        let expected = 2 + connected_sections::count(&record.options);
        ensure!(payloads.len() == expected, "unexpected connected payloads");
        ensure!(
            record.schema == 1 && record.geometry == geometry && record.seed == seed,
            "connected state identity mismatch"
        );
        let packed = payloads
            .get("shared.bin")
            .ok_or_else(|| anyhow::anyhow!("missing connected embedding"))?;
        let mut shared = decode_shared(packed, geometry, record.shared_version)?;
        shared.bias = Some(record.bias);
        let sections =
            connected_sections::load(payloads, record.shared, geometry, &record.options)?;
        let mut state = Self {
            geometry,
            options: record.options,
            seed,
            cursor: record.cursor,
            anchors: record.anchors,
            epoch: record.epoch,
            shared,
            shared_latent: record.shared_latent.map(AdamRecord::latent),
            bias_latent: record.bias_latent.map(AdamRecord::latent),
            predictor: record.predictor,
            memory: record.memory,
            gains: record.gains,
            baseline: record.baseline,
            counters: record.counters,
            ready: true,
            replay: Default::default(),
            reach: Default::default(),
            restarts: record.restarts,
            balance: record.balance,
            sections,
        };
        state.attach_view(super::connected_precision::View::Fp32)?;
        state.validate()?;
        Ok(state)
    }
}
