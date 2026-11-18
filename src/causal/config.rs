use super::geometry::{Geometry, EVIDENCE_RULE, RADIUS, START_TILE, TILES};
use anyhow::Result;
use serde::{Deserialize, Serialize};

pub(super) const PROTOCOL: &str = "causal-prefix-v3";

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Config {
    pub schema: u32,
    pub protocol: String,
    pub profile: String,
    pub seed: u64,
    pub threads: usize,
    pub anchor_interval: usize,
    pub max_visits: usize,
    pub history_capacity: usize,
    pub context_bytes: usize,
    pub evidence_rule: String,
    pub route_credit: String,
    pub predictor_rule: String,
    pub anchor_batch: String,
    pub replay_selection: String,
}

impl Config {
    pub fn candidate(seed: u64) -> Self {
        Self {
            schema: 1,
            protocol: PROTOCOL.into(),
            profile: "candidate".into(),
            seed,
            threads: 1,
            anchor_interval: 4,
            max_visits: 4,
            history_capacity: 8,
            context_bytes: 1024,
            evidence_rule: EVIDENCE_RULE.into(),
            route_credit: "detached-u-experimental".into(),
            predictor_rule: "joint-adjacent-raw-row-mean-v1".into(),
            anchor_batch: "current-anchor-forward-v1".into(),
            replay_selection: "newest4-compatible-v1".into(),
        }
    }
}

pub(super) fn plan() -> Result<serde_json::Value> {
    let geometry = Geometry::candidate();
    Ok(
        serde_json::json!({"schema":1,"protocol":PROTOCOL,"config":Config::candidate(7),
        "geometry":geometry,"tiles":TILES,"radius":RADIUS,"start_tile":START_TILE,
        "payload_ledger":geometry.ledger(1024)?,"model_allocated":false,
        "physical_admission":false,"training_admitted":false,"useful_assistant_estimate_seconds":null,
        "scope":"experimental byte-prefix candidate; original 81M/287M targets remain separate",
        "required":"independent review, integrated checks, owned physical admission and representative quality trajectory"}),
    )
}
